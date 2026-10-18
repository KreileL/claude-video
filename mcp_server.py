#!/usr/bin/env python3
"""watch MCP server (stdio): give Claude a video input.

Wraps the /watch skill — it does not reimplement it. Each call:

1. validates the URL (http/https, public addresses only — see netguard.py);
2. checks that the local Whisper service (watch-whisper) is up;
3. asks yt-dlp for the metadata only, to enforce the duration limit and to turn
   an Instagram login wall into a clear error before anything is downloaded;
4. runs skills/watch/scripts/watch.py in a fresh work dir, with every yt-dlp
   connection forced through an egress proxy that re-checks each destination
   (redirects and CDN hosts included);
5. returns one TextContent (metadata, transcript, frame list) plus the frames
   as JPEG ImageContent in temporal order, within an image budget;
6. deletes the work dir.

Runs inside the watch-mcp container (see Dockerfile / README "MCP local").
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent

import netguard

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from frames import format_time  # noqa: E402
from transcribe import format_transcript  # noqa: E402
from whisper import local_endpoint  # noqa: E402

MAX_DURATION_SECONDS = 15 * 60
MAX_FILESIZE = "500M"
# Hard cap enforced by the egress proxy on every byte yt-dlp receives (video,
# audio, subtitles, fragments). --max-filesize above only works when the server
# sends Content-Length.
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
# ffmpeg input protocols when yt-dlp delegates a download to it: http(s) go
# through the egress proxy (yt-dlp sets http_proxy); ftp, rtmp, rtsp, file and
# the rest would not, so they are refused.
FFMPEG_INPUT_WHITELIST = "http,https,tls,tcp,crypto"
# Images are budgeted on their base64 size — what actually travels in the
# tool result. A 90 s reel at the default 24 frames lands well under it.
IMAGE_BUDGET_BYTES = 800 * 1024
FRAME_LONG_SIDE = 512
MAX_FRAMES_LIMIT = 100
DESCRIPTION_LIMIT = 3000
COOKIES_PATH = Path(os.environ.get("WATCH_MCP_COOKIES", "/secrets/cookies.txt"))
CALL_TIMEOUT = float(os.environ.get("WATCH_MCP_TIMEOUT", "900"))
YTDLP_TIMEOUT = float(os.environ.get("WATCH_MCP_YTDLP_TIMEOUT", "300"))
WHISPER_TIMEOUT = os.environ.get("WATCH_WHISPER_TIMEOUT", "600")
PROGRESS_EVERY = 10.0

STACK_HINT = "rode docker compose -p watch-mcp up -d"
COOKIE_JAR_ERROR = (
    "O yt-dlp não conseguiu usar o cookies.txt (formato Netscape inválido?). Verifique o arquivo."
)
TIME_RE = re.compile(r"^\d{1,2}(:\d{1,2}){0,2}(\.\d{1,3})?$")
FRAME_NAME_RE = re.compile(r"^(frame|cue)_\d{4}\.jpg$")
JPEG_MAGIC = b"\xff\xd8\xff"
NETSCAPE_LINE_RE = re.compile(r"^[^\t\n]*(?:\t[^\t\n]*){6}$", re.M)
COOKIE_HEADER_RE = re.compile(r"(?im)^(.*?\b(?:set-)?cookies?\s*[:=]).*$")
FRAME_LINE_RE = re.compile(r"^- `(?P<path>[^`]+)` \(t=(?P<t>[0-9:]+), reason=[^)]*\)$")
DURATION_LINE_RE = re.compile(r"^- \*\*Duration:\*\* \S+ \((?P<s>[0-9.]+)s\)$", re.M)
YOUTUBE_BOT_MARKERS = ("confirm you’re not a bot", "confirm you're not a bot", "http error 429")
INSTAGRAM_LOGIN_MARKERS = (
    "login required",
    "log in",
    "login page",
    "--cookies",
    "empty media response",
    "not available",
    "rate-limit",
)

mcp = MCPServer(
    "watch",
    version="0.2.0",
    instructions=(
        "Tools to watch a video from a URL (Instagram, TikTok, YouTube, ...): "
        "watch_video returns metadata, a timestamped transcript and frames; "
        "transcribe_video returns only the transcript."
    ),
)


@dataclass
class Frame:
    path: Path
    timestamp: str


@dataclass
class RunResult:
    info: dict
    frames: list[Frame] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    transcript_source: str | None = None
    warnings: list[str] = field(default_factory=list)
    secrets: frozenset[str] = frozenset()


# ── input validation ────────────────────────────────────────────────────────


def check_time(name: str, value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    if not TIME_RE.match(value):
        raise ToolError(f"{name} inválido: use SS, MM:SS ou HH:MM:SS")
    return value


def check_url(url: str) -> str:
    try:
        return netguard.validate_url(url)
    except netguard.BlockedURL as exc:
        raise ToolError(f"URL recusada: {exc}") from None


def frame_width(width: int | None, height: int | None) -> int:
    """watch.py's --resolution is a width; pick it so the LONG side is 512px.

    A vertical 1080x1920 reel becomes 288x512 instead of 512x910, which keeps
    24 frames inside the image budget.
    """
    if not width or not height or width >= height:
        return FRAME_LONG_SIDE
    return max(64, int(round(FRAME_LONG_SIDE * width / height / 2)) * 2)


# ── whisper health ──────────────────────────────────────────────────────────


def whisper_health_url() -> str:
    parts = urlsplit(local_endpoint())
    return f"{parts.scheme}://{parts.netloc}/health"


def check_whisper() -> None:
    url = whisper_health_url()
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                return
    except Exception:
        pass
    raise ToolError(
        f"O servidor Whisper local (watch-whisper) não está respondendo em {url}: {STACK_HINT}"
    )


# ── subprocesses ────────────────────────────────────────────────────────────


def ytdlp_args(proxy_url: str, cookies: Path | None) -> list[str]:
    args = [
        "--ignore-config",
        "--no-cache-dir",
        "--proxy", proxy_url,
        "--socket-timeout", "30",
        "--max-filesize", MAX_FILESIZE,
        "--match-filter", f"duration <=? {MAX_DURATION_SECONDS} & !is_live",
        "--downloader-args", f"ffmpeg_i:-protocol_whitelist {FFMPEG_INPUT_WHITELIST}",
    ]
    if cookies is not None:
        args += ["--cookies", str(cookies)]
    return args


def cookie_secrets(path: Path | None) -> frozenset[str]:
    """The cookie values in a Netscape cookies.txt, to redact wherever they appear."""
    if path is None or not path.is_file():
        return frozenset()
    values = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.removeprefix("#HttpOnly_")
        fields = line.split("\t")
        if not line.startswith("#") and len(fields) >= 7 and len(fields[6].strip()) >= 4:
            values.add(fields[6].strip())
    return frozenset(values)


def secrets_from_args(args: list[str]) -> frozenset[str]:
    if "--cookies" in args:
        return cookie_secrets(Path(args[args.index("--cookies") + 1]))
    return frozenset()


def scrub(text: str, secrets: frozenset[str] = frozenset()) -> str:
    """Remove cookie material from text we show: the jar's own values wherever
    they appear, whole Netscape cookie lines, and Cookie/Set-Cookie headers."""
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, "<redacted>")
    text = NETSCAPE_LINE_RE.sub("<cookie line redacted>", text)
    return COOKIE_HEADER_RE.sub(r"\1 <redacted>", text)


def tail(text: str, secrets: frozenset[str] = frozenset(), lines: int = 8) -> str:
    kept = [ln for ln in text.strip().splitlines() if ln.strip()][-lines:]
    return scrub("\n".join(kept), secrets)


def cookie_jar_problem(stderr: str) -> bool:
    low = stderr.lower()
    return "cookie" in low and any(w in low for w in ("invalid", "failed to load", "could not load", "malformed"))


async def run_process(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: float,
    ctx: Context | None,
    label: str,
) -> tuple[int, str, str]:
    """Run argv (never through a shell), reporting progress while it runs.

    On timeout the whole process group is killed — watch.py spawns yt-dlp and
    ffmpeg, which must not outlive the call.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(cwd),
        start_new_session=os.name == "posix",
    )
    communicate = asyncio.ensure_future(proc.communicate())
    started = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise asyncio.TimeoutError
            done, _ = await asyncio.wait({communicate}, timeout=min(PROGRESS_EVERY, remaining))
            if done:
                stdout, stderr = communicate.result()
                return proc.returncode or 0, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")
            if ctx is not None:
                elapsed = time.monotonic() - started
                try:
                    await ctx.report_progress(elapsed, None, f"{label} ({elapsed:.0f}s)")
                except Exception:
                    pass
    except asyncio.TimeoutError:
        _kill(proc)
        try:  # with the group dead the pipes hit EOF: reap it, close the transport
            await asyncio.wait_for(communicate, 5)
        except (asyncio.TimeoutError, ProcessLookupError):
            communicate.cancel()
        raise ToolError(f"Tempo limite excedido ({timeout:.0f}s) em: {label}") from None
    except BaseException:
        _kill(proc)
        communicate.cancel()
        raise
    finally:
        # Even after a clean exit, strays (an ffmpeg left by yt-dlp) must not
        # outlive the call.
        _kill(proc)


def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill the child's whole process group (POSIX), even if the child itself
    already exited: its descendants can still hold the pipes open.

    On Windows (local development only) just the child is killed.
    """
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        elif proc.returncode is None:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def child_env(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("WATCH_WHISPER_TIMEOUT", WHISPER_TIMEOUT)
    # watch.py's report is parsed as UTF-8 whatever the platform locale is.
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def login_wall_message(url: str, stderr: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower()
    if not (host == "instagram.com" or host.endswith(".instagram.com")):
        return None
    low = stderr.lower()
    if not any(marker in low for marker in INSTAGRAM_LOGIN_MARKERS):
        return None
    return (
        "O Instagram exigiu login para este conteúdo. Para liberar: exporte os cookies de uma "
        "conta SECUNDÁRIA do Instagram em formato Netscape (cookies.txt) e salve o arquivo em "
        "%USERPROFILE%\\.config\\watch\\secrets\\cookies.txt (montado read-only no container "
        "como /secrets/cookies.txt). Não é preciso reiniciar: o arquivo é lido a cada chamada."
    )


def youtube_block_message(url: str, stderr: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower()
    if not (host in ("youtu.be", "youtube.com") or host.endswith(".youtube.com")):
        return None
    low = stderr.lower()
    if not any(marker in low for marker in YOUTUBE_BOT_MARKERS):
        return None
    return (
        "O YouTube recusou temporariamente o download (anti-bot / HTTP 429 para este IP). "
        "Tente de novo em alguns minutos. Se persistir, um cookies.txt (Netscape) de uma conta "
        "secundária em %USERPROFILE%\\.config\\watch\\secrets\\cookies.txt também vale para o YouTube."
    )


def site_block_message(url: str, stderr: str) -> str | None:
    return login_wall_message(url, stderr) or youtube_block_message(url, stderr)


async def preflight(
    url: str, extra_args: list[str], work: Path, ctx: Context | None, timeout: float | None = None,
) -> dict:
    """Metadata only: duration limit and login walls, before any download."""
    # Read before yt-dlp runs: it rewrites the jar on exit, dropping bad entries.
    secrets = secrets_from_args(extra_args)
    argv = [
        "yt-dlp", *extra_args,
        "--skip-download", "--dump-single-json", "--no-playlist", "--no-warnings",
        "--", url,
    ]
    code, stdout, stderr = await run_process(
        argv, env=child_env({}), cwd=work, timeout=timeout or YTDLP_TIMEOUT, ctx=ctx, label="lendo metadados",
    )
    info: dict = {}
    if code == 0 and stdout.strip():
        try:
            info = json.loads(stdout)
        except json.JSONDecodeError:
            info = {}
    if not info:
        if secrets and cookie_jar_problem(stderr):
            raise ToolError(COOKIE_JAR_ERROR)
        wall = site_block_message(url, stderr)
        if wall:
            raise ToolError(wall)
        raise ToolError(f"yt-dlp não conseguiu ler este vídeo:\n{tail(stderr, secrets) or f'exit {code}'}")
    if info.get("_type") == "playlist":
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise ToolError("O link aponta para uma lista/carrossel sem vídeo.")
        info = {**entries[0], "description": info.get("description") or entries[0].get("description")}
    if info.get("is_live"):
        raise ToolError("Transmissões ao vivo não são suportadas.")
    duration = info.get("duration")
    if duration and float(duration) > MAX_DURATION_SECONDS:
        raise ToolError(
            f"Vídeo muito longo ({format_time(float(duration))}); o limite é "
            f"{MAX_DURATION_SECONDS // 60} min. Use start/end para um trecho não ajuda: o download é do vídeo inteiro."
        )
    return info


def parse_frames(stdout: str, frames_dir: Path) -> list[Frame]:
    """Frame list from watch.py's report — only from its "## Frames" section and
    only files that really are this call's frames.

    The report also carries remote text (title, uploader, captions) that can
    forge a frame line, so every path must resolve (symlinks included) to a
    frame_NNNN.jpg / cue_NNNN.jpg directly inside frames_dir.
    """
    report = stdout.replace("\r\n", "\n")
    if "\n## Frames\n" not in report:
        return []
    section = report.split("\n## Frames\n", 1)[1].split("\n## Transcript\n", 1)[0]
    root = frames_dir.resolve()
    frames = []
    for line in section.splitlines():
        m = FRAME_LINE_RE.match(line.strip())
        if not m:
            continue
        path = Path(m.group("path")).resolve()
        if path.parent == root and FRAME_NAME_RE.match(path.name) and path.is_file():
            frames.append(Frame(path, m.group("t")))
    return frames


def read_transcript(run_dir: Path) -> tuple[list[dict], str | None]:
    path = run_dir / "transcript.json"
    if not path.exists():
        return [], None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments") or [], data.get("source")


async def run_watch(
    url: str,
    *,
    detail: str,
    max_frames: int,
    start: str | None,
    end: str | None,
    ctx: Context | None,
) -> tuple[RunResult, Path]:
    """Validate, preflight and run watch.py. The caller deletes the work dir."""
    deadline = time.monotonic() + CALL_TIMEOUT

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise ToolError(f"Tempo limite total excedido ({CALL_TIMEOUT:.0f}s)")
        return left

    url = check_url(url)
    start, end = check_time("start", start), check_time("end", end)
    await asyncio.to_thread(check_whisper)

    work = Path(tempfile.mkdtemp(prefix="watch-mcp-"))
    try:
        cookies = None
        if COOKIES_PATH.is_file():
            # yt-dlp writes the jar back on exit; give it a private copy so the
            # read-only mount is never written and the copy dies with the work dir.
            cookies = work / "cookies.txt"
            shutil.copyfile(COOKIES_PATH, cookies)
            os.chmod(cookies, 0o600)
        secrets = cookie_secrets(cookies)

        with netguard.EgressProxy(max_bytes=MAX_DOWNLOAD_BYTES) as proxy:
            extra = ytdlp_args(proxy.url, cookies)
            try:
                info = await preflight(url, extra, work, ctx, timeout=min(YTDLP_TIMEOUT, remaining()))
            except ToolError:
                _check_budget(proxy)
                raise

            run_dir = work / "run"
            argv = [
                sys.executable, str(SCRIPTS_DIR / "watch.py"), url,
                "--out-dir", str(run_dir),
                "--detail", detail,
                "--max-frames", str(max_frames),
                "--resolution", str(frame_width(info.get("width"), info.get("height"))),
            ]
            if start:
                argv += ["--start", start]
            if end:
                argv += ["--end", end]
            env = child_env({
                "WATCH_YTDLP_ARGS": json.dumps(extra),
                "WATCH_YTDLP_TIMEOUT": str(YTDLP_TIMEOUT),
            })
            try:
                code, stdout, stderr = await run_process(
                    argv, env=env, cwd=work, timeout=remaining(), ctx=ctx, label="baixando e analisando",
                )
            finally:
                _check_budget(proxy)
        if code != 0:
            if secrets and cookie_jar_problem(stderr):
                raise ToolError(COOKIE_JAR_ERROR)
            wall = site_block_message(url, stderr)
            raise ToolError(wall or f"watch.py falhou (exit {code}):\n{tail(stderr, secrets)}")

        if not info.get("duration"):
            # Some extractors (Instagram) omit it; watch.py measured the file.
            m = DURATION_LINE_RE.search(stdout)
            if m and float(m.group("s")) > 0:
                info = {**info, "duration": float(m.group("s"))}
        result = RunResult(info=info, frames=parse_frames(stdout, run_dir / "frames"), secrets=secrets)
        result.segments, result.transcript_source = read_transcript(run_dir)
        for line in stderr.splitlines():
            if "whisper fallback failed" in line:
                result.warnings.append(f"Transcrição Whisper falhou: {scrub(line.split('failed:', 1)[-1].strip(), secrets)}")
            elif "no audio stream found" in line:
                result.warnings.append("O vídeo não tem trilha de áudio.")
        return result, work
    except BaseException:
        cleanup(work)
        raise


def _check_budget(proxy: netguard.EgressProxy) -> None:
    if proxy.exceeded:
        raise ToolError(
            f"O download passou do limite de {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB e foi interrompido."
        )


def cleanup(work: Path) -> None:
    """Delete the per-call work dir; a failure is logged (path only), never silent."""
    try:
        shutil.rmtree(work)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[watch-mcp] could not fully remove {work}: {type(exc).__name__}", file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)


# ── output ──────────────────────────────────────────────────────────────────


def fit_budget(sizes: list[int], budget: int) -> list[int]:
    """Indices of the largest uniform subsample whose total size fits budget."""
    n = len(sizes)
    for k in range(n, 0, -1):
        if k == 1:
            picks = [0]
        else:
            picks = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
        if sum(sizes[i] for i in picks) <= budget:
            return picks
    return []


def encode_frames(frames: list[Frame], budget: int) -> tuple[list[Frame], list[str], str | None]:
    kept, encoded = [], []
    for frame in frames:
        data = frame.path.read_bytes()
        if data.startswith(JPEG_MAGIC):  # only real JPEGs leave the server
            kept.append(frame)
            encoded.append(base64.b64encode(data).decode("ascii"))
    frames = kept
    picks = fit_budget([len(e) for e in encoded], budget)
    note = None
    if len(picks) < len(frames):
        note = (
            f"Orçamento de imagens (~{budget // 1024} KB) excedido: enviados {len(picks)} de "
            f"{len(frames)} frames, escolhidos uniformemente ao longo do vídeo."
        )
    return [frames[i] for i in picks], [encoded[i] for i in picks], note


def build_text(
    url: str,
    result: RunResult,
    *,
    question: str | None,
    detail: str,
    start: str | None,
    end: str | None,
    frames: list[Frame] | None,
) -> str:
    info = result.info
    lines = [f"# Vídeo: {info.get('title') or url}", "", f"- URL: {info.get('webpage_url') or url}"]
    author = info.get("uploader") or info.get("channel") or info.get("creator")
    if author:
        # Instagram's uploader_id is numeric; its channel is the @handle.
        handle = info.get("channel") or info.get("uploader_id")
        if handle and str(handle).isdigit():
            handle = None
        lines.append(f"- Autor: {author}" + (f" (@{handle})" if handle and handle != author else ""))
    if info.get("duration"):
        d = float(info["duration"])
        lines.append(f"- Duração: {format_time(d)} ({d:.1f}s)")
    if info.get("upload_date"):
        u = str(info["upload_date"])
        lines.append(f"- Publicado em: {u[:4]}-{u[4:6]}-{u[6:]}")
    if start or end:
        lines.append(f"- Trecho analisado: {start or 'início'} → {end or 'fim'}")
    lines.append(f"- Detalhe: {detail}")
    if question:
        lines.append(f"- Pergunta do usuário: {question}")

    description = (info.get("description") or "").strip()
    if description:
        if len(description) > DESCRIPTION_LIMIT:
            description = description[:DESCRIPTION_LIMIT] + " […]"
        lines += ["", "## Legenda/descrição do post", "", description]

    lines += ["", "## Transcrição", ""]
    if result.segments:
        lines.append(f"_Fonte: {result.transcript_source or 'captions'}_")
        lines += ["", format_transcript(result.segments)]
    else:
        lines.append("_Nenhuma transcrição disponível (sem legendas e sem fala reconhecida)._")

    if frames is not None:
        lines += ["", f"## Frames ({len(frames)}, em ordem temporal — as imagens seguem nesta ordem)", ""]
        if frames:
            lines += [f"{i}. t={f.timestamp}" for i, f in enumerate(frames, 1)]
        else:
            lines.append("_Nenhum frame extraído._")

    if result.warnings:
        lines += ["", "## Avisos", ""] + [f"- {w}" for w in result.warnings]
    return "\n".join(lines)


# ── tools ───────────────────────────────────────────────────────────────────


@mcp.tool()
async def watch_video(
    url: str,
    question: str | None = None,
    start: str | None = None,
    end: str | None = None,
    detail: Literal["transcript", "efficient", "balanced"] = "balanced",
    max_frames: int = 24,
    ctx: Context | None = None,
) -> list[TextContent | ImageContent]:
    """Watch a video from a URL (Instagram reel, TikTok, YouTube, ...).

    Returns metadata (title, author, duration, post caption), a transcript with
    timestamps, and frames as JPEG images in temporal order (t=MM:SS listed in
    the text). `start`/`end` (SS, MM:SS or HH:MM:SS) focus on a range;
    `detail`: "transcript" (no frames), "efficient" (keyframes, cheapest) or
    "balanced" (scene-aware, default). Videos up to 15 minutes.
    """
    if not 1 <= max_frames <= MAX_FRAMES_LIMIT:
        raise ToolError(f"max_frames deve estar entre 1 e {MAX_FRAMES_LIMIT}")
    result, work = await run_watch(url, detail=detail, max_frames=max_frames, start=start, end=end, ctx=ctx)
    try:
        frames, encoded, note = [], [], None
        if detail != "transcript":
            frames, encoded, note = encode_frames(result.frames, IMAGE_BUDGET_BYTES)
        if note:
            result.warnings.append(note)
        text = build_text(
            url, result, question=question, detail=detail, start=start, end=end,
            frames=None if detail == "transcript" else frames,
        )
        content: list[TextContent | ImageContent] = [TextContent(type="text", text=scrub(text, result.secrets))]
        content += [ImageContent(type="image", data=data, mime_type="image/jpeg") for data in encoded]
        return content
    finally:
        cleanup(work)


@mcp.tool()
async def transcribe_video(url: str, ctx: Context | None = None) -> str:
    """Transcribe a video from a URL (captions when available, else local Whisper).

    Text only — metadata, post caption and a timestamped transcript, no frames.
    """
    result, work = await run_watch(url, detail="transcript", max_frames=1, start=None, end=None, ctx=ctx)
    try:
        if not result.segments:
            raise ToolError(
                "Nenhuma transcrição obtida: o vídeo não tem legendas e o Whisper não reconheceu fala."
                + (f" ({'; '.join(result.warnings)})" if result.warnings else "")
            )
        text = build_text(url, result, question=None, detail="transcript", start=None, end=None, frames=None)
        return scrub(text, result.secrets)
    finally:
        cleanup(work)


if __name__ == "__main__":
    mcp.run()
