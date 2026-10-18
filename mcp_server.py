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
import hashlib
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

import inbox
import jobstore
import netguard
import textsearch

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import jobview  # noqa: E402  (uses the skill's frames/transcribe)
from frames import format_time, parse_time  # noqa: E402
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

LANGUAGE_RE = re.compile(r"^[a-z]{2}$")
MAX_TERMS = 30
TERM_MAX_LEN = 60
LIST_JOBS_LIMIT = 20
LIST_INBOX_LIMIT = 200

mcp = MCPServer(
    "watch",
    version="0.3.0",
    instructions=(
        "Tools to watch videos. Short videos from a URL (Instagram, TikTok, YouTube, ...): "
        "watch_video(url=...) returns metadata, a timestamped transcript and frames; "
        "transcribe_video(url) only the transcript. Long videos or local files from the inbox "
        "(list_inbox): analyze_video(source) starts a background job and returns a job_id at once; "
        "poll job_status(job_id), then search_transcript(job_id, query) to find moments and "
        "watch_video(job_id=..., start=..., end=...) for frames and the transcript of a range. "
        "list_jobs() shows recent jobs."
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


def ytdlp_args(
    proxy_url: str,
    cookies: Path | None,
    *,
    max_duration: float = MAX_DURATION_SECONDS,
    max_filesize: str = MAX_FILESIZE,
) -> list[str]:
    """Common yt-dlp options. The limits default to the direct (short-video)
    path; the v2 worker passes its own, larger ones."""
    args = [
        "--ignore-config",
        "--no-cache-dir",
        "--proxy", proxy_url,
        "--socket-timeout", "30",
        "--max-filesize", max_filesize,
        "--match-filter", f"duration <=? {max_duration:g} & !is_live",
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
            f"Vídeo muito longo ({format_time(float(duration))}) para a resposta direta (limite de "
            f"{MAX_DURATION_SECONDS // 60} min). Use analyze_video(source=<url>) para processá-lo em "
            "segundo plano, depois job_status, search_transcript e watch_video(job_id=...)."
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
    url: str | None = None,
    question: str | None = None,
    start: str | None = None,
    end: str | None = None,
    detail: Literal["transcript", "efficient", "balanced"] = "balanced",
    max_frames: int = 24,
    job_id: str | None = None,
    ctx: Context | None = None,
) -> list[TextContent | ImageContent]:
    """Watch a video: a short one from a URL, or a range of an analyzed job.

    With `url` (Instagram reel, TikTok, YouTube, ... up to 15 minutes): downloads
    and answers directly. With `job_id` (from analyze_video): instant, from the
    job's cache, with no new download or transcription.

    Returns metadata, a transcript with timestamps, and frames as JPEG images in
    temporal order (t=MM:SS listed in the text). `start`/`end` (SS, MM:SS or
    HH:MM:SS) focus on a range; `detail`: "transcript" (no frames), "efficient"
    (scene cuts/keyframes, cheapest) or "balanced" (default).
    """
    if not 1 <= max_frames <= MAX_FRAMES_LIMIT:
        raise ToolError(f"max_frames deve estar entre 1 e {MAX_FRAMES_LIMIT}")
    if (url is None) == (job_id is None):
        raise ToolError("Informe url (vídeo curto) ou job_id (trabalho de analyze_video), um dos dois.")
    if job_id is not None:
        return watch_job(job_id, question=question, start=start, end=end, detail=detail, max_frames=max_frames)
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


# ── v2: background jobs ─────────────────────────────────────────────────────


def load_job(job_id: str) -> dict:
    try:
        return jobstore.load_job(job_id)
    except jobstore.JobError as exc:
        raise ToolError(str(exc)) from None


def check_terms(terms: list[str] | None) -> list[str]:
    if terms is None:
        return []
    if not isinstance(terms, list) or len(terms) > MAX_TERMS:
        raise ToolError(f"terms deve ser uma lista de até {MAX_TERMS} termos")
    out = []
    for term in terms:
        if not isinstance(term, str):
            raise ToolError("terms deve conter apenas textos")
        term = " ".join(term.split())
        if len(term) > TERM_MAX_LEN or any(ord(c) < 32 or ord(c) == 127 for c in term):
            raise ToolError(f"termo inválido (máximo {TERM_MAX_LEN} caracteres, sem caracteres de controle)")
        if term and term not in out:
            out.append(term)
    return out


def inbox_source(name: str) -> tuple[str, list]:
    """Validate an inbox file now (name, no symlink, real media) for a clear,
    immediate error; the worker checks it again by fd before using it."""
    try:
        fd = inbox.open_media(name)
        try:
            st = os.fstat(fd)
            posix = os.name == "posix"
            inbox.probe(inbox.fd_path(fd) if posix else str(inbox.INBOX_DIR / name),
                        pass_fds=(fd,) if posix else ())
        finally:
            os.close(fd)
    except inbox.InboxError as exc:
        raise ToolError(f"Caixa de entrada: {exc}") from None
    return name, [st.st_size, st.st_mtime_ns]


@mcp.tool()
async def analyze_video(
    source: str,
    language: str = "pt",
    quality: Literal["auto", "fast", "accurate"] = "auto",
    terms: list[str] | None = None,
) -> str:
    """Start a background analysis of a long video or a local file; returns a job_id at once.

    `source`: a URL (same rules as watch_video) or the name of a file in the
    inbox (see list_inbox; just the name, no folders). Videos up to 4 hours.
    `language`: Whisper language code, "pt" by default (Brazilian Portuguese
    context is added to the prompt), or "auto". `quality`: "auto" (fast model for
    short videos, the most accurate pt-BR model for long ones), "fast" or
    "accurate". `terms`: optional proper names/jargon that appear in the video,
    to help the transcription. Then poll job_status(job_id).
    """
    source = (source or "").strip()
    if not source:
        raise ToolError("Informe source: uma URL ou o nome de um arquivo da caixa de entrada.")
    language = (language or "pt").strip().lower()
    if language != "auto" and not LANGUAGE_RE.match(language):
        raise ToolError('language deve ser um código de 2 letras (ex.: "pt") ou "auto"')
    if quality not in ("auto", "fast", "accurate"):
        raise ToolError('quality deve ser "auto", "fast" ou "accurate"')
    terms = check_terms(terms)
    if source.lower().startswith(("http://", "https://")):
        value, fingerprint, kind = check_url(source), [], "url"
    elif "://" in source:
        raise ToolError("URL recusada: só http/https")
    else:
        (value, fingerprint), kind = await asyncio.to_thread(inbox_source, source), "inbox"
    key = hashlib.sha256(json.dumps(
        [kind, value, fingerprint, language, quality, sorted(terms)], ensure_ascii=False,
    ).encode()).hexdigest()
    cached = jobstore.find_cached(key)
    if cached is not None:
        return ("Este vídeo já foi analisado (ou está em andamento) com os mesmos parâmetros; reaproveitando.\n\n"
                + jobview.status_text(cached, worker_ok=jobview.worker_alive()))
    job = jobstore.create_job({
        "source": {"kind": kind, "value": value},
        "language": None if language == "auto" else language,
        "quality": quality,
        "terms": terms,
        "cache_key": key,
        # What the server saw (inbox: size + mtime); the worker refuses a changed file.
        "fingerprint": fingerprint or None,
    })
    text = (
        f"Trabalho criado: job_id = {job['id']}\n\n"
        "Ele roda em segundo plano (de minutos a horas, conforme a duração). Acompanhe com "
        "job_status(job_id); quando concluir, use search_transcript e watch_video(job_id=...)."
    )
    if not jobview.worker_alive():
        text += "\n\n**Aviso:** o watch-worker não está respondendo. Rode docker compose -p watch-mcp up -d"
    return text


@mcp.tool()
async def job_status(job_id: str) -> str:
    """State, progress (%), current stage and estimated time of a job from analyze_video."""
    return jobview.status_text(load_job(job_id), worker_ok=jobview.worker_alive())


@mcp.tool()
async def list_jobs() -> str:
    """Recent analysis jobs: id, state, duration, date and source."""
    jobs = jobstore.list_jobs()[:LIST_JOBS_LIMIT]
    if not jobs:
        return "Nenhum trabalho ainda. Use analyze_video(source) para começar."
    return "# Trabalhos recentes\n\n" + "\n".join(jobview.job_line(j) for j in jobs)


@mcp.tool()
async def search_transcript(job_id: str, query: str) -> str:
    """Find where something is said in a job's transcript (accent- and case-insensitive).

    Returns the matching passages with timestamps, to open with watch_video(job_id=..., start, end).
    """
    job = load_job(job_id)
    segments, partial = jobview.load_segments(jobstore.job_path(job["id"]))
    if not segments:
        raise ToolError("Ainda não há transcrição para este trabalho; veja job_status(job_id).")
    try:
        hits, total = textsearch.search(segments, query)
    except ValueError:
        raise ToolError("Informe um texto para buscar.") from None
    text = jobview.hits_text(job, query, hits, total)
    if partial:
        text += "\n\n_Transcrição parcial: o trabalho ainda está em andamento._"
    return text


@mcp.tool()
async def list_inbox() -> str:
    """Video/audio files available in the local inbox (D:\\Inbox on the host), to pass to analyze_video."""
    files = inbox.list_inbox()
    if not files:
        return ("A caixa de entrada está vazia (D:\\Inbox no Windows). Extensões aceitas: "
                + ", ".join(sorted(inbox.ALLOWED_EXTS)))
    lines = ["# Caixa de entrada", ""]
    for f in files[:LIST_INBOX_LIMIT]:
        lines.append(f"- {f['name']} — {f['size_bytes'] / 1024**2:.1f} MB — {f['modified']}")
    if len(files) > LIST_INBOX_LIMIT:
        lines.append(f"- … e mais {len(files) - LIST_INBOX_LIMIT} arquivo(s).")
    return "\n".join(lines + ["", 'Use analyze_video(source="<nome do arquivo>").'])


def watch_job(job_id: str, *, question: str | None, start: str | None, end: str | None,
              detail: str, max_frames: int) -> list[TextContent | ImageContent]:
    """watch_video for a job: index frames + transcript slice, from the cache only."""
    job = load_job(job_id)
    job_dir = jobstore.job_path(job["id"])
    if job.get("state") == "failed":
        raise ToolError(f"Este trabalho falhou: {job.get('error')}")
    index = jobstore.read_json(job_dir / "index.json")
    if index is None:
        raise ToolError("O trabalho ainda não tem o índice de cenas; veja job_status(job_id).")
    lo, hi = check_time("start", start), check_time("end", end)
    lo_s = parse_time(lo) if lo else None
    hi_s = parse_time(hi) if hi else None
    if lo_s is not None and hi_s is not None and hi_s <= lo_s:
        raise ToolError("end deve ser maior que start")
    segments, partial = jobview.load_segments(job_dir)
    segments = jobview.in_range(segments, lo_s, hi_s)
    meta = jobstore.read_json(job_dir / "meta.json") or {}
    warnings: list[str] = []
    encoded: list[str] = []
    times: list[float] | None = None
    if detail != "transcript":
        picked = jobview.pick_frames(index.get("frames") or [], job_dir, lo_s, hi_s, detail, max_frames)
        kept, encoded, note = encode_frames([Frame(path, format_time(t)) for path, t in picked],
                                            IMAGE_BUDGET_BYTES)
        seconds = {path: t for path, t in picked}
        times = [seconds[f.path] for f in kept]
        if note:
            warnings.append(note)
    text = jobview.watch_text(job, meta, segments, partial=partial, start=lo_s, end=hi_s, detail=detail,
                              question=question, frame_times=times, warnings=warnings)
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=scrub(text))]
    content += [ImageContent(type="image", data=data, mime_type="image/jpeg") for data in encoded]
    return content


if __name__ == "__main__":
    mcp.run()
