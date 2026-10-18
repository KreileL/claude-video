#!/usr/bin/env python3
"""watch-worker: processes the v2 job queue (python /app/worker.py).

Always-on compose service. The stdio MCP server only creates queued jobs in
JOBS_DIR (see jobstore.py); this process owns every later write. Per job:

1. source: a URL is downloaded through the netguard egress proxy (public
   addresses only, hard byte budget) into the job's work/ dir; an inbox file is
   opened once by fd and read in place. Either way the media is checked with
   ffprobe (protocol and demuxer whitelists) and read through that fd with the
   probed demuxer forced.
2. audio: 16 kHz mono mp3 in work/, split at silences into chunks (plan.json).
3. index: one ffmpeg pass writes frames/ (scene cuts + a time grid, JPEG with
   the long side at 512 px) and index.json. The video itself is then deleted.
4. transcript: chunk by chunk, each persisted as chunks/NNNN.json, so a restart
   resumes after the last finished chunk; then transcript.json, and work/ goes.

Every stage is skipped when its output exists, which is what makes a job
resumable after the worker restarts. Nothing secret is written under JOBS_DIR:
cookies are copied to a private dir on the container's tmpfs.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib
from pathlib import Path

import chunking
import inbox
import jobstore
import mcp_server
import netguard
import textsearch
from whisper import transcribe_local

MAX_DURATION = float(os.environ.get("WATCH_JOB_MAX_DURATION", str(4 * 3600)))
MAX_DOWNLOAD_BYTES = int(os.environ.get("WATCH_JOB_MAX_DOWNLOAD_BYTES", str(4 * 1024**3)))
JOB_TIMEOUT = float(os.environ.get("WATCH_JOB_TIMEOUT", str(8 * 3600)))
TTL_SECONDS = float(os.environ.get("WATCH_JOB_TTL", str(7 * 24 * 3600)))
MAX_JOBS_BYTES = int(os.environ.get("WATCH_JOBS_MAX_BYTES", str(20 * 1024**3)))
CHUNK_SECONDS = float(os.environ.get("WATCH_CHUNK_SECONDS", "300"))
POLL_SECONDS = float(os.environ.get("WATCH_WORKER_POLL", "2"))
CLEANUP_EVERY = 600.0
# Set from the pt-BR benchmark (NOTES-mcp.md, "v2"); overridable per deployment.
MODEL_FAST = os.environ.get("WATCH_MODEL_FAST", "Systran/faster-whisper-small")
MODEL_ACCURATE = os.environ.get("WATCH_MODEL_ACCURATE", "Systran/faster-whisper-medium")
USE_PROMPT = os.environ.get("WATCH_PTBR_PROMPT", "1") != "0"
SHORT_SECONDS = 15 * 60  # quality="auto": up to this, the fast model
PROMPT_PTBR = (
    "Transcrição em português do Brasil, com pontuação e grafia brasileira: "
    "ônibus, celular, time, trem, registro, boleto, aluguel, a gente, tô, cê."
)
PROMPT_LIMIT = 800  # characters; Whisper only keeps the last ~224 prompt tokens
FRAME_LONG_SIDE = mcp_server.FRAME_LONG_SIDE
MAX_INDEX_FRAMES = 2500
SCENE_THRESHOLD = 0.3
WHISPER_RETRIES = (30, 60, 120)  # seconds to wait before each retry of a chunk
MAX_ATTEMPTS = 3  # a job that keeps killing the worker is failed, not resumed forever
SOURCE_NAME_RE = re.compile(r"^source\.[a-z0-9]{2,5}$")
# Whisper repetition loops ("É do Léo mesmo." x150 after laughter). speaches
# takes a single temperature, so faster-whisper's fallback never runs and a
# loop carries on to the end of the request. Detected loops are re-transcribed
# in short pieces (fresh context each); leftover runs are collapsed.
LOOP_RUN = 3
LOOP_RATIO = 2.4  # zlib ratio faster-whisper uses for "too repetitive"
LOOP_MIN_BYTES = 120
LOOP_PIECE_SECONDS = 30.0
SHOWINFO_TS_RE = re.compile(r"pts_time:([0-9.]+)")
DOWNLOAD_FORMAT = "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"


class JobFailed(Exception):
    """A clear, user-facing (Portuguese) reason the job cannot finish."""


def log(message: str) -> None:
    print(f"[watch-worker] {message}", file=sys.stderr, flush=True)


# ── subprocesses ────────────────────────────────────────────────────────────


def run(argv: list[str], *, timeout: float, pass_fds: tuple[int, ...] = (), env: dict | None = None,
        cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run argv (never a shell) in its own process group; on timeout the whole
    group dies (yt-dlp spawns ffmpeg)."""
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", pass_fds=pass_fds, env=env,
        cwd=str(cwd) if cwd else None, start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = proc.communicate(timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        _kill(proc)
        proc.communicate()
        raise JobFailed(f"tempo limite excedido em {Path(argv[0]).name}") from None
    except BaseException:
        _kill(proc)
        raise
    _kill(proc)  # strays left by the child must not outlive the step
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


# ── job helpers ─────────────────────────────────────────────────────────────


class Job:
    def __init__(self, data: dict, root: Path | None = None):
        self.id = data["id"]
        self.root = root
        self.dir = jobstore.job_path(self.id, root)
        self.data = data
        started = data.get("started") or time.time()
        self.deadline = started + JOB_TIMEOUT

    @property
    def work(self) -> Path:
        return self.dir / "work"

    def update(self, **fields) -> None:
        self.data = jobstore.update_job(self.id, self.root, **fields)

    def remaining(self) -> float:
        left = self.deadline - time.time()
        if left <= 0:
            raise JobFailed(f"tempo total do trabalho excedido ({JOB_TIMEOUT / 3600:.1f} h)")
        return left


def pick_model(quality: str, duration: float) -> str:
    if quality == "fast":
        return MODEL_FAST
    if quality == "accurate":
        return MODEL_ACCURATE
    return MODEL_FAST if duration <= SHORT_SECONDS else MODEL_ACCURATE


def build_prompt(language: str | None, terms: list[str]) -> str | None:
    """Brazilian-Portuguese context plus the caller's terms (pt jobs only)."""
    if language != "pt" or not USE_PROMPT:
        return " ".join(terms)[:PROMPT_LIMIT] or None if terms else None
    prompt = PROMPT_PTBR
    if terms:
        prompt += " Termos: " + ", ".join(terms) + "."
    return prompt[:PROMPT_LIMIT]


def scale_filter() -> str:
    side = FRAME_LONG_SIDE
    return (f"scale='if(gte(iw,ih),min({side},iw),-2)':'if(gte(iw,ih),-2,min({side},ih))'")


def grid_interval(duration: float) -> float:
    """At least one index frame this often; ≤ MAX_INDEX_FRAMES grid frames per job."""
    return max(2.0, duration / MAX_INDEX_FRAMES)


# ── stage 1: source ─────────────────────────────────────────────────────────


class Source:
    """An open media file: the path ffmpeg reads (via the fd) and its probe.

    On POSIX ffmpeg reads /proc/self/fd/N, the inode validated here, whatever
    happens to the name afterwards. Windows (local development) uses the path.
    """

    def __init__(self, fd: int, info: dict, path: Path):
        self.fd = fd
        self.info = info
        self.real_path = path

    @property
    def path(self) -> str:
        return inbox.fd_path(self.fd) if os.name == "posix" else str(self.real_path)

    @property
    def fds(self) -> tuple[int, ...]:
        return (self.fd,) if os.name == "posix" else ()

    def input(self) -> list[str]:
        return [*inbox.input_args(self.info["demuxer"]), "-i", self.path]

    def limit(self) -> str:
        """Output duration cap: the probed (already limit-checked) duration, plus slack."""
        return f"{self.info['duration'] + 5:.3f}"

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


def open_checked(name: str, root: Path) -> Source:
    fd = inbox.open_media(name, root)
    src = Source(fd, {}, Path(root) / name)
    try:
        src.info = inbox.probe(src.path, pass_fds=src.fds)
        if src.info["duration"] > MAX_DURATION:
            raise JobFailed(f"mídia muito longa ({src.info['duration'] / 3600:.1f} h); "
                            f"o limite é {MAX_DURATION / 3600:.1f} h")
    except BaseException:
        src.close()
        raise
    return src


def private_cookies(tmp: Path) -> Path | None:
    """Private 0600 copy of the cookie jar on tmpfs — never under JOBS_DIR."""
    if not mcp_server.COOKIES_PATH.is_file():
        return None
    copy = tmp / "cookies.txt"
    shutil.copyfile(mcp_server.COOKIES_PATH, copy)
    os.chmod(copy, 0o600)
    return copy


def download(job: Job, url: str) -> Path:
    """Metadata check, then the download through the egress proxy; returns the file."""
    work = job.work
    work.mkdir(exist_ok=True)
    done = work / "download.done"
    if done.exists():  # finished before a restart: reuse it
        name = done.read_text(encoding="utf-8").strip()
        if not SOURCE_NAME_RE.match(name):
            raise JobFailed("estado de download inválido no cache")
        return work / name
    for stale in work.glob("source.*"):  # an interrupted download starts over
        stale.unlink()
    tmp = Path(tempfile.mkdtemp(prefix="watch-worker-"))
    try:
        cookies = private_cookies(tmp)
        secrets = mcp_server.cookie_secrets(cookies)
        with netguard.EgressProxy(max_bytes=MAX_DOWNLOAD_BYTES) as proxy:
            extra = mcp_server.ytdlp_args(
                proxy.url, cookies, max_duration=MAX_DURATION, max_filesize=f"{MAX_DOWNLOAD_BYTES}",
            )
            env = mcp_server.child_env({})
            meta = run(["yt-dlp", *extra, "--skip-download", "--dump-single-json", "--no-playlist",
                        "--no-warnings", "--", url], timeout=min(600, job.remaining()), env=env, cwd=tmp)
            info = _metadata(url, meta, secrets, proxy)
            write_meta(job, info)
            job.update(stage="baixando", progress=2.0, title=info.get("title"),
                       duration=info.get("duration"))
            got = run(["yt-dlp", *extra, "-N", "4", "-f", DOWNLOAD_FORMAT, "--merge-output-format", "mkv",
                       "--no-playlist", "-o", str(work / "source.%(ext)s"), "--", url],
                      timeout=job.remaining(), env=env, cwd=tmp)
            if proxy.exceeded:
                raise JobFailed(f"o download passou do limite de {MAX_DOWNLOAD_BYTES / 1024**3:.1f} GB")
        files = [p for p in work.glob("source.*") if p.suffix not in (".part", ".ytdl")]
        if got.returncode != 0 or not files:
            if secrets and mcp_server.cookie_jar_problem(got.stderr):
                raise JobFailed(mcp_server.COOKIE_JAR_ERROR)
            wall = mcp_server.site_block_message(url, got.stderr)
            raise JobFailed(wall or f"yt-dlp falhou:\n{mcp_server.tail(got.stderr, secrets)}")
        done.write_text(files[0].name, encoding="utf-8")
        return files[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _metadata(url: str, meta: subprocess.CompletedProcess, secrets: frozenset[str],
              proxy: netguard.EgressProxy) -> dict:
    info: dict = {}
    if meta.returncode == 0 and meta.stdout.strip():
        try:
            info = json.loads(meta.stdout)
        except json.JSONDecodeError:
            info = {}
    if not info:
        if proxy.exceeded:
            raise JobFailed("o limite de download foi atingido ao ler os metadados")
        if secrets and mcp_server.cookie_jar_problem(meta.stderr):
            raise JobFailed(mcp_server.COOKIE_JAR_ERROR)
        wall = mcp_server.site_block_message(url, meta.stderr)
        raise JobFailed(wall or f"yt-dlp não conseguiu ler este vídeo:\n{mcp_server.tail(meta.stderr, secrets)}")
    if info.get("_type") == "playlist":
        entries = [e for e in info.get("entries") or [] if e]
        if not entries:
            raise JobFailed("o link aponta para uma lista/carrossel sem vídeo")
        info = {**entries[0], "description": info.get("description") or entries[0].get("description")}
    if info.get("is_live"):
        raise JobFailed("transmissões ao vivo não são suportadas")
    if info.get("duration") and float(info["duration"]) > MAX_DURATION:
        raise JobFailed(f"vídeo muito longo ({float(info['duration']) / 3600:.1f} h); "
                        f"o limite é {MAX_DURATION / 3600:.1f} h")
    return info


def write_meta(job: Job, info: dict) -> None:
    """Only display fields; the raw yt-dlp JSON (format URLs, headers) is not kept."""
    keep = ("title", "uploader", "channel", "uploader_id", "webpage_url", "upload_date", "duration",
            "width", "height")
    meta = {k: info.get(k) for k in keep if info.get(k) is not None}
    description = (info.get("description") or "").strip()
    if description:
        meta["description"] = description[:mcp_server.DESCRIPTION_LIMIT]
    jobstore.write_json_atomic(job.dir / "meta.json", meta)


def open_source(job: Job, inbox_root: Path) -> Source:
    source = job.data["source"]
    if source["kind"] == "inbox":
        jobstore.write_json_atomic(job.dir / "meta.json", {"title": source["value"]})
        return open_checked(source["value"], inbox_root)
    path = download(job, source["value"])
    # The downloaded file is untrusted too: same probe, same fd discipline.
    return open_checked(path.name, path.parent)


# ── stage 2: audio + chunk plan ─────────────────────────────────────────────


def extract_audio(job: Job, src: Source) -> None:
    audio = job.work / "audio.mp3"
    if (job.dir / "plan.json").exists():
        return
    job.work.mkdir(exist_ok=True)
    duration = src.info["duration"]
    if not src.info["has_audio"]:
        jobstore.write_json_atomic(job.dir / "plan.json", {"chunks": [], "duration": duration})
        return
    job.update(stage="extraindo áudio", progress=15.0)
    tmp = job.work / "audio.tmp.mp3"
    got = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *src.input(), "-t", src.limit(),
               "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k", str(tmp)],
              timeout=job.remaining(), pass_fds=src.fds)
    if got.returncode != 0 or not tmp.exists():
        raise JobFailed("não foi possível extrair o áudio")
    tmp.replace(audio)
    silences = run(["ffmpeg", "-hide_banner", "-nostats", *inbox.input_args("mp3"), "-i", str(audio),
                    "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"], timeout=job.remaining())
    chunks = chunking.plan(duration, chunking.parse_silences(silences.stderr), target=CHUNK_SECONDS)
    jobstore.write_json_atomic(job.dir / "plan.json", {"chunks": chunks, "duration": duration})


# ── stage 3: frame index ────────────────────────────────────────────────────


def build_index(job: Job, src: Source) -> None:
    if (job.dir / "index.json").exists():
        return
    frames = job.dir / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    if not src.info["has_video"]:
        jobstore.write_json_atomic(job.dir / "index.json", {"frames": []})
        return
    frames.mkdir()
    job.update(stage="indexando cenas", progress=20.0)
    grid = grid_interval(src.info["duration"])
    select = f"select='eq(n\\,0)+gt(scene\\,{SCENE_THRESHOLD})+gte(t-prev_selected_t\\,{grid})'"
    got = run(["ffmpeg", "-hide_banner", "-loglevel", "info", "-y", *src.input(), "-t", src.limit(),
               "-an", "-sn", "-vf", f"{scale_filter()},{select},showinfo", "-fps_mode", "vfr",
               "-frames:v", str(MAX_INDEX_FRAMES * 3), "-q:v", "5", str(frames / "f_%05d.jpg")],
              timeout=job.remaining(), pass_fds=src.fds)
    if got.returncode != 0:
        raise JobFailed("não foi possível extrair os frames do vídeo")
    times = [float(t) for t in SHOWINFO_TS_RE.findall(got.stderr)]
    files = sorted(frames.glob("f_*.jpg"))
    index, prev = [], None
    for path, t in zip(files, times):
        kind = "first" if prev is None else ("grid" if t - prev >= grid - 0.05 else "scene")
        index.append({"t": round(t, 2), "file": path.name, "kind": kind})
        prev = t
    for extra in files[len(index):]:  # no timestamp → not addressable
        extra.unlink()
    jobstore.write_json_atomic(job.dir / "index.json", {"frames": index, "grid_seconds": grid})


# ── stage 4: transcript ─────────────────────────────────────────────────────


def whisper_ready(timeout: float = 600.0) -> None:
    """Wait for watch-whisper (it may be restarting) before giving up on the job."""
    url = mcp_server.whisper_health_url()
    end = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        if time.monotonic() > end:
            raise JobFailed(f"o Whisper local (watch-whisper) não respondeu: {mcp_server.STACK_HINT}")
        time.sleep(10)


def transcribe(job: Job) -> None:
    plan = jobstore.read_json(job.dir / "plan.json") or {}
    chunks = plan.get("chunks") or []
    chunk_dir = job.dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    duration = float(plan.get("duration") or 0)
    model = job.data.get("model") or pick_model(job.data.get("quality", "auto"), duration)
    language = job.data.get("language")
    prompt = build_prompt(language, job.data.get("terms") or [])
    job.update(model=model, chunks_total=len(chunks), prompt_used=bool(prompt))
    started = time.monotonic()
    fresh_audio = 0.0
    for i, (start, end) in enumerate(chunks):
        out = chunk_dir / f"{i:04d}.json"
        if out.exists():
            continue
        done_audio = sum(e - s for s, e in chunks[:i])
        rate = fresh_audio / (time.monotonic() - started) if fresh_audio else None
        eta = (duration - done_audio) / rate if rate else None
        job.update(stage=f"transcrevendo ({i + 1}/{len(chunks)})", chunks_done=i,
                   progress=round(30 + 70 * done_audio / max(duration, 1), 1), eta_seconds=eta)
        piece = cut_audio(job, start, end, f"chunk_{i:04d}")
        segments = transcribe_chunk(job, piece, model=model, language=language, prompt=prompt)
        piece.unlink(missing_ok=True)
        if looks_repetitive(segments):
            log(f"{job.id}: repetition loop in chunk {i + 1}; re-transcribing in {LOOP_PIECE_SECONDS:.0f} s pieces")
            segments = []
            t = start
            while t < end - 0.5:
                stop = min(end, t + LOOP_PIECE_SECONDS)
                small = cut_audio(job, t, stop, f"chunk_{i:04d}_piece")
                part = transcribe_chunk(job, small, model=model, language=language, prompt=prompt)
                small.unlink(missing_ok=True)
                segments += [{**seg, "start": seg["start"] + t - start, "end": seg["end"] + t - start} for seg in part]
                t = stop
        segments = collapse_repeats(segments)
        shifted = [{"start": round(s["start"] + start, 2), "end": round(s["end"] + start, 2), "text": s["text"]}
                   for s in segments]
        jobstore.write_json_atomic(out, {"start": start, "end": end, "segments": shifted})
        fresh_audio += end - start
    segments = []
    for i in range(len(chunks)):
        segments += (jobstore.read_json(chunk_dir / f"{i:04d}.json") or {}).get("segments") or []
    jobstore.write_json_atomic(job.dir / "transcript.json", {
        "segments": segments, "model": model, "language": language, "prompt_used": bool(prompt),
    })


def cut_audio(job: Job, start: float, end: float, name: str) -> Path:
    piece = job.work / f"{name}.mp3"
    got = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inbox.input_args("mp3"),
               "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(job.work / "audio.mp3"),
               "-c", "copy", str(piece)], timeout=min(600, job.remaining()))
    if got.returncode != 0 or not piece.exists():
        raise JobFailed(f"não foi possível cortar o áudio em {start:.0f}s")
    return piece


def looks_repetitive(segments: list[dict]) -> bool:
    """LOOP_RUN identical consecutive segments, or one highly compressible segment."""
    run_length, previous = 1, None
    for segment in segments:
        text = textsearch.normalize(segment["text"])
        run_length = run_length + 1 if text and text == previous else 1
        if run_length >= LOOP_RUN:
            return True
        previous = text
        raw = segment["text"].encode("utf-8")
        if len(raw) >= LOOP_MIN_BYTES and len(raw) / len(zlib.compress(raw)) > LOOP_RATIO:
            return True
    return False


def collapse_repeats(segments: list[dict], keep: int = 2) -> list[dict]:
    """Drop the 3rd and later of identical consecutive segments (a person may repeat once)."""
    out, run_length, previous = [], 0, None
    for segment in segments:
        text = textsearch.normalize(segment["text"])
        run_length = run_length + 1 if text and text == previous else 1
        previous = text
        if run_length <= keep:
            out.append(segment)
    return out


def transcribe_chunk(job: Job, piece: Path, *, model: str, language: str | None, prompt: str | None) -> list[dict]:
    for wait in (*WHISPER_RETRIES, None):
        whisper_ready()
        try:
            return transcribe_local(piece, model=model, language=language, prompt=prompt)
        except SystemExit as exc:
            if wait is None or job.remaining() < wait:
                raise JobFailed(f"a transcrição falhou: {exc}") from None
            log(f"{job.id}: chunk {piece.name} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


# ── the job ─────────────────────────────────────────────────────────────────


def process(data: dict, *, root: Path | None = None, inbox_root: Path | None = None) -> None:
    job = Job(data, root)
    attempts = int(data.get("attempts") or 0) + 1
    if not data.get("started"):
        job.update(state="running", started=time.time(), stage="preparando", progress=1.0, attempts=attempts)
    else:
        job.update(state="running", stage="retomando", attempts=attempts)
    src = None
    try:
        if attempts > MAX_ATTEMPTS:
            raise JobFailed(f"o trabalho foi interrompido {MAX_ATTEMPTS} vezes no meio; desisti dele")
        if not ((job.dir / "plan.json").exists() and (job.dir / "index.json").exists()):
            src = open_source(job, inbox_root or inbox.INBOX_DIR)
            job.update(duration=src.info["duration"], has_video=src.info["has_video"])
            check_disk(job)
            extract_audio(job, src)
            build_index(job, src)
            src.close()
            src = None
            for leftover in [*job.work.glob("source.*"), job.work / "download.done"]:
                leftover.unlink(missing_ok=True)  # the whole video is not kept
        transcribe(job)
        shutil.rmtree(job.work, ignore_errors=True)
        job.update(state="done", stage="concluído", progress=100.0, eta_seconds=None, finished=time.time())
    except (JobFailed, inbox.InboxError, netguard.BlockedURL) as exc:
        fail(job, str(exc))
    except Exception as exc:  # noqa: BLE001 - recorded on the job, worker keeps running
        log(f"{job.id}: unexpected {type(exc).__name__}: {exc}")
        fail(job, f"erro interno no worker ({type(exc).__name__})")
    finally:
        if src is not None:
            src.close()


def fail(job: Job, message: str) -> None:
    shutil.rmtree(job.work, ignore_errors=True)
    job.update(state="failed", stage="falhou", error=mcp_server.scrub(message), eta_seconds=None,
               finished=time.time())


def check_disk(job: Job) -> None:
    used = sum(jobstore.dir_size(jobstore.job_path(j["id"], job.root)) for j in jobstore.list_jobs(job.root))
    if used > MAX_JOBS_BYTES:
        raise JobFailed(f"o cache de trabalhos passou do teto de {MAX_JOBS_BYTES / 1024**3:.0f} GB")


def heartbeat(every: float = 15.0) -> None:
    """Lets the stdio server tell "queued" from "nobody is processing"."""
    while True:
        try:
            jobstore.write_json_atomic(jobstore.JOBS_DIR / jobstore.HEARTBEAT_FILE, {"t": time.time()})
        except OSError as exc:
            log(f"heartbeat failed: {type(exc).__name__}")
        time.sleep(every)


def main() -> None:
    threading.Thread(target=heartbeat, daemon=True).start()
    log(f"started: jobs={jobstore.JOBS_DIR} inbox={inbox.INBOX_DIR} fast={MODEL_FAST} accurate={MODEL_ACCURATE}")
    last_cleanup = 0.0
    while True:
        if time.monotonic() - last_cleanup > CLEANUP_EVERY:
            removed = jobstore.cleanup(ttl_seconds=TTL_SECONDS, max_bytes=MAX_JOBS_BYTES)
            if removed:
                log(f"cleanup removed {len(removed)} job(s)")
            last_cleanup = time.monotonic()
        job = jobstore.next_job()
        if job is None:
            time.sleep(POLL_SECONDS)
            continue
        log(f"{job['id']}: {job.get('state')} → processing")
        process(job)
        log(f"{job['id']}: {jobstore.load_job(job['id']).get('state')}")


if __name__ == "__main__":
    main()
