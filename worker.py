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

v3: a done job then goes through postproc.py (LLM summary, a folder in the
Holocron's "00 - Entrada", archived state). `python /app/worker.py export
<job_id> [--again]` does the same for a job finished earlier.
"""
from __future__ import annotations

import json
import math
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
import hashlib
import zlib
from pathlib import Path

import chunking
import entrada
import inbox
import jobstore
import mcp_server
import netguard
import postproc
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
RETRY_EVERY = float(os.environ.get("WATCH_RETRY_EVERY", "60"))  # v3: pending notes/summaries
BOOT_ID = os.urandom(8).hex()  # tells this process's heartbeat from another worker's
OTHER_WORKER_STALE = 45.0
# Chosen by the pt-BR benchmark (NOTES-mcp.md, "v2"); overridable per deployment.
MODEL_FAST = os.environ.get("WATCH_MODEL_FAST", "Systran/faster-whisper-small")
MODEL_ACCURATE = os.environ.get("WATCH_MODEL_ACCURATE", "deepdml/faster-whisper-large-v3-turbo-ct2")
# Seconds of compute per second of audio measured on this CPU (8 cores, warm):
# the ETA before the first chunk of a run has finished.
SPEED_HINT = {
    "Systran/faster-whisper-small": 0.20,
    "Systran/faster-whisper-medium": 0.57,
    "deepdml/faster-whisper-large-v3-turbo-ct2": 0.53,
}
SHORT_SECONDS = 15 * 60  # quality="auto": up to this, the fast model
PROMPT_PTBR = (
    "Transcrição em português do Brasil, com pontuação e grafia brasileira: "
    "ônibus, celular, time, trem, registro, boleto, aluguel, a gente, tô, cê."
)
PROMPT_LIMIT = 800  # characters; Whisper only keeps the last ~224 prompt tokens
FRAME_LONG_SIDE = mcp_server.FRAME_LONG_SIDE
# Index size: one grid frame at least every duration/MAX_INDEX_FRAMES s (≥ 2 s)
# and scene cuts no closer than half that, so a job never exceeds
# ~3 x MAX_INDEX_FRAMES frames and the index always reaches the end.
MAX_INDEX_FRAMES = 2500
INDEX_FRAME_BYTES = 60_000  # generous per-JPEG estimate for the disk check
AUDIO_BYTES_PER_SECOND = 8_000  # 64 kb/s mp3
TITLE_LIMIT = 300
DISK_POLL = 1.0
NO_ROOM = ("sem espaço no cache de trabalhos (teto WATCH_JOBS_MAX_BYTES); "
           "apague trabalhos antigos ou aumente o teto")
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
TRANSIENT_RE = re.compile(r"HTTP Error (403|429|5\d\d)")


class JobFailed(Exception):
    """A clear, user-facing (Portuguese) reason the job cannot finish."""


def log(message: str) -> None:
    print(f"[watch-worker] {message}", file=sys.stderr, flush=True)


# ── subprocesses ────────────────────────────────────────────────────────────


def run(argv: list[str], *, timeout: float, pass_fds: tuple[int, ...] = (), env: dict | None = None,
        cwd: Path | None = None, disk: tuple[Path, int] | None = None) -> subprocess.CompletedProcess:
    """Run argv (never a shell) in its own process group; on timeout the whole
    group dies (yt-dlp spawns ffmpeg). `disk=(dir, room)`: the step is killed as
    soon as what it wrote under dir exceeds room bytes (real bytes, checked
    every DISK_POLL s — not an estimate)."""
    limit = jobstore.dir_size(disk[0]) + disk[1] if disk else None  # before the child writes
    end = time.monotonic() + max(1.0, timeout)
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", pass_fds=pass_fds, env=env,
        cwd=str(cwd) if cwd else None, start_new_session=os.name == "posix",
    )
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=max(0.05, min(DISK_POLL, end - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if disk and jobstore.dir_size(disk[0]) > limit:
                    _kill(proc)
                    proc.communicate()
                    raise JobFailed(NO_ROOM) from None
                if time.monotonic() >= end:
                    _kill(proc)
                    proc.communicate()
                    raise JobFailed(f"tempo limite excedido em {Path(argv[0]).name}") from None
        if disk and jobstore.dir_size(disk[0]) > limit:
            raise JobFailed(NO_ROOM)
    finally:
        _kill(proc)  # every path, the final size check included: strays must not outlive the step
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


def _finite(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def validate(data: dict) -> None:
    """Reject a job.json the worker did not write the way it expects (JobFailed)."""
    source = data.get("source")
    problems = []
    if not (isinstance(source, dict) and source.get("kind") in ("url", "inbox")
            and isinstance(source.get("value"), str) and source["value"]):
        problems.append("source")
    if data.get("language") is not None and not isinstance(data.get("language"), str):
        problems.append("language")
    if data.get("quality", "auto") not in ("auto", "fast", "accurate"):
        problems.append("quality")
    terms = data.get("terms") or []
    if not (isinstance(terms, list) and all(isinstance(t, str) for t in terms)):
        problems.append("terms")
    for key in ("started", "created", "updated"):
        if data.get(key) is not None and not _finite(data[key]):
            problems.append(key)
    attempts = data.get("attempts")
    if attempts is not None and not (isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 0):
        problems.append("attempts")
    for key in ("model", "fingerprint", "source_identity"):
        value = data.get(key)
        if value is not None and not isinstance(value, (str, list, dict)):
            problems.append(key)
    if problems:
        raise JobFailed(f"trabalho corrompido no cache (campos inválidos: {', '.join(problems)})")


class Job:
    def __init__(self, data: dict, root: Path | None = None):
        self.id = data["id"]
        self.root = root
        self.dir = jobstore.job_path(self.id, root)
        self.data = data
        started = data.get("started") if _finite(data.get("started")) else time.time()
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


def cache_used(root: Path | None) -> int:
    """Bytes of the working cache; archived jobs (v3) are kept apart from the cap."""
    jobs = sum(jobstore.dir_size(jobstore.job_path(j["id"], root)) for j in jobstore.list_jobs(root)
               if j.get("state") != jobstore.ARCHIVED)
    trash = (root or jobstore.JOBS_DIR) / jobstore.TRASH_DIR  # leftovers of a failed rmtree still use disk
    return jobs + (jobstore.dir_size(trash) if trash.is_dir() and not jobstore.is_link(trash) else 0)


def ensure_room(job: Job, need: int) -> int:
    """Free room under WATCH_JOBS_MAX_BYTES, checked to fit `need` more bytes.

    Finished jobs are evicted first (oldest first); queued/running never are.
    """
    used = cache_used(job.root)
    if used + need > MAX_JOBS_BYTES:
        jobstore.cleanup(job.root, ttl_seconds=TTL_SECONDS, max_bytes=max(0, MAX_JOBS_BYTES - need))
        used = cache_used(job.root)
    if used + need > MAX_JOBS_BYTES:
        raise JobFailed(
            f"sem espaço no cache de trabalhos (teto de {MAX_JOBS_BYTES / 1024**3:.1f} GB, "
            f"em uso {used / 1024**3:.2f} GB); apague trabalhos antigos ou aumente WATCH_JOBS_MAX_BYTES"
        )
    return MAX_JOBS_BYTES - used


def pick_model(quality: str, duration: float) -> str:
    if quality == "fast":
        return MODEL_FAST
    if quality == "accurate":
        return MODEL_ACCURATE
    return MODEL_FAST if duration <= SHORT_SECONDS else MODEL_ACCURATE


def build_prompt(language: str | None, terms: list[str]) -> str | None:
    """Initial prompt, only when the caller gave terms (the benchmark: the
    Brazilian context alone did not help consistently; with the terms, names
    and jargon improved). pt jobs get the Brazilian context in front of them."""
    if not terms:
        return None
    if language != "pt":
        return ", ".join(terms)[:PROMPT_LIMIT]
    return f"{PROMPT_PTBR} Termos: {', '.join(terms)}."[:PROMPT_LIMIT]


def scale_filter() -> str:
    side = FRAME_LONG_SIDE
    return (f"scale='if(gte(iw,ih),min({side},iw),-2)':'if(gte(iw,ih),-2,min({side},ih))'")


def grid_interval(duration: float) -> float:
    """At least one index frame this often; ≤ MAX_INDEX_FRAMES grid frames per job."""
    return max(2.0, duration / MAX_INDEX_FRAMES)


def index_bound(duration: float) -> int:
    """Upper bound of index frames: consecutive frames are ≥ grid/2 apart."""
    return int(duration / (grid_interval(duration) / 2)) + 2


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

    def identity(self) -> dict:
        """Content identity of the open file (SHA-256 through the fd)."""
        digest, offset = hashlib.sha256(), 0
        while True:
            if hasattr(os, "pread"):
                block = os.pread(self.fd, 1 << 20, offset)
            else:  # Windows (development)
                os.lseek(self.fd, offset, os.SEEK_SET)
                block = os.read(self.fd, 1 << 20)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        return {"size": offset, "sha256": digest.hexdigest(), "duration": round(self.info["duration"], 3)}


def open_checked(name: str, root: Path, timeout: float = 120.0) -> Source:
    fd = inbox.open_media(name, root)
    src = Source(fd, {}, Path(root) / name)
    try:
        src.info = inbox.probe(src.path, pass_fds=src.fds, timeout=min(120.0, timeout))
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
    room = ensure_room(job, 0)
    budget = min(MAX_DOWNLOAD_BYTES, room)
    tmp = Path(tempfile.mkdtemp(prefix="watch-worker-"))
    try:
        cookies = private_cookies(tmp)
        secrets = mcp_server.cookie_secrets(cookies)
        with netguard.EgressProxy(max_bytes=budget) as proxy:
            extra = mcp_server.ytdlp_args(
                proxy.url, cookies, max_duration=MAX_DURATION, max_filesize=f"{budget}",
            )
            env = mcp_server.child_env({})
            meta = run(["yt-dlp", *extra, "--skip-download", "--dump-single-json", "--no-playlist",
                        "--no-warnings", "--", url], timeout=min(600, job.remaining()), env=env, cwd=tmp)
            info = _metadata(url, meta, secrets, proxy)
            shown = write_meta(job, info, secrets)
            job.update(stage="baixando", progress=2.0, title=shown.get("title"),
                       duration=info.get("duration") if _finite(info.get("duration")) else None)
            for attempt in (1, 2):
                got = run(["yt-dlp", *extra, "-N", "4", "-f", DOWNLOAD_FORMAT, "--merge-output-format", "mkv",
                           "--no-playlist", "-o", str(work / "source.%(ext)s"), "--", url],
                          timeout=job.remaining(), env=env, cwd=tmp, disk=(work, room))
                # A media 403/5xx is often transient (signed CDN URLs); one fresh run
                # re-extracts them. Same proxy, so the byte budget stays aggregate.
                if got.returncode == 0 or attempt == 2 or proxy.exceeded or not TRANSIENT_RE.search(got.stderr):
                    break
                log(f"{job.id}: transient download error; retrying once")
                for partial in work.glob("source.*"):
                    partial.unlink()
            if proxy.exceeded:
                if budget < MAX_DOWNLOAD_BYTES:
                    raise JobFailed(f"sem espaço no cache de trabalhos para o download "
                                    f"(cabiam {budget / 1024**3:.2f} GB; teto WATCH_JOBS_MAX_BYTES)")
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


def write_meta(job: Job, info: dict, secrets: frozenset[str] = frozenset()) -> dict:
    """Only display fields, with every cookie value redacted (a site can echo the
    jar into its title or description); the raw yt-dlp JSON is not kept."""
    meta: dict = {}
    for key in ("title", "uploader", "channel", "uploader_id", "webpage_url", "upload_date"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            meta[key] = mcp_server.scrub(value.strip(), secrets)[:TITLE_LIMIT]
    for key in ("duration", "width", "height"):
        if _finite(info.get(key)):
            meta[key] = info[key]
    description = info.get("description")
    if isinstance(description, str) and description.strip():
        meta["description"] = mcp_server.scrub(description.strip(), secrets)[:mcp_server.DESCRIPTION_LIMIT]
    jobstore.write_json_atomic(job.dir / "meta.json", meta)
    return meta


def open_source(job: Job, inbox_root: Path) -> Source:
    source = job.data["source"]
    if source["kind"] == "inbox":
        jobstore.write_json_atomic(job.dir / "meta.json", {"title": source["value"][:TITLE_LIMIT]})
        src = open_checked(source["value"], inbox_root, job.remaining())
    else:
        path = download(job, source["value"])
        # The downloaded file is untrusted too: same probe, same fd discipline.
        src = open_checked(path.name, path.parent, job.remaining())
    try:
        check_identity(job, src)
    except BaseException:
        src.close()
        raise
    return src


def check_identity(job: Job, src: Source) -> None:
    """Artifacts of one job come from one file: the one the server saw (inbox
    fingerprint) and the one opened first (resume after a restart)."""
    changed = JobFailed("o arquivo de origem mudou durante o processamento; rode analyze_video de novo")
    seen = job.data.get("fingerprint")
    if job.data["source"]["kind"] == "inbox" and seen:
        st = os.fstat(src.fd)
        if list(seen) != [st.st_size, st.st_mtime_ns]:
            raise changed
    job.update(stage="verificando o arquivo")
    now = src.identity()
    first = job.data.get("source_identity")
    if first is None:
        job.update(source_identity=now)
    elif first != now:
        raise changed


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
    need = int((duration + 10) * AUDIO_BYTES_PER_SECOND * 1.2) + 65_536
    room = ensure_room(job, need)
    tmp = job.work / "audio.tmp.mp3"
    got = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *src.input(), "-t", src.limit(),
               "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k", "-fs", str(need),
               str(tmp)], timeout=job.remaining(), pass_fds=src.fds, disk=(job.work, room))
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
    duration = src.info["duration"]
    bound = index_bound(duration)
    room = ensure_room(job, bound * INDEX_FRAME_BYTES)
    frames.mkdir()
    job.update(stage="indexando cenas", progress=20.0)
    grid = grid_interval(duration)
    # Grid frames keep coverage to the end; scene cuts only count when at least
    # grid/2 after the previous frame, which bounds the total (no -frames:v cap
    # that would stop before the end).
    select = (f"select='eq(n\\,0)+gte(t-prev_selected_t\\,{grid})"
              f"+gt(scene\\,{SCENE_THRESHOLD})*gte(t-prev_selected_t\\,{grid / 2})'")
    got = run(["ffmpeg", "-hide_banner", "-loglevel", "info", "-y", *src.input(), "-t", src.limit(),
               "-an", "-sn", "-vf", f"{scale_filter()},{select},showinfo", "-fps_mode", "vfr",
               "-frames:v", str(bound), "-q:v", "5", str(frames / "f_%05d.jpg")],
              timeout=job.remaining(), pass_fds=src.fds, disk=(frames, room))
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
    """Wait for watch-whisper (it may be restarting), never past `timeout`."""
    url = mcp_server.whisper_health_url()
    end = time.monotonic() + timeout
    while True:
        left = end - time.monotonic()
        if left <= 0:
            raise JobFailed(f"o Whisper local (watch-whisper) não respondeu a tempo: {mcp_server.STACK_HINT}")
        try:
            # Bounded as a whole (connect, headers): a socket timeout is per operation.
            if within(min(5.0, left), lambda: _health_ok(url, min(5.0, left))):
                return
        except JobFailed:
            pass
        time.sleep(max(0.0, min(10.0, end - time.monotonic())))


def _health_ok(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


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
        if valid_segments((jobstore.read_json(out) or {}).get("segments")):
            continue  # finished before a restart (a corrupt file is redone)
        done_audio = sum(e - s for s, e in chunks[:i])
        rate = fresh_audio / (time.monotonic() - started) if fresh_audio else None
        eta = (duration - done_audio) / rate if rate else (duration - done_audio) * SPEED_HINT.get(model, 0.6)
        job.update(stage=f"transcrevendo ({i + 1}/{len(chunks)})", chunks_done=i,
                   progress=round(30 + 70 * done_audio / max(duration, 1), 1), eta_seconds=eta)
        room = ensure_room(job, int((end - start) * AUDIO_BYTES_PER_SECOND * 1.2) + 200_000)
        piece = cut_audio(job, start, end, f"chunk_{i:04d}", room)
        segments = transcribe_chunk(job, piece, model=model, language=language, prompt=prompt)
        piece.unlink(missing_ok=True)
        if looks_repetitive(segments):
            log(f"{job.id}: repetition loop in chunk {i + 1}; re-transcribing in {LOOP_PIECE_SECONDS:.0f} s pieces")
            segments = []
            t = start
            while t < end - 0.5:
                stop = min(end, t + LOOP_PIECE_SECONDS)
                small = cut_audio(job, t, stop, f"chunk_{i:04d}_piece", room)
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


def cut_audio(job: Job, start: float, end: float, name: str, room: int) -> Path:
    piece = job.work / f"{name}.mp3"
    got = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inbox.input_args("mp3"),
               "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(job.work / "audio.mp3"),
               "-c", "copy", str(piece)], timeout=min(600, job.remaining()), disk=(job.work, room))
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


def within(seconds: float, call):
    """Run call() but stop waiting after `seconds`: a socket timeout is per
    operation, so a slow upload or a trickling reply could otherwise hold the
    single worker long past the job deadline. The abandoned request ends on its
    own socket timeout."""
    box: dict = {}

    def target():
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)  # an abandoned call never blocks exit
    thread.start()
    thread.join(max(0.0, seconds))
    if thread.is_alive():
        raise JobFailed(f"tempo total do trabalho excedido ({JOB_TIMEOUT / 3600:.1f} h): "
                        "o Whisper não respondeu a tempo")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def valid_segments(segments) -> bool:
    return isinstance(segments, list) and all(
        isinstance(s, dict) and _finite(s.get("start")) and _finite(s.get("end"))
        and 0 <= s["start"] <= s["end"] and isinstance(s.get("text"), str)
        for s in segments
    )


def transcribe_chunk(job: Job, piece: Path, *, model: str, language: str | None, prompt: str | None) -> list[dict]:
    """One chunk, entirely inside the job deadline (health wait, request, retries)."""
    for wait in (*WHISPER_RETRIES, None):
        whisper_ready(timeout=min(600.0, job.remaining()))
        try:
            left = job.remaining()
            segments = within(left, lambda: transcribe_local(
                piece, model=model, language=language, prompt=prompt, deadline=time.monotonic() + left))
            job.remaining()  # a reply that lands after the deadline does not count
            if not valid_segments(segments):
                raise SystemExit("resposta do Whisper em formato inesperado")
            return segments
        except SystemExit as exc:
            if wait is None or job.remaining() < wait:
                raise JobFailed(f"a transcrição falhou: {exc}") from None
            log(f"{job.id}: chunk {piece.name} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


# ── the job ─────────────────────────────────────────────────────────────────


def process(data: dict, *, root: Path | None = None, inbox_root: Path | None = None) -> None:
    try:
        validate(data)
    except JobFailed as exc:
        jobstore.update_job(data["id"], root, state="failed", stage="falhou", error=str(exc), finished=time.time())
        return
    job = Job(data, root)
    attempts = (data.get("attempts") or 0) + 1
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
            extract_audio(job, src)
            build_index(job, src)
            src.close()
            src = None
        # Runs on every pass, also when a restart skipped the stages above.
        for leftover in [*job.work.glob("source.*"), job.work / "download.done"]:
            leftover.unlink(missing_ok=True)  # the whole video is not kept
        transcribe(job)
        shutil.rmtree(job.work, ignore_errors=True)
        # export="pendente" first: a crash before postproc.finish still gets the note later.
        job.update(state="done", stage="concluído", progress=100.0, eta_seconds=None, finished=time.time(),
                   export="pendente")
    except (JobFailed, inbox.InboxError, netguard.BlockedURL) as exc:
        fail(job, str(exc))
    except Exception as exc:  # noqa: BLE001 - recorded on the job, worker keeps running
        log(f"{job.id}: unexpected {type(exc).__name__}: {exc}")
        fail(job, f"erro interno no worker ({type(exc).__name__})")
    finally:
        if src is not None:
            src.close()


def fail(job: Job, message: str) -> None:
    # A failed job is never resumed: keep only job.json/meta.json, free the rest
    # (including what a step wrote before it was stopped for lack of room).
    for heavy in (job.work, job.dir / "frames", job.dir / "chunks"):
        shutil.rmtree(heavy, ignore_errors=True)
    job.update(state="failed", stage="falhou", error=mcp_server.scrub(message), eta_seconds=None,
               finished=time.time())


def heartbeat(every: float = 15.0) -> None:
    """Lets the stdio server tell "queued" from "nobody is processing"."""
    while True:
        try:
            jobstore.write_json_atomic(jobstore.JOBS_DIR / jobstore.HEARTBEAT_FILE,
                                       {"t": time.time(), "worker": "v3", "boot": BOOT_ID})
        except OSError as exc:
            log(f"heartbeat failed: {type(exc).__name__}")
        time.sleep(every)


WORKER_LOCK = ".worker.lock"


def lock_worker(root: Path | None = None, sleep=time.sleep):
    """Exclusive ownership of the queue among v3 workers (flock, held for the
    process lifetime; released by the kernel if the process dies). The v2 worker
    does not take it, hence also wait_for_other_worker()."""
    try:
        import fcntl
    except ImportError:  # Windows test runs
        return None
    fd = os.open((root or jobstore.JOBS_DIR) / WORKER_LOCK, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    warned = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if not warned:
                log("another v3 worker holds the queue; waiting")
                warned = True
            sleep(5)


def wait_for_other_worker(root: Path | None = None, sleep=time.sleep) -> None:
    """Two workers on one queue would process the same job twice: while another
    worker's heartbeat is fresh (e.g. the v2 container still running), wait."""
    warned = False
    while True:
        beat = jobstore.read_json((root or jobstore.JOBS_DIR) / jobstore.HEARTBEAT_FILE) or {}
        t = jobstore.as_number(beat.get("t"))
        if beat.get("boot") == BOOT_ID or time.time() - t > OTHER_WORKER_STALE:
            return
        if not warned:
            log("another worker's heartbeat is fresh; waiting until it stops")
            warned = True
        sleep(5)


def export_cli(argv: list[str]) -> int:
    """python /app/worker.py export <job_id> [--again]"""
    args = [a for a in argv if a != "--again"]
    if len(args) != 1:
        print("uso: worker.py export <job_id> [--again]", file=sys.stderr)
        return 2
    try:
        folder = postproc.run_export(args[0], again="--again" in argv)
    except (jobstore.JobError, entrada.EntradaError, postproc.Busy) as exc:
        print(f"recusado: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - its text could carry a secret: type only
        print(f"erro interno ({type(exc).__name__})", file=sys.stderr)
        return 4
    if folder is None:
        job = jobstore.load_job(args[0])
        print(f"nota pendente: {job.get('export_error')}", file=sys.stderr)
        return 3
    job = jobstore.load_job(args[0])
    print(f"ok: {postproc.jobview.ENTRADA_LABEL}/{folder} (resumo: {job.get('resumo')})")
    return 0


def main() -> None:
    if sys.argv[1:2] == ["export"]:
        sys.exit(export_cli(sys.argv[2:]))
    lock = lock_worker()  # noqa: F841 - kept open for the process lifetime
    wait_for_other_worker()
    threading.Thread(target=heartbeat, daemon=True).start()
    log(f"started: jobs={jobstore.JOBS_DIR} inbox={inbox.INBOX_DIR} fast={MODEL_FAST} accurate={MODEL_ACCURATE}")
    log(f"v3: entrada={entrada.ENTRADA_DIR} ({'ok' if entrada.available() else 'indisponível'}) "
        f"resumo={postproc.resumo.PROVIDER}:{postproc.resumo.GEMINI_MODEL} "
        f"caixa_de_entrada={'sim' if postproc.RESUMO_CAIXA_ENTRADA else 'não'}")
    removed = entrada.clean_orphans()
    if removed:
        log(f"removed {len(removed)} orphan staging dir(s) in {entrada.TMP_NAME}")
    last_cleanup = last_retry = 0.0
    while True:
        if time.monotonic() - last_retry > RETRY_EVERY:
            last_retry = time.monotonic()
            postproc.retry_due()
        try:
            if time.monotonic() - last_cleanup > CLEANUP_EVERY:
                last_cleanup = time.monotonic()
                removed = jobstore.cleanup(ttl_seconds=TTL_SECONDS, max_bytes=MAX_JOBS_BYTES)
                if removed:
                    log(f"cleanup removed {len(removed)} job(s)")
            job = jobstore.next_job()
        except Exception as exc:  # noqa: BLE001 - a corrupt queue entry must not stop the worker
            log(f"queue scan failed ({type(exc).__name__}); retrying")
            time.sleep(POLL_SECONDS)
            continue
        if job is None:
            time.sleep(POLL_SECONDS)
            continue
        log(f"{job['id']}: {job.get('state')} → processing")
        try:
            process(job)
            state = jobstore.load_job(job["id"]).get("state")
            log(f"{job['id']}: {state}")
            if state == "done":
                postproc.finish(job["id"])
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the queue
            log(f"{job['id']}: worker error {type(exc).__name__}; marking it failed")
            try:
                jobstore.update_job(job["id"], state="failed", stage="falhou",
                                    error=f"erro interno no worker ({type(exc).__name__})", finished=time.time())
            except Exception as again:  # noqa: BLE001
                log(f"{job['id']}: could not mark failed ({type(again).__name__})")
                time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
