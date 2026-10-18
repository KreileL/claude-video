#!/usr/bin/env python3
"""Transcribe a video via Groq or OpenAI Whisper API.

Strategy: extract audio (mono 16kHz mp3, tiny payload), upload to whichever
API has a key. Returns segments in the same shape as transcribe.parse_vtt so
the rest of the pipeline (filter_range, format_transcript) doesn't care where
the transcript came from.

Pure stdlib — no `pip install groq` or `pip install openai` needed.
"""
from __future__ import annotations

import io
import json
import math
import mimetypes
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

OPENAI_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"

# Both Groq's free tier and OpenAI whisper-1 cap uploads at 25 MB. We target a
# margin under that so multipart framing overhead never pushes a chunk over.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

# A local OpenAI-compatible server (speaches, faster-whisper-server, LM Studio,
# whisper.cpp's whisper-server, …) speaks the same multipart protocol as the
# hosted APIs, so pointing at one needs no new backend — only these overrides.
# Set WATCH_WHISPER_BASE_URL to the server's /v1 root to switch.
BASE_URL_KEY = "WATCH_WHISPER_BASE_URL"
MODEL_KEY = "WATCH_WHISPER_MODEL"
MAX_UPLOAD_KEY = "WATCH_WHISPER_MAX_UPLOAD_MB"
# Deliberately separate from OPENAI_API_KEY: the base URL can point at any host,
# so forwarding the real OpenAI credential there would leak it to a third party.
API_KEY_KEY = "WATCH_WHISPER_API_KEY"
# Full transcription endpoint of a local server (wins over BASE_URL_KEY), for
# servers whose route isn't <base>/audio/transcriptions or that sit on a
# compose-network hostname.
URL_KEY = "WATCH_WHISPER_URL"
DEFAULT_LOCAL_URL = "http://watch-whisper:8000/v1/audio/transcriptions"
DEFAULT_LOCAL_MODEL = "Systran/faster-whisper-small"
TIMEOUT_KEY = "WATCH_WHISPER_TIMEOUT"
DEFAULT_TIMEOUT = 300.0
# Stands in for the key on a local server — a placeholder, not a credential, so
# it must never be sent as an Authorization header.
LOCAL_NO_KEY = "local"


def _dotenv_paths() -> list[Path]:
    # Resolved per call, not at import: cwd can change between invocations.
    return [Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"]


def _from_dotenv(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return value or None
    except OSError:
        return None
    return None


def setting(name: str) -> str | None:
    """Read a setting from the environment, falling back to the .env files."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for path in _dotenv_paths():
        value = _from_dotenv(path, name)
        if value:
            return value
    return None


def local_endpoint() -> str:
    """Local server endpoint: explicit URL, else <base>/audio/transcriptions,
    else the compose-stack default."""
    url = setting(URL_KEY)
    if url:
        return url
    base = setting(BASE_URL_KEY)
    if base:
        return f"{base.rstrip('/')}/audio/transcriptions"
    return DEFAULT_LOCAL_URL


def local_model() -> str:
    return setting(MODEL_KEY) or DEFAULT_LOCAL_MODEL


def request_timeout() -> float:
    """Per-request timeout in seconds; CPU Whisper on long audio can be slow."""
    raw = setting(TIMEOUT_KEY)
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        seconds = float(raw)
    except ValueError:
        print(
            f"[watch] ignoring invalid {TIMEOUT_KEY}={raw!r} (want a number of seconds)",
            file=sys.stderr,
        )
        return DEFAULT_TIMEOUT
    if seconds <= 0:
        print(f"[watch] ignoring non-positive {TIMEOUT_KEY}={raw!r}", file=sys.stderr)
        return DEFAULT_TIMEOUT
    return seconds


def max_upload_bytes() -> int:
    """Upload cap. The 24 MB default exists only because the hosted APIs
    enforce 25 MB; a local server has no such limit, so raising this avoids
    pointless chunking (and the seam artifacts that come with it)."""
    raw = setting(MAX_UPLOAD_KEY)
    if not raw:
        return MAX_UPLOAD_BYTES
    try:
        megabytes = float(raw)
    except ValueError:
        print(
            f"[watch] ignoring invalid {MAX_UPLOAD_KEY}={raw!r} (want a number of MB)",
            file=sys.stderr,
        )
        return MAX_UPLOAD_BYTES
    if megabytes <= 0:
        print(f"[watch] ignoring non-positive {MAX_UPLOAD_KEY}={raw!r}", file=sys.stderr)
        return MAX_UPLOAD_BYTES
    return int(megabytes * 1024 * 1024)


def plan_chunks(
    total_seconds: float,
    total_bytes: int,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> list[tuple[float, float]]:
    """Split a duration into contiguous (offset, duration) chunks under max_bytes.

    Size scales linearly with duration (constant-bitrate mono mp3), so an even
    time split yields evenly-sized chunks. Returns a single full-length chunk
    when the audio already fits.
    """
    if total_bytes <= max_bytes or total_seconds <= 0:
        return [(0.0, total_seconds)]

    n = math.ceil(total_bytes / max_bytes)
    chunk = total_seconds / n
    plan: list[tuple[float, float]] = []
    for i in range(n):
        offset = i * chunk
        # The last chunk absorbs any rounding remainder so durations sum exactly.
        duration = (total_seconds - offset) if i == n - 1 else chunk
        plan.append((round(offset, 3), round(duration, 3)))
    return plan


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Return (backend, api_key). Prefers a local server, then Groq, then OpenAI.

    If `preferred` is "groq", "openai" or "local", only that backend is considered.

    A configured local server outranks a hosted key: it costs nothing, keeps the
    audio on the machine, and having set it is an explicit choice. `--whisper
    groq|openai` is the escape hatch when both are configured, and then goes to
    that hosted API itself, never to the local endpoint. Local servers generally
    ignore Authorization, so a placeholder stands in for the key rather than
    blocking on a credential that carries no meaning. OPENAI_API_KEY is never
    reused here — the base URL can name any host, and the real credential must
    not follow it there. A server that does want auth takes WATCH_WHISPER_API_KEY.

    A local server counts as configured via WATCH_WHISPER_URL or
    WATCH_WHISPER_BASE_URL. Forcing `preferred="local"` with neither set means
    "use the default endpoint" (DEFAULT_LOCAL_URL), so it still returns local.
    """
    local_configured = setting(URL_KEY) or setting(BASE_URL_KEY)
    if preferred == "local" or (local_configured and preferred is None):
        return "local", setting(API_KEY_KEY) or LOCAL_NO_KEY

    candidates = (("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"))
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = setting(key_name)
        if value:
            return backend, value

    return None, None


def extract_audio(video_path: str, out_path: Path) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits any Whisper limit."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


def audio_duration(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")
    fmt = json.loads(result.stdout or "{}").get("format", {})
    return float(fmt.get("duration") or 0.0)


def split_audio(
    full_audio: Path,
    work_dir: Path,
    plan: list[tuple[float, float]],
) -> list[tuple[Path, float]]:
    """Slice full_audio into per-plan chunk files, returning (path, offset) pairs.

    Uses stream copy (`-c copy`) so there is no re-encode and no quality loss;
    mp3 frame boundaries are close enough for transcription's purposes.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    work_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, float]] = []
    for index, (offset, duration) in enumerate(plan):
        out_path = work_dir / f"chunk_{index:03d}.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{offset:.3f}",
            "-i", str(full_audio.resolve()),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(out_path.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise SystemExit(
                f"ffmpeg failed to split audio chunk {index + 1}: {result.stderr.strip()}"
            )
        chunks.append((out_path, offset))
    return chunks


def _build_multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body the Whisper APIs accept.

    Whisper's multipart upload is small and predictable — doing it by hand
    keeps us on pure stdlib instead of pulling requests/groq/openai SDKs.
    """
    boundary = f"----WatchBoundary{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()

    for name, value in fields.items():
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)

    mimetype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mimetype}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(file_path.read_bytes())
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    return buf.getvalue(), boundary


MAX_ATTEMPTS = 4       # initial + 3 retries
MAX_429_RETRIES = 2
RETRY_BASE_DELAY = 2.0


def _post_whisper(
    endpoint: str,
    api_key: str | None,
    model: str,
    audio_path: Path,
    extra: dict[str, str | None] | None = None,
    deadline: float | None = None,
) -> dict:
    """POST one file, retrying transient failures. `deadline` (time.monotonic())
    bounds every attempt and backoff; past it, SystemExit."""
    fields = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    # Optional OpenAI-API form fields (language, prompt); empty ones are not sent.
    fields.update({k: v for k, v in (extra or {}).items() if v})
    body, boundary = _build_multipart(fields, audio_path)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        # Groq sits behind Cloudflare — the default `Python-urllib/3.x` UA
        # trips WAF rule 1010 (403) before auth even runs. Any non-default
        # UA clears it; we identify honestly.
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = request_timeout()
    context = ssl.create_default_context()
    rate_limit_hits = 0
    last_exc: Exception | None = None
    last_detail = ""

    def time_left() -> float:
        if deadline is None:
            return timeout
        left = deadline - time.monotonic()
        if left <= 0:
            raise SystemExit(f"Whisper request stopped: deadline exceeded ({last_exc or 'no attempt left'})")
        return min(timeout, left)

    def backoff(delay: float) -> None:
        if deadline is not None and time.monotonic() + delay >= deadline:
            raise SystemExit(f"Whisper request failed before the deadline: {last_exc}{last_detail}")
        time.sleep(delay)

    for attempt in range(MAX_ATTEMPTS):
        request = Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=time_left(), context=context) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            last_exc, last_detail = exc, detail

            # 4xx other than 429 are client errors — no retry will fix them.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"Whisper request failed: {exc}{detail}")

            if exc.code == 429:
                rate_limit_hits += 1
                if rate_limit_hits >= MAX_429_RETRIES:
                    raise SystemExit(f"Whisper request failed: {exc}{detail}")
                delay = _retry_after(exc) or RETRY_BASE_DELAY * (2 ** attempt) + 1
            else:
                delay = RETRY_BASE_DELAY * (2 ** attempt)

            if attempt < MAX_ATTEMPTS - 1:
                print(
                    f"[watch] whisper HTTP {exc.code} — retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                backoff(delay)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc, last_detail = exc, ""
            if attempt < MAX_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(
                    f"[watch] whisper network error ({type(exc).__name__}: {exc}) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                backoff(delay)
            continue

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Whisper returned non-JSON response: {exc}: {payload[:200]}")

    raise SystemExit(
        f"Whisper request failed after {MAX_ATTEMPTS} attempts: {last_exc}{last_detail}"
    )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        # One line: a multi-line body must not be able to forge a status line of watch.py.
        return f" — {' '.join(body.decode('utf-8', errors='replace').split())[:400]}"
    except Exception:
        return ""


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def shift_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Return a copy of segments with start/end shifted by offset_seconds.

    Each chunk is transcribed in isolation, so Whisper returns 0-based timestamps
    per chunk; shifting by the chunk's offset stitches them into source time.
    """
    if offset_seconds == 0:
        return segments
    return [
        {
            "start": round(seg["start"] + offset_seconds, 2),
            "end": round(seg["end"] + offset_seconds, 2),
            "text": seg["text"],
        }
        for seg in segments
    ]


def _valid_segment(seg) -> bool:
    # Each segment carries its text as a string ("" is legitimate silence); anything
    # else (null, missing, an error object) is a broken answer, not silence.
    return isinstance(seg, dict) and isinstance(seg.get("text"), str) and "error" not in seg


def require_transcription(data) -> dict:
    """A body that is really a transcription: an error object (even with HTTP 200)
    must not read as "no speech" (MCP v3: silence is reported, errors fail)."""
    valid = (isinstance(data, dict) and "error" not in data and ("segments" in data or "text" in data)
             and ("text" not in data or isinstance(data["text"], str))
             and ("segments" not in data or (isinstance(data["segments"], list)
                                            and all(_valid_segment(s) for s in data["segments"]))))
    if not valid:
        raise SystemExit("Whisper returned an invalid response (no transcription in it)")
    return data


def _segments_from_response(data: dict) -> list[dict]:
    """Convert Whisper verbose_json into our {start, end, text} segment format."""
    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start") or 0.0), 2),
            "end": round(float(seg.get("end") or 0.0), 2),
            "text": text,
        })

    if not out:
        full = (data.get("text") or "").strip()
        if full:
            out.append({"start": 0.0, "end": 0.0, "text": full})

    return out


def transcribe_chunks(
    chunks: list[tuple[Path, float]],
    transcribe_one,
) -> list[dict]:
    """Transcribe each chunk, shift its segments by the chunk offset, concatenate.

    A chunk that fails after its own retries is logged and skipped so one bad
    slice doesn't discard the whole transcript. Raises only if every chunk fails.
    """
    segments: list[dict] = []
    failures = 0
    for index, (path, offset) in enumerate(chunks):
        try:
            chunk_segments = transcribe_one(path)
        except SystemExit as exc:
            failures += 1
            print(
                f"[watch] chunk {index + 1}/{len(chunks)} failed — skipping ({exc})",
                file=sys.stderr,
            )
            continue
        segments.extend(shift_segments(chunk_segments, offset))
        print(
            f"[watch] chunk {index + 1}/{len(chunks)} → {len(chunk_segments)} segments",
            file=sys.stderr,
        )

    if failures == len(chunks):
        raise SystemExit("Whisper failed on every audio chunk")
    if failures and not segments:  # an error must not pass for silence
        raise SystemExit(f"Whisper failed on {failures} of {len(chunks)} audio chunks; the others had no speech")
    return segments


def _transcribe_file(backend: str, api_key: str, audio_path: Path) -> list[dict]:
    """Upload one audio file and return its 0-based segments."""
    if backend == "groq":
        response = _post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path)
    elif backend == "local":
        response = _post_whisper(
            local_endpoint(),
            None if api_key == LOCAL_NO_KEY else api_key,
            local_model(),
            audio_path,
            _forced_language(),
        )
    elif backend == "openai":
        # Always OpenAI itself: local overrides must never redirect the real key.
        response = _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")
    return _segments_from_response(require_transcription(response))


def transcribe_local(
    audio_path: Path,
    *,
    model: str,
    language: str | None = None,
    prompt: str | None = None,
    deadline: float | None = None,
) -> list[dict]:
    """One request to the local server with an explicit model, language and
    initial prompt; returns 0-based segments. For callers that do their own
    chunking (the MCP v2 worker). Raises SystemExit like the rest of this module.
    """
    response = _post_whisper(
        local_endpoint(),
        setting(API_KEY_KEY),
        model,
        audio_path,
        {"language": language, "prompt": prompt},
        deadline=deadline,
    )
    return _segments_from_response(require_transcription(response))


LANGUAGE_NAMES = {  # OpenAI-style servers answer full names, faster-whisper ones codes
    "portuguese": "pt", "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "japanese": "ja", "chinese": "zh", "korean": "ko", "russian": "ru",
    "dutch": "nl", "polish": "pl", "turkish": "tr", "arabic": "ar", "hindi": "hi",
    "swedish": "sv", "ukrainian": "uk", "greek": "el", "hebrew": "he", "catalan": "ca",
    "czech": "cs", "danish": "da", "finnish": "fi", "norwegian": "no", "romanian": "ro",
    "hungarian": "hu", "indonesian": "id", "vietnamese": "vi", "thai": "th", "galician": "gl",
    "afrikaans": "af", "malay": "ms", "persian": "fa", "bulgarian": "bg", "croatian": "hr",
    "serbian": "sr", "slovak": "sk", "slovenian": "sl", "lithuanian": "lt", "latvian": "lv",
    "estonian": "et", "tagalog": "tl", "urdu": "ur", "bengali": "bn", "tamil": "ta",
    "basque": "eu", "welsh": "cy", "swahili": "sw", "icelandic": "is", "armenian": "hy",
}


def detect_language_local(audio_path: Path, *, model: str, deadline: float | None = None) -> tuple[str | None, bool]:
    """(language code, has_speech) for one sample, with no language forced: the
    server detects it. For the MCP worker's majority vote over several samples."""
    response = require_transcription(
        _post_whisper(local_endpoint(), setting(API_KEY_KEY), model, audio_path, deadline=deadline))
    raw = str(response.get("language") or "").strip().lower()
    code = LANGUAGE_NAMES.get(raw, raw)
    has_speech = bool(_segments_from_response(response))
    return (code if len(code) == 2 and code.isalpha() else None), has_speech


def _forced_language() -> dict[str, str | None]:
    # MCP v3: watch_video/transcribe_video(language=...) overrides detection.
    return {"language": os.environ.get("WATCH_WHISPER_LANGUAGE") or None}


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], str]:
    """Run the full flow: extract audio → upload → parse segments.

    Returns (segments, backend_used). Raises SystemExit on any failure.
    """
    if backend is None or api_key is None:
        detected_backend, detected_key = load_api_key()
        backend = backend or detected_backend
        api_key = api_key or detected_key

    if not backend or not api_key:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No Whisper backend available. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY "
            f"in the environment or in ~/.config/watch/.env, or set {URL_KEY} / "
            f"{BASE_URL_KEY} to a local OpenAI-compatible server "
            "(e.g. http://localhost:8000/v1/audio/transcriptions). "
            f"Run `python3 {setup_py}` to configure."
        )

    print(f"[watch] extracting audio for Whisper ({backend})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)
    audio_bytes = audio_path.stat().st_size
    upload_cap = max_upload_bytes()

    def transcribe_one(path: Path) -> list[dict]:
        return _transcribe_file(backend, api_key, path)

    if audio_bytes <= upload_cap:
        print(
            f"[watch] audio: {audio_bytes / 1024:.0f} kB — uploading to {backend} Whisper…",
            file=sys.stderr,
        )
        segments = transcribe_one(audio_path)
    else:
        duration = audio_duration(audio_path)
        plan = plan_chunks(duration, audio_bytes, upload_cap)
        print(
            f"[watch] audio: {audio_bytes / (1024 * 1024):.0f} MB exceeds "
            f"{upload_cap // (1024 * 1024)} MB — splitting into {len(plan)} chunks…",
            file=sys.stderr,
        )
        chunks = split_audio(audio_path, audio_out.parent / "chunks", plan)
        segments = transcribe_chunks(chunks, transcribe_one)

    if not segments:
        raise SystemExit("Whisper returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: whisper.py <video-path> [<audio-out.mp3>] [--backend groq|openai|local]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    segments, backend = transcribe_video(video, audio_out, backend=backend_override)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
