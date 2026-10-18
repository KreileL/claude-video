#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _subtitle_lang(path: Path) -> str:
    """The language token of a `video.<lang>.vtt` file, lowercased.

    "video.pt-orig.vtt" -> "pt-orig";  "video.en-US.vtt" -> "en-us"
    """
    parts = path.name.split(".")
    return parts[1].lower() if len(parts) >= 3 else ""


def _video_language(out_dir: Path) -> str | None:
    """The source language yt-dlp reported for the video itself, e.g. "pt-BR"."""
    info = out_dir / "video.info.json"
    if not info.exists():
        return None
    try:
        return (json.loads(info.read_text(encoding="utf-8")) or {}).get("language")
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None

    # Original-language captions beat YouTube's machine translation, so take an
    # *-orig track first and only fall back to English.
    original = [c for c in candidates if "-orig." in c.name]
    if original:
        # A video carrying YouTube's AI dubbing exposes one *-orig track per
        # dubbed language (en-US-orig, es-US-orig, pt-orig...), so "the -orig
        # one" is ambiguous and picking by sort order silently prefers English —
        # handing back the very machine translation this branch exists to avoid.
        # Match the language the extractor reported for the video itself.
        lang = (_video_language(out_dir) or "").lower()
        if lang:
            base = lang.split("-")[0]
            for c in original:
                if _subtitle_lang(c).removesuffix("-orig") == lang:
                    return c
            for c in original:
                if _subtitle_lang(c).removesuffix("-orig").split("-")[0] == base:
                    return c
        return original[0]

    preferred = [
        c for c in candidates
        if any(marker in c.name for marker in (".en.", ".en-US.", ".en-GB."))
    ]
    return preferred[0] if preferred else candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


# Operator-set hooks for the MCP server (mcp_server.py), which runs watch.py in
# a locked-down container: WATCH_YTDLP_ARGS injects yt-dlp options (egress
# proxy, cookies file, size/duration limits) and WATCH_YTDLP_TIMEOUT bounds the
# run. They come from the container environment, never from user input.
def _extra_args() -> list[str]:
    raw = os.environ.get("WATCH_YTDLP_ARGS", "").strip()
    if not raw:
        return []
    try:
        args = json.loads(raw)
    except ValueError:
        args = None
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        # Don't echo the value: it may contain a path the operator keeps private.
        raise SystemExit("WATCH_YTDLP_ARGS must be a JSON array of strings")
    return args


def _timeout() -> float | None:
    raw = os.environ.get("WATCH_YTDLP_TIMEOUT", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
        if not value > 0 or value == float("inf"):
            raise ValueError(raw)
    except ValueError:
        print(
            "[watch] warning: WATCH_YTDLP_TIMEOUT must be a positive number of seconds, ignoring",
            file=sys.stderr,
        )
        return None
    return value


def _run_ytdlp(cmd: list[str]) -> subprocess.CompletedProcess:
    timeout = _timeout()
    try:
        return subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"yt-dlp timed out after {timeout:g}s")


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        *_extra_args(),
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", ".*-orig,en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    _run_ytdlp(cmd)
    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    cmd = [
        "yt-dlp",
        *_extra_args(),
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", ".*-orig,en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = _run_ytdlp(cmd)
    video = _pick_video(out_dir)
    if video is None:
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode})"
        )

    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
