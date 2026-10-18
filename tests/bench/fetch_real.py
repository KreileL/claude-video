"""Fetch the real Brazilian-speech clips of the benchmark (host side, yt-dlp).

    python tests/bench/fetch_real.py            # → tests/bench/data/real/

Each clip becomes <id>.mp3 (16 kHz mono 64 kb/s, what the worker sends to
Whisper). When the video has official (manually made) pt captions, they are
saved as <id>.ref.txt — the exact reference. Clips without them get their
approximate reference from refs/<id>.txt (reviewed by hand; see NOTES-mcp.md).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "real"
YTDLP = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-warnings"]

CLIPS = [
    # (id, url, section or None, kind)
    ("reel-DeniXRTs5Mw", "https://www.instagram.com/reel/DeniXRTs5Mw/", None, "narração (reel)"),
    ("nerdologia-kyGu9lTr_jM", "https://www.youtube.com/watch?v=kyGu9lTr_jM", None, "narração com legenda oficial"),
    ("podpah-gwAwKfW-Q8w", "https://www.youtube.com/watch?v=gwAwKfW-Q8w", "*0-180", "conversa espontânea"),
]
TAG_RE = re.compile(r"<[^>]+>")


def vtt_text(path: Path) -> str:
    """Caption text in order, without timing lines, tags or repeated lines."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = TAG_RE.sub("", raw).strip()
        if not line or "-->" in line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) or line.isdigit():
            continue
        if not lines or lines[-1] != line:
            lines.append(line)
    return " ".join(lines)


def fetch(clip_id: str, url: str, section: str | None) -> None:
    mp3 = OUT / f"{clip_id}.mp3"
    if mp3.exists():
        return
    src = OUT / f"{clip_id}.src"
    argv = [*YTDLP, "-f", "ba/b", "-o", f"{src}.%(ext)s", "--write-subs", "--sub-langs", "pt.*",
            "--sub-format", "vtt", "--convert-subs", "vtt"]
    if section:
        argv += ["--download-sections", section]
    subprocess.run([*argv, "--", url], check=True)
    media = next(p for p in OUT.glob(f"{clip_id}.src.*") if not p.name.endswith(".vtt"))
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
         "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(mp3)],
        check=True,
    )
    media.unlink()
    subs = sorted(OUT.glob(f"{clip_id}.src.*.vtt"))
    if subs and not section:
        (OUT / f"{clip_id}.ref.txt").write_text(vtt_text(subs[0]) + "\n", encoding="utf-8")
    for sub in subs:
        sub.unlink()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for clip_id, url, section, _kind in CLIPS:
        fetch(clip_id, url, section)
        print(clip_id, "ok", flush=True)
