"""Tests for inbox: validation of user-dropped media files (ffmpeg-synthesized, no network)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inbox  # noqa: E402
from inbox import InboxError  # noqa: E402


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def _ff(*args: str) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", *args]


VIDEO_IN = ["-f", "lavfi", "-i", "testsrc=duration=1.5:size=160x120:rate=10"]
AUDIO_IN = ["-f", "lavfi", "-i", "sine=frequency=440:duration=1.5"]


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("media")
    out = {k: d / k for k in ("a.mp4", "a.mkv", "a.wav", "a.mp3", "cover.png", "cover.mp3")}
    _run(_ff(*VIDEO_IN, *AUDIO_IN, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(out["a.mp4"])))
    _run(_ff(*VIDEO_IN, *AUDIO_IN, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(out["a.mkv"])))
    _run(_ff(*AUDIO_IN, str(out["a.wav"])))
    _run(_ff(*AUDIO_IN, "-c:a", "libmp3lame", str(out["a.mp3"])))
    _run(_ff("-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=1",
             "-frames:v", "1", str(out["cover.png"])))
    _run(_ff(*AUDIO_IN, "-i", str(out["cover.png"]), "-map", "0:a", "-map", "1:v",
             "-c:a", "libmp3lame", "-c:v", "mjpeg", "-id3v2_version", "3",
             "-disposition:v", "attached_pic", str(out["cover.mp3"])))
    return out


def _put(root: Path, name: str, src: Path) -> Path:
    dst = root / name
    dst.write_bytes(src.read_bytes())
    return dst


# --- check_name -------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "a.mp4", "x.MP4", "Aula 1 - parte 2.mkv", "áudio çã.m4a", "a.b.c.wav",
    "x.mov", "x.webm", "x.m4v", "x.mp3", "x.ogg", "x.opus", "a" * 251 + ".mp4",
])
def test_check_name_valid(name):
    assert inbox.check_name(name) == name


@pytest.mark.parametrize("name", [
    "", "   ", None, 5, b"a.mp4",
    "../a.mp4", "a/b.mp4", "a\\b.mp4", "C:x.mp4", "C:\\x.mp4", "\\\\server\\x.mp4",
    "/etc/x.mp4", "a\x00.mp4", "a\n.mp4", "a\t.mp4", "a\x7f.mp4",
    "a\u200b.mp4", "a\u202e.mp4", "a\u2028.mp4",
    ".hidden.mp4", ".", "..", " a.mp4", "a.mp4 ", "a:b.mp4",
    "x.txt", "x", "x.mp4.exe", "x.avi", "a" * 252 + ".mp4",
])
def test_check_name_rejected(name):
    with pytest.raises(InboxError):
        inbox.check_name(name)


def test_256_char_name_rejected():
    with pytest.raises(InboxError):
        inbox.check_name("a" * 252 + ".mp4")


# --- resolve ----------------------------------------------------------------

def test_resolve_ok(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"x")
    assert inbox.resolve("v.mp4", root=tmp_path) == tmp_path / "v.mp4"


def test_resolve_missing(tmp_path):
    with pytest.raises(InboxError, match="não encontrado"):
        inbox.resolve("nope.mp4", root=tmp_path)


def test_resolve_directory_rejected(tmp_path):
    (tmp_path / "x.mp4").mkdir()
    with pytest.raises(InboxError):
        inbox.resolve("x.mp4", root=tmp_path)


def _symlink_or_skip(link: Path, target: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("symlinks not reliable on Windows")
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("cannot create symlinks")


def test_resolve_symlink_rejected(tmp_path):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    _symlink_or_skip(tmp_path / "link.mp4", real)
    with pytest.raises(InboxError, match="simbólicos"):
        inbox.resolve("link.mp4", root=tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fd semantics")
def test_open_media_symlink_rejected(tmp_path):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    _symlink_or_skip(tmp_path / "link.mp4", real)
    with pytest.raises(InboxError, match="simbólicos"):
        inbox.open_media("link.mp4", root=tmp_path)


def test_open_media_returns_fd(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"hello")
    fd = inbox.open_media("v.mp4", root=tmp_path)
    try:
        assert os.read(fd, 5) == b"hello"
    finally:
        os.close(fd)


def test_open_media_missing(tmp_path):
    with pytest.raises(InboxError, match="não encontrado"):
        inbox.open_media("v.mp4", root=tmp_path)


def test_fd_path():
    assert inbox.fd_path(7) == "/proc/self/fd/7"


# --- list_inbox -------------------------------------------------------------

def test_list_inbox_missing_root(tmp_path):
    assert inbox.list_inbox(root=tmp_path / "nope") == []


def test_list_inbox_filters_and_sorts(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"12345")
    (tmp_path / "A.wav").write_bytes(b"1")
    (tmp_path / "c.txt").write_bytes(b"1")
    (tmp_path / ".hidden.mp4").write_bytes(b"1")
    (tmp_path / "dir.mp4").mkdir()
    real = tmp_path / "real.mkv"
    real.write_bytes(b"1")
    if sys.platform != "win32":
        try:
            os.symlink(real, tmp_path / "link.mkv")
        except OSError:
            pass
    items = inbox.list_inbox(root=tmp_path)
    assert [i["name"] for i in items] == ["A.wav", "b.mp4", "real.mkv"]
    assert items[1]["size_bytes"] == 5
    assert items[1]["modified"].endswith("Z") and "T" in items[1]["modified"]


# --- probe ------------------------------------------------------------------

def test_probe_text_file_rejected(tmp_path):
    p = tmp_path / "t.mp4"
    p.write_text("hello world, not media\n")
    with pytest.raises(InboxError, match="válido"):
        inbox.probe(str(p))


def test_probe_hls_playlist_rejected(tmp_path):
    p = tmp_path / "pl.mp4"
    p.write_text("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n"
                 "#EXTINF:10,\nhttp://127.0.0.1:9/seg.ts\n#EXT-X-ENDLIST\n")
    with pytest.raises(InboxError):
        inbox.probe(str(p))


def test_probe_ffconcat_rejected(tmp_path):
    p = tmp_path / "c.mp4"
    p.write_text("ffconcat version 1.0\nfile '/etc/passwd'\n")
    with pytest.raises(InboxError):
        inbox.probe(str(p))


def test_probe_mp4(media):
    r = inbox.probe(str(media["a.mp4"]))
    assert r["demuxer"] == "mov"
    assert r["has_video"] and r["has_audio"]
    assert (r["width"], r["height"]) == (160, 120)
    assert 1.0 < r["duration"] < 2.5


def test_probe_mkv(media):
    r = inbox.probe(str(media["a.mkv"]))
    assert r["demuxer"] == "matroska"
    assert r["has_video"] and r["has_audio"]


def test_probe_wav(media):
    r = inbox.probe(str(media["a.wav"]))
    assert r["demuxer"] == "wav"
    assert r["has_audio"] and not r["has_video"]
    assert r["width"] is None and r["height"] is None


def test_probe_mp3_cover_is_not_video(media):
    r = inbox.probe(str(media["cover.mp3"]))
    assert r["demuxer"] == "mp3"
    assert r["has_audio"] is True
    assert r["has_video"] is False


@pytest.mark.skipif(sys.platform == "win32", reason="/proc/self/fd is Linux-only")
def test_probe_via_fd(tmp_path, media):
    _put(tmp_path, "a.mp4", media["a.mp4"])
    fd = inbox.open_media("a.mp4", root=tmp_path)
    try:
        r = inbox.probe(inbox.fd_path(fd), pass_fds=(fd,))
    finally:
        os.close(fd)
    assert r["demuxer"] == "mov"


def test_input_args_whitelist_protocols_and_demuxers():
    assert inbox.input_args() == [
        "-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,mp3,wav,ogg",
    ]
    assert inbox.input_args("mov")[-2:] == ["-f", "mov"]


def test_probe_never_lets_concat_open_other_files(tmp_path, monkeypatch):
    # With only a protocol whitelist, ffprobe happily runs the concat demuxer
    # and opens the files it names; the demuxer whitelist refuses it outright.
    target = tmp_path / "secret.wav"
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
          "-f", "lavfi", "-i", "sine=d=1", str(target)])
    evil = tmp_path / "evil.mp4"
    evil.write_text(f"ffconcat version 1.0\nfile '{target.name}'\n")
    seen = []
    real_run = inbox.subprocess.run

    def spy(argv, **kw):
        seen.append(argv)
        return real_run(argv, **kw)

    monkeypatch.setattr(inbox.subprocess, "run", spy)
    with pytest.raises(inbox.InboxError):
        inbox.probe(str(evil))
    assert "-format_whitelist" in seen[0]
    result = real_run(seen[0], capture_output=True, text=True)
    assert result.returncode != 0 and "whitelist" in result.stderr.lower()
