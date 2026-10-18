"""Local inbox: validate media files the user drops in a folder before ffmpeg sees them.

Threat model: the file may be crafted (playlist/concat disguised as .mp4) and may
be replaced between validation and use. Callers pass only a bare file name.
Defence: strict name check, lstat (no symlinks), open once by fd (O_NOFOLLOW) and
hand ffmpeg that fd, and probe with a protocol whitelist plus a demuxer allow-list.

Pure stdlib.
"""
from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

INBOX_DIR = Path(os.environ.get("WATCH_INBOX_DIR", "/inbox"))
ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".mp3", ".m4a", ".wav", ".ogg", ".opus"}
FORMAT_DEMUXERS = {
    "mov,mp4,m4a,3gp,3g2,mj2": "mov",
    "matroska,webm": "matroska",
    "mp3": "mp3",
    "wav": "wav",
    "ogg": "ogg",
}
# Demuxers ffmpeg/ffprobe may pick for a local input. A protocol whitelist alone
# still lets concat (and friends) open other local files named inside the input.
INPUT_FORMATS = ",".join(FORMAT_DEMUXERS.values())


class InboxError(ValueError):
    """Validation failure; the message is shown to the user (Portuguese)."""


def check_name(name: str) -> str:
    """Validate a bare file name; return it unchanged or raise InboxError."""
    bad = InboxError("nome inválido: use só o nome do arquivo, sem pastas")
    if not isinstance(name, str) or not name.strip() or len(name) > 255:
        raise bad
    if name != name.strip() or name.startswith("."):
        raise bad
    for ch in name:
        if ch in "/\\:" or ord(ch) < 32 or unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp"):
            raise bad
    win = PureWindowsPath(name)
    if os.path.isabs(name) or win.drive or win.anchor:
        raise bad
    if os.path.splitext(name)[1].lower() not in ALLOWED_EXTS:
        raise InboxError(f"extensão não permitida (aceitas: {', '.join(sorted(ALLOWED_EXTS))})")
    return name


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else INBOX_DIR


def resolve(name: str, root: Path | None = None) -> Path:
    """Return the path of a regular, non-symlink file directly inside the inbox."""
    check_name(name)
    base = _root(root)
    path = base / name
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise InboxError("arquivo não encontrado na caixa de entrada") from None
    except OSError:
        raise InboxError("não foi possível acessar o arquivo") from None
    if stat.S_ISLNK(st.st_mode):
        raise InboxError("links simbólicos não são aceitos")
    if not stat.S_ISREG(st.st_mode):
        raise InboxError("não é um arquivo comum")
    if path.resolve().parent != base.resolve():
        raise InboxError("nome inválido: use só o nome do arquivo, sem pastas")
    return path


def list_inbox(root: Path | None = None) -> list[dict]:
    """List allowed regular files: [{name, size_bytes, modified (ISO UTC)}], sorted by name."""
    base = _root(root)
    out: list[dict] = []
    try:
        with os.scandir(base) as it:
            for entry in it:
                try:
                    check_name(entry.name)
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    st = entry.stat(follow_symlinks=False)
                except (InboxError, OSError):
                    continue
                modified = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                out.append({
                    "name": entry.name,
                    "size_bytes": st.st_size,
                    "modified": modified.strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
    except OSError:
        return []
    out.sort(key=lambda d: (d["name"].lower(), d["name"]))
    return out


def open_media(name: str, root: Path | None = None) -> int:
    """Open the validated file read-only without following symlinks; caller closes the fd."""
    path = resolve(name, root)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise InboxError("arquivo não encontrado na caixa de entrada") from None
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise InboxError("links simbólicos não são aceitos") from None
        raise InboxError("não foi possível abrir o arquivo") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise InboxError("não é um arquivo comum")
    except BaseException:
        os.close(fd)
        raise
    return fd


def fd_path(fd: int) -> str:
    """Path through which a child process can read an open fd (Linux).

    Pass the fd to the child with subprocess ``pass_fds=(fd,)``: /proc/self/fd/N
    is a magic link to the open file, so the child reads the SAME inode even if
    the name in the inbox is replaced afterwards.
    """
    return f"/proc/self/fd/{fd}"


def input_args(demuxer: str | None = None) -> list[str]:
    """ffmpeg/ffprobe input options for a local file: local protocol only,
    allowed demuxers only, and (once probed) that exact demuxer forced."""
    args = ["-protocol_whitelist", "file", "-format_whitelist", INPUT_FORMATS]
    return args + (["-f", demuxer] if demuxer else [])


def probe(path: str, pass_fds: tuple[int, ...] = (), timeout: float = 120.0) -> dict:
    """ffprobe ``path`` and verify it is real audio/video in an allowed container."""
    invalid = InboxError("o arquivo não é um vídeo/áudio válido")
    argv = ["ffprobe", "-v", "error", *input_args(),
            "-print_format", "json", "-show_format", "-show_streams", path]
    kwargs: dict = {}
    if pass_fds:
        kwargs["pass_fds"] = pass_fds
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        raise InboxError("tempo esgotado ao analisar o arquivo") from None
    except OSError:
        raise InboxError("ffprobe indisponível") from None
    if res.returncode != 0:
        raise invalid
    try:
        data = json.loads(res.stdout)
        fmt = data.get("format") or {}
        demuxer = FORMAT_DEMUXERS.get(fmt.get("format_name"))
        streams = data.get("streams") or []
        duration = float(fmt.get("duration"))
    except (ValueError, TypeError, AttributeError):
        raise invalid from None
    if demuxer is None:
        raise invalid
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [s for s in streams if s.get("codec_type") == "video"
             and not (s.get("disposition") or {}).get("attached_pic") == 1]
    if not audio and not video:
        raise invalid
    if not duration > 0 or duration != duration or duration == float("inf"):
        raise invalid
    width = height = None
    if video:
        width, height = video[0].get("width"), video[0].get("height")
    return {
        "duration": duration,
        "demuxer": demuxer,
        "has_video": bool(video),
        "has_audio": bool(audio),
        "width": width,
        "height": height,
    }
