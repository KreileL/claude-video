"""On-disk job queue and cache shared by the MCP server and the worker.

Layout: JOBS_DIR/<job_id>/job.json (+ whatever artifacts the worker writes).

Ownership rule: the stdio server only ever calls create_job (and reads); every
later write to a job (update_job, artifacts, cleanup) belongs to the worker.
Because there is a single writer per job, no locking is needed; job.json is
replaced atomically so readers never see a half-written file.
"""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

JOBS_DIR = Path(os.environ.get("WATCH_JOBS_DIR", "/jobs"))
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
STATES = ("queued", "running", "done", "failed")
ORPHAN_AGE = 3600.0
ARCHIVED = "archived"  # v3: exported to the Holocron; never expires (v2 cleanup also skips it)
EXPORT_LOCK = ".export.lock"  # v3: held by the exporter, and by cleanup while it removes a job
TRASH_DIR = ".trash"  # v3: a job is renamed here (atomically, under its lock) before rmtree
# Touched by the worker every few seconds; not a job dir, so every scan skips it.
HEARTBEAT_FILE = ".heartbeat"


class JobError(ValueError):
    """Invalid job id or missing job (message is user-facing, Portuguese)."""


def new_job_id() -> str:
    return secrets.token_hex(16)


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else JOBS_DIR


def job_path(job_id: str, root: Path | None = None) -> Path:
    """Directory of a job; validates the id before building any path."""
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise JobError("job_id inválido")
    return _root(root) / job_id


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via a temp file in the same directory + os.replace."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> dict | None:
    """Parsed JSON object, or None if missing/corrupt/not an object."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def create_job(fields: dict, root: Path | None = None) -> dict:
    """Create a queued job directory + job.json; the only write the server makes."""
    base = _root(root)
    job_id = new_job_id()
    path = job_path(job_id, base)
    os.mkdir(path, 0o700)
    now = time.time()
    job = {
        "id": job_id,
        "state": "queued",
        "created": now,
        "updated": now,
        "progress": 0.0,
        "stage": "na fila",
        "error": None,
    }
    job.update(fields)
    write_json_atomic(path / "job.json", job)
    return job


def is_link(path: Path) -> bool:
    """Symlink, or a Windows junction (development only)."""
    return os.path.islink(path) or bool(getattr(os.path, "isjunction", lambda _p: False)(path))


def load_job(job_id: str, root: Path | None = None) -> dict:
    path = job_path(job_id, root)
    # A job is a real directory with a real job.json: links could point anywhere.
    if is_link(path) or is_link(path / "job.json"):
        raise JobError("trabalho não encontrado")
    job = read_json(path / "job.json")
    if job is None or job.get("id") != job_id:
        raise JobError("trabalho não encontrado")
    return job


def update_job(job_id: str, root: Path | None = None, **fields: Any) -> dict:
    """Read-modify-write job.json (worker only)."""
    job = load_job(job_id, root)
    job.update(fields)
    job["updated"] = time.time()
    write_json_atomic(job_path(job_id, root) / "job.json", job)
    return job


def _scan(root: Path | None) -> list[tuple[str, os.DirEntry]]:
    """Real (non-symlink) directories whose name is a valid job id."""
    out = []
    try:
        with os.scandir(_root(root)) as it:
            for entry in it:
                try:
                    if entry.is_symlink() or is_link(Path(entry.path)) or not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if JOB_ID_RE.fullmatch(entry.name):
                    out.append((entry.name, entry))
    except OSError:
        pass
    return out


def _valid_job(name: str, entry: os.DirEntry) -> dict | None:
    if is_link(Path(entry.path) / "job.json"):
        return None
    job = read_json(Path(entry.path) / "job.json")
    return job if job is not None and job.get("id") == name else None


def list_jobs(root: Path | None = None) -> list[dict]:
    """All valid jobs, newest created first."""
    jobs = []
    for name, entry in _scan(root):
        job = _valid_job(name, entry)
        if job is not None:
            jobs.append(job)
    jobs.sort(key=lambda j: as_number(j.get("created")), reverse=True)
    return jobs


def as_number(v: Any) -> float:
    """A sortable number; anything else (bool, str, huge int, NaN) counts as 0."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0.0
    try:
        f = float(v)
    except OverflowError:
        return 0.0
    return f if math.isfinite(f) else 0.0


def find_cached(cache_key: str, root: Path | None = None) -> dict | None:
    """Newest queued/running/done/archived job with this cache_key."""
    for job in list_jobs(root):
        if job.get("cache_key") == cache_key and job.get("state") in ("queued", "running", "done", ARCHIVED):
            return job
    return None


def next_job(root: Path | None = None) -> dict | None:
    """Oldest running job (resume after restart), else oldest queued."""
    jobs = sorted(list_jobs(root), key=lambda j: as_number(j.get("created")))
    for state in ("running", "queued"):
        for job in jobs:
            if job.get("state") == state:
                return job
    return None


def dir_size(path: Path) -> int:
    """Total bytes of regular files under path; symlinks and errors ignored."""
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
    return total


def _remove(path: str) -> bool:
    if os.path.islink(path):
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def _discard(path: str, name: str) -> bool:
    """Atomically move a job out of the job namespace, then delete it. Called under the
    job's lock: once renamed, nobody can find (or re-create a lock in) the job, so an
    rmtree that unlinks .export.lock early can no longer race an exporter."""
    if os.path.islink(path):
        return False
    trash = Path(path).parent / TRASH_DIR
    try:
        if is_link(trash):
            return False
        trash.mkdir(mode=0o700, exist_ok=True)
        target = trash / f"{name}-{secrets.token_hex(4)}"
        os.rename(path, target)
    except OSError:
        return False
    shutil.rmtree(target, ignore_errors=True)
    return True


def _empty_trash(root: Path | None) -> None:
    """Leftovers of a cleanup that died between the rename and the rmtree."""
    trash = _root(root) / TRASH_DIR
    if is_link(trash) or not trash.is_dir():
        return
    for entry in os.scandir(trash):
        if not entry.is_symlink() and entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path, ignore_errors=True)


def _evictable(job: dict | None) -> bool:
    # v3: a done job still waiting for its Holocron note is kept until it is archived.
    return (job is not None and job.get("state") in ("done", "failed")
            and job.get("export") != "pendente")


def try_lock_job(job_dir: Path) -> int | None:
    """The job's exclusive lock: an OS lock on <job>/.export.lock (flock in the Linux
    container, msvcrt.locking on Windows test runs). Released by close() or by the
    kernel when the process dies, so a dead holder never needs reaping, and the file
    is never deleted to take ownership (that raced in Codex v3 pass 3). Returns the
    fd, or None when someone else holds it. FileNotFoundError: the job is gone."""
    fd = os.open(Path(job_dir) / EXPORT_LOCK, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            import fcntl
        except ImportError:
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def release_job(fd: int) -> None:
    try:
        os.close(fd)  # drops the lock
    except OSError:
        pass


def _remove_claimed(path: str, name: str) -> bool:
    """Remove a job only while holding its lock, after re-checking on disk that it
    may still go: the exporter holds the same lock, so a job is never deleted while
    it is being exported."""
    try:
        fd = try_lock_job(Path(path))
    except OSError:
        return False
    if fd is None:
        return False
    job = read_json(Path(path) / "job.json")
    if not _evictable(job if job is not None and job.get("id") == name else None):
        release_job(fd)
        return False
    if os.name == "nt":  # Windows cannot rename a dir with an open file (test runs only)
        release_job(fd)
        return _discard(path, name)
    try:
        return _discard(path, name)
    finally:
        release_job(fd)


def cleanup(root: Path | None = None, *, ttl_seconds: float, max_bytes: int,
            now: float | None = None) -> list[str]:
    """Delete expired/over-cap finished jobs and stale orphan dirs; returns removed ids.

    Never touches queued, running or archived jobs.
    """
    now = time.time() if now is None else now
    try:
        _empty_trash(root)
    except OSError:
        pass
    removed: list[str] = []
    finished: list[tuple[float, str, str]] = []  # (updated, id, path)
    sizes: dict[str, int] = {}

    for name, entry in _scan(root):
        job = _valid_job(name, entry)
        if job is None:
            try:
                age = now - entry.stat(follow_symlinks=False).st_mtime
            except OSError:
                continue
            if age > ORPHAN_AGE and _remove(entry.path):
                removed.append(name)
            continue
        if job.get("state") == ARCHIVED:
            continue  # v3: archived jobs are kept forever and do not count against the cap
        if not _evictable(job):
            sizes[name] = dir_size(Path(entry.path))
            continue
        updated = as_number(job.get("updated"))
        if now - updated > ttl_seconds:
            if _remove_claimed(entry.path, name):
                removed.append(name)
            continue
        sizes[name] = dir_size(Path(entry.path))
        finished.append((updated, name, entry.path))

    total = sum(sizes.values())
    for _updated, name, path in sorted(finished):
        if total <= max_bytes:
            break
        if _remove_claimed(path, name):
            removed.append(name)
            total -= sizes[name]
    return removed
