"""v3 post-processing of a finished job: summary → Holocron folder → archived.

Called by worker.py right after a job is done, periodically for pending work,
and by `python /app/worker.py export <job_id>`. Nothing here may fail the job:
the transcript and frames are already safe in JOBS_DIR. Job fields written:

- resumo: ok | pendente | desligado | erro (+ resumo_next, resumo_attempts,
  resumo_error, resumo_calls, resumo_model, resumo_file); the text itself is
  kept in the job dir as summary.json, so an export retry never calls the LLM again.
- export: ok | pendente (+ export_next, export_attempts, export_error),
  entrada_folder, entrada_index, exported_at; on success state → archived.

Jobs finished before v3 (no "export" field) are only exported on request.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import entrada
import jobstore
import jobview
import resumo

RESUMO_CAIXA_ENTRADA = os.environ.get("WATCH_RESUMO_CAIXA_ENTRADA", "1").strip().lower() not in ("0", "false", "nao", "não", "off")
UTC_OFFSET = timezone(timedelta(hours=float(os.environ.get("WATCH_UTC_OFFSET_HOURS", "-3"))))
SUMMARY_TIMEOUT = float(os.environ.get("WATCH_RESUMO_JOB_TIMEOUT", "900"))
# Transient summary failures: 5 min, doubling, up to 6 h (env only to shorten end-to-end tests).
RETRY_FIRST = float(os.environ.get("WATCH_RESUMO_RETRY_FIRST", "300"))
RETRY_MAX = 6 * 3600.0
PERMANENT_TRIES = 3  # a non-transient summary error is retried this many times, then "erro"
EXPORT_RETRY = float(os.environ.get("WATCH_EXPORT_RETRY_FIRST", "300"))
EXPORT_RETRY_MAX = 3600.0
LOCK_NAME = jobstore.EXPORT_LOCK
RESERVED_NOTES = ("transcrição.md", "resumo.md")  # our fixed file names, compared casefolded
SUMMARY_FILE = "summary.json"
SUMMARY_FRAMES = 6
HIGHLIGHTS = 8
READY = ("done", jobstore.ARCHIVED)


def log(message: str) -> None:
    print(f"[postproc] {message}", flush=True)


class Busy(Exception):
    """Another process is exporting this job right now."""


# ── helpers ────────────────────────────────────────────────────────────────


def _lock(job_dir: Path) -> int:
    """The job's exclusive OS lock, shared with jobstore.cleanup (which never removes a
    locked job). Taken BEFORE reading job.json, so every decision uses fresh state."""
    if jobstore.is_link(job_dir) or not job_dir.is_dir():
        raise jobstore.JobError("trabalho não encontrado")
    try:
        fd = jobstore.try_lock_job(job_dir)
    except FileNotFoundError:
        raise jobstore.JobError("trabalho não encontrado") from None
    except OSError:
        raise Busy("trava do trabalho indisponível") from None
    if fd is None:
        raise Busy("exportação já em andamento para este trabalho")
    return fd


def _casefold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _backoff(attempts: int, first: float, cap: float) -> float:
    return min(cap, first * (2 ** max(0, attempts - 1)))


def title_of(job: dict, meta: dict) -> str:
    source = job.get("source") or {}
    title = str(meta.get("title") or source.get("value") or "Vídeo")
    if source.get("kind") == "inbox":
        stem = title.rsplit(".", 1)[0] if "." in title else title
        title = stem.replace("_", " ")
    return title


def origem_of(job: dict) -> str:
    source = job.get("source") or {}
    if source.get("kind") == "inbox":
        return f"caixa de entrada: {source.get('value', '?')}"
    return str(source.get("value", "?"))


def summary_wanted(job: dict) -> bool:
    if resumo.PROVIDER == "none":
        return False
    return (job.get("source") or {}).get("kind") != "inbox" or RESUMO_CAIXA_ENTRADA


def highlights(segments: list[dict], n: int = HIGHLIGHTS) -> list[dict]:
    """Evenly spread transcript lines, for a note without a summary."""
    if not segments:
        return []
    step = max(1, len(segments) // n)
    return [{"t": s["start"], "text": s["text"]} for s in segments[::step][:n]]


def demote(summary: str) -> str:
    """The summary sits under the note's "## Resumo": its ## sections become ###."""
    return re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", summary, flags=re.M)


def _frame_bytes(job_dir: Path, name: str) -> bytes | None:
    if not entrada.SRC_FRAME_RE.fullmatch(name):
        return None
    frames = job_dir / "frames"
    if jobstore.is_link(frames) or jobstore.is_link(frames / name):
        return None
    try:
        fd = os.open(frames / name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        return f.read(2_000_000)


def _date(job: dict) -> str:
    stamp = jobstore.as_number(job.get("finished")) or time.time()
    return datetime.fromtimestamp(stamp, UTC_OFFSET).strftime("%Y-%m-%d")


def _material(job: dict, root: Path | None) -> dict:
    job_dir = jobstore.job_path(job["id"], root)
    if jobstore.is_link(job_dir):
        raise entrada.EntradaError("trabalho inválido")
    meta = jobstore.read_json(job_dir / "meta.json") or {}
    index = jobstore.read_json(job_dir / "index.json") or {}
    segments, _partial = jobview.load_segments(job_dir)
    key = entrada.pick_key_frames(index.get("frames") or [])
    return {"dir": job_dir, "meta": meta, "segments": segments, "key_frames": key,
            "title": title_of(job, meta)}


# ── summary ────────────────────────────────────────────────────────────────


def stored_summary(job_dir: Path) -> str | None:
    data = jobstore.read_json(job_dir / SUMMARY_FILE) or {}
    text = data.get("text")
    return text if isinstance(text, str) and text.strip() else None


def try_summary(job: dict, mat: dict, root: Path | None) -> str | None:
    """One summary attempt; updates the job's resumo fields. Returns the text or None."""
    if not summary_wanted(job):
        jobstore.update_job(job["id"], root, resumo="desligado")
        return None
    frames = []
    for f in mat["key_frames"][:: max(1, len(mat["key_frames"]) // SUMMARY_FRAMES)][:SUMMARY_FRAMES]:
        data = _frame_bytes(mat["dir"], f["file"])
        if data:
            frames.append((f["t"], data))
    attempts = int(jobstore.as_number(job.get("resumo_attempts"))) + 1
    duration = jobstore.as_number(job.get("duration")) or None
    before = sum(resumo.CALLS.values())  # real HTTP calls only (a missing key makes none)
    calls = int(jobstore.as_number(job.get("resumo_calls")))
    try:
        result = resumo.summarize(mat["segments"], title=mat["title"], duration=duration, frames=frames,
                                  deadline=time.monotonic() + SUMMARY_TIMEOUT)
    except resumo.ResumoDisabled:
        jobstore.update_job(job["id"], root, resumo="desligado")
        return None
    except resumo.ResumoError as exc:
        calls += sum(resumo.CALLS.values()) - before
        if not exc.transient and attempts >= PERMANENT_TRIES:
            jobstore.update_job(job["id"], root, resumo="erro", resumo_error=str(exc)[:300],
                                resumo_attempts=attempts, resumo_next=None, resumo_calls=calls)
        else:
            wait = _backoff(attempts, RETRY_FIRST, RETRY_MAX)
            jobstore.update_job(job["id"], root, resumo="pendente", resumo_error=str(exc)[:300],
                                resumo_attempts=attempts, resumo_next=time.time() + wait, resumo_calls=calls)
        log(f"{job['id']}: resumo falhou ({exc}); tentativa {attempts}")
        return None
    jobstore.write_json_atomic(mat["dir"] / SUMMARY_FILE, {
        "text": result.text, "provider": result.provider, "model": result.model, "at": time.time()})
    jobstore.update_job(job["id"], root, resumo="ok", resumo_error=None, resumo_next=None,
                        resumo_attempts=attempts, resumo_model=f"{result.provider}:{result.model}",
                        resumo_calls=calls + result.calls)
    log(f"{job['id']}: resumo ok ({result.calls} chamada(s) ao {result.provider})")
    return result.text


# ── export ─────────────────────────────────────────────────────────────────


def _status_for_note(job: dict) -> str:
    state = job.get("resumo")
    return state if state in ("ok", "desligado") else "pendente"


def _index_file(title: str) -> str:
    index_file = entrada.index_filename(title)
    if _casefold(index_file) in RESERVED_NOTES:  # Windows: case-insensitive names
        index_file = "Nota - " + index_file
    return index_file


def _head(path: Path, size: int = 8192) -> str:
    """First bytes of a file we may have written, never following links; '' on any problem."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    except OSError:
        return ""
    try:
        return os.read(fd, size).decode("utf-8", errors="replace")
    except OSError:
        return ""
    finally:
        os.close(fd)


def _find_published(job: dict, mat: dict, entrada_root: Path | None) -> str | None:
    """The folder a previous attempt published before job.json recorded it (crash
    between the rename and the update): matched by the attempt's export_id."""
    export_id = (job.get("export_intent") or {}).get("id")
    if not isinstance(export_id, str):
        return None
    root = entrada._root(entrada_root)
    base = entrada.folder_name(_date(job), mat["title"], job["id"])
    marker = f'export_id: "{export_id}"'
    for n in range(1, 100):
        name = base if n == 1 else f"{base} ({n})"
        d = root / name
        if jobstore.is_link(d) or not d.is_dir():
            continue
        if marker in _head(d / _index_file(mat["title"])):
            return name
    return None


def publish_once(job: dict, mat: dict, summary: str | None, root: Path | None,
                 entrada_root: Path | None) -> tuple[str, str, bool]:
    """Publish a folder exactly once per attempt: an earlier attempt that crashed after
    its rename is adopted instead of published again. Caller holds the lock and must
    clear export_intent when it records the result."""
    found = _find_published(job, mat, entrada_root)
    if found is not None:
        log(f"{job['id']}: pasta de uma tentativa anterior reconhecida: {found}")
        return found, _index_file(mat["title"]), bool((job.get("export_intent") or {}).get("with_summary"))
    export_id = os.urandom(8).hex()
    jobstore.update_job(job["id"], root, export_intent={"id": export_id, "with_summary": bool(summary)})
    folder, index_file = build_folder(job, mat, summary, entrada_root, export_id=export_id)
    return folder, index_file, bool(summary)


def build_folder(job: dict, mat: dict, summary: str | None, entrada_root: Path | None,
                 export_id: str | None = None) -> tuple[str, str]:
    """Stage and publish one complete folder; returns (folder, index file name)."""
    staged = entrada.staging(entrada_root, job["id"])
    frames_dir = staged / "frames"
    frames_dir.mkdir()
    used: set[str] = set()
    note_frames = []
    for f in mat["key_frames"]:
        data = _frame_bytes(mat["dir"], f["file"])
        if not data:
            continue
        name = entrada.frame_filename(f["t"])
        n = 2
        while name in used:
            name = entrada.frame_filename(f["t"])[:-4] + f"-{n}.jpg"
            n += 1
        used.add(name)
        entrada.create_file(frames_dir / name, data)
        note_frames.append({"t": f["t"], "file": name})
    meta = mat["meta"]
    title = mat["title"]
    status = "ok" if summary else _status_for_note(job)
    info = {
        "resumo_erro": job.get("resumo") == "erro", "export_id": export_id,
        "title": title, "job_id": job["id"], "date": _date(job), "origem": origem_of(job),
        "duration": jobstore.as_number(job.get("duration")) or None,
        "autor": meta.get("uploader") or meta.get("channel"), "modelo": job.get("model"),
        "idioma": job.get("language") or "automático", "resumo_status": status,
        "summary": demote(summary) if summary else None,
        "highlights": [] if summary else highlights(mat["segments"]), "frames": note_frames,
        "origin_kind": (job.get("source") or {}).get("kind"),
    }
    index_file = _index_file(title)
    entrada.create_file(staged / "Transcrição.md", entrada.render_transcript(title, mat["segments"]))
    entrada.create_file(staged / index_file, entrada.render_index(info))
    folder = entrada.publish(entrada_root, staged, entrada.folder_name(info["date"], title, job["id"]))
    return folder, index_file


def _mark_exported(job: dict, root: Path | None, folder: str, index_file: str, with_summary: bool) -> None:
    fields = {"export": "ok", "export_error": None, "export_next": None, "entrada_folder": folder,
              "entrada_index": index_file, "exported_at": time.time(), "resumo_entregue": with_summary,
              "export_intent": None}
    if job.get("state") != jobstore.ARCHIVED:
        fields.update(state=jobstore.ARCHIVED, archived=True, archived_at=time.time(),
                      archived_from=job.get("state"))
    jobstore.update_job(job["id"], root, **fields)


def _export_pending(job: dict, root: Path | None, reason: str) -> None:
    attempts = int(jobstore.as_number(job.get("export_attempts"))) + 1
    jobstore.update_job(job["id"], root, export="pendente", export_error=reason[:300], export_attempts=attempts,
                        export_next=time.time() + _backoff(attempts, EXPORT_RETRY, EXPORT_RETRY_MAX))
    log(f"{job['id']}: nota pendente ({reason})")


def run_export(job_id: str, root: Path | None = None, entrada_root: Path | None = None, *,
               again: bool = False, summarize: bool = True, if_done: str = "refuse") -> str | None:
    """Summary (if due) + folder. Returns the folder name, or None when left pending.
    if_done="return": an already exported job is a no-op (finish() after a manual export).
    Raises Busy, jobstore.JobError, or entrada.EntradaError for a refused request."""
    job_dir = jobstore.job_path(job_id, root)
    lock = _lock(job_dir)
    try:
        job = jobstore.load_job(job_id, root)  # read under the lock
        if job.get("state") not in READY:
            raise entrada.EntradaError("o trabalho ainda não terminou (ou falhou)")
        if job.get("entrada_folder") and job.get("export") == "ok" and not again:
            if if_done == "return":
                return job["entrada_folder"]
            raise entrada.EntradaError(f"já exportado em {jobview.ENTRADA_LABEL}/{job['entrada_folder']} "
                                       "(use --again para uma nova versão)")
        if again and job.get("resumo") == "erro":  # an explicit retry also retries the summary
            job = jobstore.update_job(job_id, root, resumo=None, resumo_attempts=0, resumo_error=None)
        mat = _material(job, root)
        summary = stored_summary(job_dir)
        due = job.get("resumo") != "pendente" or jobstore.as_number(job.get("resumo_next")) <= time.time()
        if summary is None and summarize and due and job.get("resumo") not in ("desligado", "erro"):
            summary = try_summary(job, mat, root)
            job = jobstore.load_job(job_id, root)
        if not entrada.available(entrada_root):
            _export_pending(job, root, "pasta de entrada do Holocron indisponível ou sem escrita")
            return None
        try:
            folder, index_file, with_summary = publish_once(job, mat, summary, root, entrada_root)
        except (entrada.EntradaError, OSError) as exc:
            _export_pending(jobstore.load_job(job_id, root), root, f"falha ao gravar a nota ({type(exc).__name__})")
            return None
        _mark_exported(job, root, folder, index_file, with_summary=with_summary)
        log(f"{job_id}: nota em {jobview.ENTRADA_LABEL}/{folder}")
        return folder
    except (entrada.EntradaError, jobstore.JobError):
        raise
    except Exception as exc:  # noqa: BLE001 - recorded under the lock, then reported
        try:
            current = jobstore.load_job(job_id, root)
            if current.get("export") != "ok":
                _export_pending(current, root, f"erro interno ({type(exc).__name__})")
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        jobstore.release_job(lock)


def _delivered(entrada_root: Path | None, folder: str, job_id: str) -> str | None:
    """Idempotent delivery: a Resumo*.md of this job already in the folder."""
    d = entrada._root(entrada_root) / folder
    if not entrada._component_ok(folder) or jobstore.is_link(d) or not d.is_dir():
        return None
    marker = f'job_id: "{job_id}"'
    for p in sorted(d.glob("Resumo*.md")):
        head = _head(p, 4096)  # bounded, no links, any encoding: a user's file can't block us
        if "subtipo: resumo-video" in head and marker in head:
            return p.name
    return None


def add_late_summary(job_id: str, root: Path | None = None, entrada_root: Path | None = None) -> None:
    """A summary that arrived after the note: Resumo.md in the published folder, or a
    new complete folder when the user already moved/renamed it (create-only).

    "Available" (summary.json + resumo=ok) and "delivered" (resumo_entregue) are
    separate: a crash or a write error between them is retried, and a Resumo.md
    already written for this job is recognized instead of written again."""
    job_dir = jobstore.job_path(job_id, root)
    lock = _lock(job_dir)
    try:
        job = jobstore.load_job(job_id, root)
        if job.get("export") != "ok" or job.get("resumo_entregue") is not False:
            return  # absent = exported before this field existed: treat as delivered
        mat = _material(job, root)
        summary = stored_summary(job_dir)
        if summary is None:
            summary = try_summary(job, mat, root)
            if summary is None:
                return
        if job.get("resumo") != "ok":  # summary.json was saved but job.json not yet updated
            jobstore.update_job(job_id, root, resumo="ok", resumo_next=None, resumo_error=None)
        job = jobstore.load_job(job_id, root)
        try:
            if not entrada.available(entrada_root):
                raise entrada.EntradaError("pasta de entrada indisponível")
            folder = job.get("entrada_folder") or ""
            name = _delivered(entrada_root, folder, job_id)
            if name is None:
                info = {"title": mat["title"], "job_id": job_id, "date": _date(job),
                        "index_file": job.get("entrada_index") or entrada.index_filename(mat["title"])}
                name = entrada.add_file(entrada_root, folder, "Resumo.md", entrada.render_resumo(info, summary))
            if name is not None:
                jobstore.update_job(job_id, root, resumo_file=name, resumo_entregue=True,
                                    resumo_entrega_next=None, resumo_entrega_attempts=None)
                log(f"{job_id}: {name} em {folder}")
                return
            folder, index_file, _ = publish_once(job, mat, summary, root, entrada_root)
            jobstore.update_job(job_id, root, resumo_file=None, resumo_entregue=True, entrada_folder=folder,
                                entrada_index=index_file, exported_at=time.time(), export_intent=None,
                                resumo_entrega_next=None, resumo_entrega_attempts=None)
            log(f"{job_id}: pasta original não encontrada; nova versão em {folder}")
        except (entrada.EntradaError, OSError) as exc:
            attempts = int(jobstore.as_number(job.get("resumo_entrega_attempts"))) + 1
            jobstore.update_job(job_id, root, resumo_entrega_attempts=attempts, resumo_entrega_next=time.time()
                                + _backoff(attempts, EXPORT_RETRY, EXPORT_RETRY_MAX))
            log(f"{job_id}: entrega do resumo adiada ({type(exc).__name__})")
    finally:
        jobstore.release_job(lock)


def finish(job_id: str, root: Path | None = None, entrada_root: Path | None = None) -> None:
    """Right after a job is done (process() already set export="pendente"). Idempotent:
    a job a manual export already published is left alone. Never raises."""
    try:
        run_export(job_id, root, entrada_root, if_done="return")
    except Busy:
        return  # someone else is exporting it; the state is theirs to write
    except Exception as exc:  # noqa: BLE001 - run_export already recorded it under the lock
        log(f"{job_id}: pós-processamento falhou ({type(exc).__name__}); fica pendente")


def awaiting_summary(job: dict) -> bool:
    """Only an explicit False (written by an export without summary) means a summary
    is owed to the note. A missing field is a job exported before the field existed,
    whose note already has its summary: treating it as owed re-delivered summaries
    (and re-created folders that had gone to another vault) in the acceptance run."""
    return job.get("resumo_entregue") is False


def late_summary_due(job: dict, root: Path | None, now: float) -> bool:
    state = job.get("resumo")
    if state == "ok":  # available, not delivered yet
        return jobstore.as_number(job.get("resumo_entrega_next")) <= now
    if state == "pendente":
        saved = (jobstore.job_path(job["id"], root) / SUMMARY_FILE).exists()  # crash before job.json
        return saved or jobstore.as_number(job.get("resumo_next")) <= now
    return False


def retry_due(root: Path | None = None, entrada_root: Path | None = None, now: float | None = None) -> int:
    """Pending exports and summaries whose time has come. Never raises; returns how many ran."""
    now = time.time() if now is None else now
    ran = 0
    try:
        jobs = jobstore.list_jobs(root)
    except Exception:  # noqa: BLE001
        return 0
    for job in jobs:
        if job.get("state") not in READY:
            continue
        try:
            if job.get("export") == "pendente" and jobstore.as_number(job.get("export_next")) <= now:
                ran += 1
                run_export(job["id"], root, entrada_root)
            elif job.get("export") == "ok" and awaiting_summary(job) and late_summary_due(job, root, now):
                ran += 1
                add_late_summary(job["id"], root, entrada_root)
        except Busy:
            continue
        except Exception as exc:  # noqa: BLE001
            log(f"{job['id']}: nova tentativa falhou ({type(exc).__name__})")
    return ran
