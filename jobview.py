"""Read side of v2 jobs for the stdio server: what a job's files say, as text.

Pure functions over the job directory written by worker.py; nothing here
writes or runs ffmpeg (the server only enqueues, reads and serves).
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path

import jobstore
from frames import format_time
from transcribe import format_transcript

INDEX_FRAME_RE = re.compile(r"^f_\d{5}\.jpg$")
TRANSCRIPT_TEXT_LIMIT = 60_000
SEARCH_TEXT_LIMIT = 40_000
HIT_TEXT_LIMIT = 500
TITLE_LIMIT = 300
HEARTBEAT_STALE = 90.0
STATE_PT = {"queued": "na fila", "running": "processando", "done": "concluído", "failed": "falhou"}


def clip(text, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def source_label(job: dict) -> str:
    source = job.get("source") or {}
    prefix = "caixa de entrada: " if source.get("kind") == "inbox" else ""
    return prefix + str(source.get("value", "?"))


def eta_text(seconds) -> str | None:
    seconds = jobstore.as_number(seconds)
    if seconds <= 0:
        return None
    return f"~{format_time(seconds)}" if seconds >= 60 else "menos de 1 min"


def worker_alive(root: Path | None = None, now: float | None = None) -> bool:
    beat = jobstore.read_json((root or jobstore.JOBS_DIR) / jobstore.HEARTBEAT_FILE) or {}
    t = beat.get("t")
    return isinstance(t, (int, float)) and (now or time.time()) - t < HEARTBEAT_STALE


def status_text(job: dict, *, worker_ok: bool = True) -> str:
    state = job.get("state")
    lines = [
        f"# Trabalho {job['id']}",
        "",
        f"- Origem: {source_label(job)}",
        f"- Estado: {STATE_PT.get(state, state)}",
        f"- Etapa: {job.get('stage') or '-'}",
        f"- Progresso: {jobstore.as_number(job.get('progress')):.0f}%",
    ]
    if job.get("title"):
        lines.append(f"- Título: {clip(job['title'], TITLE_LIMIT)}")
    if jobstore.as_number(job.get("duration")) > 0:
        lines.append(f"- Duração: {format_time(jobstore.as_number(job['duration']))}")
    if job.get("model"):
        lines.append(f"- Modelo de transcrição: {job['model']} (idioma: {job.get('language') or 'automático'})")
    eta = eta_text(job.get("eta_seconds")) if state == "running" else None
    if eta:
        lines.append(f"- Estimativa para terminar: {eta}")
    if state == "failed":
        lines += ["", f"**Erro:** {job.get('error') or 'desconhecido'}"]
    elif state == "done":
        lines += ["", "Pronto: use search_transcript(job_id, query) para achar trechos e "
                      "watch_video(job_id=..., start=..., end=...) para ver frames e a transcrição de um trecho."]
    if state in ("queued", "running") and not worker_ok:
        lines += ["", "**Aviso:** o watch-worker não está respondendo; o trabalho não vai andar. "
                      "Rode docker compose -p watch-mcp up -d"]
    return "\n".join(lines)


def job_line(job: dict) -> str:
    seconds = jobstore.as_number(job.get("duration"))
    duration = format_time(seconds) if seconds > 0 else "?"
    try:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(jobstore.as_number(job.get("created"))))
    except (OverflowError, OSError, ValueError):
        created = "?"
    state = STATE_PT.get(job.get("state"), job.get("state"))
    progress = f" {jobstore.as_number(job.get('progress')):.0f}%" if job.get("state") == "running" else ""
    title = (f" — {clip(job['title'], 120)}" if job.get("title")
             and job.get("title") != (job.get("source") or {}).get("value") else "")
    return f"- `{job['id']}` {state}{progress} · {duration} · {created} · {source_label(job)}{title}"


def _usable(segments) -> list[dict]:
    """Only well-formed segments are served (a damaged cache file must not crash a tool)."""
    if not isinstance(segments, list):
        return []
    out = []
    for s in segments:
        if isinstance(s, dict) and isinstance(s.get("text"), str):
            try:
                start, end = float(s["start"]), float(s["end"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(start) and math.isfinite(end) and 0 <= start <= end:
                out.append({"start": start, "end": end, "text": s["text"]})
    return out


def load_segments(job_dir: Path) -> tuple[list[dict], bool]:
    """(segments, partial): the final transcript, or the chunks finished so far."""
    final = jobstore.read_json(job_dir / "transcript.json")
    if final is not None:
        return _usable(final.get("segments")), False
    segments: list[dict] = []
    chunk_dir = job_dir / "chunks"
    if chunk_dir.is_dir():
        for path in sorted(chunk_dir.glob("[0-9][0-9][0-9][0-9].json")):
            segments += _usable((jobstore.read_json(path) or {}).get("segments"))
    return segments, True


def in_range(segments: list[dict], start: float | None, end: float | None) -> list[dict]:
    lo = start or 0.0
    hi = end if end is not None else float("inf")
    return [s for s in segments if float(s["end"]) >= lo and float(s["start"]) <= hi]


def pick_frames(index: list[dict], job_dir: Path, start: float | None, end: float | None,
                detail: str, max_frames: int) -> list[tuple[Path, float]]:
    """Index frames inside [start, end], evenly thinned to max_frames.

    "efficient" keeps scene cuts (plus the first frame); "balanced" also the
    time-grid frames. Only frames/f_NNNNN.jpg files of this job are returned.
    """
    lo = start or 0.0
    hi = end if end is not None else float("inf")
    if jobstore.is_link(job_dir) or jobstore.is_link(job_dir / "frames"):
        return []  # frames must be this job's own directory, not a link to another
    root = (job_dir / "frames").resolve()
    picked = []
    for entry in index:
        try:
            t, name = float(entry["t"]), str(entry["file"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (lo <= t <= hi) or not INDEX_FRAME_RE.match(name):
            continue
        if detail == "efficient" and entry.get("kind") == "grid":
            continue
        path = (root / name).resolve()
        if path.parent == root and path.is_file():
            picked.append((path, t))
    if len(picked) > max_frames:
        if max_frames == 1:
            return picked[:1]
        step = (len(picked) - 1) / (max_frames - 1)
        picked = [picked[round(i * step)] for i in range(max_frames)]
    return picked


def watch_text(job: dict, meta: dict, segments: list[dict], *, partial: bool, start: float | None,
               end: float | None, detail: str, question: str | None, frame_times: list[float] | None,
               warnings: list[str]) -> str:
    title = clip(meta.get("title") or source_label(job), TITLE_LIMIT)
    lines = [f"# Vídeo: {title}", "", f"- Trabalho: `{job['id']}`", f"- Origem: {source_label(job)}"]
    author = meta.get("uploader") or meta.get("channel")
    if author:
        lines.append(f"- Autor: {author}")
    d = jobstore.as_number(job.get("duration"))
    if d > 0:
        lines.append(f"- Duração: {format_time(d)} ({d:.1f}s)")
    if meta.get("upload_date"):
        u = str(meta["upload_date"])
        lines.append(f"- Publicado em: {u[:4]}-{u[4:6]}-{u[6:]}")
    if start is not None or end is not None:
        lines.append(f"- Trecho: {format_time(start or 0)} → {format_time(end) if end is not None else 'fim'}")
    lines.append(f"- Detalhe: {detail}")
    if question:
        lines.append(f"- Pergunta do usuário: {question}")
    if meta.get("description") and start is None and end is None:
        lines += ["", "## Legenda/descrição do post", "", meta["description"]]

    lines += ["", "## Transcrição", ""]
    if segments:
        if job.get("model"):
            lines.append(f"_Fonte: Whisper local ({job['model']})_")
        text = format_transcript(segments)
        if len(text) > TRANSCRIPT_TEXT_LIMIT:
            text = text[:TRANSCRIPT_TEXT_LIMIT].rsplit("\n", 1)[0]
            warnings = [*warnings, "Transcrição do trecho cortada por tamanho: use start/end menores "
                                   "ou search_transcript para achar o ponto exato."]
        lines += ["", text]
    else:
        lines.append("_Nenhuma fala transcrita neste trecho._")
    if partial:
        warnings = [*warnings, "Transcrição parcial: o trabalho ainda está em andamento."]

    if frame_times is not None:
        lines += ["", f"## Frames ({len(frame_times)}, em ordem temporal — as imagens seguem nesta ordem)", ""]
        lines += [f"{i}. t={format_time(t)}" for i, t in enumerate(frame_times, 1)] or ["_Nenhum frame neste trecho._"]
    if warnings:
        lines += ["", "## Avisos", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines)


def hits_text(job: dict, query: str, hits: list[dict], total: int) -> str:
    lines = [f"# Busca por “{query}” em `{job['id']}`", "", f"{total} trecho(s) encontrado(s)"
             + (f"; mostrando os {len(hits)} primeiros." if total > len(hits) else "."), ""]
    size = 0
    for shown, hit in enumerate(hits):
        entry = [f"- [{format_time(float(hit['start']))}–{format_time(float(hit['end']))}] "
                 f"{clip(hit['text'], HIT_TEXT_LIMIT)}"]
        context = " … ".join(clip(x, HIT_TEXT_LIMIT // 2) for x in (hit.get("before"), hit.get("after")) if x)
        if context:
            entry.append(f"  - contexto: {context}")
        size += sum(len(e) for e in entry)
        if size > SEARCH_TEXT_LIMIT:
            lines.append(f"- … resposta cortada por tamanho ({len(hits) - shown} trecho(s) não mostrados).")
            break
        lines += entry
    if hits:
        first = float(hits[0]["start"])
        lines += ["", f"Para ver um trecho: watch_video(job_id=\"{job['id']}\", start=\"{format_time(max(0, first - 10))}\", "
                      f"end=\"{format_time(first + 30)}\")."]
    return "\n".join(lines)
