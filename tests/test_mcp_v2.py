"""v2 MCP tools: inbox, analyze → status → search → watch(job_id), list_jobs.

Real jobs are produced by worker.process against synthesized media and the
fake Whisper server from test_worker.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent

import inbox
import jobstore
import mcp_server
import worker
from test_worker import dirs, fake_whisper, make_clip  # noqa: F401  (fixtures)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def v2(dirs, monkeypatch):  # noqa: F811
    jobs, box = dirs
    monkeypatch.setattr(jobstore, "JOBS_DIR", jobs)
    monkeypatch.setattr(inbox, "INBOX_DIR", box)
    return jobs, box


def job_id_from(text: str) -> str:
    return text.split("job_id = ", 1)[1].split()[0]


# ── inbox ───────────────────────────────────────────────────────────────────


def test_list_inbox_shows_only_allowed_media(v2):
    _jobs, box = v2
    make_clip(box / "aula.mp4", seconds=2)
    (box / "notas.txt").write_text("x")
    text = run(mcp_server.list_inbox())
    assert "aula.mp4" in text and "notas.txt" not in text


@pytest.mark.parametrize("name, message", [
    ("../segredo.mp4", "nome inválido"),
    ("pasta/aula.mp4", "nome inválido"),
    ("C:\\Windows\\x.mp4", "nome inválido"),
    ("/etc/passwd.mp4", "nome inválido"),
    ("aula\x00.mp4", "nome inválido"),
    ("roteiro.txt", "extensão não permitida"),
    ("sumiu.mp4", "não encontrado"),
])
def test_analyze_refuses_bad_inbox_names(v2, name, message):
    with pytest.raises(ToolError, match=message):
        run(mcp_server.analyze_video(name))
    assert jobstore.list_jobs() == []


def test_analyze_refuses_non_media_and_playlists(v2):
    _jobs, box = v2
    (box / "falso.mp4").write_text("não sou vídeo")
    (box / "lista.mp4").write_text("#EXTM3U\n#EXTINF:1,\nhttp://watch-whisper:8000/health\n")
    (box / "concat.mkv").write_text("ffconcat version 1.0\nfile 'falso.mp4'\n")
    for name in ("falso.mp4", "lista.mp4", "concat.mkv"):
        with pytest.raises(ToolError, match="não é um vídeo/áudio válido"):
            run(mcp_server.analyze_video(name))
    assert jobstore.list_jobs() == []


@pytest.mark.skipif(os.name != "posix", reason="symlinks need POSIX here")
def test_analyze_refuses_symlink(v2, tmp_path):
    _jobs, box = v2
    make_clip(tmp_path / "fora.mp4", seconds=2)
    (box / "link.mp4").symlink_to(tmp_path / "fora.mp4")
    with pytest.raises(ToolError, match="links simbólicos"):
        run(mcp_server.analyze_video("link.mp4"))


def test_analyze_validates_url_language_and_terms(v2):
    with pytest.raises(ToolError, match="URL recusada"):
        run(mcp_server.analyze_video("file:///etc/passwd"))
    with pytest.raises(ToolError, match="URL recusada"):
        run(mcp_server.analyze_video("http://127.0.0.1/x.mp4"))
    with pytest.raises(ToolError, match="language"):
        run(mcp_server.analyze_video("https://example.com/v", language="pt-BR; rm"))
    with pytest.raises(ToolError, match="termo inválido"):
        run(mcp_server.analyze_video("https://example.com/v", terms=["a\x00b"]))
    with pytest.raises(ToolError, match="terms"):
        run(mcp_server.analyze_video("https://example.com/v", terms=["x"] * 31))


def test_analyze_url_creates_job_and_reuses_cache(v2):
    first = run(mcp_server.analyze_video("https://example.com/video", terms=["Embrapa"]))
    job_id = job_id_from(first)
    job = jobstore.load_job(job_id)
    assert job["state"] == "queued" and job["source"] == {"kind": "url", "value": "https://example.com/video"}
    assert job["language"] == "pt" and job["terms"] == ["Embrapa"]
    again = run(mcp_server.analyze_video("https://example.com/video", terms=["Embrapa"]))
    assert "reaproveitando" in again and job_id in again
    assert len(jobstore.list_jobs()) == 1
    other = run(mcp_server.analyze_video("https://example.com/video", quality="fast"))
    assert job_id_from(other) != job_id


def test_worker_warning_when_no_heartbeat(v2):
    jobs, _box = v2
    text = run(mcp_server.analyze_video("https://example.com/a"))
    assert "watch-worker não está respondendo" in text
    jobstore.write_json_atomic(jobs / jobstore.HEARTBEAT_FILE, {"t": time.time()})
    status = run(mcp_server.job_status(job_id_from(text)))
    assert "não está respondendo" not in status and "na fila" in status


def test_job_ids_are_validated(v2):
    for bad in ("../../etc", "A" * 32, "0" * 31, ""):
        with pytest.raises(ToolError, match="job_id inválido"):
            run(mcp_server.job_status(bad))
    with pytest.raises(ToolError, match="não encontrado"):
        run(mcp_server.job_status("0" * 32))


# ── full flow ───────────────────────────────────────────────────────────────


def test_analyze_status_search_watch_flow(v2, fake_whisper):  # noqa: F811
    _jobs, box = v2
    make_clip(box / "palestra.mp4", seconds=65)
    job_id = job_id_from(run(mcp_server.analyze_video("palestra.mp4", terms=["Joinville"])))
    worker.process(jobstore.load_job(job_id), inbox_root=box)

    status = run(mcp_server.job_status(job_id))
    assert "concluído" in status and "100%" in status
    assert job_id in run(mcp_server.list_jobs())

    found = run(mcp_server.search_transcript(job_id, "BLOCO 2"))
    assert "[00:21–00:22]" in found and "bloco 2" in found
    assert "1 trecho" in found

    content = run(mcp_server.watch_video(job_id=job_id, start="00:15", end="00:45", max_frames=6))
    text = content[0].text
    assert isinstance(content[0], TextContent) and "Trecho: 00:15 → 00:45" in text
    assert "bloco 2" in text and "bloco 1" not in text and "bloco 3" in text
    images = content[1:]
    assert 1 <= len(images) <= 6 and all(isinstance(i, ImageContent) for i in images)
    times = [line.split("t=")[1] for line in text.splitlines() if ". t=" in line]
    assert len(times) == len(images) and all("00:15" <= t <= "00:45" for t in times)

    only_text = run(mcp_server.watch_video(job_id=job_id, detail="transcript"))
    assert len(only_text) == 1


def test_watch_job_respects_image_budget(v2, fake_whisper, monkeypatch):  # noqa: F811
    _jobs, box = v2
    make_clip(box / "longo.mp4", seconds=60)
    job_id = job_id_from(run(mcp_server.analyze_video("longo.mp4")))
    worker.process(jobstore.load_job(job_id), inbox_root=box)
    monkeypatch.setattr(mcp_server, "IMAGE_BUDGET_BYTES", 6000)
    content = run(mcp_server.watch_video(job_id=job_id, max_frames=30))
    payload = sum(len(c.data) for c in content[1:])
    assert payload <= 6000 and "Orçamento de imagens" in content[0].text


def test_watch_job_never_serves_files_outside_its_frames(v2, fake_whisper):  # noqa: F811
    jobs, box = v2
    make_clip(box / "x.mp4", seconds=10)
    job_id = job_id_from(run(mcp_server.analyze_video("x.mp4")))
    worker.process(jobstore.load_job(job_id), inbox_root=box)
    job_dir = jobstore.job_path(job_id)
    secret = jobs / "segredo.jpg"
    secret.write_bytes(b"\xff\xd8\xff" + b"SEGREDO")
    index = json.loads((job_dir / "index.json").read_text())
    index["frames"] = [{"t": 1.0, "file": "../../segredo.jpg", "kind": "scene"},
                       {"t": 2.0, "file": "/etc/passwd", "kind": "scene"}]
    (job_dir / "index.json").write_text(json.dumps(index))
    content = run(mcp_server.watch_video(job_id=job_id))
    assert len(content) == 1


def test_watch_video_needs_exactly_one_of_url_or_job_id(v2):
    with pytest.raises(ToolError, match="url .* ou job_id"):
        run(mcp_server.watch_video())
    with pytest.raises(ToolError, match="url .* ou job_id"):
        run(mcp_server.watch_video(url="https://example.com/v", job_id="0" * 32))


def test_search_before_transcript_and_failed_job(v2):
    job_id = job_id_from(run(mcp_server.analyze_video("https://example.com/b")))
    with pytest.raises(ToolError, match="Ainda não há transcrição"):
        run(mcp_server.search_transcript(job_id, "x"))
    with pytest.raises(ToolError, match="índice de cenas"):
        run(mcp_server.watch_video(job_id=job_id))
    jobstore.update_job(job_id, state="failed", error="o download passou do limite")
    with pytest.raises(ToolError, match="falhou: o download passou do limite"):
        run(mcp_server.watch_video(job_id=job_id))
    assert "o download passou do limite" in run(mcp_server.job_status(job_id))


def test_too_long_url_error_points_to_analyze_video():
    import inspect
    assert "analyze_video" in inspect.getsource(mcp_server.preflight)
