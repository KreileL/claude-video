"""v2 worker: the whole job pipeline against synthesized media and a fake
Whisper server (no network; real ffmpeg, real yt-dlp for the URL path)."""
from __future__ import annotations

import asyncio
import functools
import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jobstore  # noqa: E402
import mcp_server  # noqa: E402
import whisper  # noqa: E402
import worker  # noqa: E402

JPEG_MAGIC = b"\xff\xd8\xff"


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_clip(path: Path, seconds: int = 65, video: bool = True) -> Path:
    inputs = ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if video:
        inputs = ["-f", "lavfi", "-i", f"testsrc2=size=160x90:rate=5:duration={seconds}", *inputs]
        _ffmpeg(*inputs, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path))
    else:
        _ffmpeg(*inputs, str(path))
    return path


@pytest.fixture
def fake_whisper(monkeypatch):
    # v3: a request without a language field is a language-detection sample
    # (language="auto"); it answers state["language"] and does not count as a block.
    state = {"requests": [], "detections": [], "status": 200, "language": "pt"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if state["status"] == 200 and b'name="language"' not in body:
                state["detections"].append(body)
                payload = json.dumps({"language": state["language"],
                                      "segments": [{"start": 0.0, "end": 1.0, "text": " amostra"}]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload.encode())
                return
            state["requests"].append(body)
            n = len(state["requests"])
            payload = json.dumps({"segments": [{"start": 1.0, "end": 2.5, "text": f" bloco {n}"}]})
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("WATCH_WHISPER_URL", f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions")
    monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])
    monkeypatch.setattr(whisper, "RETRY_BASE_DELAY", 0.0)
    yield state
    server.shutdown()


@pytest.fixture
def dirs(tmp_path, monkeypatch, fake_whisper):
    jobs, box = tmp_path / "jobs", tmp_path / "inbox"
    jobs.mkdir()
    box.mkdir()
    monkeypatch.setattr(worker, "CHUNK_SECONDS", 20.0)
    monkeypatch.setattr(worker, "WHISPER_RETRIES", ())
    monkeypatch.setattr(mcp_server, "COOKIES_PATH", tmp_path / "no-cookies.txt")
    return jobs, box


def new_job(jobs: Path, value: str = "clip.mp4", kind: str = "inbox", **fields) -> dict:
    return jobstore.create_job({
        "source": {"kind": kind, "value": value}, "language": "pt", "quality": "auto",
        "terms": ["Joinville"], "cache_key": "k", **fields,
    }, jobs)


def form_field(body: bytes, name: str) -> str | None:
    marker = f'name="{name}"\r\n\r\n'.encode()
    if marker not in body:
        return None
    return body.split(marker, 1)[1].split(b"\r\n", 1)[0].decode()


def test_inbox_job_runs_to_done(dirs, fake_whisper):
    jobs, box = dirs
    make_clip(box / "clip.mp4")
    job = new_job(jobs)
    worker.process(job, root=jobs, inbox_root=box)

    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "done" and data["progress"] == 100.0, data
    assert data["model"] == worker.MODEL_FAST  # 65 s ≤ 15 min under quality=auto
    assert data["chunks_total"] == 3 and data["has_video"] is True
    path = jobstore.job_path(job["id"], jobs)
    segments = json.loads((path / "transcript.json").read_text(encoding="utf-8"))["segments"]
    assert [s["start"] for s in segments] == [1.0, 21.0, 41.0]  # shifted to source time
    index = json.loads((path / "index.json").read_text(encoding="utf-8"))["frames"]
    assert index[0]["t"] == 0.0 and len(index) >= 10
    assert all((path / "frames" / f["file"]).read_bytes().startswith(JPEG_MAGIC) for f in index)
    assert not (path / "work").exists()  # audio and video are gone
    assert (box / "clip.mp4").exists()  # the inbox is only read
    for body in fake_whisper["requests"]:
        assert form_field(body, "language") == "pt"
        prompt = form_field(body, "prompt")
        assert "português do Brasil" in prompt and "Joinville" in prompt


def test_resume_does_not_retranscribe_finished_chunks(dirs, fake_whisper, monkeypatch):
    jobs, box = dirs
    make_clip(box / "clip.mp4")
    job = new_job(jobs)
    real = worker.transcribe_local
    calls = []

    def dies_on_third(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt  # the worker is killed mid-job
        return real(*args, **kwargs)

    monkeypatch.setattr(worker, "transcribe_local", dies_on_third)
    with pytest.raises(KeyboardInterrupt):
        worker.process(job, root=jobs, inbox_root=box)
    path = jobstore.job_path(job["id"], jobs)
    interrupted = jobstore.load_job(job["id"], jobs)
    assert interrupted["state"] == "running"
    done = sorted((path / "chunks").glob("*.json"))
    assert [p.name for p in done] == ["0000.json", "0001.json"]
    stamps = [p.stat().st_mtime_ns for p in done]

    monkeypatch.setattr(worker, "transcribe_local", real)
    assert jobstore.next_job(jobs)["id"] == job["id"]  # running jobs are resumed first
    worker.process(jobstore.load_job(job["id"], jobs), root=jobs, inbox_root=box)
    assert jobstore.load_job(job["id"], jobs)["state"] == "done"
    assert len(fake_whisper["requests"]) == 3  # 2 before the crash + 1 after
    assert [p.stat().st_mtime_ns for p in done] == stamps


def test_audio_only_inbox_file(dirs):
    jobs, box = dirs
    make_clip(box / "voz.mp3", seconds=25, video=False)
    job = new_job(jobs, value="voz.mp3")
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "done" and data["has_video"] is False
    index = json.loads((jobstore.job_path(job["id"], jobs) / "index.json").read_text())
    assert index["frames"] == []


def test_non_media_file_fails_with_clear_error(dirs):
    jobs, box = dirs
    (box / "fake.mp4").write_text("isto não é vídeo")
    job = new_job(jobs, value="fake.mp4")
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed" and "não é um vídeo/áudio válido" in data["error"]


def test_whisper_failure_fails_the_job(dirs, fake_whisper):
    jobs, box = dirs
    fake_whisper["status"] = 500
    make_clip(box / "clip.mp4", seconds=10)
    job = new_job(jobs)
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed" and "transcrição falhou" in data["error"]
    assert not (jobstore.job_path(job["id"], jobs) / "work").exists()


def test_disk_cap_fails_the_job(dirs, monkeypatch):
    jobs, box = dirs
    monkeypatch.setattr(worker, "MAX_JOBS_BYTES", 1)
    make_clip(box / "clip.mp4", seconds=5)
    job = new_job(jobs)
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed" and "teto" in data["error"]


def test_job_timeout_fails_the_job(dirs, monkeypatch):
    jobs, box = dirs
    monkeypatch.setattr(worker, "JOB_TIMEOUT", 0.0)
    make_clip(box / "clip.mp4", seconds=5)
    job = new_job(jobs)
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed" and "tempo total" in data["error"]


def test_too_long_media_is_refused(dirs, monkeypatch):
    jobs, box = dirs
    monkeypatch.setattr(worker, "MAX_DURATION", 3.0)
    make_clip(box / "clip.mp4", seconds=5)
    job = new_job(jobs)
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed" and "muito longa" in data["error"]


def test_model_policy_and_prompt():
    assert worker.pick_model("auto", 60) == worker.MODEL_FAST
    assert worker.pick_model("auto", 3600) == worker.MODEL_ACCURATE
    assert worker.pick_model("fast", 3600) == worker.MODEL_FAST
    assert worker.pick_model("accurate", 60) == worker.MODEL_ACCURATE
    prompt = worker.build_prompt("pt", ["Embrapa", "Uberlândia"])
    assert prompt.startswith(worker.PROMPT_PTBR) and "Embrapa, Uberlândia" in prompt
    assert len(worker.build_prompt("pt", ["x" * 50] * 40)) <= worker.PROMPT_LIMIT
    assert worker.build_prompt("en", []) is None
    assert worker.build_prompt("pt", []) is None  # no terms, no prompt (benchmark)
    assert worker.build_prompt("en", ["Embrapa"]) == "Embrapa"


def test_url_job_downloads_through_proxy_and_keeps_no_cookies(dirs, tmp_path, monkeypatch):
    if shutil.which("yt-dlp") is None:
        pytest.skip("yt-dlp not installed")
    jobs, _box = dirs
    site = tmp_path / "site"
    site.mkdir()
    make_clip(site / "clip.mp4", seconds=12)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site)))
    server.RequestHandlerClass.log_message = lambda *a: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    jar = tmp_path / "cookies.txt"
    jar.write_text("# Netscape HTTP Cookie File\n.example.com\tTRUE\t/\tTRUE\t0\tsid\tSEGREDO-COOKIE-123\n")
    monkeypatch.setattr(mcp_server, "COOKIES_PATH", jar)
    monkeypatch.setenv("WATCH_MCP_TEST_ALLOW_HOSTS", "127.0.0.1")
    try:
        job = new_job(jobs, value=f"http://127.0.0.1:{server.server_port}/clip.mp4", kind="url")
        worker.process(job, root=jobs)
    finally:
        server.shutdown()
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "done", data
    path = jobstore.job_path(job["id"], jobs)
    assert (path / "meta.json").exists() and not (path / "work").exists()
    for f in path.rglob("*"):
        assert f.name != "cookies.txt"
        if f.is_file():
            assert b"SEGREDO-COOKIE-123" not in f.read_bytes()


def test_url_to_internal_host_is_blocked(dirs, monkeypatch):
    jobs, _box = dirs
    monkeypatch.delenv("WATCH_MCP_TEST_ALLOW_HOSTS", raising=False)
    job = new_job(jobs, value="http://127.0.0.1:9/x.mp4", kind="url")
    worker.process(job, root=jobs)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed"


def test_tampered_download_marker_is_refused(dirs, tmp_path):
    jobs, _box = dirs
    job = new_job(jobs, value="https://example.com/v", kind="url")
    work = jobstore.job_path(job["id"], jobs) / "work"
    work.mkdir()
    (work / "download.done").write_text("../../../etc/passwd", encoding="utf-8")
    worker.process(job, root=jobs)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed"


def test_job_that_keeps_crashing_the_worker_is_failed(dirs):
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=5)
    job = new_job(jobs)
    jobstore.update_job(job["id"], jobs, state="running", started=1.0e12, attempts=worker.MAX_ATTEMPTS)
    worker.process(jobstore.load_job(job["id"], jobs), root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed" and "interrompido" in data["error"]


def seg(start, text):
    return {"start": start, "end": start + 1.0, "text": text}


def test_repetition_loop_detection():
    normal = [seg(i, f"frase número {i} sobre a safra") for i in range(10)]
    assert not worker.looks_repetitive(normal)
    looped = normal[:3] + [seg(10 + i, "É do Léo mesmo.") for i in range(5)]
    assert worker.looks_repetitive(looped)
    one_long = [seg(0, "é do léo mesmo " * 30)]
    assert worker.looks_repetitive(one_long)
    assert not worker.looks_repetitive([seg(0, "Não ri não."), seg(1, "Não ri não."), seg(2, "Tá bom.")])


def test_collapse_repeats_keeps_at_most_two():
    segs = [seg(0, "a"), seg(1, "É do Léo mesmo."), seg(2, "é do léo mesmo"), seg(3, "É do Léo mesmo!"),
            seg(4, "É do Léo mesmo."), seg(5, "fim")]
    assert [s["start"] for s in worker.collapse_repeats(segs)] == [0, 1, 2, 5]


def test_looping_chunk_is_retranscribed_in_short_pieces(dirs, monkeypatch):
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=40)
    monkeypatch.setattr(worker, "CHUNK_SECONDS", 300.0)
    monkeypatch.setattr(worker, "LOOP_PIECE_SECONDS", 15.0)
    calls = []

    def fake(path, **kwargs):
        calls.append(path.name)
        if len(calls) == 1:  # the whole chunk loops
            return [seg(i, "É do Léo mesmo.") for i in range(20)]
        return [seg(0.5, f"pedaço {len(calls)}")]

    monkeypatch.setattr(worker, "transcribe_local", fake)
    job = new_job(jobs)
    worker.process(job, root=jobs, inbox_root=box)
    assert jobstore.load_job(job["id"], jobs)["state"] == "done"
    segments = json.loads((jobstore.job_path(job["id"], jobs) / "transcript.json").read_text(encoding="utf-8"))["segments"]
    assert [s["text"] for s in segments] == ["pedaço 2", "pedaço 3", "pedaço 4"]
    assert [s["start"] for s in segments] == [0.5, 15.5, 30.5]


def test_transient_403_on_media_is_retried_once(dirs, tmp_path, monkeypatch):
    if shutil.which("yt-dlp") is None:
        pytest.skip("yt-dlp not installed")
    jobs, _box = dirs
    site = tmp_path / "site"
    site.mkdir()
    make_clip(site / "clip.mp4", seconds=4)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site)))
    server.RequestHandlerClass.log_message = lambda *a: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("WATCH_MCP_TEST_ALLOW_HOSTS", "127.0.0.1")
    real_run, downloads = worker.run, []

    def flaky(argv, **kwargs):
        if argv[0] == "yt-dlp" and "-o" in argv:
            downloads.append(1)
            if len(downloads) == 1:
                return subprocess.CompletedProcess(argv, 1, "", "ERROR: unable to download video data: HTTP Error 403: Forbidden")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(worker, "run", flaky)
    try:
        job = new_job(jobs, value=f"http://127.0.0.1:{server.server_port}/clip.mp4", kind="url")
        worker.process(job, root=jobs)
    finally:
        server.shutdown()
    assert len(downloads) == 2
    assert jobstore.load_job(job["id"], jobs)["state"] == "done"


# ── v3 adjustments: automatic language, no speech ───────────────────────────


def test_decide_language_by_majority_of_samples_with_speech():
    assert worker.decide_language(["en", "pt", "en"]) == "en"
    assert worker.decide_language([None, "pt", None]) == "pt"
    assert worker.decide_language(["es", "pt"]) == "es"  # tie: the earliest
    assert worker.decide_language([None, None, None]) is None


@pytest.mark.parametrize("heard,prompt_ptbr", [("en", False), ("pt", True)])
def test_auto_language_detects_on_three_samples_and_gates_the_brazilian_prompt(dirs, fake_whisper, heard, prompt_ptbr):
    jobs, box = dirs
    fake_whisper["language"] = heard
    make_clip(box / "clip.mp4")
    job = new_job(jobs, language=None)  # analyze_video's new default: "auto"
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "done" and data["language_detected"] == heard
    assert data["language_votes"] == [heard] * 3 and len(fake_whisper["detections"]) == 3
    for body in fake_whisper["requests"]:
        assert form_field(body, "language") == heard
        prompt = form_field(body, "prompt") or ""
        assert ("português do Brasil" in prompt) is prompt_ptbr and "Joinville" in prompt
    transcript = json.loads((jobstore.job_path(job["id"], jobs) / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["language"] == heard


def test_explicit_language_skips_detection_and_resume_does_not_detect_again(dirs, fake_whisper):
    jobs, box = dirs
    make_clip(box / "clip.mp4")
    job = new_job(jobs, language="pt")
    worker.process(job, root=jobs, inbox_root=box)
    assert fake_whisper["detections"] == []
    auto = new_job(jobs, language=None, language_detected="en", language_votes=["en", None, "en"])
    make_clip(box / "clip.mp4")
    worker.process(auto, root=jobs, inbox_root=box)
    assert fake_whisper["detections"] == []  # already decided before the restart
    assert all(form_field(b, "language") == "en" for b in fake_whisper["requests"][-3:])


def test_silent_media_is_no_speech_not_a_failure(dirs, fake_whisper, monkeypatch):
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=25)
    monkeypatch.setattr(worker, "transcribe_local", lambda *a, **k: [])
    monkeypatch.setattr(worker, "detect_language_local", lambda *a, **k: (None, False))
    job = new_job(jobs, language=None)
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "done" and data["speech"] is False and data["language_detected"] is None
    monkeypatch.setattr(jobstore, "JOBS_DIR", jobs)
    status = asyncio.run(mcp_server.job_status(job["id"]))
    assert "sem fala no áudio" in status and "falhou" not in status
    text = asyncio.run(mcp_server.watch_video(job_id=job["id"], detail="transcript"))[0].text
    assert "Sem fala no áudio" in text


def test_p6_an_error_body_with_http_200_fails_the_job_not_no_speech(dirs, monkeypatch):
    """Codex v3 pass 6: {"error": ...} with 200 used to read as an empty transcript,
    and the job ended as "sem fala" with a note."""
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=25)
    monkeypatch.setattr(whisper, "_post_whisper", lambda *a, **k: {"error": "model unavailable"})
    job = new_job(jobs, language=None)
    worker.process(job, root=jobs, inbox_root=box)
    data = jobstore.load_job(job["id"], jobs)
    assert data["state"] == "failed" and data.get("speech") is not False
