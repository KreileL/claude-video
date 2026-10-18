"""Regressions for the Codex counter-review of v2 (.codex/review-mcp-v2.md).

Each test is adapted from Codex's reproduction and asserts the fixed behaviour.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import http.server
import json
import os
import shutil
import subprocess
import threading
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import jobstore
import jobview
import mcp_server
import worker
from test_worker import dirs, fake_whisper, make_clip, new_job  # noqa: F401  (fixtures)


def run(coro):
    return asyncio.run(coro)


def dirlink(link, target):
    """A directory link: junction on Windows (no privilege needed), symlink elsewhere."""
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if result.returncode:
            pytest.skip("directory junction unavailable")
    else:
        link.symlink_to(target, target_is_directory=True)


# ── B1: cookies reflected in metadata ───────────────────────────────────────


def test_b1_cookie_values_are_redacted_from_display_metadata(tmp_path, monkeypatch):
    secret = "REVIEW-COOKIE-SYNTHETIC-12345"
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(jobstore, "JOBS_DIR", jobs)
    data = new_job(jobs, kind="url", value="https://example.test/video")
    job = worker.Job(data, jobs)
    meta = worker.write_meta(job, {"title": f"t {secret}", "description": secret, "uploader": secret,
                                   "http_headers": {"Cookie": secret}}, frozenset({secret}))
    job.update(title=meta.get("title"), state="done")
    jobstore.write_json_atomic(job.dir / "index.json", {"frames": []})
    for path in job.dir.rglob("*.json"):
        assert secret not in path.read_text(encoding="utf-8")
    assert secret not in run(mcp_server.job_status(job.id))
    assert secret not in run(mcp_server.list_jobs())
    assert secret not in run(mcp_server.watch_video(job_id=job.id, detail="transcript"))[0].text


def test_b1_real_yt_dlp_cookie_reflection_never_reaches_the_cache(dirs, monkeypatch, tmp_path):  # noqa: F811
    if shutil.which("yt-dlp") is None:
        pytest.skip("yt-dlp unavailable")
    jobs, box = dirs
    site = tmp_path / "site"
    site.mkdir()
    make_clip(site / "clip.mp4", seconds=3)
    secret = "REVIEW-REFLECTED-COOKIE-98765"
    seen = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/page":
                cookie = self.headers.get("Cookie", "")
                seen.append(cookie)
                body = (f"<html><head><title>Title {cookie}</title></head>"
                        '<body><video src="/clip.mp4" controls></video></body></html>').encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    jar = tmp_path / "cookies.txt"
    jar.write_text("# Netscape HTTP Cookie File\n127.0.0.1\tFALSE\t/\tFALSE\t2147483647\tsid\t" + secret + "\n")
    monkeypatch.setattr(mcp_server, "COOKIES_PATH", jar)
    monkeypatch.setenv("WATCH_MCP_TEST_ALLOW_HOSTS", "127.0.0.1")
    data = new_job(jobs, kind="url", value=f"http://127.0.0.1:{server.server_port}/page")
    try:
        worker.process(data, root=jobs, inbox_root=box)
    finally:
        server.shutdown()
    assert any(secret in c for c in seen), seen  # the site really reflected it
    assert jobstore.load_job(data["id"], jobs)["state"] == "done"
    for path in (jobs / data["id"]).rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


# ── B2: disk cap during the job ─────────────────────────────────────────────


def test_b2_disk_cap_holds_while_the_job_runs(dirs, monkeypatch):  # noqa: F811
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=8)
    data = new_job(jobs)
    cap = jobstore.dir_size(jobs) + 1024
    monkeypatch.setattr(worker, "MAX_JOBS_BYTES", cap)
    worker.process(data, root=jobs, inbox_root=box)
    final = jobstore.load_job(data["id"], jobs)
    assert final["state"] == "failed" and "espaço" in final["error"]
    assert jobstore.dir_size(jobs) <= cap


def test_b2_room_check_evicts_finished_jobs_first(dirs, monkeypatch):  # noqa: F811
    jobs, box = dirs
    old = new_job(jobs, value="old.mp4")
    (jobstore.job_path(old["id"], jobs) / "big.bin").write_bytes(b"x" * 2_400_000)
    jobstore.update_job(old["id"], jobs, state="done")
    make_clip(box / "clip.mp4", seconds=4)
    data = new_job(jobs)
    monkeypatch.setattr(worker, "MAX_JOBS_BYTES", 2_500_000)
    worker.process(data, root=jobs, inbox_root=box)
    final = jobstore.load_job(data["id"], jobs)
    assert final["state"] == "done", final.get("error")
    assert not jobstore.job_path(old["id"], jobs).exists()


# ── B3: the total deadline covers Whisper ───────────────────────────────────


def test_b3_no_whisper_call_after_the_deadline(tmp_path, monkeypatch):
    data = jobstore.create_job({}, tmp_path)
    job = worker.Job(data, tmp_path)
    job.deadline = time.time() - 60
    calls = []
    monkeypatch.setattr(worker, "whisper_ready", lambda *a, **k: calls.append("health"))
    monkeypatch.setattr(worker, "transcribe_local", lambda *a, **k: calls.append("post") or [])
    with pytest.raises(worker.JobFailed, match="tempo total"):
        worker.transcribe_chunk(job, tmp_path / "x.mp3", model="m", language="pt", prompt=None)
    assert calls == []


def test_b3_slow_whisper_cannot_finish_a_job_past_its_deadline(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    data = new_job(jobs)
    d = jobs / data["id"]
    (d / "work").mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=1",
                    str(d / "work" / "audio.mp3")], check=True)
    jobstore.write_json_atomic(d / "plan.json", {"duration": 1, "chunks": [[0, 1]]})
    jobstore.write_json_atomic(d / "index.json", {"frames": []})

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(2.5)
            body = json.dumps({"segments": [{"start": 0, "end": 1, "text": "fala"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("WATCH_WHISPER_URL", f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions")
    monkeypatch.setenv("WATCH_WHISPER_TIMEOUT", "10")
    monkeypatch.setattr(worker, "JOB_TIMEOUT", 1.5)
    started = time.monotonic()
    try:
        worker.process(data, root=jobs)
    finally:
        server.shutdown()
    final = jobstore.load_job(data["id"], jobs)
    assert final["state"] == "failed" and "tempo" in final["error"], final
    assert time.monotonic() - started < 2.4


# ── B4: resume never mixes two sources ──────────────────────────────────────


def test_b4_replaced_inbox_file_on_resume_fails_the_job(dirs, monkeypatch):  # noqa: F811
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=6)
    data = new_job(jobs)
    original_index = worker.build_index

    def crash_before_index(job, src):
        raise KeyboardInterrupt("audio and plan committed")

    monkeypatch.setattr(worker, "build_index", crash_before_index)
    with pytest.raises(KeyboardInterrupt):
        worker.process(data, root=jobs, inbox_root=box)
    make_clip(box / "clip.mp4", seconds=12)  # same name, different file
    monkeypatch.setattr(worker, "build_index", original_index)
    worker.process(jobstore.load_job(data["id"], jobs), root=jobs, inbox_root=box)
    final = jobstore.load_job(data["id"], jobs)
    assert final["state"] == "failed" and "mudou" in final["error"]


def test_b4_file_changed_between_analyze_and_worker_fails(dirs):  # noqa: F811
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=3)
    data = new_job(jobs, fingerprint=[1, 2])  # what the server saw differs from the file now
    worker.process(data, root=jobs, inbox_root=box)
    final = jobstore.load_job(data["id"], jobs)
    assert final["state"] == "failed" and "mudou" in final["error"]


# ── B5: the index covers the whole video ────────────────────────────────────


def test_b5_scene_heavy_video_is_indexed_to_the_end(dirs, monkeypatch):  # noqa: F811
    jobs, box = dirs
    out = box / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "nullsrc=s=160x90:r=10:d=10,geq=lum='if(mod(floor(N/2),2),255,0)':cb=128:cr=128",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    monkeypatch.setattr(worker, "MAX_INDEX_FRAMES", 5)
    data = new_job(jobs)
    worker.process(data, root=jobs, inbox_root=box)
    assert jobstore.load_job(data["id"], jobs)["state"] == "done"
    index = jobstore.read_json(jobs / data["id"] / "index.json")
    frames = index["frames"]
    assert frames[-1]["t"] >= 8 and len(frames) <= 2 * 5 + 2
    assert jobview.pick_frames(frames, jobs / data["id"], 5, 9, "balanced", 24)


# ── B6: a corrupt job never takes the worker down ───────────────────────────


@pytest.mark.parametrize("field,value", [("started", "broken"), ("attempts", "broken"),
                                         ("source", "x"), ("terms", [1, 2])])
def test_b6_semantically_corrupt_job_is_failed_not_crashing(tmp_path, field, value):
    data = jobstore.create_job({"source": {"kind": "inbox", "value": "a.mp4"}, field: value}, tmp_path)
    worker.process(data, root=tmp_path)
    final = jobstore.load_job(data["id"], tmp_path)
    assert final["state"] == "failed" and "corrompido" in final["error"]


# ── non-blocking ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("location", ["job", "frames"])
def test_n1_directory_links_inside_jobs_are_refused(tmp_path, monkeypatch, location):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(jobstore, "JOBS_DIR", jobs)
    victim = new_job(jobs)
    victim_dir = jobs / victim["id"]
    (victim_dir / "frames").mkdir()
    jpeg = b"\xff\xd8\xffSYNTHETIC-OTHER-JOB-PRIVATE-IMAGE"
    (victim_dir / "frames" / "f_00001.jpg").write_bytes(jpeg)
    data = new_job(jobs)
    d = jobs / data["id"]
    if location == "job":
        outside = tmp_path / "outside-jobs"
        d.rename(outside)
        dirlink(d, outside)
        (outside / "frames").mkdir()
        (outside / "frames" / "f_00001.jpg").write_bytes(jpeg)
    else:
        dirlink(d / "frames", victim_dir / "frames")
    jobstore.write_json_atomic(d / "index.json", {"frames": [{"t": 0, "file": "f_00001.jpg"}]})
    try:
        content = run(mcp_server.watch_video(job_id=data["id"]))
    except ToolError:
        return
    assert all(base64.b64decode(c.data) != jpeg for c in content[1:])


def test_n2_search_response_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(jobstore, "JOBS_DIR", tmp_path)
    data = new_job(tmp_path)
    d = jobstore.job_path(data["id"], tmp_path)
    segment = {"start": 0, "end": 1, "text": "alvo " + "x" * 100_000}
    jobstore.write_json_atomic(d / "transcript.json", {"segments": [segment] * 20})
    result = run(mcp_server.search_transcript(data["id"], "alvo"))
    assert len(result) < jobview.SEARCH_TEXT_LIMIT + 2000


def test_n3_corrupt_chunk_is_transcribed_again(dirs, monkeypatch):  # noqa: F811
    jobs, box = dirs
    make_clip(box / "clip.mp4", seconds=6)
    data = new_job(jobs)
    real = worker.transcribe_local
    monkeypatch.setattr(worker, "transcribe_local", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        worker.process(data, root=jobs, inbox_root=box)
    d = jobs / data["id"]
    (d / "chunks").mkdir(exist_ok=True)
    (d / "chunks" / "0000.json").write_text("{corrupt")
    calls = []
    monkeypatch.setattr(worker, "transcribe_local", lambda *a, **k: calls.append(1) or real(*a, **k))
    worker.process(jobstore.load_job(data["id"], jobs), root=jobs, inbox_root=box)
    assert calls and jobstore.load_job(data["id"], jobs)["state"] == "done"
    assert jobstore.read_json(d / "transcript.json")["segments"]


def test_n4_video_is_deleted_before_transcription_even_after_a_restart(dirs, monkeypatch):  # noqa: F811
    jobs, box = dirs
    data = new_job(jobs)
    d = jobs / data["id"]
    make_clip(box / "clip.mp4", seconds=4)
    original_transcribe = worker.transcribe
    monkeypatch.setattr(worker, "transcribe", lambda job: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        worker.process(data, root=jobs, inbox_root=box)
    (d / "work" / "source.mkv").write_bytes(b"leftover video")  # as if the delete never ran
    seen = []

    def inspect(job):
        seen.append((d / "work" / "source.mkv").exists())
        original_transcribe(job)

    monkeypatch.setattr(worker, "transcribe", inspect)
    worker.process(jobstore.load_job(data["id"], jobs), root=jobs, inbox_root=box)
    assert seen == [False]
