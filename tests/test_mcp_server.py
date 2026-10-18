"""mcp_server.py: input checks, budget, output shape, cleanup, timeouts.

The integration tests run the real pipeline (preflight yt-dlp → egress proxy →
watch.py → frames) against a clip served from 127.0.0.1, with the test-mode
allow-list naming that host. No Docker, no external network.
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mcp_server  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.types import ImageContent, TextContent  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ── pure helpers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["5", "1:05", "01:02:03", "90.5", None, "  "])
def test_check_time_accepts(value):
    mcp_server.check_time("start", value)


@pytest.mark.parametrize("value", ["-5", "--help", "1;rm -rf /", "abc", "1:2:3:4", "00:10 "+"x"])
def test_check_time_rejects(value):
    with pytest.raises(ToolError):
        mcp_server.check_time("start", value)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "/etc/passwd", "C:\\video.mp4", "http://localhost/", "http://127.0.0.1:8000/",
    "http://169.254.169.254/latest/meta-data", "ftp://example.com/a.mp4",
])
def test_check_url_rejects_local_and_non_http(url, monkeypatch):
    monkeypatch.delenv("WATCH_MCP_TEST_ALLOW_HOSTS", raising=False)
    with pytest.raises(ToolError, match="URL recusada"):
        mcp_server.check_url(url)


def test_test_mode_is_off_by_default(monkeypatch):
    monkeypatch.delenv("WATCH_MCP_TEST_ALLOW_HOSTS", raising=False)
    import netguard
    assert netguard.test_allow_hosts() == frozenset()


def test_frame_width_keeps_long_side_at_512():
    assert mcp_server.frame_width(1080, 1920) == 288
    assert mcp_server.frame_width(1920, 1080) == 512
    assert mcp_server.frame_width(None, None) == 512


def test_fit_budget_all_fit():
    assert mcp_server.fit_budget([10] * 5, 100) == [0, 1, 2, 3, 4]


def test_fit_budget_reduces_uniformly_keeping_ends():
    picks = mcp_server.fit_budget([10] * 24, 120)
    assert len(picks) == 12
    assert picks[0] == 0 and picks[-1] == 23
    gaps = {b - a for a, b in zip(picks, picks[1:])}
    assert gaps <= {2, 3}


def test_encode_frames_warns_when_over_budget(tmp_path):
    frames = []
    for i in range(10):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"x" * 3000)
        frames.append(mcp_server.Frame(p, f"00:0{i}"))
    kept, encoded, note = mcp_server.encode_frames(frames, budget=4000 * 5)
    assert len(kept) == len(encoded) == 5
    assert sum(len(e) for e in encoded) <= 20000
    assert note and "5 de 10" in note


def test_login_wall_message_only_for_instagram():
    err = "ERROR: [Instagram] abc: Requested content is not available, rate-limit reached or login required. Use --cookies"
    msg = mcp_server.login_wall_message("https://www.instagram.com/reel/abc/", err)
    assert msg and "cookies.txt" in msg and "SECUNDÁRIA" in msg
    assert mcp_server.login_wall_message("https://www.youtube.com/watch?v=x", err) is None
    assert mcp_server.login_wall_message("https://www.instagram.com/reel/abc/", "ERROR: HTTP 500") is None


def test_scrub_redacts_cookie_values():
    assert "s3cr3t" not in mcp_server.scrub("Cookie: sessionid=s3cr3t")
    assert "s3cr3t" not in mcp_server.scrub("cookies=s3cr3t;")


def test_ytdlp_args_include_limits_proxy_and_optional_cookies(tmp_path):
    args = mcp_server.ytdlp_args("http://127.0.0.1:1", None)
    assert args[args.index("--proxy") + 1] == "http://127.0.0.1:1"
    assert "--max-filesize" in args and "--ignore-config" in args
    assert "--cookies" not in args
    with_cookies = mcp_server.ytdlp_args("http://127.0.0.1:1", tmp_path / "c.txt")
    assert with_cookies[with_cookies.index("--cookies") + 1] == str(tmp_path / "c.txt")


def test_whisper_down_gives_compose_hint(monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_URL", "http://127.0.0.1:9/v1/audio/transcriptions")
    with pytest.raises(ToolError, match=r"docker compose -p watch-mcp up -d"):
        mcp_server.check_whisper()


def test_run_process_timeout_kills_child(tmp_path):
    marker = tmp_path / "survived"
    code = f"import time,pathlib; time.sleep(5); pathlib.Path({str(marker)!r}).write_text('x')"
    started = time.monotonic()
    with pytest.raises(ToolError, match="Tempo limite"):
        run(mcp_server.run_process(
            [sys.executable, "-c", code], env=mcp_server.child_env({}), cwd=tmp_path,
            timeout=1, ctx=None, label="teste",
        ))
    assert time.monotonic() - started < 4
    time.sleep(5)
    assert not marker.exists()


# ── integration (real yt-dlp + proxy + watch.py, local only) ────────────────


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


def _serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def local_stack(cut_clip, tmp_path, monkeypatch):
    if shutil.which("yt-dlp") is None:
        pytest.skip("yt-dlp not installed")
    site = tmp_path / "site"
    site.mkdir()
    shutil.copy(cut_clip, site / "clip.mp4")
    files = _serve(functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site)))
    files.RequestHandlerClass.log_message = lambda *a: None
    health = _serve(_Health)
    monkeypatch.setenv("WATCH_MCP_TEST_ALLOW_HOSTS", "127.0.0.1")
    monkeypatch.setenv("WATCH_WHISPER_URL", f"http://127.0.0.1:{health.server_port}/v1/audio/transcriptions")
    monkeypatch.setattr(mcp_server, "COOKIES_PATH", tmp_path / "no-cookies.txt")
    made: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        made.append(Path(path))
        return path

    monkeypatch.setattr(mcp_server.tempfile, "mkdtemp", tracking_mkdtemp)
    yield f"http://127.0.0.1:{files.server_port}/clip.mp4", made, tmp_path
    files.shutdown()
    health.shutdown()


def test_watch_video_returns_text_then_images_and_cleans_up(local_stack):
    url, made, _ = local_stack
    content = run(mcp_server.watch_video(url, question="o que aparece?", max_frames=6))
    assert isinstance(content[0], TextContent)
    images = content[1:]
    assert images and all(isinstance(c, ImageContent) and c.mime_type == "image/jpeg" for c in images)
    text = content[0].text
    assert "Pergunta do usuário: o que aparece?" in text
    stamps = [line.split("t=", 1)[1] for line in text.splitlines() if ". t=" in line]
    assert len(stamps) == len(images)
    assert stamps == sorted(stamps)
    assert made and all(not p.exists() for p in made)


def test_parse_frames_matches_real_watch_report(local_stack):
    url, _, tmp_path = local_stack
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, str(mcp_server.SCRIPTS_DIR / "watch.py"), url, "--out-dir", str(out),
         "--no-whisper", "--max-frames", "5"],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
        env=mcp_server.child_env({}),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    frames = mcp_server.parse_frames(proc.stdout)
    assert frames and all(f.path.exists() for f in frames)


def test_cookies_are_copied_never_echoed(local_stack, monkeypatch):
    url, made, tmp_path = local_stack
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n.instagram.com\tTRUE\t/\tTRUE\t0\tsessionid\tTOPSECRET\n")
    monkeypatch.setattr(mcp_server, "COOKIES_PATH", cookies)
    seen: list[list[str]] = []
    real = mcp_server.run_process

    async def spy(argv, **kwargs):
        seen.append(list(argv) + [kwargs["env"].get("WATCH_YTDLP_ARGS", "")])
        return await real(argv, **kwargs)

    monkeypatch.setattr(mcp_server, "run_process", spy)
    content = run(mcp_server.watch_video(url, max_frames=2))
    flat = json.dumps(seen)
    assert str(cookies) not in flat  # only the private copy is handed to yt-dlp
    assert "cookies.txt" in flat
    assert all("TOPSECRET" not in getattr(c, "text", "") for c in content)
    assert cookies.read_text().endswith("TOPSECRET\n")  # original untouched
    assert all(not p.exists() for p in made)


def test_work_dir_removed_on_failure(local_stack):
    url, made, _ = local_stack
    with pytest.raises(ToolError):
        run(mcp_server.watch_video(url.replace("clip.mp4", "missing.mp4")))
    assert made and all(not p.exists() for p in made)


def test_redirect_to_internal_host_is_blocked(local_stack):
    """Allowed test host redirects to a host outside the allow-list → refused by the proxy."""
    hits: list[str] = []

    class Target(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    target = _serve(Target)

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{target.server_port}/clip.mp4")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirect = _serve(Redirect)
    try:
        with pytest.raises(ToolError, match="403"):
            run(mcp_server.watch_video(f"http://127.0.0.1:{redirect.server_port}/r.mp4"))
        assert hits == []
    finally:
        redirect.shutdown()
        target.shutdown()


def test_youtube_block_message():
    err = "ERROR: [youtube] x: Sign in to confirm you’re not a bot. Use --cookies"
    assert "anti-bot" in mcp_server.site_block_message("https://www.youtube.com/watch?v=x", err)
    assert mcp_server.site_block_message("https://youtu.be/x", "ERROR: HTTP Error 429: Too Many Requests")
    assert mcp_server.youtube_block_message("https://vimeo.com/1", err) is None


def test_duration_falls_back_to_watch_report():
    report = "- **Source:** u\n- **Duration:** 01:15 (75.3s)\n"
    assert float(mcp_server.DURATION_LINE_RE.search(report).group("s")) == 75.3


def test_author_handle_skips_numeric_ids():
    result = mcp_server.RunResult(info={"uploader": "Leo Messi", "uploader_id": "427553890", "channel": "leomessi"})
    text = mcp_server.build_text("u", result, question=None, detail="balanced", start=None, end=None, frames=[])
    assert "Autor: Leo Messi (@leomessi)" in text
    result = mcp_server.RunResult(info={"uploader": "X", "uploader_id": "123"})
    text = mcp_server.build_text("u", result, question=None, detail="balanced", start=None, end=None, frames=[])
    assert "- Autor: X\n" in text
