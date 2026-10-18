"""Local Whisper backend: endpoint/model/timeout resolution and the wire format,
exercised against a real in-process HTTP server (no network, no ffmpeg)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import whisper

ENV_NAMES = (
    "WATCH_WHISPER_URL",
    "WATCH_WHISPER_BASE_URL",
    "WATCH_WHISPER_MODEL",
    "WATCH_WHISPER_API_KEY",
    "WATCH_WHISPER_TIMEOUT",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
)

RESPONSE = {
    "text": "olá mundo",
    "language": "pt",
    "duration": 4.0,
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": " olá"},
        {"id": 1, "start": 2.0, "end": 4.0, "text": " mundo"},
    ],
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [])
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def server():
    state = {"requests": [], "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            state["requests"].append(
                {"path": self.path, "headers": dict(self.headers), "body": self.rfile.read(length)}
            )
            payload = json.dumps(RESPONSE if state["status"] == 200 else {"error": "boom"})
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload.encode())))
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{httpd.server_address[1]}/v1/audio/transcriptions"
    yield state
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"\x00\x01fake-mp3")
    return path


def test_url_setting_selects_local_over_groq(monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_URL", "http://x:1/v1/audio/transcriptions")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    backend, key = whisper.load_api_key()
    assert backend == "local"
    assert key == whisper.LOCAL_NO_KEY


def test_transcribe_local_wire_format(monkeypatch, server, audio):
    monkeypatch.setenv("WATCH_WHISPER_URL", server["url"])
    monkeypatch.setenv("WATCH_WHISPER_MODEL", "my/model-x")
    backend, key = whisper.load_api_key()
    segments = whisper._transcribe_file(backend, key, audio)

    req = server["requests"][0]
    assert req["path"] == "/v1/audio/transcriptions"
    assert b"my/model-x" in req["body"]
    assert b"verbose_json" in req["body"]
    assert not any(h.lower() == "authorization" for h in req["headers"])
    assert segments == [
        {"start": 0.0, "end": 2.0, "text": "olá"},
        {"start": 2.0, "end": 4.0, "text": "mundo"},
    ]


def test_default_model_when_unset(monkeypatch, server, audio):
    monkeypatch.setenv("WATCH_WHISPER_URL", server["url"])
    whisper._transcribe_file("local", whisper.LOCAL_NO_KEY, audio)
    assert whisper.DEFAULT_LOCAL_MODEL.encode() in server["requests"][0]["body"]


def test_explicit_api_key_sends_bearer(monkeypatch, server, audio):
    monkeypatch.setenv("WATCH_WHISPER_URL", server["url"])
    monkeypatch.setenv("WATCH_WHISPER_API_KEY", "sekret-value")
    backend, key = whisper.load_api_key()
    whisper._transcribe_file(backend, key, audio)
    headers = {k.lower(): v for k, v in server["requests"][0]["headers"].items()}
    assert headers["authorization"] == "Bearer sekret-value"


def test_base_url_derives_endpoint(monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_BASE_URL", "http://h:9/v1/")
    assert whisper.local_endpoint() == "http://h:9/v1/audio/transcriptions"


def test_url_wins_over_base_url(monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_BASE_URL", "http://h:9/v1")
    monkeypatch.setenv("WATCH_WHISPER_URL", "http://full:1/custom")
    assert whisper.local_endpoint() == "http://full:1/custom"


def test_forced_local_with_nothing_configured_uses_default():
    backend, key = whisper.load_api_key("local")
    assert backend == "local" and key
    assert whisper.local_endpoint() == whisper.DEFAULT_LOCAL_URL


def test_hosted_selection_unchanged_without_local(monkeypatch):
    assert whisper.load_api_key() == (None, None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    assert whisper.load_api_key() == ("openai", "sk-o")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-g")
    assert whisper.load_api_key() == ("groq", "gsk-g")


def test_request_timeout(monkeypatch):
    assert whisper.request_timeout() == 300
    monkeypatch.setenv("WATCH_WHISPER_TIMEOUT", "900")
    assert whisper.request_timeout() == 900.0
    for bad in ("abc", "-1"):
        monkeypatch.setenv("WATCH_WHISPER_TIMEOUT", bad)
        assert whisper.request_timeout() == 300


def test_server_500_retries_then_exits(monkeypatch, server, audio):
    server["status"] = 500
    monkeypatch.setattr(whisper.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit):
        whisper._post_whisper(server["url"], None, "m", audio)
    assert len(server["requests"]) == whisper.MAX_ATTEMPTS
