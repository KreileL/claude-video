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


def test_explicit_openai_is_honoured_even_with_local_server(monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_BASE_URL", "http://local.test:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic")
    assert whisper.load_api_key("openai") == ("openai", "sk-synthetic")


def test_openai_backend_never_posts_to_local_endpoint(monkeypatch, tmp_path):
    # The OpenAI key must only ever travel to OpenAI, whatever local overrides are set.
    monkeypatch.setenv("WATCH_WHISPER_BASE_URL", "http://local.test:1234/v1")
    monkeypatch.setenv("WATCH_WHISPER_URL", "http://local.test:1234/v1/audio/transcriptions")
    monkeypatch.setenv("WATCH_WHISPER_MODEL", "Systran/faster-whisper-small")
    seen = {}

    def fake_post(endpoint, api_key, model, audio_path):
        seen.update(endpoint=endpoint, model=model)
        return {"segments": []}

    monkeypatch.setattr(whisper, "_post_whisper", fake_post)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    whisper._transcribe_file("openai", "sk-synthetic", audio)
    assert seen == {"endpoint": whisper.OPENAI_ENDPOINT, "model": whisper.OPENAI_MODEL}


def _form_fields(body: bytes) -> dict[str, str]:
    fields = {}
    for part in body.split(b"--")[1:]:
        head, _, value = part.partition(b"\r\n\r\n")
        if b'name="' in head and b"filename=" not in head:
            name = head.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
            fields[name] = value.rsplit(b"\r\n", 1)[0].decode()
    return fields


def test_transcribe_local_sends_language_and_prompt(server, audio, monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_URL", server["url"])
    segments = whisper.transcribe_local(audio, model="m-x", language="pt", prompt="Ônibus, celular.")
    assert [s["text"] for s in segments] == ["olá", "mundo"]
    fields = _form_fields(server["requests"][0]["body"])
    assert fields["model"] == "m-x"
    assert fields["language"] == "pt"
    assert fields["prompt"] == "Ônibus, celular."
    assert "Authorization" not in server["requests"][0]["headers"]


def test_transcribe_local_omits_empty_options(server, audio, monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_URL", server["url"])
    whisper.transcribe_local(audio, model="m-x")
    fields = _form_fields(server["requests"][0]["body"])
    assert "language" not in fields and "prompt" not in fields


def test_transcribe_local_respects_a_deadline(server, audio, monkeypatch):
    import time
    monkeypatch.setenv("WATCH_WHISPER_URL", server["url"])
    server["status"] = 500  # every attempt fails and would normally back off and retry
    monkeypatch.setattr(whisper, "RETRY_BASE_DELAY", 5.0)
    started = time.monotonic()
    with pytest.raises(SystemExit):
        whisper.transcribe_local(audio, model="m", deadline=time.monotonic() + 1.0)
    assert time.monotonic() - started < 3.0  # no 5 s backoff past the deadline
    with pytest.raises(SystemExit, match="deadline"):
        whisper.transcribe_local(audio, model="m", deadline=time.monotonic() - 1.0)


# ── Codex v3 pass 6: an error must not read as "no speech" ─────────────────


def test_p6_error_bodies_are_not_transcriptions():
    import pytest as _pytest
    for bad in ({"error": "model unavailable"}, {"detail": "x"}, [], "texto"):
        with _pytest.raises(SystemExit, match="invalid response"):
            whisper.require_transcription(bad)
    assert whisper.require_transcription({"text": "", "segments": []}) == {"text": "", "segments": []}


def test_p6_a_partial_failure_with_silent_rest_is_an_error(tmp_path):
    import pytest as _pytest
    calls = iter([SystemExit("HTTP Error 503"), []])

    def one(path):
        result = next(calls)
        if isinstance(result, BaseException):
            raise result
        return result

    with _pytest.raises(SystemExit) as err:
        whisper.transcribe_chunks([(tmp_path / "a", 0.0), (tmp_path / "b", 600.0)], one)
    assert "no transcript segments" not in str(err.value) and "1 of 2" in str(err.value)
