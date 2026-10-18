"""Tests for telegram_bot.py; local http.server stubs only, fake tokens."""
import contextlib
import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jobstore  # noqa: E402
import telegram_bot as tb  # noqa: E402

TOKEN = "123456:FAKE-token_ABCdef0123456789abcdefXYZ"
ME = tb.ALLOWED_USER


@contextlib.contextmanager
def server_for(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


class Stub:
    """Scripted Telegram: `script` is a list of (status, updates) for getUpdates."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []  # (method, params)
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                method = self.path.rsplit("/", 1)[-1]
                stub.requests.append((method, body, self.path))
                status, result = 200, True
                if method == "getUpdates":
                    status, result = stub.script.pop(0) if stub.script else (200, [])
                data = json.dumps({"ok": status == 200, "result": result}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.handler = H

    def sent(self):
        return [p["text"] for m, p, _ in self.requests if m == "sendMessage"]


def msg(text, uid=ME, chat_type="private", chat_id=None, **extra):
    m = {"message_id": 1, "from": {"id": uid, "is_bot": False},
         "chat": {"id": uid if chat_id is None else chat_id, "type": chat_type}, "text": text}
    m.update(extra)
    return m


def upd(n, message):
    return {"update_id": n, "message": message}


def run_once(stub, enqueue, tmp_path, sleep=lambda s: None, batches=1):
    """Run the loop until the stub's script is consumed."""
    with server_for(stub.handler) as base:
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return not stub.script and calls["n"] > 2

        tb.run(enqueue=enqueue, api=tb.Api(TOKEN, base), root=tmp_path, stop=stop, sleep=sleep)


def fake_enqueue(url):
    return {"id": "a" * 32}, False


# --- is_allowed ---

def test_is_allowed_cases():
    assert tb.is_allowed(upd(1, msg("x")))
    assert not tb.is_allowed(upd(1, msg("x", uid=123)))
    assert not tb.is_allowed(upd(1, msg("x", chat_type="group")))
    assert not tb.is_allowed(upd(1, msg("x", chat_type="supergroup")))
    assert not tb.is_allowed({"update_id": 1, "channel_post": msg("x")})
    assert not tb.is_allowed({"update_id": 1, "edited_message": msg("x")})
    bot = msg("x")
    bot["from"]["is_bot"] = True
    assert not tb.is_allowed(upd(1, bot))
    assert not tb.is_allowed(upd(1, msg("x", chat_id=999)))


# --- extract_urls ---

def test_extract_instagram_share():
    m = {"text": "Veja este reel https://www.instagram.com/reel/AbC123/?igsh=xyz"}
    assert tb.extract_urls(m) == ["https://www.instagram.com/reel/AbC123/?igsh=xyz"]


def test_extract_caption_and_text_link():
    m = {"caption": "olha https://youtu.be/abc",
         "text": "clique aqui", "entities": [{"type": "text_link", "offset": 0, "length": 6,
                                              "url": "https://tiktok.com/@a/video/1"}]}
    assert tb.extract_urls(m) == ["https://tiktok.com/@a/video/1", "https://youtu.be/abc"]
    assert tb.extract_urls({"caption": "olha https://youtu.be/abc"}) == ["https://youtu.be/abc"]


def test_extract_utf16_offset_entity():
    # Each emoji is 2 UTF-16 units; the entity offset counts those.
    m = {"text": "🎬🎬 https://youtu.be/x", "entities": [{"type": "url", "offset": 5, "length": 18}]}
    assert tb.extract_urls(m) == ["https://youtu.be/x"]
    # Entity-only (no regex help): url with a non-matching-by-regex wrapper still exact.
    m2 = {"text": "🎬 youtu.be/x", "entities": [{"type": "url", "offset": 3, "length": 10}]}
    assert tb.extract_urls(m2) == []  # no scheme -> not http/https


def test_extract_punctuation_dupes_cap_scheme():
    assert tb.extract_urls({"text": "(veja https://a.com/x), depois https://a.com/x!"}) == ["https://a.com/x"]
    many = " ".join(f"https://a.com/{i}" for i in range(9))
    assert len(tb.extract_urls({"text": many})) == tb.MAX_URLS
    assert tb.extract_urls({"text": "ftp://a.com/x"}) == []


# --- handle_update ---

class FakeApi:
    token = TOKEN

    def __init__(self):
        self.sent = []

    def send(self, chat_id, text):
        self.sent.append(text)


def test_multiple_urls_enqueue_each():
    api, state, seen = FakeApi(), {"offset": None, "pending": []}, []

    def enq(url):
        seen.append(url)
        return {"id": f"{len(seen):x}".zfill(32)}, len(seen) == 2

    tb.handle_update(upd(1, msg("https://a.com/1 https://a.com/2")), api=api, state=state, enqueue=enq)
    assert seen == ["https://a.com/1", "https://a.com/2"]
    assert api.sent[0].startswith("Na fila") and api.sent[1].startswith("Já processado")
    assert len(state["pending"]) == 2


def test_help_status_and_no_url():
    api, state = FakeApi(), {"offset": None, "pending": []}
    for text in ("/ajuda", "/start", "/help"):
        tb.handle_update(upd(1, msg(text)), api=api, state=state, enqueue=fake_enqueue)
    tb.handle_update(upd(1, msg("/status")), api=api, state=state, enqueue=fake_enqueue)
    tb.handle_update(upd(1, msg("oi")), api=api, state=state, enqueue=fake_enqueue)
    assert "00 - Entrada" in api.sent[0]
    assert api.sent[3] == "Nada em andamento."
    assert "link" in api.sent[4]


def test_enqueue_error_reply_is_redacted():
    api, state = FakeApi(), {"offset": None, "pending": []}

    def boom(url):
        raise Exception(f"URL recusada: {TOKEN}")

    tb.handle_update(upd(1, msg("https://a.com/1")), api=api, state=state, enqueue=boom)
    assert api.sent[0].startswith("Não consegui enfileirar")
    assert TOKEN not in api.sent[0]


# --- run ---

def test_foreign_sender_ignored_but_offset_advances(tmp_path, capsys):
    stub = Stub([(200, [upd(7, msg("segredo https://a.com/x", uid=123))])])
    called = []
    run_once(stub, lambda u: called.append(u) or ({"id": "a" * 32}, False), tmp_path)
    assert not stub.sent() and not called
    assert tb.load_state(tmp_path)["offset"] == 8
    out = capsys.readouterr()
    assert "segredo" not in out.out + out.err and "a.com" not in out.out + out.err


def test_offset_persistence(tmp_path):
    stub = Stub([(200, [upd(5, msg("oi"))])])
    run_once(stub, fake_enqueue, tmp_path)
    assert tb.load_state(tmp_path)["offset"] == 6
    stub2 = Stub([(200, [])])
    run_once(stub2, fake_enqueue, tmp_path)
    first = [p for m, p, _ in stub2.requests if m == "getUpdates"][0]
    assert first["offset"] == 6 and first["allowed_updates"] == ["message"]


def test_backoff_on_502(tmp_path):
    stub = Stub([(502, None), (502, None), (200, [upd(1, msg("oi"))])])
    sleeps = []
    run_once(stub, fake_enqueue, tmp_path, sleep=sleeps.append)
    assert sleeps == [1, 2]
    assert len(stub.sent()) == 1


def test_token_never_leaks(tmp_path, capsys):
    stub = Stub([(401, None)])
    with server_for(stub.handler) as base:
        api = tb.Api(TOKEN, base)
        with pytest.raises(tb.TelegramError) as ei:
            api.get_updates(None)
        assert TOKEN not in str(ei.value) and TOKEN not in repr(ei.value)
        assert TOKEN in stub.requests[0][2]  # the token really was in the URL
        stub.script.append((401, None))
        stub.script.append((200, []))
        tb.run(enqueue=fake_enqueue, api=api, root=tmp_path,
               stop=iter([False, False, True, True, True, True]).__next__, sleep=lambda s: None)
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err
    assert tb.redact(f"x bot{TOKEN}/y {TOKEN}", TOKEN) == "x bot<redacted>/y <redacted>"


def test_read_token(tmp_path):
    f = tmp_path / "t"
    f.write_text(f" {TOKEN}\n")
    assert tb.read_token(f) == TOKEN
    with pytest.raises(tb.TelegramError, match="token do Telegram ausente"):
        tb.read_token(tmp_path / "missing")


# --- check_pending ---

def pending_for(job):
    return {"offset": None, "pending": [{"job_id": job["id"], "chat_id": ME,
                                         "short": job["id"][:8], "since": 0}]}


def test_check_pending_ready_failed_missing(tmp_path):
    ok = jobstore.create_job({"state": "archived", "entrada_folder": "2026-10-18 Nota"}, tmp_path)
    bad = jobstore.create_job({"state": "failed", "error": f"linha1 {TOKEN}\nlinha2"}, tmp_path)
    state = {"offset": None, "pending": pending_for(ok)["pending"] + pending_for(bad)["pending"]
             + [{"job_id": "f" * 32, "chat_id": ME, "short": "ffffffff", "since": 0}]}
    api = FakeApi()
    tb.check_pending(api=api, state=state, root=tmp_path)
    assert api.sent[0] == "Pronto — nota em 00 - Entrada/2026-10-18 Nota"
    assert api.sent[1].startswith("Falhou") and "linha2" not in api.sent[1] and TOKEN not in api.sent[1]
    assert api.sent[2] == "Trabalho <code>ffffffff</code> não encontrado."
    assert state["pending"] == []


def test_check_pending_export_warning_once_and_status(tmp_path):
    import time
    job = jobstore.create_job({"state": "done", "export": "pendente",
                               "finished": time.time() - 3600, "progress": 100.0}, tmp_path)
    state, api = pending_for(job), FakeApi()
    tb.check_pending(api=api, state=state, root=tmp_path)
    tb.check_pending(api=api, state=state, root=tmp_path)
    assert len(api.sent) == 1 and "Holocron" in api.sent[0]
    assert len(state["pending"]) == 1
    assert "concluído 100%" in tb.status_text(state, tmp_path)


# --- save_state ---

def test_save_state_refuses_symlink(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    try:
        os.symlink(target, tmp_path / tb.STATE_DIR_NAME, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted")
    with pytest.raises(tb.TelegramError):
        tb.save_state({"offset": 1, "pending": []}, tmp_path)
    assert not list(target.iterdir())


# ── Codex v3 pass 1 ─────────────────────────────────────────────────────────


def test_p1_malformed_token_is_refused_without_echoing_it(tmp_path):
    f = tmp_path / "t"
    f.write_text("123456:FIRST-FAKE-token-aaaaaaaaaaaaaaaaaaaaaaa\n999:SECOND-FAKE\n")
    with pytest.raises(tb.TelegramError) as err:
        tb.read_token(f)
    assert "FIRST-FAKE" not in str(err.value) and "formato inválido" in str(err.value)


def test_p1_failed_reply_keeps_enqueuing_and_tracking(monkeypatch):
    sent, queued = [], []

    class FlakyApi:
        token = TOKEN

        def send(self, chat_id, text):
            sent.append(text)
            raise tb.TelegramError("Telegram respondeu HTTP 503 em sendMessage", status=503)

    def enqueue(url):
        queued.append(url)
        return {"id": f"{len(queued):032x}", "state": "queued"}, False

    state = {"offset": None, "pending": []}
    update = {"update_id": 1, "message": {"chat": {"id": tb.ALLOWED_USER, "type": "private"},
                                          "from": {"id": tb.ALLOWED_USER, "is_bot": False},
                                          "text": "https://youtu.be/a https://youtu.be/b"}}
    tb.handle_update(update, api=FlakyApi(), state=state, enqueue=enqueue)
    assert queued == ["https://youtu.be/a", "https://youtu.be/b"]
    assert [p["job_id"] for p in state["pending"]] == [f"{1:032x}", f"{2:032x}"]


def test_links_from_telegram_use_language_detection_and_hide_internal_errors():
    import types
    calls = []

    class ToolError(Exception):
        pass

    def enqueue_analysis(**kw):
        calls.append(kw)
        if kw["source"].endswith("bad"):
            raise ToolError("URL recusada: host privado")
        if kw["source"].endswith("bug"):
            raise KeyError("/app/segredo/interno")
        return {"id": "a" * 32}, False

    fake = types.SimpleNamespace(enqueue_analysis=enqueue_analysis, ToolError=ToolError)
    enqueue = tb.make_enqueue(fake)
    enqueue("https://www.instagram.com/reel/x/")
    assert calls[0] == {"source": "https://www.instagram.com/reel/x/", "language": "auto", "origin": "telegram"}
    with pytest.raises(ToolError, match="URL recusada"):
        enqueue("https://x/bad")
    with pytest.raises(RuntimeError) as err:
        enqueue("https://x/bug")
    assert "segredo" not in str(err.value)


def test_p2_ready_job_is_tracked_so_a_failed_reply_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(jobstore, "JOBS_DIR", tmp_path)
    job = jobstore.create_job({"source": {"kind": "url", "value": "https://x"}}, root=tmp_path)
    jobstore.update_job(job["id"], tmp_path, state="archived", entrada_folder="2026-10-18 - X (abcdef)")
    outbox, fail = [], {"on": True}

    class Api:
        token = TOKEN

        def send(self, chat_id, text):
            if fail["on"]:
                raise tb.TelegramError("Telegram respondeu HTTP 503 em sendMessage", status=503)
            outbox.append(text)

    state = {"offset": None, "pending": []}
    update = {"update_id": 1, "message": {"chat": {"id": tb.ALLOWED_USER, "type": "private"},
                                          "from": {"id": tb.ALLOWED_USER, "is_bot": False},
                                          "text": "https://x"}}
    tb.handle_update(update, api=Api(), state=state,
                     enqueue=lambda url: (jobstore.load_job(job["id"], tmp_path), True))
    tb.check_pending(api=Api(), state=state, root=tmp_path)
    assert len(state["pending"]) == 1  # the reply failed: still tracked
    fail["on"] = False
    tb.check_pending(api=Api(), state=state, root=tmp_path)
    assert state["pending"] == [] and outbox == ["Pronto — nota em 00 - Entrada/2026-10-18 - X (abcdef)"]


def test_p2_bad_status_line_never_carries_the_token():
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        request = conn.recv(65536).split(b"\r\n")[0]  # "POST /bot<TOKEN>/getUpdates HTTP/1.1"
        conn.sendall(b"GARBAGE " + request + b"\r\n\r\n")
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    api = tb.Api(TOKEN, base=f"http://127.0.0.1:{srv.getsockname()[1]}")
    with pytest.raises(tb.TelegramError) as err:
        api.call("getUpdates", {}, 5)
    assert TOKEN not in str(err.value) and TOKEN.split(":")[1] not in str(err.value)
    srv.close()


def test_p3_a_trickling_answer_respects_a_total_deadline():
    class Trickle(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"ok": True, "result": []}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for byte in body:
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.2)  # under the per-operation timeout, over the total

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        api = tb.Api(TOKEN, base=f"http://127.0.0.1:{server.server_port}")
        start = time.monotonic()
        with pytest.raises(tb.TelegramError, match="tempo esgotado"):
            api.call("getUpdates", {}, 0.5)
        assert time.monotonic() - start < 4.0  # total = 0.5 * 1.2 + 2 s
    finally:
        server.shutdown()


def test_p4_timed_out_calls_are_capped(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(tb, "_ABANDONED", [])
    monkeypatch.setattr(tb, "_read", lambda req, timeout: release.wait(5) or b'{"ok": true, "result": []}')
    api = tb.Api(TOKEN, base="http://127.0.0.1:9")
    try:
        for _ in range(tb.MAX_ABANDONED + 2):
            with pytest.raises(tb.TelegramError, match="tempo esgotado"):
                api.call("getUpdates", {}, 0.01)
        assert len(tb._ABANDONED) == tb.MAX_ABANDONED  # refused calls start no new thread
    finally:
        release.set()


# ── v3 adjustments: HTML formatting ─────────────────────────────────────────


def test_ids_go_in_code_tags_and_dynamic_text_is_escaped():
    sent = []

    class Api:
        token = TOKEN

        def send(self, chat_id, text):
            sent.append(text)

    state = {"offset": None, "pending": []}
    msg = {"chat": {"id": tb.ALLOWED_USER, "type": "private"}, "from": {"id": tb.ALLOWED_USER, "is_bot": False}}
    tb.handle_update({"update_id": 1, "message": {**msg, "text": "https://youtu.be/a"}}, api=Api(), state=state,
                     enqueue=lambda url: ({"id": "ab12cd34" + "0" * 24, "state": "queued"}, False))
    assert sent[-1] == "Na fila — ID <code>ab12cd34</code>" and "`" not in sent[-1]

    def refuse(url):
        raise Exception("URL recusada: <script>&x</script>")

    tb.handle_update({"update_id": 2, "message": {**msg, "text": "https://evil/x"}}, api=Api(), state=state,
                     enqueue=refuse)
    assert sent[-1] == "Não consegui enfileirar: URL recusada: &lt;script&gt;&amp;x&lt;/script&gt;"


def test_send_uses_html_and_falls_back_to_plain_text_when_refused():
    calls = []

    class Api(tb.Api):
        def call(self, method, params, timeout):
            calls.append(params)
            if params.get("parse_mode") == "HTML" and "quebrado" in params["text"]:
                raise tb.TelegramError("Telegram respondeu HTTP 400 em sendMessage", status=400)
            return {}

    api = Api(TOKEN)
    api.send(1, "Na fila — ID <code>ab12</code>")
    assert calls[-1]["parse_mode"] == "HTML"
    api.send(1, "texto quebrado <code>x &amp; y</code>")
    assert "parse_mode" not in calls[-1] and calls[-1]["text"] == "texto quebrado x & y"
