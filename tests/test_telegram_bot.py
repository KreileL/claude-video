"""Tests for telegram_bot.py; local http.server stubs only, fake tokens."""
import contextlib
import http.server
import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jobstore  # noqa: E402
import telegram_bot as tb  # noqa: E402

TOKEN = "123456:FAKE-token_ABCdef"
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
    assert api.sent[2] == "Trabalho ffffffff não encontrado."
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
