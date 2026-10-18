"""Telegram bot (InstWhisperBot): receives video links from ONE allowed user,
enqueues analysis jobs and reports back. Long polling only, outbound only.

mcp_server is imported lazily in main(); the enqueue function is injected so
the rest of this module stays testable without it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import jobstore

API_BASE = os.environ.get("WATCH_TELEGRAM_API", "https://api.telegram.org")
TOKEN_FILE = Path(os.environ.get("WATCH_TELEGRAM_TOKEN_FILE", "/run/secrets/telegram_bot_token"))
ALLOWED_USER = int(os.environ.get("WATCH_TELEGRAM_ALLOWED_USER", "1779779810"))
STATE_DIR_NAME = ".telegram"  # not a job id, so the job scans ignore it
POLL_TIMEOUT = 50
MAX_URLS = 5
ENTRADA_LABEL = "00 - Entrada"
PENDING_CHECK_EVERY = 10.0
EXPORT_WARN_AFTER = 30 * 60
URL_RE = re.compile(r"https?://[^\s<>\"']+")
TRAILING = ".,;:!?)»”'\""
HELP = ("Mande um link de vídeo do Instagram, YouTube ou TikTok. "
        f"A nota aparece em \"{ENTRADA_LABEL}\".\nComandos: /status e /ajuda")
NO_URL = "Mande um link de vídeo (Instagram, YouTube, TikTok). /ajuda"
STATE_PT = {"queued": "na fila", "running": "processando", "done": "concluído",
            "archived": "arquivado", "failed": "falhou"}


class TelegramError(Exception):
    """Telegram/API failure; the message never contains the token."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def redact(text: str, token: str) -> str:
    """Remove the token (and its "bot<token>" URL form) from text."""
    text = str(text)
    if token:
        text = text.replace("bot" + token, "bot<redacted>").replace(token, "<redacted>")
    return text


def read_token(path: Path = TOKEN_FILE) -> str:
    try:
        token = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if not token:
        raise TelegramError("token do Telegram ausente")
    return token


class Api:
    def __init__(self, token: str, base: str = API_BASE):
        self.token = token
        self.base = base.rstrip("/")

    def call(self, method: str, params: dict, timeout: float) -> object:
        url = f"{self.base}/bot{self.token}/{method}"
        req = urllib.request.Request(url, data=json.dumps(params).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise TelegramError(redact(f"Telegram respondeu HTTP {exc.code} em {method}", self.token),
                                status=exc.code) from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise TelegramError(redact(f"falha de rede em {method}: {type(exc).__name__}", self.token)) from None
        if not isinstance(body, dict) or not body.get("ok"):
            raise TelegramError(f"Telegram recusou {method}")
        return body.get("result")

    def get_updates(self, offset: int | None) -> list[dict]:
        params = {"offset": offset, "timeout": POLL_TIMEOUT, "allowed_updates": ["message"]}
        result = self.call("getUpdates", params, POLL_TIMEOUT + 15)
        return [u for u in result if isinstance(u, dict)] if isinstance(result, list) else []

    def send(self, chat_id: int, text: str) -> None:
        self.call("sendMessage", {"chat_id": chat_id, "text": text,
                                  "disable_web_page_preview": True}, 20)


def is_allowed(update: dict, allowed: int = ALLOWED_USER) -> bool:
    msg = update.get("message") if isinstance(update, dict) else None
    if not isinstance(msg, dict):
        return False
    chat, sender = msg.get("chat"), msg.get("from")
    if not isinstance(chat, dict) or not isinstance(sender, dict):
        return False
    return (chat.get("type") == "private" and sender.get("id") == allowed
            and chat.get("id") == allowed and sender.get("is_bot") is False)


def _entity_text(text: str, ent: dict) -> str:
    # Telegram offsets/lengths are UTF-16 code units.
    raw = text.encode("utf-16-le")
    off, length = ent.get("offset"), ent.get("length")
    if not isinstance(off, int) or not isinstance(length, int) or off < 0 or length < 0:
        return ""
    return raw[2 * off:2 * (off + length)].decode("utf-16-le", errors="ignore")


def _clean_url(url: str) -> str | None:
    url = url.strip().rstrip(TRAILING)
    return url if re.match(r"https?://\S", url, re.I) else None


def extract_urls(message: dict) -> list[str]:
    found: list[str] = []
    for text_key, ent_key in (("text", "entities"), ("caption", "caption_entities")):
        text = message.get(text_key)
        if not isinstance(text, str):
            continue
        for ent in message.get(ent_key) or []:
            if not isinstance(ent, dict):
                continue
            if ent.get("type") == "url":
                found.append(_entity_text(text, ent))
            elif ent.get("type") == "text_link" and isinstance(ent.get("url"), str):
                found.append(ent["url"])
        found.extend(URL_RE.findall(text))
    out: list[str] = []
    for raw in found:
        url = _clean_url(raw)
        if url and url not in out:
            out.append(url)
    return out[:MAX_URLS]


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else jobstore.JOBS_DIR


def load_state(root: Path | None = None) -> dict:
    data = jobstore.read_json(_root(root) / STATE_DIR_NAME / "state.json") or {}
    offset = data.get("offset")
    pending = [p for p in data.get("pending") or [] if isinstance(p, dict) and "job_id" in p]
    return {"offset": offset if isinstance(offset, int) else None, "pending": pending}


def save_state(state: dict, root: Path | None = None) -> None:
    d = _root(root) / STATE_DIR_NAME
    if jobstore.is_link(d):
        raise TelegramError("diretório de estado é um link; recusado")
    d.mkdir(mode=0o700, exist_ok=True)
    jobstore.write_json_atomic(d / "state.json", state)


def _job_state(job_id: str, root: Path | None) -> dict | None:
    try:
        return jobstore.load_job(job_id, root)
    except jobstore.JobError:
        return None


def _is_ready(job: dict) -> bool:
    return job.get("state") in ("done", "archived") and bool(job.get("entrada_folder"))


def _ready_text(job: dict) -> str:
    return f"Pronto — nota em {ENTRADA_LABEL}/{job['entrada_folder']}"


def status_text(state: dict, root: Path | None = None) -> str:
    lines = []
    for p in state.get("pending", []):
        job = _job_state(p["job_id"], root)
        if job is None:
            lines.append(f"{p.get('short', '?')} — desconhecido")
            continue
        pt = STATE_PT.get(job.get("state"), "desconhecido")
        lines.append(f"{p.get('short', '?')} — {pt} {jobstore.as_number(job.get('progress')):.0f}%")
    return "\n".join(lines) if lines else "Nada em andamento."


def handle_update(update: dict, *, api: Api, state: dict,
                  enqueue: Callable[[str], tuple[dict, bool]]) -> None:
    if not is_allowed(update):
        return
    msg = update["message"]
    chat_id = msg["chat"]["id"]
    text = msg.get("text") if isinstance(msg.get("text"), str) else ""
    cmd = text.strip().split(None, 1)[0].split("@")[0].lower() if text.strip() else ""
    if cmd in ("/ajuda", "/start", "/help"):
        return api.send(chat_id, HELP)
    if cmd == "/status":
        return api.send(chat_id, status_text(state))
    urls = extract_urls(msg)
    if not urls:
        return api.send(chat_id, NO_URL)
    for url in urls:
        try:
            job, reused = enqueue(url)
            short = str(job["id"])[:8]
        except Exception as exc:  # noqa: BLE001 - shown to the user, redacted
            api.send(chat_id, "Não consegui enfileirar: " + redact(str(exc), api.token)[:300])
            continue
        if _is_ready(job):
            api.send(chat_id, _ready_text(job))
            continue
        api.send(chat_id, (f"Já processado ou em andamento — ID `{short}`" if reused
                           else f"Na fila — ID `{short}`"))
        if not any(p["job_id"] == job["id"] for p in state["pending"]):
            state["pending"].append({"job_id": job["id"], "chat_id": chat_id,
                                     "short": short, "since": time.time()})


def check_pending(*, api: Api, state: dict, root: Path | None = None) -> None:
    keep = []
    for p in state.get("pending", []):
        chat_id, short = p.get("chat_id"), p.get("short", "?")
        job = _job_state(p["job_id"], root)
        if job is None:
            api.send(chat_id, f"Trabalho {short} não encontrado.")
            continue
        if _is_ready(job):
            api.send(chat_id, _ready_text(job))
            continue
        if job.get("state") == "failed":
            err = str(job.get("error") or "erro desconhecido").splitlines() or [""]
            api.send(chat_id, f"Falhou — {short}: {redact(err[0], api.token)[:300]}")
            continue
        if job.get("state") in ("done", "archived") and job.get("export") == "pendente" \
                and not p.get("warned"):
            done_at = jobstore.as_number(job.get("finished")) or jobstore.as_number(job.get("updated"))
            if time.time() - done_at > EXPORT_WARN_AFTER:
                api.send(chat_id, f"Pronto — {short} concluído; a nota no Holocron está pendente "
                                  "e sai assim que a pasta estiver acessível.")
                p["warned"] = True
        keep.append(p)
    state["pending"] = keep


def _log(msg: str, token: str) -> None:
    print(redact(msg, token)[:300], file=sys.stderr, flush=True)


def run(*, enqueue: Callable[[str], tuple[dict, bool]], api: Api | None = None,
        root: Path | None = None, stop: Callable[[], bool] = lambda: False,
        sleep=time.sleep) -> None:
    if api is None:
        api = Api(read_token())
    state = load_state(root)
    backoff, last_check = 1.0, 0.0
    while not stop():
        if time.monotonic() - last_check >= PENDING_CHECK_EVERY:
            last_check = time.monotonic()
            try:
                check_pending(api=api, state=state, root=root)
                save_state(state, root)
            except Exception as exc:  # noqa: BLE001
                _log(f"erro ao checar pendentes: {type(exc).__name__}: {exc}", api.token)
        try:
            updates = api.get_updates(state["offset"])
        except TelegramError as exc:
            _log(f"erro de rede: {exc}", api.token)
            sleep(30 if exc.status == 409 else backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        backoff = 1.0
        for update in sorted(updates, key=lambda u: u.get("update_id", 0)):
            uid = update.get("update_id")
            if not isinstance(uid, int):
                continue
            try:
                handle_update(update, api=api, state=state, enqueue=enqueue)
            except Exception as exc:  # noqa: BLE001 - never log message content
                _log(f"erro ao tratar update: {type(exc).__name__}", api.token)
            state["offset"] = uid + 1
            try:
                save_state(state, root)
            except Exception as exc:  # noqa: BLE001
                _log(f"erro ao salvar estado: {type(exc).__name__}", api.token)


def main() -> None:
    api = Api(read_token())
    import mcp_server  # lazy: heavy import, and keeps this module testable

    def enqueue(url: str) -> tuple[dict, bool]:
        try:
            return mcp_server.enqueue_analysis(source=url, origin="telegram")
        except mcp_server.ToolError:
            raise  # a user-facing reason (e.g. "URL recusada: ...")
        except Exception as exc:  # noqa: BLE001 - no internal details in the chat
            raise RuntimeError("erro interno ao enfileirar") from exc

    run(enqueue=enqueue, api=api)


if __name__ == "__main__":
    main()
