#!/usr/bin/env python3
"""v2 end-to-end: inbox attacks over the real MCP stdio server ("watch-v2" args).

Creates clearly test-only files in D:\\Inbox (removed at the end) and checks that
analyze_video refuses, with a clear error and without creating a job:
path traversal, absolute paths, drive paths, a forbidden extension, a text
file disguised as .mp4, an HLS playlist disguised as .mp4, an ffconcat file
disguised as .mkv, a missing file, and a symlink (when Windows allows creating
one). list_inbox must list only the allowed media.

    python tests/e2e/e2e_v2_inbox.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from desktop_entry import INBOX_DIR, entry_v2  # noqa: E402
from e2e_mcp_stdio import PROTOCOL_VERSION, Session  # noqa: E402

PREFIX = "watch-mcp-e2e-"
INBOX = Path(INBOX_DIR)


def text_of(result: dict) -> str:
    return "\n".join(c["text"] for c in result.get("content") or [] if c.get("type") == "text")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = []

    def put(name: str, data: bytes) -> None:
        (INBOX / name).write_bytes(data)
        made.append(INBOX / name)

    good = f"{PREFIX}ok.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=160x90:rate=5:duration=3", "-f", "lavfi", "-i", "sine=duration=3",
                    "-shortest", str(INBOX / good)], check=True)
    made.append(INBOX / good)
    put(f"{PREFIX}texto.mp4", "isto não é um vídeo\n".encode())
    put(f"{PREFIX}lista.mp4", b"#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\nhttp://watch-whisper:8000/health\n#EXT-X-ENDLIST\n")
    put(f"{PREFIX}concat.mkv", f"ffconcat version 1.0\nfile '{good}'\n".encode())
    put(f"{PREFIX}roteiro.txt", b"roteiro")
    link = INBOX / f"{PREFIX}link.mp4"
    symlink_made = False
    try:
        os.symlink(INBOX / good, link)
        made.append(link)
        symlink_made = True
    except OSError:
        pass  # needs admin or Developer Mode on Windows; covered by the Linux unit tests

    cases = [
        ("../" + good, "nome inválido"),
        ("..\\" + good, "nome inválido"),
        ("/etc/passwd", "nome inválido"),
        ("C:\\Windows\\win.ini", "nome inválido"),
        (f"sub/{good}", "nome inválido"),
        (f"{PREFIX}roteiro.txt", "extensão não permitida"),
        (f"{PREFIX}texto.mp4", "não é um vídeo/áudio válido"),
        (f"{PREFIX}lista.mp4", "não é um vídeo/áudio válido"),
        (f"{PREFIX}concat.mkv", "não é um vídeo/áudio válido"),
        (f"{PREFIX}sumiu.mp4", "não encontrado"),
        ("file:///inbox/" + good, "URL recusada"),
    ]
    if symlink_made:
        cases.append((link.name, "links simbólicos"))

    desktop = entry_v2()
    session = Session(["docker", *desktop["args"]], {"PATH": os.environ["PATH"], **desktop["env"]})
    report: dict = {"symlink_tested": symlink_made, "cases": []}
    try:
        session.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                       "clientInfo": {"name": "watch-e2e-inbox", "version": "1.0"}}, timeout=60)
        session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        listing = text_of(session.request("tools/call", {"name": "list_inbox", "arguments": {}}, timeout=60))
        report["list_inbox"] = listing
        assert good in listing and f"{PREFIX}roteiro.txt" not in listing, listing
        jobs_before = text_of(session.request("tools/call", {"name": "list_jobs", "arguments": {}}, timeout=60))
        for source, expected in cases:
            result = session.request("tools/call", {"name": "analyze_video", "arguments": {"source": source}},
                                     timeout=60)
            message = text_of(result)
            ok = bool(result.get("isError")) and expected in message
            report["cases"].append({"source": source, "isError": bool(result.get("isError")),
                                    "message": message, "ok": ok})
            print(("OK  " if ok else "BAD ") + f"{source!r}: {message}", flush=True)
        jobs_after = text_of(session.request("tools/call", {"name": "list_jobs", "arguments": {}}, timeout=60))
        report["no_job_created"] = jobs_before == jobs_after
    finally:
        session.close()
        for path in made:
            path.unlink(missing_ok=True)
        (out / "inbox_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    assert all(c["ok"] for c in report["cases"]), [c for c in report["cases"] if not c["ok"]]
    assert report["no_job_created"], "a refused source created a job"
    print(f"V2 INBOX E2E: OK ({len(report['cases'])} refusals; symlink tested: {symlink_made})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
