#!/usr/bin/env python3
"""v2 end-to-end: a symlink in the inbox is refused by the real stdio server.

Windows cannot create symlinks without privilege, so this runs the "watch-v2"
server with a throwaway Linux volume (watch-mcp-e2e-inbox, removed at the end)
as /inbox, holding a real clip, a symlink to /etc/passwd and a symlink to the
clip. Everything else is the exact watch-v2 argument list.

    python tests/e2e/e2e_v2_symlink.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from desktop_entry import INBOX_DIR, entry_v2  # noqa: E402
from e2e_mcp_stdio import PROTOCOL_VERSION, Session  # noqa: E402

VOLUME = "watch-mcp-e2e-inbox"


def text_of(result: dict) -> str:
    return "\n".join(c["text"] for c in result.get("content") or [] if c.get("type") == "text")


def main() -> int:
    subprocess.run(["docker", "volume", "create", VOLUME], check=True, capture_output=True)
    try:
        subprocess.run([
            "docker", "run", "--rm", "--network", "none", "-v", f"{VOLUME}:/inbox", "--user", "0",
            "--entrypoint", "sh", "watch-mcp:v2", "-c",
            "ffmpeg -loglevel error -y -f lavfi -i testsrc=size=160x90:rate=5:duration=2 "
            "-f lavfi -i sine=duration=2 -shortest /inbox/clip.mp4 && "
            "ln -s /etc/passwd /inbox/senhas.mp4 && ln -s /inbox/clip.mp4 /inbox/atalho.mp4 && ls -la /inbox",
        ], check=True)
        desktop = entry_v2()
        args = ["docker", *desktop["args"]]
        i = args.index(f"type=bind,src={INBOX_DIR},dst=/inbox,readonly")
        args[i] = f"type=volume,src={VOLUME},dst=/inbox,readonly"
        session = Session(args, {"PATH": os.environ["PATH"], **desktop["env"]})
        results = {}
        try:
            session.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                           "clientInfo": {"name": "watch-e2e-symlink", "version": "1.0"}}, timeout=60)
            session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            listing = text_of(session.request("tools/call", {"name": "list_inbox", "arguments": {}}, timeout=60))
            for name in ("senhas.mp4", "atalho.mp4"):
                r = session.request("tools/call", {"name": "analyze_video", "arguments": {"source": name}}, timeout=60)
                results[name] = (bool(r.get("isError")), text_of(r))
        finally:
            session.close()
    finally:
        subprocess.run(["docker", "volume", "rm", VOLUME], capture_output=True)
    print("list_inbox:", listing.replace("\n", " | "))
    for name, (is_error, message) in results.items():
        print(f"{name}: isError={is_error} :: {message}")
    assert "clip.mp4" in listing and "senhas.mp4" not in listing and "atalho.mp4" not in listing, listing
    assert all(e and "links simbólicos" in m for e, m in results.values()), results
    print("V2 SYMLINK E2E: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
