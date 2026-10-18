#!/usr/bin/env python3
"""Real MCP protocol test against the watch-mcp container over stdio.

Spawns `docker <args>` with exactly the Claude Desktop entry (and Claude
Desktop's minimal environment: PATH + the entry's env block), then speaks raw
newline-delimited JSON-RPC: initialize → notifications/initialized →
tools/list → tools/call. Validates TextContent + ImageContent and saves the
result under --out.

    python tests/e2e/e2e_mcp_stdio.py --url URL [--tool watch_video] [--allow-host H] --out DIR
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desktop_entry import docker_args, entry  # noqa: E402

PROTOCOL_VERSION = "2025-06-18"


class Session:
    def __init__(self, argv: list[str], env: dict[str, str]):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        self.lines: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        self.next_id = 1
        self.progress = 0

    def _pump_out(self):
        for raw in self.proc.stdout:
            self.lines.put(raw.decode("utf-8", "replace"))
        self.lines.put(None)

    def _pump_err(self):
        for raw in self.proc.stderr:
            self.stderr.append(raw.decode("utf-8", "replace").rstrip())

    def send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def request(self, method: str, params: dict, timeout: float) -> dict:
        rid = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{method}: no response in {timeout:.0f}s")
            line = self.lines.get(timeout=remaining)
            if line is None:
                raise RuntimeError(f"server exited; stderr:\n" + "\n".join(self.stderr[-30:]))
            msg = json.loads(line)
            if msg.get("method") == "notifications/progress":
                self.progress += 1
                continue
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"{method} error: {msg['error']}")
                return msg["result"]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tool", default="watch_video", choices=["watch_video", "transcribe_video"])
    ap.add_argument("--question")
    ap.add_argument("--detail")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--allow-host", help="test mode only: WATCH_MCP_TEST_ALLOW_HOSTS for the container")
    ap.add_argument("--expect-error", action="store_true", help="the call must return isError")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    desktop = entry()
    argv = ["docker", *desktop["args"]]
    if args.allow_host:
        argv = ["docker", *docker_args(extra_env={"WATCH_MCP_TEST_ALLOW_HOSTS": args.allow_host})]
    # Claude Desktop's environment: PATH plus the entry's env block, nothing else.
    env = {"PATH": os.environ["PATH"], **desktop["env"]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = Session(argv, env)
    started = time.monotonic()
    try:
        init = session.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "watch-e2e", "version": "1.0"},
        }, timeout=60)
        session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = {t["name"]: t for t in session.request("tools/list", {}, timeout=30)["tools"]}
        assert {"watch_video", "transcribe_video"} <= set(tools), tools.keys()
        assert tools["watch_video"]["inputSchema"]["properties"]["detail"].get("enum") or \
            "transcript" in json.dumps(tools["watch_video"]["inputSchema"]), "detail enum missing"

        call_args: dict = {"url": args.url}
        for key in ("question", "detail", "max_frames"):
            value = getattr(args, key)
            if value is not None:
                call_args[key] = value
        result = session.request("tools/call", {
            "name": args.tool, "arguments": call_args, "_meta": {"progressToken": "e2e"},
        }, timeout=args.timeout)
    finally:
        session.close()
    elapsed = time.monotonic() - started

    content = result.get("content") or []
    texts = [c for c in content if c.get("type") == "text"]
    images = [c for c in content if c.get("type") == "image"]
    (out / "result.txt").write_text("\n\n".join(t["text"] for t in texts), encoding="utf-8")
    summary = {
        "negotiated_protocol": init.get("protocolVersion"),
        "server": init.get("serverInfo"),
        "tool": args.tool,
        "isError": bool(result.get("isError")),
        "content_types": [c.get("type") for c in content],
        "images": len(images),
        "image_b64_bytes": sum(len(i["data"]) for i in images),
        "progress_notifications": session.progress,
        "elapsed_s": round(elapsed, 1),
    }
    for i, image in enumerate(images, 1):
        data = base64.b64decode(image["data"])
        assert image.get("mimeType") == "image/jpeg" and data[:2] == b"\xff\xd8", f"image {i} is not JPEG"
        (out / f"frame_{i:02d}.jpg").write_bytes(data)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.expect_error:
        assert summary["isError"], "expected an error result"
        print(texts[0]["text"] if texts else "")
        return 0
    assert not summary["isError"], texts[0]["text"] if texts else result
    assert content and content[0]["type"] == "text", "first block must be TextContent"
    if args.tool == "watch_video" and args.detail != "transcript":
        assert images, "watch_video returned no ImageContent"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
