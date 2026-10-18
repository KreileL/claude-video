#!/usr/bin/env python3
"""v2 end-to-end: a long video from a URL through the background worker.

Over MCP stdio with the "watch-v2" entry: analyze_video(url) → job_status until
done → search_transcript(query) → watch_video(job_id, a range around the hit).

    python tests/e2e/e2e_v2_url.py --url URL --query WORDS --out DIR
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from desktop_entry import entry_v2  # noqa: E402
from e2e_mcp_stdio import PROTOCOL_VERSION, Session  # noqa: E402


def text_of(result: dict) -> str:
    return "\n".join(c["text"] for c in result.get("content") or [] if c.get("type") == "text")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=3 * 3600)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    desktop = entry_v2()
    session = Session(["docker", *desktop["args"]], {"PATH": os.environ["PATH"], **desktop["env"]})
    report: dict = {"url": args.url}
    try:
        session.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                       "clientInfo": {"name": "watch-e2e-url", "version": "1.0"}}, timeout=60)
        session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        def call(name, arguments, timeout=120):
            return session.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

        created = text_of(call("analyze_video", {"source": args.url}))
        job_id = re.search(r"job_id = ([0-9a-f]{32})", created).group(1)
        started, samples = time.monotonic(), []
        while True:
            status = text_of(call("job_status", {"job_id": job_id}))
            state = re.search(r"- Estado: (.+)", status).group(1)
            stage = re.search(r"- Etapa: (.+)", status).group(1)
            eta = re.search(r"- Estimativa para terminar: (.+)", status)
            samples.append({"t": round(time.monotonic() - started), "stage": stage, "eta": eta and eta.group(1)})
            print(samples[-1], flush=True)
            if state in ("concluído", "falhou") or time.monotonic() - started > args.timeout:
                break
            time.sleep(30)
        report.update(job_id=job_id, elapsed_s=round(time.monotonic() - started), status=status, samples=samples)
        assert state == "concluído", status
        found = text_of(call("search_transcript", {"job_id": job_id, "query": args.query}))
        report["search"] = found
        hit = re.search(r"^- \[(\d+):(\d+)", found, flags=re.M)
        assert hit, found
        t = int(hit.group(1)) * 60 + int(hit.group(2))
        result = call("watch_video", {"job_id": job_id, "start": f"0:{max(0, t - 5) // 60:02d}:{max(0, t - 5) % 60:02d}",
                                      "end": f"0:{(t + 25) // 60:02d}:{(t + 25) % 60:02d}", "max_frames": 6})
        images = [c for c in result["content"] if c.get("type") == "image"]
        for i, image in enumerate(images, 1):
            (out / f"frame_{i:02d}.jpg").write_bytes(base64.b64decode(image["data"]))
        (out / "watch.txt").write_text(text_of(result), encoding="utf-8")
        report["watch_frames"] = len(images)
        report["watch_b64_bytes"] = sum(len(i["data"]) for i in images)
        assert images and args.query.split()[0].lower() in text_of(result).lower()
    finally:
        session.close()
        (out / "url_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("job_id", "elapsed_s", "watch_frames", "watch_b64_bytes")}, indent=2))
    print("V2 URL E2E: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
