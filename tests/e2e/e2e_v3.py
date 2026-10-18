#!/usr/bin/env python3
"""v3 end-to-end on a THROWAWAY jobs volume (never watch-mcp-jobs).

Starts a test watch-worker-v3 container (same hardening as the compose service)
and drives the v3 stdio server over MCP (the Desktop entry's exact args, only
the jobs volume swapped): analyze_video(url) → job_status until archived →
checks the folder written into --entrada.

    python tests/e2e/e2e_v3.py --volume watch-mcp-v3test-jobs --entrada DIR --key FILE --url URL
    # summary failure then recovery: --key EMPTYFILE first, then
    python tests/e2e/e2e_v3.py --volume ... --entrada DIR --key FILE --job JOB_ID --wait resumo
    # vault mount missing then back: --no-entrada first, then --job JOB_ID --wait export

--key is a path handed to `docker --mount`; this script never opens it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from desktop_entry import INBOX_DIR, WHISPER_URL_V2, entry_v3, secrets_dir  # noqa: E402
from e2e_mcp_stdio import PROTOCOL_VERSION, Session  # noqa: E402

WORKER = "watch-mcp-v3test-worker"
REAL_VOLUME = "watch-mcp-jobs"


def text_of(result: dict) -> str:
    return "\n".join(c["text"] for c in result.get("content") or [] if c.get("type") == "text")


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", check=check)


def start_worker(volume: str, entrada: str | None, key: str) -> None:
    docker("rm", "-f", WORKER, check=False)
    args = ["run", "-d", "--name", WORKER, "--init", "--network", "watch-mcp-net", "--user", "10001:10001",
            "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--cpus", "4", "--memory", "3g", "--pids-limit", "256",
            "--entrypoint", "python",
            "--mount", f"type=volume,src={volume},dst=/jobs",
            "--mount", f"type=bind,src={INBOX_DIR},dst=/inbox,readonly",
            "--mount", f"type=bind,src={secrets_dir().as_posix()},dst=/secrets,readonly",
            "--mount", f"type=bind,src={Path(key).as_posix()},dst=/run/secrets/gemini_api_key,readonly"]
    if entrada:
        args += ["--mount", f"type=bind,src={Path(entrada).as_posix()},dst=/entrada"]
    for k, v in {"WATCH_WHISPER_URL": WHISPER_URL_V2, "WATCH_WHISPER_TIMEOUT": "1800",
                 "WATCH_MODEL_FAST": "Systran/faster-whisper-small",
                 "WATCH_MODEL_ACCURATE": "deepdml/faster-whisper-large-v3-turbo-ct2",
                 "WATCH_RESUMO_PROVIDER": "gemini", "WATCH_GEMINI_MODEL": "gemini-3.8-flash",
                 "WATCH_ENTRADA_DIR": "/entrada", "WATCH_RETRY_EVERY": "10",
                 "WATCH_RESUMO_RETRY_FIRST": "20", "WATCH_EXPORT_RETRY_FIRST": "20"}.items():
        args += ["-e", f"{k}={v}"]
    docker(*args, "watch-mcp:v3", "/app/worker.py")


def job_json(volume: str, job_id: str) -> dict:
    got = docker("run", "--rm", "--network", "none", "--user", "10001:10001", "--entrypoint", "cat",
                 "--mount", f"type=volume,src={volume},dst=/jobs,readonly", "watch-mcp:v3", f"/jobs/{job_id}/job.json")
    return json.loads(got.stdout)


def wait_for(volume: str, job_id: str, done, timeout: float) -> dict:
    start = time.monotonic()
    while True:
        job = job_json(volume, job_id)
        if done(job) or job.get("state") == "failed" or time.monotonic() - start > timeout:
            return job
        time.sleep(10)


def check_folder(entrada: Path, job: dict) -> dict:
    folder = entrada / job["entrada_folder"]
    notes = [p.name for p in folder.glob("*.md")]
    index = folder / job["entrada_index"]
    text = index.read_text(encoding="utf-8")
    resumo = re.search(r"^resumo: (\S+)", text, re.M).group(1)
    frames = sorted(p.name for p in (folder / "frames").iterdir())
    return {"folder": folder.name, "notes": notes, "resumo_in_note": resumo, "frames": len(frames),
            "transcript_lines": sum(1 for line in (folder / "Transcrição.md").read_text(encoding="utf-8").splitlines()
                                    if line.startswith("[")),
            "resumo_md": (folder / "Resumo.md").exists()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--entrada")
    ap.add_argument("--no-entrada", action="store_true")
    ap.add_argument("--key", required=True)
    ap.add_argument("--url")
    ap.add_argument("--job")
    ap.add_argument("--wait", choices=["archived", "resumo", "export", "done"], default="archived")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--keep-worker", action="store_true")
    args = ap.parse_args()
    if args.volume == REAL_VOLUME:
        sys.exit("refusing to run against the real jobs volume")
    entrada = None if args.no_entrada else args.entrada
    start_worker(args.volume, entrada, args.key)
    report: dict = {"volume": args.volume, "entrada": entrada}
    try:
        job_id = args.job
        if args.url:
            desktop = entry_v3(jobs_volume=args.volume)
            session = Session(["docker", *desktop["args"]], {"PATH": os.environ["PATH"], **desktop["env"]})
            try:
                init = session.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                                      "clientInfo": {"name": "watch-e2e-v3", "version": "1.0"}},
                                       timeout=60)
                session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
                report["server"] = init["serverInfo"]
                created = text_of(session.request("tools/call", {"name": "analyze_video",
                                                                 "arguments": {"source": args.url}}, timeout=120))
                job_id = re.search(r"job_id = ([0-9a-f]{32})", created).group(1)
                print("job", job_id, flush=True)
                job = wait_for(args.volume, job_id, lambda j: j.get("state") in ("archived", "done")
                               and j.get("export") in ("ok", "pendente") and j.get("resumo"), args.timeout)
                report["status_tool"] = text_of(session.request(
                    "tools/call", {"name": "job_status", "arguments": {"job_id": job_id}}, timeout=60))
                report["list_jobs"] = text_of(session.request("tools/call", {"name": "list_jobs", "arguments": {}},
                                                              timeout=60))[:600]
            finally:
                session.close()
        conditions = {
            "archived": lambda j: j.get("state") == "archived",
            "done": lambda j: j.get("state") in ("done", "archived") and j.get("export"),
            "resumo": lambda j: j.get("resumo_file") or (j.get("resumo") == "ok" and not j.get("resumo_file_pending")
                                                          and j.get("export") == "ok" and j.get("resumo_file") is not None),
            "export": lambda j: j.get("export") == "ok",
        }
        job = wait_for(args.volume, job_id, conditions[args.wait], args.timeout)
        report["job"] = {k: job.get(k) for k in ("id", "state", "export", "export_error", "resumo", "resumo_error",
                                                 "resumo_calls", "resumo_model", "resumo_file", "entrada_folder",
                                                 "duration", "model", "origin")}
        if entrada and job.get("entrada_folder"):
            report["folder"] = check_folder(Path(entrada), job)
        report["worker_log"] = docker("logs", "--tail", "25", WORKER, check=False).stdout.splitlines()
    finally:
        if not args.keep_worker:
            docker("rm", "-f", WORKER, check=False)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
