"""The Claude Desktop "watch" server entry — single source of truth.

Used by the end-to-end harness (so the protocol test spawns the container with
exactly these arguments) and by install_desktop_config.py.
"""
from __future__ import annotations

import os
from pathlib import Path

IMAGE = "watch-mcp:latest"
WHISPER_URL = "http://watch-whisper:8000/v1/audio/transcriptions"
WHISPER_MODEL = "Systran/faster-whisper-small"
# Claude Desktop starts MCP servers with only PATH plus this block; docker.exe
# needs these to find its config and named pipe.
ENV_KEYS = ("SystemRoot", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "ProgramData")


def secrets_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config" / "watch" / "secrets"


def docker_args(home: Path | None = None, extra_env: dict[str, str] | None = None) -> list[str]:
    # A directory mount, not the file: the config is static, and a missing
    # bind source would stop the server from starting. cookies.txt is optional
    # inside it. Forward slashes: --mount on Windows needs src=C:/...
    secrets = secrets_dir(home).as_posix()
    args = [
        "run", "-i", "--rm", "--init",
        "--network", "watch-mcp-net",
        "-e", f"WATCH_WHISPER_URL={WHISPER_URL}",
        "-e", f"WATCH_WHISPER_MODEL={WHISPER_MODEL}",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "2g", "--pids-limit", "256",
        "--mount", f"type=bind,src={secrets},dst=/secrets,readonly",
    ]
    for key, value in (extra_env or {}).items():
        args += ["-e", f"{key}={value}"]
    return args + [IMAGE]


def entry(home: Path | None = None, environ: dict[str, str] | None = None) -> dict:
    environ = dict(os.environ if environ is None else environ)
    return {
        "command": "docker",
        "args": docker_args(home),
        "env": {k: environ[k] for k in ENV_KEYS if k in environ},
    }
