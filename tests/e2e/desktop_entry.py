"""The Claude Desktop "watch" server entries (v1, v2, v3) — single source of truth.

Used by the end-to-end harness (so the protocol test spawns the container with
exactly these arguments) and by install_desktop_config.py.
"""
from __future__ import annotations

import os
from pathlib import Path

IMAGE = "watch-mcp:latest"
WHISPER_URL = "http://watch-whisper:8000/v1/audio/transcriptions"
WHISPER_MODEL = "Systran/faster-whisper-small"
# v2: new image, the background-job cache volume and the inbox (read-only).
IMAGE_V2 = "watch-mcp:v2"
# v2 has its own Whisper service (docker-compose.yml, watch-whisper-v2).
WHISPER_URL_V2 = "http://watch-whisper-v2:8000/v1/audio/transcriptions"
WHISPER_MODEL_V2 = "Systran/faster-whisper-small"
JOBS_VOLUME = "watch-mcp-jobs"
# v3: same server mounts as v2 (the vault is mounted only into watch-worker-v3).
IMAGE_V3 = "watch-mcp:v3"
INBOX_DIR = "D:/Inbox"
# Claude Desktop starts MCP servers with only PATH plus this block; docker.exe
# needs these to find its config and named pipe.
ENV_KEYS = ("SystemRoot", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "ProgramData")


def secrets_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config" / "watch" / "secrets"


def _args(home: Path | None, extra_env: dict[str, str] | None, *, image: str, model: str,
          mounts: list[str], whisper_url: str = WHISPER_URL) -> list[str]:
    # A directory mount, not the file: the config is static, and a missing
    # bind source would stop the server from starting. cookies.txt is optional
    # inside it. Forward slashes: --mount on Windows needs src=C:/...
    secrets = secrets_dir(home).as_posix()
    args = [
        "run", "-i", "--rm", "--init",
        "--network", "watch-mcp-net",
        "-e", f"WATCH_WHISPER_URL={whisper_url}",
        "-e", f"WATCH_WHISPER_MODEL={model}",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "2g", "--pids-limit", "256", "--cpus", "4",
        "--mount", f"type=bind,src={secrets},dst=/secrets,readonly",
    ]
    for mount in mounts:
        args += ["--mount", mount]
    for key, value in (extra_env or {}).items():
        args += ["-e", f"{key}={value}"]
    return args + [image]


def docker_args(home: Path | None = None, extra_env: dict[str, str] | None = None) -> list[str]:
    return _args(home, extra_env, image=IMAGE, model=WHISPER_MODEL, mounts=[])


def docker_args_v2(home: Path | None = None, extra_env: dict[str, str] | None = None) -> list[str]:
    # The server only enqueues and reads jobs; the watch-worker service does the work.
    return _args(home, extra_env, image=IMAGE_V2, model=WHISPER_MODEL_V2, whisper_url=WHISPER_URL_V2, mounts=[
        f"type=volume,src={JOBS_VOLUME},dst=/jobs",
        f"type=bind,src={INBOX_DIR},dst=/inbox,readonly",
    ])


def docker_args_v3(home: Path | None = None, extra_env: dict[str, str] | None = None, *,
                   jobs_volume: str = JOBS_VOLUME) -> list[str]:
    # jobs_volume: end-to-end tests point the server at a throwaway volume.
    return _args(home, extra_env, image=IMAGE_V3, model=WHISPER_MODEL_V2, whisper_url=WHISPER_URL_V2, mounts=[
        f"type=volume,src={jobs_volume},dst=/jobs",
        f"type=bind,src={INBOX_DIR},dst=/inbox,readonly",
    ])


def _env(environ: dict[str, str] | None) -> dict[str, str]:
    # Windows env names are case-insensitive and os.environ upper-cases them
    # (SYSTEMROOT), so match without case and write the conventional spelling.
    upper = {k.upper(): v for k, v in (os.environ if environ is None else environ).items()}
    return {k: upper[k.upper()] for k in ENV_KEYS if k.upper() in upper}


def entry(home: Path | None = None, environ: dict[str, str] | None = None) -> dict:
    return {"command": "docker", "args": docker_args(home), "env": _env(environ)}


def entry_v2(home: Path | None = None, environ: dict[str, str] | None = None) -> dict:
    return {"command": "docker", "args": docker_args_v2(home), "env": _env(environ)}


def entry_v3(home: Path | None = None, environ: dict[str, str] | None = None, *,
             jobs_volume: str = JOBS_VOLUME) -> dict:
    return {"command": "docker", "args": docker_args_v3(home, jobs_volume=jobs_volume), "env": _env(environ)}
