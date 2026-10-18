"""The Claude Desktop entries: v1 ("watch") is frozen, v2 ("watch-v2") adds its mounts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))

import desktop_entry  # noqa: E402

HOME = Path("C:/Users/someone")


def test_v1_entry_is_exactly_what_production_runs():
    assert desktop_entry.docker_args(HOME) == [
        "run", "-i", "--rm", "--init", "--network", "watch-mcp-net",
        "-e", "WATCH_WHISPER_URL=http://watch-whisper:8000/v1/audio/transcriptions",
        "-e", "WATCH_WHISPER_MODEL=Systran/faster-whisper-small",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "2g", "--pids-limit", "256", "--cpus", "4",
        "--mount", "type=bind,src=C:/Users/someone/.config/watch/secrets,dst=/secrets,readonly",
        "watch-mcp:latest",
    ]


def test_v2_entry_uses_its_image_whisper_and_mounts():
    args = desktop_entry.docker_args_v2(HOME)
    assert args[-1] == "watch-mcp:v2"
    assert "WATCH_WHISPER_URL=http://watch-whisper-v2:8000/v1/audio/transcriptions" in args
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "--mount"]
    assert mounts == [
        "type=bind,src=C:/Users/someone/.config/watch/secrets,dst=/secrets,readonly",
        "type=volume,src=watch-mcp-jobs,dst=/jobs",
        "type=bind,src=D:/Inbox,dst=/inbox,readonly",
    ]
    for flag in ("--read-only", "--init", "no-new-privileges", "ALL"):
        assert flag in args


def test_env_block_matches_names_case_insensitively():
    env = desktop_entry.entry_v2(HOME, {"SYSTEMROOT": "C:\\Windows", "USERPROFILE": "x", "PATH": "p"})["env"]
    assert env == {"SystemRoot": "C:\\Windows", "USERPROFILE": "x"}
