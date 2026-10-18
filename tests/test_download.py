"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. We only support English, so the request must stay
bounded to the English-only pattern.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub subprocess.run inside download.py and record every argv."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _assert_original_and_english(langs: str) -> None:
    """Guard the subtitle selection: the original-language track first, English
    as the fallback, and never every language on the platform.

    The breadth guard is the point — `all` (or a bare `.*`) pulls down a
    subtitle file per language, dozens on a popular video. The original-language
    requirement is what keeps a non-English video from being read through
    YouTube's machine translation when the real audio is right there.
    """
    tokens = [t.strip() for t in langs.split(",")]
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert ".*" not in tokens, f"a bare '.*' is every language, got {langs!r}"
    assert any(t.endswith("-orig") for t in tokens), (
        f"sub-langs must request the original-language track, got {langs!r}"
    )
    assert any(t.startswith("en") for t in tokens), (
        f"sub-langs must keep English as a fallback, got {langs!r}"
    )


def test_fetch_captions_requests_original_and_english(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_original_and_english(_sub_langs(calls[0]))


def test_download_url_requests_original_and_english(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_original_and_english(_sub_langs(calls[0]))


def test_pick_subtitle_matches_video_language_among_dubbed_orig_tracks(tmp_path):
    # AI-dubbed videos expose one *-orig track per dubbed language; sort order
    # would pick en-US-orig. The video's reported language must win.
    for name in ("video.en-US-orig.vtt", "video.es-US-orig.vtt", "video.pt-orig.vtt", "video.en.vtt"):
        (tmp_path / name).write_text("WEBVTT\n", encoding="utf-8")
    (tmp_path / "video.info.json").write_text('{"language": "pt-BR"}', encoding="utf-8")
    assert download._pick_subtitle(tmp_path).name == "video.pt-orig.vtt"


def test_pick_subtitle_without_language_keeps_first_orig(tmp_path):
    for name in ("video.en-US-orig.vtt", "video.pt-orig.vtt"):
        (tmp_path / name).write_text("WEBVTT\n", encoding="utf-8")
    assert download._pick_subtitle(tmp_path).name == "video.en-US-orig.vtt"


# --- operator-set env hooks (WATCH_YTDLP_ARGS / WATCH_YTDLP_TIMEOUT) ---------


def _capture_run(monkeypatch: pytest.MonkeyPatch, raises: Exception | None = None):
    """Like _capture_argv but records (argv, kwargs) and can raise from run()."""
    calls: list[tuple[list[str], dict]] = []

    class _Result:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        calls.append((list(cmd), kwargs))
        if raises is not None:
            raise raises
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    return calls


def _run_both(tmp_path):
    """Run fetch_captions and download_url; return nothing (stub records)."""
    download.fetch_captions(URL, tmp_path / "a")
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "b")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("WATCH_YTDLP_ARGS", raising=False)
    monkeypatch.delenv("WATCH_YTDLP_TIMEOUT", raising=False)


def test_extra_args_injected_after_ytdlp_before_end_of_options(monkeypatch, tmp_path):
    calls = _capture_run(monkeypatch)
    monkeypatch.setenv("WATCH_YTDLP_ARGS", '["--proxy", "http://127.0.0.1:8899", "--max-filesize", "500M"]')
    _run_both(tmp_path)
    assert len(calls) == 2
    for argv, _ in calls:
        assert argv[:5] == ["yt-dlp", "--proxy", "http://127.0.0.1:8899", "--max-filesize", "500M"]
        assert argv.index("--proxy") < argv.index("--")
        assert argv[-2:] == ["--", URL]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_args_leave_argv_unchanged(monkeypatch, tmp_path, value):
    baseline = _capture_run(monkeypatch)
    _run_both(tmp_path)
    expected = [argv for argv, _ in baseline]
    if value is not None:
        monkeypatch.setenv("WATCH_YTDLP_ARGS", value)
    again = _capture_run(monkeypatch)
    _run_both(tmp_path)
    assert [argv for argv, _ in again] == expected


@pytest.mark.parametrize("value", ["--proxy secret-cookies.txt", '{"a": "secret-cookies.txt"}', '["--proxy", 5, "secret-cookies.txt"]'])
def test_invalid_extra_args_exit_with_fixed_message(monkeypatch, tmp_path, value):
    _capture_run(monkeypatch)
    monkeypatch.setenv("WATCH_YTDLP_ARGS", value)
    for fn in (download.fetch_captions, download.download_url):
        with pytest.raises(SystemExit) as exc:
            fn(URL, tmp_path / "x")
        assert str(exc.value) == "WATCH_YTDLP_ARGS must be a JSON array of strings"
        assert "secret-cookies" not in str(exc.value)


def test_timeout_passed_through(monkeypatch, tmp_path):
    calls = _capture_run(monkeypatch)
    monkeypatch.setenv("WATCH_YTDLP_TIMEOUT", "120")
    _run_both(tmp_path)
    assert [kw.get("timeout") for _, kw in calls] == [120.0, 120.0]


@pytest.mark.parametrize("value", [None, "abc", "0", "-5"])
def test_timeout_absent_or_invalid_is_none(monkeypatch, tmp_path, value):
    calls = _capture_run(monkeypatch)
    if value is not None:
        monkeypatch.setenv("WATCH_YTDLP_TIMEOUT", value)
    _run_both(tmp_path)
    assert [kw.get("timeout") for _, kw in calls] == [None, None]


def test_invalid_timeout_warns_on_stderr(monkeypatch, tmp_path, capsys):
    _capture_run(monkeypatch)
    monkeypatch.setenv("WATCH_YTDLP_TIMEOUT", "abc")
    download.fetch_captions(URL, tmp_path / "a")
    assert "WATCH_YTDLP_TIMEOUT" in capsys.readouterr().err


def test_timeout_expired_becomes_systemexit(monkeypatch, tmp_path):
    _capture_run(monkeypatch, raises=subprocess.TimeoutExpired("yt-dlp", 5))
    monkeypatch.setenv("WATCH_YTDLP_TIMEOUT", "5")
    for fn in (download.fetch_captions, download.download_url):
        with pytest.raises(SystemExit) as exc:
            fn(URL, tmp_path / "x")
        assert "timed out after 5s" in str(exc.value)


def test_audio_only_download_falls_back_to_a_single_file_format(monkeypatch, tmp_path):
    # A direct .mp4 has no separate audio stream; "ba" alone fails with
    # "Requested format is not available" (found by the v2 counter-review).
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download", audio_only=True)
    argv = calls[0]
    assert argv[argv.index("-f") + 1] == "ba/b"
