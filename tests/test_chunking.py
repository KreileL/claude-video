"""Tests for chunking: silence-aware audio chunk planning."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunking import parse_silences, plan  # noqa: E402

STDERR = """\
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
Input #0, mp3, from 'a.mp3':
[silencedetect @ 0x55d0c8a3f2c0] silence_start: 12.345
[silencedetect @ 0x55d0c8a3f2c0] silence_end: 13.9 | silence_duration: 1.55
size=N/A time=00:00:30.00 bitrate=N/A speed=200x
[silencedetect @ 0x55d0c8a3f2c0] silence_start: 100
[silencedetect @ 0x55d0c8a3f2c0] silence_end: 101.25 | silence_duration: 1.25
[silencedetect @ 0x55d0c8a3f2c0] silence_start: 500.5
"""


def test_parse_silences():
    assert parse_silences(STDERR) == [(12.345, 13.9), (100.0, 101.25)]
    assert parse_silences("") == []


def assert_covers(chunks, duration):
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == round(duration, 3)
    for (_, e), (s, _) in zip(chunks, chunks[1:]):
        assert e == s
    assert all(e > s for s, e in chunks)


def test_no_silences_even_cuts():
    chunks = plan(1300.0, [])
    assert chunks == [(0.0, 300.0), (300.0, 600.0), (600.0, 900.0), (900.0, 1300.0)]
    assert_covers(chunks, 1300.0)


def test_exact_even_cuts_with_merge():
    # remainder 100 < 150 -> merged into previous chunk
    assert plan(1000.0, []) == [(0.0, 300.0), (300.0, 600.0), (600.0, 1000.0)]


def test_silence_near_boundary_chosen():
    chunks = plan(700.0, [(290.0, 292.0), (305.0, 307.0)])
    assert chunks[0] == (0.0, 306.0)  # mid 306 is 6 from 300, mid 291 is 9
    assert_covers(chunks, 700.0)


def test_silence_outside_window_ignored():
    chunks = plan(700.0, [(200.0, 202.0)])
    assert chunks[0] == (0.0, 300.0)


def test_remainder_merge():
    assert plan(740.0, []) == [(0.0, 300.0), (300.0, 740.0)]  # 140 < 150: merged
    assert plan(760.0, []) == [(0.0, 300.0), (300.0, 600.0), (600.0, 760.0)]  # 160 kept


def test_short_audio_single_chunk():
    assert plan(120.0, []) == [(0.0, 120.0)]
    assert plan(300.0, []) == [(0.0, 300.0)]


def test_nonpositive_duration():
    assert plan(0, []) == []
    assert plan(-5, []) == []


def test_contiguous_with_many_silences():
    sil = [(i * 40.0, i * 40.0 + 1.5) for i in range(1, 100)]
    chunks = plan(3601.2345, sil)
    assert_covers(chunks, 3601.2345)
    assert all(e - s >= 150 for s, e in chunks)
