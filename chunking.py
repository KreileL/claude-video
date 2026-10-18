"""Split long audio into transcription chunks at silences."""
from __future__ import annotations

import re

_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silences(stderr: str) -> list[tuple[float, float]]:
    """Parse ffmpeg silencedetect output into (start, end) pairs; unmatched start dropped."""
    out: list[tuple[float, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        if "silencedetect" not in line:
            continue
        m = _START_RE.search(line)
        if m:
            start = float(m.group(1))
            continue
        m = _END_RE.search(line)
        if m and start is not None:
            out.append((start, float(m.group(1))))
            start = None
    return out


def plan(
    duration: float,
    silences: list[tuple[float, float]],
    target: float = 300.0,
    window: float = 30.0,
) -> list[tuple[float, float]]:
    """Contiguous (start, end) chunks covering [0, duration], cutting at silences."""
    if duration <= 0:
        return []
    if duration <= target:
        return [(0.0, round(duration, 3))]
    mids = [(s + e) / 2 for s, e in silences]
    cuts: list[float] = []
    prev = 0.0
    k = 1
    while k * target < duration:
        ideal = k * target
        cands = [m for m in mids if abs(m - ideal) <= window and m > prev + target / 2]
        cut = min(cands, key=lambda m: abs(m - ideal)) if cands else ideal
        cuts.append(cut)
        prev = cut
        k += 1
    if cuts and duration - cuts[-1] < target / 2:
        cuts.pop()
    bounds = [0.0, *cuts, duration]
    return [(round(a, 3), round(b, 3)) for a, b in zip(bounds, bounds[1:])]
