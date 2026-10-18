"""Summarize the pt-BR benchmark: one row per model and prompt mode.

    python tests/bench/report.py            # reads tests/bench/data/, prints Markdown

Hypotheses are re-scored here against the current references, so the
approximate references (refs/) can be revised without re-running Whisper.
WER is aggregated per group (total edits / total reference words):
TTS (exact references), real-official (manual captions), real-approx
(reviewed large-v3 drafts). Speed is seconds of compute per minute of audio,
warm (model already loaded). RAM is the peak of watch-whisper while the model
ran (docker stats; includes the process baseline).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402
from run_bench import clips  # noqa: E402

DATA = HERE / "data"
GROUPS = ("tts", "real-official", "real-approx")
SIZE_RE = re.compile(r"([\d.]+)\s*([KMG]i?B)")
UNITS = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "KB": 1 / 1000, "MB": 1, "GB": 1000}


def mem_samples() -> list[tuple[float, float]]:
    out = []
    path = DATA / "mem.log"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(" ", 1)
        m = SIZE_RE.search(parts[1]) if len(parts) == 2 else None
        if m and parts[0].isdigit():
            out.append((float(parts[0]), float(m.group(1)) * UNITS[m.group(2)]))
    return out


def windows() -> dict[str, tuple[float, float]]:
    spans: dict[str, list[float]] = {}
    path = DATA / "windows.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            spans.setdefault(event["model"], []).append(event["t"])
    return {model: (min(ts), max(ts)) for model, ts in spans.items() if len(ts) >= 2}


def summarize() -> list[dict]:
    refs = {c["id"]: c for c in clips()}
    samples, spans = mem_samples(), windows()
    rows = []
    for path in sorted((DATA / "results").glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        model = records[0]["model"]
        span = spans.get(model)
        peak = max((mb for t, mb in samples if span and span[0] <= t <= span[1]), default=None)
        for mode in ("none", "ptbr", "ptbr+terms"):
            mine = [r for r in records if r["prompt_mode"] == mode]
            if not mine:
                continue
            row: dict = {"model": model, "prompt": mode, "ram_peak_mb": peak}
            for group in GROUPS:
                edits = words = 0
                for r in mine:
                    clip = refs[r["clip"]]
                    if clip["kind"] != group or not clip["ref"]:
                        continue
                    d = metrics.wer_details(clip["ref"], r["hyp"])
                    edits += d["substitutions"] + d["deletions"] + d["insertions"]
                    words += d["ref_words"]
                row[f"wer_{group}"] = edits / words if words else None
            for key in ("numbers", "names", "terms"):
                total = missed = 0
                for r in mine:
                    clip = refs[r["clip"]]
                    if clip["kind"] != "tts" or not clip[key]:
                        continue
                    total += len(clip[key])
                    missed += len(metrics.missing_terms(clip[key], r["hyp"]))
                row[f"{key}_missed"] = f"{missed}/{total}" if total else "-"
            seconds = sum(r["seconds"] for r in mine)
            minutes = sum(r["audio_seconds"] for r in mine) / 60
            row["s_per_audio_min"] = seconds / minutes if minutes else None
            row["warmup_s"] = mine[0]["warmup_seconds"]
            rows.append(row)
    return rows


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def markdown(rows: list[dict]) -> str:
    head = ("| Modelo | Prompt | WER TTS | WER real (legenda oficial) | WER real (ref. aproximada) | "
            "Números perdidos (TTS) | Nomes perdidos (TTS) | Termos perdidos (TTS) | s por min de áudio | "
            "RAM pico (MiB) |")
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        ram = f"{r['ram_peak_mb']:.0f}" if r["ram_peak_mb"] else "-"
        speed = f"{r['s_per_audio_min']:.1f}" if r["s_per_audio_min"] else "-"
        lines.append(
            f"| {r['model']} | {r['prompt']} | {pct(r['wer_tts'])} | {pct(r['wer_real-official'])} | "
            f"{pct(r['wer_real-approx'])} | {r['numbers_missed']} | {r['names_missed']} | {r['terms_missed']} | "
            f"{speed} | {ram} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = summarize()
    (DATA / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(markdown(rows))
