"""pt-BR Whisper benchmark runner. Runs in a container on watch-mcp-net, next
to watch-whisper (see tests/bench/README.md):

    python /bench/run_bench.py tts            # synthesize the TTS corpus (once)
    python /bench/run_bench.py run <model>... # transcribe every clip, all prompt modes
    python /bench/run_bench.py refs           # large-v3 drafts for approximate references

Writes data/results/<model>.json (one record per clip and prompt mode) and
data/windows.jsonl (when each model ran, to match `docker stats` samples).
Every request uses language="pt".
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "skills" / "watch" / "scripts"))

import metrics  # noqa: E402
import whisper  # noqa: E402

DATA = HERE / "data"
TTS_DIR = DATA / "tts"
REAL_DIR = DATA / "real"
RESULTS = DATA / "results"
BASE = "http://watch-whisper:8000"
# Kokoro's phonemizer dlopens a copy of libespeak-ng from /tmp, which the
# hardened watch-whisper (noexec tmpfs) refuses; synthesis can point at a
# throwaway speaches container instead (see README.md).
TTS_BASE = os.environ.get("WATCH_BENCH_TTS", BASE)
# The prompt under test: Brazilian spelling and vocabulary, plus the clip's terms.
PROMPT_PTBR = (
    "Transcrição em português do Brasil, com pontuação e grafia brasileira: "
    "ônibus, celular, time, trem, registro, boleto, aluguel, a gente, tô, cê."
)
REAL_TERMS = {
    "reel-DeniXRTs5Mw": ["Project Zomboid", "Build 42", "zumbis"],
    "nerdologia-kyGu9lTr_jM": ["Nerdologia", "Atila", "Hubble", "Efeito Borboleta"],
    "podpah-gwAwKfW-Q8w": ["Podpah", "UFC", "MMA"],
}


def corpus() -> dict:
    return json.loads((HERE / "corpus_ptbr.json").read_text(encoding="utf-8"))


def voice_items(data: dict) -> list[tuple[dict, list[dict]]]:
    """Round-robin: each voice reads every len(voices)-th sentence."""
    voices = data["voices"]
    return [(v, data["sentences"][i::len(voices)]) for i, v in enumerate(voices)]


def synthesize() -> None:
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    for voice, items in voice_items(corpus()):
        out = TTS_DIR / f"{voice['id']}.mp3"
        if out.exists():
            continue
        body = json.dumps({
            "model": voice["model"], "voice": voice["voice"], "response_format": "wav",
            "input": " ".join(item["text"] for item in items),
        }).encode()
        req = urllib.request.Request(f"{TTS_BASE}/v1/audio/speech", data=body,
                                     headers={"Content-Type": "application/json"})
        wav = TTS_DIR / f"{voice['id']}.wav"
        with urllib.request.urlopen(req, timeout=600) as response:
            wav.write_bytes(response.read())
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                        "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out)], check=True)
        wav.unlink()
        print(f"tts {voice['id']}: {duration(out):.1f}s", flush=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out)


def clips() -> list[dict]:
    """Every clip with its reference and key items (TTS: exact; real: see kind)."""
    out = []
    for voice, items in voice_items(corpus()):
        out.append({
            "id": f"tts-{voice['id']}", "audio": TTS_DIR / f"{voice['id']}.mp3", "kind": "tts",
            "ref": " ".join(i["text"] for i in items),
            "numbers": [n for i in items for n in i["numbers"]],
            "names": [n for i in items for n in i["names"]],
            "terms": [t for i in items for t in i["terms"]],
        })
    for clip_id, terms in REAL_TERMS.items():
        official = REAL_DIR / f"{clip_id}.ref.txt"
        approx = HERE / "refs" / f"{clip_id}.txt"
        ref_path = official if official.exists() else approx
        out.append({
            "id": clip_id, "audio": REAL_DIR / f"{clip_id}.mp3",
            "kind": "real-official" if official.exists() else "real-approx",
            "ref": ref_path.read_text(encoding="utf-8") if ref_path.exists() else None,
            "numbers": [], "names": terms, "terms": [],
        })
    return out


def prompts(clip: dict) -> dict[str, str | None]:
    terms = sorted(set(clip["names"] + clip["terms"]))
    modes = {"none": None, "ptbr": PROMPT_PTBR}
    if terms:
        modes["ptbr+terms"] = f"{PROMPT_PTBR} Termos: {', '.join(terms)}."
    return modes


def unload_all(limit: float = 300.0) -> None:
    """Wait until speaches' idle TTL (120 s) has unloaded every model.

    Never DELETE /api/ps/{model}: in speaches 0.8.3 an explicit unload while
    the TTL offload is scheduled deadlocked the model manager (every later
    transcription hung until the container was restarted; NOTES-mcp.md).
    """
    end = time.monotonic() + limit
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{BASE}/api/ps", timeout=30) as response:
                if not json.load(response).get("models"):
                    return
        except Exception:
            pass
        time.sleep(10)


def transcribe(audio: Path, model: str, prompt: str | None, pieces: float | None = None) -> tuple[str, float]:
    """language="pt". With `pieces`, the audio is sent as consecutive pieces of
    that many seconds, each a fresh request (distil models degrade past ~30 s
    of conditioned decoding); the time includes every piece."""
    started = time.monotonic()
    if not pieces:
        segments = whisper.transcribe_local(audio, model=model, language="pt", prompt=prompt)
        return " ".join(s["text"] for s in segments), time.monotonic() - started
    folder = Path("/tmp") / f"pieces-{audio.stem}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio), "-f", "segment",
                    "-segment_time", f"{pieces:g}", "-c", "copy", str(folder / "p%03d.mp3")], check=True)
    texts = []
    for piece in sorted(folder.glob("p*.mp3")):
        segments = whisper.transcribe_local(piece, model=model, language="pt", prompt=prompt)
        texts.append(" ".join(s["text"] for s in segments))
    return " ".join(texts), time.monotonic() - started


def score(clip: dict, hyp: str) -> dict:
    result: dict = {}
    if clip["ref"]:
        result.update(metrics.wer_details(clip["ref"], hyp))
    for key in ("numbers", "names", "terms"):
        if clip[key]:
            missing = metrics.missing_terms(clip[key], hyp)
            result[f"{key}_missing"] = missing
            result[f"{key}_total"] = len(clip[key])
    return result


def log_window(event: str, model: str) -> None:
    with open(DATA / "windows.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "event": event, "model": model}) + "\n")


def run(models: list[str], pieces: float | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for model in models:
        suffix = f"__p{pieces:g}" if pieces else ""
        out = RESULTS / f"{model.replace('/', '__')}{suffix}.json"
        records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        done = {(r["clip"], r["prompt_mode"]) for r in records}
        unload_all()
        label = model + (f" (pedaços de {pieces:g} s)" if pieces else "")
        log_window("start", label)
        # Warm-up loads the model; its time is reported separately.
        _hyp, load_seconds = transcribe(TTS_DIR / "piper-faber.mp3", model, None)
        for clip in clips():
            for mode, prompt in prompts(clip).items():
                if (clip["id"], mode) in done:
                    continue
                hyp, seconds = transcribe(clip["audio"], model, prompt, pieces)
                audio_seconds = duration(clip["audio"])
                records.append({
                    "model": label,
                    "clip": clip["id"], "kind": clip["kind"], "prompt_mode": mode,
                    "audio_seconds": audio_seconds, "seconds": round(seconds, 2),
                    "seconds_per_audio_minute": round(seconds / audio_seconds * 60, 2),
                    "warmup_seconds": round(load_seconds, 2), "hyp": hyp, **score(clip, hyp),
                })
                out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{model} {clip['id']} {mode}: wer={records[-1].get('wer')} {seconds:.0f}s", flush=True)
        log_window("end", label)
        unload_all()


def drafts() -> None:
    """large-v3 (not a candidate) drafts the approximate references, reviewed by hand."""
    out_dir = DATA / "drafts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips():
        if clip["kind"] != "real-approx":
            continue
        # 30 s pieces, each a fresh request: a repetition loop (large-v3 looped
        # on the podcast's laughter) cannot run past its own piece.
        pieces = Path("/tmp") / clip["id"]
        pieces.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip["audio"]),
                        "-f", "segment", "-segment_time", "30", "-c", "copy", str(pieces / "p%03d.mp3")],
                       check=True)
        texts = [transcribe(p, "Systran/faster-whisper-large-v3", None)[0] for p in sorted(pieces.glob("p*.mp3"))]
        (out_dir / f"{clip['id']}.txt").write_text(" ".join(texts) + "\n", encoding="utf-8")
        print(f"draft {clip['id']}", flush=True)
    unload_all()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "tts":
        synthesize()
    elif command == "run":
        run(sys.argv[2:])
    elif command == "run-pieces":  # run-pieces <seconds> <model>...
        run(sys.argv[3:], pieces=float(sys.argv[2]))
    elif command == "refs":
        drafts()
    else:
        raise SystemExit(__doc__)
