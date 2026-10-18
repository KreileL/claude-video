#!/usr/bin/env python3
"""Offline end-to-end test: synthetic pt-BR video served inside watch-mcp-net.

1. Speech: POST /v1/audio/speech on watch-whisper (Piper pt-BR), from a
   throwaway watch-mcp container on the network.
2. Video (host ffmpeg): testsrc + "CENA n" text and a colored box that change
   every 3 s, plus the speech; mp4 with no subtitle track.
3. Serve it from a throwaway container (watch-mcp-testsrv) on watch-mcp-net.
4. Call watch_video over real MCP stdio (e2e_mcp_stdio.py) with the test-mode
   allow-list naming only that host.
5. Check: transcript contains the phrase, frames show the different scenes
   (box color per frame matches the scene for its timestamp), timestamps
   ordered and within the clip.

All Docker steps run sequentially. The serving container is removed at the end.

    python tests/e2e/e2e_offline.py --work DIR
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = "watch-mcp-testsrv"
NETWORK = "watch-mcp-net"
IMAGE = "watch-mcp:latest"
TTS_MODEL = "speaches-ai/piper-pt_BR-faber-medium"
PHRASE = "Olá! Este é um teste de ponta a ponta. O gato azul dorme no sofá da sala."
EXPECTED_WORDS = ("teste", "ponta", "gato", "azul", "sofa", "sala")
SEGMENT = 3
SCENES = [("CENA 1", "red", (255, 0, 0)), ("CENA 2", "lime", (0, 255, 0)),
          ("CENA 3", "blue", (0, 0, 255)), ("CENA 4", "yellow", (255, 255, 0))]
DURATION = SEGMENT * len(SCENES)
FONT = "C\\:/Windows/Fonts/arial.ttf"


def sh(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(argv[:6]), "…" if len(argv) > 6 else "", flush=True)
    return subprocess.run(argv, check=True, **kwargs)


def make_speech(out: Path) -> None:
    code = (
        "import json,sys,urllib.request;"
        f"body=json.dumps({{'model':{TTS_MODEL!r},'voice':'faber','input':{PHRASE!r},'response_format':'wav'}}).encode();"
        "r=urllib.request.Request('http://watch-whisper:8000/v1/audio/speech',data=body,headers={'Content-Type':'application/json'});"
        "sys.stdout.buffer.write(urllib.request.urlopen(r,timeout=120).read())"
    )
    wav = sh(["docker", "run", "--rm", "--network", NETWORK, "--read-only", "--cap-drop", "ALL",
              "--entrypoint", "python", IMAGE, "-c", code], capture_output=True).stdout
    assert wav[:4] == b"RIFF", "TTS did not return a WAV"
    out.write_bytes(wav)


def make_video(speech: Path, out: Path) -> None:
    filters = []
    for i, (label, color, _) in enumerate(SCENES):
        a, b = i * SEGMENT, (i + 1) * SEGMENT
        filters.append(f"drawbox=x=0:y=0:w=200:h=120:color={color}:t=fill:enable='between(t,{a},{b})'")
        filters.append(
            f"drawtext=fontfile='{FONT}':text='{label}':fontsize=72:fontcolor=white:box=1:boxcolor=black"
            f":x=(w-text_w)/2:y=h-text_h-40:enable='between(t,{a},{b})'"
        )
    sh([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=25:duration={DURATION}",
        "-i", str(speech),
        "-filter_complex", f"[0:v]{','.join(filters)}[v];[1:a]apad[a]",
        "-map", "[v]", "-map", "[a]", "-t", str(DURATION), "-sn",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
    ])


def box_color(frame: Path) -> tuple[int, int, int]:
    """Mean RGB of the top-left box region (frames are scaled, so use ratios)."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(frame),
         "-vf", "crop=iw*200/640*0.8:ih*120/360*0.8:iw*0.02:ih*0.03,scale=1:1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return raw[0], raw[1], raw[2]


def nearest_scene(rgb: tuple[int, int, int]) -> int:
    return min(range(len(SCENES)), key=lambda i: sum((a - b) ** 2 for a, b in zip(rgb, SCENES[i][2])))


def plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    site = work / "site"
    site.mkdir(parents=True, exist_ok=True)

    make_speech(work / "speech.wav")
    make_video(work / "speech.wav", site / "teste.mp4")
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(site / "teste.mp4")], capture_output=True, text=True).stdout.split()
    assert "subtitle" not in streams and "audio" in streams, streams

    subprocess.run(["docker", "rm", "-f", SERVER], capture_output=True)
    sh(["docker", "run", "-d", "--rm", "--name", SERVER, "--network", NETWORK, "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--mount", f"type=bind,src={site.resolve().as_posix()},dst=/srv,readonly",
        "--entrypoint", "python", IMAGE, "-m", "http.server", "8080", "-d", "/srv"], capture_output=True)
    try:
        time.sleep(2)
        out = work / "mcp"
        sh([sys.executable, str(HERE / "e2e_mcp_stdio.py"), "--url", f"http://{SERVER}:8080/teste.mp4",
            "--allow-host", SERVER, "--question", "Quais textos aparecem na tela?", "--out", str(out)])
    finally:
        subprocess.run(["docker", "rm", "-f", SERVER], capture_output=True)

    text = (out / "result.txt").read_text(encoding="utf-8")
    transcript = text.split("## Transcrição", 1)[1].split("## Frames", 1)[0]
    missing = [w for w in EXPECTED_WORDS if w not in plain(transcript)]
    stamps = re.findall(r"^\d+\. t=(\d+):(\d+)$", text, flags=re.M)
    seconds = [int(m) * 60 + int(s) for m, s in stamps]
    frames = sorted(out.glob("frame_*.jpg"))
    checks = []
    for t, frame in zip(seconds, frames):
        rgb = box_color(frame)
        got = nearest_scene(rgb)
        # t is rounded to whole seconds, so a frame on a boundary may be either scene.
        ok_scenes = {min(t // SEGMENT, len(SCENES) - 1)}
        if t % SEGMENT == 0 and t:
            ok_scenes.add(t // SEGMENT - 1)
        checks.append({"t": t, "rgb": rgb, "scene": SCENES[got][0], "ok": got in ok_scenes})

    report = {
        "transcript_missing_words": missing,
        "frames": len(frames),
        "timestamps": seconds,
        "distinct_scenes": sorted({c["scene"] for c in checks}),
        "frame_checks": checks,
    }
    (work / "offline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert not missing, f"transcript missing {missing}:\n{transcript}"
    assert len(frames) == len(seconds) and frames, "frame list and images differ"
    assert seconds == sorted(seconds) and all(0 <= t <= DURATION for t in seconds), seconds
    assert len(report["distinct_scenes"]) >= 3, report["distinct_scenes"]
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
    print("OFFLINE E2E: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
