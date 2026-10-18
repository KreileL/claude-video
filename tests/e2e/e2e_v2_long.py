#!/usr/bin/env python3
"""v2 end-to-end: a long synthetic inbox video through the real MCP stdio server.

1. Speech: one pt-BR TTS block per minute (Piper on watch-whisper, from a
   throwaway watch-mcp:v2 container), "Bloco do minuto N. <frase única>",
   starting 5 s into each minute; silence in between.
2. Video (host ffmpeg): MINUTES minutes, "MINUTO N" + clock on screen and a
   background colour that changes every minute; .mkv.
3. The file goes into D:\\Inbox under a clearly test-only name (removed at the end).
4. Over MCP stdio with exactly the "watch-v2" Desktop entry: list_inbox →
   analyze_video → job_status until done → search_transcript finds the sentence
   said at minute 25 → watch_video(job_id, range) shows minute-25 frames.
5. --restart-at-chunk K: restarts watch-mcp-worker once chunk K is being
   transcribed, then checks the chunks finished before were not redone.

    python tests/e2e/e2e_v2_long.py --work DIR [--minutes 35] [--restart-at-chunk 3]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))
from desktop_entry import INBOX_DIR, entry_v2  # noqa: E402
from e2e_mcp_stdio import PROTOCOL_VERSION, Session  # noqa: E402
from metrics import number_to_words  # noqa: E402

IMAGE = "watch-mcp:v2"
NETWORK = "watch-mcp-net"
TTS_MODEL = "speaches-ai/piper-pt_BR-faber-medium"
TEST_NAME = "watch-mcp-e2e-longo.mkv"
TARGET_MINUTE = 25
TARGET_SENTENCE = "A colheitadeira amarela chegou à fazenda de Uberlândia."
QUERY = "colheitadeira amarela"
SENTENCES = [
    "O café da manhã tinha pão de queijo e suco de laranja.", "A chuva começou forte no fim da tarde.",
    "O ônibus atrasou por causa do trânsito na avenida.", "A professora explicou a matéria com calma.",
    "O time treinou pênaltis durante uma hora.", "A feira de domingo vende frutas e verduras frescas.",
    "O servidor da empresa foi atualizado de madrugada.", "Os alunos visitaram o museu da cidade.",
    "A padaria da esquina abre às seis horas.", "O celular ficou sem bateria no meio da viagem.",
    "A plantação de milho cresceu depois da chuva.", "O cachorro do vizinho latiu a noite inteira.",
    "A reunião foi remarcada para a semana que vem.", "O trem chegou na estação antes do horário.",
    "A biblioteca ganhou novos livros de ciência.", "O mercado estava cheio no sábado de manhã.",
    "A obra da ponte deve terminar em dezembro.", "O médico recomendou caminhar todos os dias.",
    "A festa junina teve quentão e pipoca.", "O avião decolou com uma hora de atraso.",
    "A loja de roupas fez uma promoção de inverno.", "O jardim da praça foi replantado.",
    "A internet caiu durante a videochamada.", "O pescador voltou com o barco cheio.",
    "O registro da obra foi feito no cartório.", "A orquestra tocou no teatro municipal.",
    "O motorista parou para abastecer o carro.", "A escola organizou uma gincana de ciências.",
    "O sorvete derreteu rápido no calor.", "A ciclovia nova liga o centro ao parque.",
    "O relógio da igreja marcou meio-dia.", "A vizinha plantou tomates no quintal.",
    "O jogo terminou empatado sem gols.", "A ponte de madeira foi trocada por concreto.",
    "O filme estreou no cinema do shopping.", "A colheita de café começou em maio.",
    "O telhado foi consertado antes das chuvas.", "A estrada de terra virou asfalto.",
    "O piloto testou o drone no campo.", "A sorveteria fechou para reforma.",
]
PALETTE = [("red", (255, 0, 0)), ("lime", (0, 255, 0)), ("blue", (0, 0, 255)), ("yellow", (255, 255, 0)),
           ("magenta", (255, 0, 255)), ("cyan", (0, 255, 255)), ("orange", (255, 165, 0)),
           ("white", (255, 255, 255)), ("purple", (128, 0, 128)), ("gray", (128, 128, 128))]
FONT = "C\\:/Windows/Fonts/arial.ttf"


def sh(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(str(a) for a in argv[:7]), "…" if len(argv) > 7 else "", flush=True)
    return subprocess.run(argv, check=True, **kwargs)


def phrase(minute: int) -> str:
    sentence = TARGET_SENTENCE if minute == TARGET_MINUTE else SENTENCES[minute % len(SENTENCES)]
    return f"Bloco do minuto {number_to_words(minute)}. {sentence}"


def make_speech(minutes: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    phrases = {m: phrase(m) for m in range(minutes)}
    code = (
        "import json,sys,urllib.request\n"
        f"phrases = {json.dumps(phrases, ensure_ascii=False)}\n"
        "for m, text in phrases.items():\n"
        f"    body = json.dumps({{'model': {TTS_MODEL!r}, 'voice': 'faber', 'input': text, 'response_format': 'wav'}}).encode()\n"
        "    r = urllib.request.Request('http://watch-whisper:8000/v1/audio/speech', data=body, headers={'Content-Type': 'application/json'})\n"
        "    open(f'/out/m{int(m):03d}.wav', 'wb').write(urllib.request.urlopen(r, timeout=300).read())\n"
    )
    sh(["docker", "run", "--rm", "--network", NETWORK, "--read-only", "--cap-drop", "ALL",
        "--mount", f"type=bind,src={out_dir.resolve().as_posix()},dst=/out",
        "--entrypoint", "python", IMAGE, "-c", code])


def make_video(minutes: int, speech: Path, out: Path) -> None:
    # One 60 s audio file per minute: the phrase 5 s in, silence after.
    parts = []
    for m in range(minutes):
        part = speech / f"p{m:03d}.wav"
        sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(speech / f"m{m:03d}.wav"),
            "-af", "adelay=5000:all=1,apad=whole_dur=60", "-ar", "22050", "-ac", "1", "-t", "60", str(part)])
        parts.append(part)
    (speech / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(speech / "list.txt"), "-c", "copy", str(speech / "all.wav")])
    boxes = ",".join(
        f"drawbox=x=0:y=0:w=iw:h=ih:color={PALETTE[m % len(PALETTE)][0]}:t=fill"
        f":enable='between(t,{m * 60},{m * 60 + 59.99})'" for m in range(minutes)
    )
    text = (f"drawtext=fontfile='{FONT}':text='MINUTO %{{eif\\:floor(t/60)\\:d}}':fontsize=64:fontcolor=white"
            ":box=1:boxcolor=black:x=(w-text_w)/2:y=h/2-80,"
            f"drawtext=fontfile='{FONT}':text='%{{pts\\:hms}}':fontsize=40:fontcolor=white"
            ":box=1:boxcolor=black:x=(w-text_w)/2:y=h/2+20")
    seconds = minutes * 60
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=black:size=640x360:rate=2:duration={seconds}",
        "-i", str(speech / "all.wav"),
        "-filter_complex", f"[0:v]{boxes},{text}[v]", "-map", "[v]", "-map", "1:a", "-t", str(seconds),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "48k",
        str(out)])


def corner_color(frame: Path) -> tuple[int, int, int]:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(frame),
         "-vf", "crop=iw*0.1:ih*0.1:iw*0.02:ih*0.02,scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return raw[0], raw[1], raw[2]


def nearest_minute_color(rgb) -> int:
    return min(range(len(PALETTE)), key=lambda i: sum((a - b) ** 2 for a, b in zip(rgb, PALETTE[i][1])))


def plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def call(session: Session, name: str, arguments: dict, timeout: float = 120) -> dict:
    return session.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)


def text_of(result: dict) -> str:
    return "\n".join(c["text"] for c in result.get("content") or [] if c.get("type") == "text")


def chunk_stamps(job_id: str) -> dict[str, str]:
    """Chunk file → mtime inside the jobs volume (read-only look)."""
    out = subprocess.run(
        ["docker", "run", "--rm", "--read-only", "--network", "none", "-v", "watch-mcp-jobs:/jobs:ro",
         "--entrypoint", "sh", IMAGE, "-c", f"cd /jobs/{job_id}/chunks 2>/dev/null && stat -c '%n %Y.%y' *.json"],
        capture_output=True, text=True,
    ).stdout
    return dict(line.split(" ", 1) for line in out.splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--minutes", type=int, default=35)
    ap.add_argument("--restart-at-chunk", type=int)
    ap.add_argument("--timeout", type=float, default=4 * 3600)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    video = work / TEST_NAME
    if not video.exists():
        make_speech(args.minutes, work / "speech")
        make_video(args.minutes, work / "speech", video)
    inbox_copy = Path(INBOX_DIR) / TEST_NAME
    shutil.copyfile(video, inbox_copy)
    report: dict = {"minutes": args.minutes, "video_bytes": video.stat().st_size}
    desktop = entry_v2()
    session = Session(["docker", *desktop["args"]], {"PATH": __import__("os").environ["PATH"], **desktop["env"]})
    try:
        init = session.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                              "clientInfo": {"name": "watch-e2e-v2", "version": "1.0"}}, timeout=60)
        session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = {t["name"] for t in session.request("tools/list", {}, timeout=30)["tools"]}
        report["protocol"] = init.get("protocolVersion")
        report["tools"] = sorted(tools)
        assert {"analyze_video", "job_status", "list_jobs", "search_transcript", "watch_video",
                "list_inbox", "transcribe_video"} <= tools, tools
        assert TEST_NAME in text_of(call(session, "list_inbox", {}))

        started = time.monotonic()
        created = text_of(call(session, "analyze_video", {"source": TEST_NAME}))
        job_id = re.search(r"job_id = ([0-9a-f]{32})", created).group(1)
        report["job_id"] = job_id
        samples, restarted, before_restart = [], False, {}
        while True:
            status = text_of(call(session, "job_status", {"job_id": job_id}))
            state = re.search(r"- Estado: (.+)", status).group(1)
            stage = re.search(r"- Etapa: (.+)", status).group(1)
            progress = re.search(r"- Progresso: (\d+)%", status).group(1)
            samples.append({"t": round(time.monotonic() - started), "state": state, "stage": stage, "progress": int(progress)})
            print(samples[-1], flush=True)
            m = re.search(r"transcrevendo \((\d+)/(\d+)\)", stage)
            if args.restart_at_chunk and not restarted and m and int(m.group(1)) > args.restart_at_chunk:
                before_restart = chunk_stamps(job_id)
                sh(["docker", "restart", "watch-mcp-worker"], capture_output=True)
                restarted = True
                report["restart"] = {"at_stage": stage, "chunks_done_before": sorted(before_restart)}
            if state in ("concluído", "falhou"):
                break
            if time.monotonic() - started > args.timeout:
                raise TimeoutError("job did not finish")
            time.sleep(20)
        report["status_final"] = status
        report["elapsed_s"] = round(time.monotonic() - started)
        report["progress_samples"] = samples
        assert state == "concluído", status
        if restarted:
            after = chunk_stamps(job_id)
            unchanged = {k: before_restart[k] == after.get(k) for k in before_restart}
            report["restart"]["chunks_unchanged"] = unchanged
            assert before_restart and all(unchanged.values()), unchanged

        found = text_of(call(session, "search_transcript", {"job_id": job_id, "query": QUERY}))
        report["search"] = found
        hit = re.search(r"^- \[(\d+):(\d+)(?::(\d+))?–", found, flags=re.M)
        assert hit, found
        if hit.group(3):
            hit_s = int(hit.group(1)) * 3600 + int(hit.group(2)) * 60 + int(hit.group(3))
        else:
            hit_s = int(hit.group(1)) * 60 + int(hit.group(2))
        assert TARGET_MINUTE * 60 <= hit_s < (TARGET_MINUTE + 1) * 60, hit_s

        start, end = hit_s - 3, hit_s + 20
        hms = lambda x: f"{x // 3600}:{x % 3600 // 60:02d}:{x % 60:02d}"  # noqa: E731
        result = call(session, "watch_video", {"job_id": job_id, "start": hms(start), "end": hms(end),
                                               "max_frames": 6}, timeout=60)
        text = text_of(result)
        images = [c for c in result["content"] if c.get("type") == "image"]
        out = work / "watch"
        out.mkdir(exist_ok=True)
        (out / "result.txt").write_text(text, encoding="utf-8")
        checks = []
        stamps = re.findall(r"^\d+\. t=(\d+):(\d+)$", text, flags=re.M)
        for i, (image, (mm, ss)) in enumerate(zip(images, stamps), 1):
            path = out / f"frame_{i:02d}.jpg"
            path.write_bytes(__import__("base64").b64decode(image["data"]))
            t = int(mm) * 60 + int(ss)
            color = nearest_minute_color(corner_color(path))
            checks.append({"t": t, "minute": t // 60, "color_minute_ok": color == (t // 60) % len(PALETTE)})
        report["watch"] = {"frames": len(images), "checks": checks,
                           "transcript_has_sentence": "colheitadeira" in plain(text)}
        assert images and all(c["minute"] == TARGET_MINUTE and c["color_minute_ok"] for c in checks), checks
        assert report["watch"]["transcript_has_sentence"], text
    finally:
        session.close()
        inbox_copy.unlink(missing_ok=True)
        (work / "v2_long_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("progress_samples", "status_final")},
                     indent=2, ensure_ascii=False))
    print("V2 LONG E2E: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
