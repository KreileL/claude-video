# NOTES — branch mcp-local (local MCP + Whisper stack)

Decisions and assumptions made while building the local Docker stack, in the
order they came up. Anything here that turns out wrong is a one-line change.

## Origin of the branch

- `mcp-local` started at upstream `main` (83da59f). It now merges `origin/lab`
  (the fork's working branch from the ClaudeWatch install, already on GitHub):
  local OpenAI-compatible Whisper backend, `--force-whisper`, `-fps_mode` for
  ffmpeg 9, original-language captions, sparse-coverage warning.
- The ClaudeWatch checkout had one uncommitted change (pick the `-orig` caption
  matching the video's language on AI-dubbed videos). It was copied here as a
  patch, with tests, and committed. ClaudeWatch itself was not touched.

## Phase 0 — watch-whisper

- Image: `ghcr.io/speaches-ai/speaches:0.8.3-cpu`, pinned by digest. 0.8.3 is the
  latest stable; 0.9.0 only has release candidates.
- Resource limits: **8 CPUs, 4 GB** (not the suggested 2 CPUs). The host CPU is a
  Xeon E5-2680 v4 (Broadwell, no AVX-512/VNNI); with 2 CPUs even the small model
  ran at 1.3× real time (9.5 s for 7.5 s of speech). 8 CPUs → 3.9 s.
- `OMP_NUM_THREADS` and `WHISPER__CPU_THREADS` match the CPU limit: `nproc`
  reports all 28 host cores inside the quota.
- `read_only: true` + tmpfs `/tmp`. `HF_HUB_DISABLE_XET=1` is required: the xet
  downloader writes a chunk cache outside the model volume, which fails on a
  read-only rootfs.
- `WHISPER__TTL=120`: in speaches 0.8.x this single TTL also governs the Piper
  and Kokoro TTS managers. Idle models unload after 2 minutes.
- UI disabled (`ENABLE_UI=false`). No published ports.

### Models

| Role | Model | Notes |
|------|-------|-------|
| STT (default) | `Systran/faster-whisper-small` | multilingual; ~620 MB RAM loaded, ~0.5× real time on 8 CPUs |
| STT (alt) | `Systran/faster-whisper-medium` | kept in the volume; better Portuguese, ~1.75× real time on 8 CPUs, ~2 GB RAM |
| TTS (tests) | `speaches-ai/piper-pt_BR-faber-medium` (voice `faber`) | pt-BR |

Benchmark (7.5 s Piper pt-BR clip, "…O gato azul pulou a cerca amarela às três horas da tarde."):

- small: "…pulou a ser camarelas três horas…" — 3.9 s warm on 8 CPUs.
- medium: "…pulou a cerca amarelas 3 horas…" — 13 s warm on 8 CPUs (int8 ≈ float32 on this CPU).

Small was chosen because an Instagram reel almost never has captions, so every
reel goes through Whisper, and medium would take ~2.5 min for a 90 s reel.
Switching is one env var (`WATCH_WHISPER_MODEL`) — see the README.

`deepdml/faster-whisper-large-v3-turbo-ct2` (used by the ClaudeWatch install)
was not adopted: the brief asked for small/medium, and its encoder is heavier
than medium's on this CPU.

Measured memory (`docker stats`): idle 118 MiB; Piper loaded 312 MiB; small +
Piper 739 MiB; small + medium loaded together 2.4 GiB.
