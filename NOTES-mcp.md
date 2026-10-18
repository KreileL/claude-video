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
After the 120 s TTL the models unload (`/api/ps` empty); the process keeps
~590 MiB RSS because glibc does not hand freed heap back to the OS.

- `watch-mcp-net` has IPv6 enabled (Docker-assigned ULA subnet). See "YouTube
  and IPv6" below.

## Phase 1 — local Whisper backend (`whisper.py`)

- `WATCH_WHISPER_URL` is the full endpoint; it wins over the lab branch's
  `WATCH_WHISPER_BASE_URL`. Either one makes `local` the preferred backend.
  `--whisper local` with neither set uses `http://watch-whisper:8000/v1/audio/transcriptions`.
- No API key: the internal placeholder `local` is never sent as `Authorization`
  (`WATCH_WHISPER_API_KEY` is sent when set).
- Behaviour change vs. `lab`: a local server with no `WATCH_WHISPER_MODEL` now
  gets `Systran/faster-whisper-small` instead of OpenAI's `whisper-1`.
- `WATCH_WHISPER_TIMEOUT` (default 300 s; the MCP server sets 600 s) bounds each
  HTTP request; CPU transcription of long audio is slow.

## Phase 2 — MCP server (`mcp_server.py`)

- SDK: `mcp==2.3.0`. In mcp 2.x `FastMCP` is called `MCPServer`
  (`mcp.server.mcpserver`); same decorator API.
- It **wraps** `watch.py` as a subprocess instead of reimplementing it. It
  parses the frame list from watch.py's report (a test runs the real watch.py
  to guard that format) and reads `transcript.json`.
- yt-dlp options reach `download.py` through two operator-only env hooks:
  `WATCH_YTDLP_ARGS` (JSON array) and `WATCH_YTDLP_TIMEOUT`.
- A metadata-only yt-dlp preflight runs first. It checks the duration limit and
  live streams, turns the Instagram login wall / YouTube anti-bot into clear
  errors, and gives width/height so frames come out with the **long side at
  512 px** (288×512 for a vertical reel). Assumption: "JPEG 512px" means the long
  side, so 24 reel frames fit the budget.
- Image budget: 800 KiB measured on the **base64** payload (what travels in the
  tool result). Over budget → largest uniform subsample that fits (first and
  last frame kept) plus a warning in the text. Measured: Instagram reel (77 s,
  24 frames) 282 KB; YouTube 19 s (19 frames, landscape 512 px) 348 KB.
- The text and messages are in Portuguese (the user's language).
- `question` is echoed in the text so the model knows what to look for; the
  server does not use it otherwise.
- Progress notifications every 10 s while a subprocess runs (helps clients that
  reset their timeout on progress).
- Server version reported as 0.2.0 (the plugin version).

### Security controls (as implemented)

| Control | Where | Test |
|---|---|---|
| Only `http`/`https`; no userinfo, whitespace or control characters; ≤ 2048 chars | `netguard.validate_url` | `test_netguard.py`, `test_mcp_server.py::test_check_url_*` |
| Public addresses only (loopback, private, link-local, CGNAT, multicast, reserved, IPv4-mapped/6to4/Teredo-embedded all refused; any non-public record refuses the whole name) | `netguard.resolve_checked` | `test_netguard.py` |
| Redirects and every yt-dlp connection re-checked; connects to the vetted IP (no re-resolution → no DNS rebinding) | `netguard.EgressProxy`, `--proxy` in the yt-dlp args | `test_netguard.py` (redirect to loopback / private name), `test_mcp_server.py::test_redirect_to_internal_host_is_blocked` (real yt-dlp) |
| Test mode `WATCH_MCP_TEST_ALLOW_HOSTS`: exact hostnames only, **empty when unset** | `netguard.test_allow_hosts` | `test_allow_hosts_empty_by_default`, `test_test_mode_is_off_by_default` |
| No `shell=True` anywhere; argv lists; URL after `--`; `start`/`end` validated by regex | `mcp_server.run_process`, `download.py` | `test_check_time_*` |
| Timeouts: whole call 900 s (process-group kill), each yt-dlp 300 s, Whisper request 600 s, yt-dlp socket 30 s, proxy connect 10 s / idle 60 s | `mcp_server`, `download.py`, `whisper.py` | `test_run_process_timeout_kills_child`, download/whisper timeout tests |
| Duration ≤ 15 min (preflight + `--match-filter "duration <=? 900 & !is_live"`), download ≤ 500 MB (`--max-filesize`), `/tmp` tmpfs 1 GB | `mcp_server`, `desktop_entry.py` | unit (args); see limitation below |
| Cookies: directory mounted read-only at `/secrets`; private 0600 copy in the per-call work dir; never logged; cookie-looking values scrubbed from errors | `mcp_server.run_watch`, `scrub` | `test_cookies_are_copied_never_echoed`, `test_scrub_redacts_cookie_values` |
| Work dir removed after every call, success or failure | `mcp_server` | `test_watch_video_returns_text_then_images_and_cleans_up`, `test_work_dir_removed_on_failure` |
| Whisper down → clear error with `docker compose -p watch-mcp up -d` | `mcp_server.check_whisper` | unit + e2e with the service stopped |
| Container: non-root (uid 10001), `--read-only`, `--tmpfs /tmp`, `--cap-drop ALL`, `no-new-privileges`, `--memory 2g`, `--pids-limit 256`, no published ports | `Dockerfile`, `tests/e2e/desktop_entry.py` | the e2e harness spawns exactly these args |
| Compose: no published ports, `read_only`, `cap_drop: ALL`, `no-new-privileges`, CPU/memory/pids limits | `docker-compose.yml` | checked by inspection (`docker inspect`) |

Known limitations, accepted:

- When the site reports no duration (Instagram), the 15 min limit cannot be
  checked before downloading; the 500 MB cap, the 1 GB tmpfs and the call
  timeout still bound the work. Instagram reels themselves cap at 20 min.
- The proxy is reachable by any process inside the container (loopback). The
  only processes there are the server and its children.
- The egress proxy only governs yt-dlp. The server's own requests go to the
  operator-configured `WATCH_WHISPER_URL`.

## Phase 3 — image

- `python:3.12-slim` (Debian 13: ffmpeg 7.1, so `-fps_mode` works), `mcp==2.3.0`,
  `yt-dlp[default,deno]` **unpinned** on purpose: the README's fix for a broken
  extractor is `docker build --no-cache`, which only helps if a rebuild pulls
  the newest yt-dlp. `deno` is the JS runtime YouTube now requires.
- `HOME`, caches and `DENO_DIR` point at `/tmp` (tmpfs); the rootfs is read-only.
- Cookies: mounted as a **directory** (`%USERPROFILE%\.config\watch\secrets` →
  `/secrets`, read-only), not the file. Claude Desktop's config is static, and a
  missing bind source would stop the container from starting, so "mount only if
  the file exists" becomes "the file is optional inside a mounted directory".
  The empty directory was created outside the repo (the only such change
  besides the Claude Desktop config).
- Image size 1.08 GB (ffmpeg + deno).

## Phase 4 — end-to-end results

| Test | Result |
|---|---|
| Offline: Piper pt-BR speech + testsrc, "CENA n" text and colored box every 3 s, served by `watch-mcp-testsrv` on watch-mcp-net, real MCP stdio | OK, 13 s. Transcript "…teste de ponta a ponta. O gato azul dorme no sofá da sala." (first word misheard: "Olá! Este" → "Exi"); 12 frames, the box color matches the scene for every timestamp; testsrc's own counter in the frames equals t |
| YouTube "Me at the zoo" (captions path) | OK after the IPv6 fix: transcript source `captions`, 19 frames, 348 KB, 35 s |
| MCP protocol (initialize → tools/list → tools/call), exact Desktop args and minimal env | OK: TextContent first, then image/jpeg ImageContent; negotiated 2025-06-18; progress notifications received |
| Instagram reel `DKw2J6TMZd7` (Leo Messi, 77 s), no cookies | **OK without login**: 24 frames, 282 KB, Spanish auto-detected by Whisper, post caption included, 23–26 s |
| `transcribe_video` on the same reel | OK, text only, 22 s |
| watch-whisper stopped | clear error naming `docker compose -p watch-mcp up -d` |
| Other containers | unchanged (same IDs, creation times, states; `speaches` has the same container ID and start time) |

### YouTube and IPv6 (debugged)

The first YouTube runs failed with "Sign in to confirm you're not a bot" while
the same yt-dlp worked from the host. Root cause: the host reaches YouTube over
IPv6, and the line's IPv4 (shared, likely CGNAT) is throttled: from the host,
IPv4 gets a 302 to the "sorry" page; from the IPv4-only container, a 429. Fix:
`enable_ipv6` on `watch-mcp-net`, and the proxy tries the first vetted IPv6
address before IPv4 (a ULA-only container would otherwise prefer IPv4). If the
host ever loses IPv6, the proxy falls back to IPv4 and the clear "anti-bot"
error explains the situation.
