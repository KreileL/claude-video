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
| Timeouts: one 900 s deadline for the whole call (process-group kill, descendants included), each yt-dlp 300 s, Whisper request 600 s, yt-dlp socket 30 s, proxy connect 10 s / idle 60 s | `mcp_server`, `download.py`, `whisper.py` | `test_call_timeout_spans_the_whole_call`, `test_timeout_kills_grandchildren`, `test_strays_killed_even_when_parent_already_exited` (Linux), download/whisper timeout tests |
| Duration ≤ 15 min when the site reports it (preflight + `--match-filter`); download ≤ 500 MiB **hard, aggregate**, counted by the egress proxy (plus `--max-filesize` for early abort); `/tmp` tmpfs 1 GB; ffmpeg-delegated downloads limited to proxied protocols | `mcp_server`, `netguard.EgressProxy(max_bytes)`, `desktop_entry.py` | `test_proxy_byte_budget_is_hard_and_aggregate`, `test_download_budget_is_enforced` (real yt-dlp), `test_ytdlp_args_restrict_ffmpeg_input_protocols`, Docker N4 probe |
| Cookies: directory mounted read-only at `/secrets`; private 0600 copy in the per-call work dir; never logged; every jar value, Netscape line and Cookie header redacted from errors, warnings and output | `mcp_server.run_watch`, `cookie_secrets`, `scrub` | `test_cookies_are_copied_never_echoed`, `test_cookie_parse_error_does_not_leak_values` (real yt-dlp), `test_scrub_covers_multi_cookie_headers_and_jar_lines` |
| Work dir removed after every call, success or failure; a failed removal is logged | `mcp_server.cleanup` | `test_watch_video_returns_text_then_images_and_cleans_up`, `test_work_dir_removed_on_failure`, `test_cleanup_failure_is_logged` |
| Only this call's frames are returned: `## Frames` section only, path confined to the call's `frames/` (symlinks resolved), JPEG bytes only, none at transcript detail | `mcp_server.parse_frames`, `encode_frames` | `test_title_injection_cannot_exfiltrate_files` (real yt-dlp), `test_parse_frames_rejects_*`, `test_encode_frames_skips_non_jpeg` |
| Whisper down → clear error with `docker compose -p watch-mcp up -d` | `mcp_server.check_whisper` | unit + e2e with the service stopped |
| Container: non-root (uid 10001), `--read-only`, `--tmpfs /tmp`, `--cap-drop ALL`, `no-new-privileges`, `--memory 2g`, `--pids-limit 256`, `--cpus 4`, no published ports | `Dockerfile`, `tests/e2e/desktop_entry.py` | the e2e harness spawns exactly these args |
| Compose: no published ports, `user: "1000:1000"`, `read_only`, `cap_drop: ALL`, `no-new-privileges`, CPU/memory/pids limits | `docker-compose.yml` | checked by inspection (`docker inspect`, `docker exec … id`) |

Known limitations, accepted:

- When the site reports no duration (Instagram), the 15 min limit cannot be
  checked before downloading; the hard 500 MiB download cap, the 1 GB tmpfs and
  the 900 s call deadline still bound the work. Instagram reels cap at 20 min.
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

## Phase 5 — Codex counter-review

- Run 1, the command exactly as specified: Codex (codex-cli 0.163.0) could not
  create its worktree. The `workspace-write` sandbox keeps `.git` and `.codex`
  read-only even inside the workspace, so the worktree under `.codex/worktrees/`
  could not be created. It reviewed nothing and said so (archived as
  `.codex/review-mcp-local-run1-*.md`, git-ignored).
- Run 2, minimal deviation: the operator pre-created the detached worktree
  `.codex/worktrees/contraprova-codex` (with a `.contraprova-marker` identity
  file). The same command got one extra override,
  `-c sandbox_workspace_write.writable_roots=[<that worktree>]`, and the prompt
  got one leading sentence explaining this. Still `workspace-write`, no network,
  `approval_policy="never"`, no danger-full-access. The operator removes the
  worktree afterwards (`git worktree remove`), because `.git` is read-only
  for Codex.
- Run 2 still failed: the Windows elevated sandbox refuses a writable
  directory nested under a read-only carveout ("cannot reopen writable
  descendants under read-only carveouts"). A writable worktree **inside**
  `.codex/` is impossible under `workspace-write` on Windows.
- Run 3 (the one that counts): the same command and prompt (plus the operator
  note), launched from the repo root with `-C .codex/worktrees/contraprova-codex`,
  so the worktree is Codex's workspace root and the outer `.codex` is no longer
  a carveout for it. `-o .codex/review-mcp-local.md` stayed at the root. Codex
  could only write its full report to `$TMPDIR`; the operator copied it to
  `.codex/review-mcp-local.md` (git-ignored, with the 46 reproduction tests).

### Codex verdict (run 3) and what was done

Blocking — all fixed, each with a regression test adapted from Codex's reproduction:

- **B1 (P1) — a remote title could exfiltrate any readable file.** `parse_frames`
  matched frame-shaped lines anywhere in watch.py's report, and `encode_frames`
  read whatever path they named. A forged `og:title` with a newline returned
  `/secrets/cookies.txt` as an image. Fix: only the `## Frames` section; each
  path must resolve (symlinks included) to `frame_NNNN.jpg`/`cue_NNNN.jpg`
  directly inside the call's `frames/`; only JPEG bytes are encoded; nothing is
  encoded at `detail="transcript"`.
- **B2 (P1) — cookie values could leak in errors.** yt-dlp echoes an invalid jar
  line, and the old scrub only redacted one token after `Cookie:`. Fix: every
  value in the jar is redacted wherever it appears (read **before** yt-dlp runs,
  because yt-dlp rewrites the jar on exit), plus whole Netscape lines and whole
  `Cookie`/`Set-Cookie` header lines, in errors, warnings and the final text. A
  jar that fails to load gets a fixed message.
- **B3 (P2) — `--max-filesize` was not a hard cap** (it relies on
  Content-Length). Fix: the egress proxy enforces a hard 500 MiB budget on all
  bytes received across every connection of the call (video, audio, subtitles,
  fragments); over budget, tunnels are cut, new requests get 403, and the tool
  returns a clear error.

Non-blocking:

- N1, no whole-call deadline: fixed. One deadline covers preflight and
  watch.py.
- N2, kill skipped once the parent had exited; grandchildren survived: fixed.
  On POSIX, `killpg` always runs, also after a clean exit. The killed child is
  reaped (no dangling transport). Verified on Linux in the image. On Windows
  (dev only), just the child is killed.
- N3, 15 min only when metadata has a duration: the README now says so. The
  hard download cap and the total timeout bound the rest.
- N4, delegated protocols: `--downloader-args "ffmpeg_i:-protocol_whitelist
  http,https,tls,tcp,crypto"`. Reproduced in Docker (below).
- N5, silent cleanup failure: a failure is now logged (path only).
- N6, the health check ignored `WATCH_WHISPER_BASE_URL`: it now uses
  `whisper.local_endpoint()`.
- N7, `--whisper openai` was overridden by a local server: fixed. An explicit
  backend is honoured, and the OpenAI backend always posts to OpenAI itself
  (the old base-URL override could have sent `OPENAI_API_KEY` to a local URL).
- N8, compose had no explicit `user:`: now `user: "1000:1000"` (the image's own
  `ubuntu` user, owner of the model volume; verified with `id`, which also
  dropped the image's supplementary groups such as `sudo`). The MCP container
  also gets `--cpus 4`.

### Reproduced with Docker (what Codex could not run)

- Full unit suite on **Linux / Python 3.12 inside the image**: 211 passed,
  including the POSIX-only grandchild and stray-after-exit kill tests and the
  symlink-escape test. Windows: 208 passed + those 3 skipped.
- N4 probe inside the image on watch-mcp-net (ffmpeg forced with
  `--hls-prefer-ffmpeg`):
  - A legit HLS playlist downloads through the proxy.
  - A segment pointing at `http://watch-whisper:8000/...` was **blocked by
    netguard**, because ffmpeg honoured `http_proxy`.
  - A `file:///etc/hostname.ts` segment was refused.
  - A local "mp4" that is really an m3u8 was refused by ffprobe and ffmpeg.
  - watch-whisper logged **0** requests from the probe.
- E2E re-run on the final image: offline OK (12 frames, 4 scenes, phrase
  found); YouTube OK (captions, 19 frames, 348 KB, 15 s); Instagram
  `watch_video` OK (24 frames, 282 KB, 23 s) and `transcribe_video` OK (21 s).

## Phase 6 — Claude Desktop

- Pristine backup: `%APPDATA%\Claude\claude_desktop_config.backup-20261017-223822.json`
  (a second backup, `…-223837.json`, was taken when the entry was rewritten to
  fix the env block). `filesystem`, `MCP_DOCKER` and every other key are
  unchanged versus the pristine backup (checked programmatically); the file
  parses with `ConvertFrom-Json`.
- The `watch` entry is generated by `tests/e2e/desktop_entry.py`, the same code
  the protocol harness uses. Its env block holds SystemRoot, USERPROFILE,
  APPDATA, LOCALAPPDATA and ProgramData. os.environ upper-cases names on Windows,
  so they are matched case-insensitively (the first write missed two of them).
- Claude Desktop was not restarted.
