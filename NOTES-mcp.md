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

# v2 (branch mcp-v2)

v2 adds a local inbox, long videos as background jobs and better Brazilian
Portuguese transcription, as image `watch-mcp:v2` and Desktop entry
`watch-v2`. v1 (`watch-mcp:latest`, entry `watch`) was not touched: the image
was never rebuilt or retagged, and the `watch` entry is byte-for-byte what was
installed. `mcp-v2` started from `mcp-local` (16dc2ed); nothing was merged back.

## Architecture — why an always-on worker with a disk queue

Claude Desktop starts the stdio server per session (`docker run -i --rm`), and
one tool call cannot last hours, so the work must outlive the call.
Alternatives considered:

- **A detached process inside the stdio container.** It dies with the
  container when Desktop closes (`--rm`), so this was rejected.
- **A worker with an HTTP API.** It would open a listening port inside
  watch-mcp-net and need auth. More attack surface and more code than a
  directory, so this was rejected.
- **A worker that polls a queue of job directories on a named volume
  (chosen).** It needs no port and no new protocol. State survives restarts
  and can be inspected. Resume is per stage, because each stage is skipped when
  its output exists.

How the chosen design works:

- **Ownership:** the stdio server only creates `job.json` (state `queued`)
  and reads. Every later write belongs to the worker. job.json is replaced
  atomically (temp file + `os.replace`), so there is a single writer and no
  locks (`jobstore.py`).
- **Service:** `watch-worker` (compose) runs the same image with entrypoint
  `worker.py`. It runs one job at a time, because Whisper on 8 CPUs is the
  bottleneck anyway. It writes `/jobs/.heartbeat` every 15 s, and `job_status`
  / `analyze_video` warn when it is stale ("rode docker compose -p watch-mcp
  up -d").
- **Job directory** (`/jobs/<32-hex id>/`):
  - `job.json`
  - `meta.json`: display fields only, never the raw yt-dlp JSON with format
    URLs or headers
  - `plan.json`: the chunk plan
  - `index.json` + `frames/f_NNNNN.jpg`
  - `chunks/NNNN.json`: one per transcribed chunk
  - `transcript.json`
  - `work/`: download and audio, removed when no longer needed

### Source file: neither kept nor re-read — the frame index is kept

The brief left the choice between keeping the source for a TTL and re-reading
it from the origin. Decision: neither. The index (scene cuts + a time grid,
512 px JPEG) is computed once, and the video is deleted right after audio
extraction and indexing.

- **Re-reading the origin** would need network again, and the origin can
  change, disappear or rate-limit. Inbox files can also be moved.
- **Keeping the source** costs gigabytes per job.
- **The index:**
  - Its grid interval is `max(2 s, duration/2500)`: 2 s for anything up to
    83 min, 5.8 s for 4 h.
  - Scene cuts (threshold 0.3) are added on top.
  - A cap of 7500 frames bounds noisy video.
- **Trade-off:** inside a short range, frames are at most one grid interval
  apart. For "what is on screen at 1:23:45" that is enough; slide or scene
  changes are cuts.
- **Cost:** `watch_video(job_id=…)` never runs ffmpeg in the server; it reads
  JPEGs and JSON.

### Short videos keep the direct path

The brief keeps `watch_video(url)` / `transcribe_video(url)` as in v1. That
path still runs watch.py inside the stdio container for videos up to 15 min,
under every v1 limit: 500 MiB, 900 s, 1 GB tmpfs. This is the one place the
stdio server does real work. It is deliberate, for compatibility. A longer
URL gets an error pointing to `analyze_video`.

### Long-job limits (env on watch-worker; defaults)

| Limit | Default | Enforced by |
|---|---|---|
| Media duration | 4 h (`WATCH_JOB_MAX_DURATION`) | yt-dlp metadata + `--match-filter`; ffprobe of the file (inbox and downloaded); decoding capped with `-t` at the probed duration |
| URL download | 4 GiB (`WATCH_JOB_MAX_DOWNLOAD_BYTES`) | netguard proxy budget (hard, aggregate) + `--max-filesize` |
| Time per job | 8 h (`WATCH_JOB_TIMEOUT`) | deadline from the first start; every subprocess gets the remaining time; process-group kill |
| Whisper request | 1800 s (`WATCH_WHISPER_TIMEOUT`) | per chunk; 3 retries (30/60/120 s) after waiting for `/health` |
| Disk | 20 GiB for the whole cache (`WATCH_JOBS_MAX_BYTES`) | checked before audio/index; cleanup evicts oldest finished jobs |
| Retention | 7 days after the last update (`WATCH_JOB_TTL`) | cleanup every 10 min; never queued/running; orphan dirs after 1 h |
| Resume attempts | 3 | a job that keeps killing the worker fails instead of looping |
| Work dir | volume, not tmpfs | `/jobs/<id>/work`; worker tmpfs is 512 MiB (yt-dlp/deno caches, cookie copy) |

Transcription:
- **Chunks:** ~300 s (`WATCH_CHUNK_SECONDS`), cut at the silence midpoint
  nearest each boundary (±30 s, ffmpeg `silencedetect`).
- **Resume:** each chunk is written to `chunks/NNNN.json` as soon as it
  finishes, and a restart skips the existing ones (verified by the E2E test
  below).
- **Progress and ETA:** progress follows transcribed audio seconds. The ETA
  comes from the measured speed of the chunks done in this run.

### Cache reuse

The cache key is a hash of:
- the source
- for an inbox file, its size and mtime
- language
- quality
- the sorted terms

`analyze_video` with the same key returns the existing queued, running or done
job instead of a new one.

### Security controls added in v2 (all v1 controls kept)

| Control | Where | Test |
|---|---|---|
| Inbox names: bare name only; no `/ \ :`, control/format characters, leading dot, absolute or drive paths; ≤ 255 chars; allowed extensions only | `inbox.check_name` | `test_inbox.py`, `test_mcp_v2.py::test_analyze_refuses_bad_inbox_names` |
| Inbox file: `lstat` (no symlink), regular file, parent must be the inbox; opened once with `O_NOFOLLOW`; ffmpeg/ffprobe read `/proc/self/fd/N` (the validated inode, whatever happens to the name: TOCTOU) | `inbox.resolve/open_media`, `worker.Source` | symlink tests (Linux), Docker probe |
| Local media: `-protocol_whitelist file` **and** `-format_whitelist mov,matroska,mp3,wav,ogg`, then the probed demuxer forced with `-f`; ffprobe must find real audio/video (attached cover art does not count) and a positive duration | `inbox.input_args/probe` | concat/HLS/text-as-mp4 tests; `test_probe_never_lets_concat_open_other_files` |
| Downloaded files get the same probe + fd discipline | `worker.open_source` | `test_url_job_downloads_through_proxy…` |
| job_id = `secrets.token_hex(16)`, validated by `^[0-9a-f]{32}$` before any path is built | `jobstore.job_path` | `test_jobstore.py`, `test_job_ids_are_validated` |
| Nothing outside `/jobs`; scans skip symlinks and foreign names; corrupt job.json is skipped, never served | `jobstore` | `test_jobstore.py` |
| No cookies or secrets in the cache: the cookie copy lives on the worker tmpfs and is deleted; `meta.json` keeps display fields only; errors are scrubbed | `worker.private_cookies/write_meta/fail` | `test_url_job_downloads_through_proxy_and_keeps_no_cookies` |
| `watch_video(job_id)` serves only `frames/f_NNNNN.jpg` of that job (resolved path, JPEG bytes), within the 800 KiB budget | `jobview.pick_frames`, `encode_frames` | `test_watch_job_never_serves_files_outside_its_frames`, `test_watch_job_respects_image_budget` |
| `terms`: ≤ 30, ≤ 60 chars, no control characters; `language`: 2 letters or `auto`; prompt ≤ 800 chars | `mcp_server.check_terms`, `worker.build_prompt` | `test_analyze_validates_url_language_and_terms` |
| Worker container: uid 10001, `read_only`, noexec tmpfs, `cap_drop: ALL`, `no-new-privileges`, 4 CPUs / 3 GB / 256 pids, no ports, inbox and secrets read-only | `docker-compose.yml` | `docker inspect` (below) |

## Environment findings (outside the code)

- **Kaspersky intercepts part of Docker's HTTPS.** About one TLS handshake
  in three, from containers to huggingface.co, came back signed by "Kaspersky
  Anti-Virus Personal Root Certificate". Containers do not trust that root,
  so verification fails ("self-signed certificate in certificate chain"):
  correct behaviour, and nothing was changed to trust it.
  - **Downloads here:** model files were fetched on the host instead (pinned
    revisions) and installed offline.
  - **Impact on v1/v2:** the same interception can make an occasional yt-dlp
    or model download fail.
  - **Option for the user:** exclude Docker traffic from Kaspersky's
    encrypted-connection scanning.
- **Kokoro TTS cannot run inside watch-mcp-whisper.** Its phonemizer copies
  `libespeak-ng.so` to `/tmp` and dlopens it, and the hardened container's
  tmpfs is `noexec`. watch-whisper was left as is, because changing it would
  recreate the v1 service. The benchmark's Kokoro voices were synthesized by a
  throwaway speaches container (`watch-mcp-tts`, models volume read-only,
  removed right after).
- **Incident: the benchmark hung watch-whisper for about 30 min (02:26–02:55,
  2026-10-18).**
  - **Trigger:** the first benchmark runner unloaded models explicitly
    (`DELETE /api/ps/{model}`) between candidates. In speaches 0.8.3 that call,
    made while the idle-TTL offload was already scheduled, logged "unloaded"
    but never returned, and left the model manager locked.
  - **Impact:** every later transcription request hung. `/health` stayed
    green, so the healthcheck did not notice. v1 transcription was unavailable
    during that window.
  - **Detection:** confirmed with a 15 s clip that timed out at 30 s.
  - **Fix:** the stuck benchmark container was stopped and watch-mcp-whisper
    restarted with `docker restart`: same container ID and same compose config
    hash; only the start time changed. A real transcription worked right after.
  - **Prevention:** the runner never unloads explicitly. It waits for the
    120 s TTL to empty `/api/ps`. Neither the MCP server nor the worker ever
    calls `/api/ps`.

## Codex counter-review (v2)

- **Procedure:** the one that worked in Phase 5. The worktree was pre-created
  as `.codex/worktrees/contraprova-v2`, detached at the commit under review
  (f028d14). The command is the one from the brief:
  ```
  codex exec -m gpt-6.1-sol -c model_reasoning_effort="high" --disable unified_exec
    -s workspace-write -c approval_policy="never"
    -c sandbox_workspace_write.network_access=false
    -C .codex/worktrees/contraprova-v2 -o .codex/review-mcp-v2.md "<prompt>"
  ```
  The prompt starts with an operator note about the worktree and asks for the
  brief's focus, three rounds, reproduction of every claim, separate lists and
  "Fora do diff". No danger-full-access.
- **Parallel run:** Codex ran while the benchmark used Docker. Its review
  covers the code before the benchmark's model choice; that later change is
  configuration only.
- **What Codex ran:** the existing suite (394 passed) plus 22 reproduction
  tests of its own, in `.codex/worktrees/contraprova-v2/.codex-repro/`
  (git-ignored).

Verdict: "não promover ainda" — 6 blocking, 5 non-blocking, many confirmed. All fixed in 14fdc34, each with a
regression test adapted from Codex's reproduction (`tests/test_v2_counter_review.py`, 18 tests):

| # | Finding | Fix |
|---|---|---|
| B1 | A cookie value the site echoes into title/description was written to meta.json/job.json and served | display fields are redacted with the jar's values before anything is persisted |
| B2 | `WATCH_JOBS_MAX_BYTES` was only checked once, before audio/index, after the download | room check (evicting finished jobs first) before download, audio and index; proxy budget = min(download cap, room); ffmpeg `-fs` on the audio |
| B3 | The job deadline did not cover Whisper (health wait, request, retries); a late reply still finished the job | `whisper.py` takes a `deadline` for every attempt and backoff; the worker passes the remaining time and fails a reply that lands after it; ffprobe gets the remaining time |
| B4 | A resume after the inbox file was replaced mixed old audio with new frames | source identity (size, mtime, duration) stored at first open, compared on resume and against the fingerprint the server saw |
| B5 | The `-frames:v` cap stopped the index before the end of scene-heavy videos | grid frames always; scene cuts only ≥ grid/2 after the previous frame, so the total is bounded by spacing and the index always reaches the end |
| B6 | A job.json with wrong types (`"started": "broken"`) crashed the worker in a loop | job.json validated before use; the main loop marks any job that raises as failed |
| N1 | A job dir or `frames/` that is a link (junction) could serve another job's frame | links refused for the job dir, job.json and frames/ |
| N2 | No overall text budget (5.8 M chars with a synthetic transcript) | search capped at 40 k chars (500 per hit), titles 300, inbox listing 200 entries |
| N3 | A corrupt chunk file was skipped silently | only a valid chunk counts as done; a corrupt one is transcribed again |
| N4 | A restart between index and video deletion kept the video during transcription | the cleanup runs on every pass |
| N5 | The download budget restarts on each attempt | kept and documented: per attempt, ≤ 3 attempts, and B2's room check bounds the disk |

Confirmed by Codex:
- inbox name validation;
- refusal of disguised content (text/HLS/ffconcat as media);
- unguessable job_id validated before any path is built;
- normal queue and retention behaviour;
- chunk-level resume;
- pt-BR request fields;
- the image budget;
- every v1 control it could run;
- the v1 Desktop entry identical to mcp-local's;
- container flags, by inspection.

"Fora do diff": nothing new.

## pt-BR benchmark (tests/bench/)

Reproducible pipeline (all scripts in `tests/bench/`, command list in RESULTS.md's header and the README):

1. **Fetch:** `fetch_models.py` downloads the pinned revisions on the host.
2. **Install:** `install_models.py` converts the fine-tunes with ctranslate2
   4.5.0 (the version inside speaches) and installs them into the
   watch-mcp-models cache. It runs offline in a throwaway container.
3. **Real clips:** `fetch_real.py` downloads them.
4. **Run:** `run_bench.py` transcribes, from a container on watch-mcp-net:
   - `tts`: synthesize the corpus
   - `refs`: draft the approximate references
   - `run <models>` / `run-pieces 30 <models>`
5. **Report:** `report.py` re-scores everything against the current
   references and writes the table.

Every request is `language="pt"`. Metrics:
- **WER:** normalized (casefold, no punctuation, numbers, money, dates,
  percentages and ordinals spelled out in Brazilian Portuguese on both sides,
  so "R$ 1.500,00" equals "mil e quinhentos reais"; `metrics.py`).
- **Separate counts:** numbers, proper names and jargon missed on the TTS
  corpus.
- **Speed:** compute seconds per minute of audio, warm, on the Xeon E5-2680 v4
  with 8 CPUs.
- **RAM:** the speaches container's peak while the model ran.

### Audio (always Brazilian)

- **TTS:** 20 sentences with numbers, dates, money in reais, Brazilian names,
  slang, IT and agro terms (`corpus_ptbr.json`). Five Brazilian voices read
  them in turns: Piper pt_BR faber and edresson, Kokoro pf_dora, pm_alex and
  pm_santa. Kokoro was synthesized in a throwaway speaches, see "Environment
  findings". The reference is exact (the TTS text).
- **Real speech:**
  - Reel DeniXRTs5Mw (70 s, a gamer's narration).
  - Nerdologia `kyGu9lTr_jM` (6.4 min narration). Reference: its official,
    manually made pt captions. Manual captions sometimes condense speech,
    which puts a floor of a few % under everyone.
  - Podpah `gwAwKfW-Q8w`, first 3 min (spontaneous conversation, laughter,
    crosstalk).
- **Approximate references (reel and Podpah):**
  - Drafted by `Systran/faster-whisper-large-v3`, which is not a candidate.
  - The Podpah draft was made in 30 s pieces: the whole-clip draft fell into a
    repetition loop.
  - Reviewed by reading. I cannot listen to audio, so only context-certain
    fixes were made: reel "sobrevivo"→"sobrevivi" and "palio"→"falou";
    Podpah "Quanto o Edwards"→"Contra o Edwards" (×2) and
    "esterinha"→"esteirinha".
  - **They are approximate.** Large-v3 shares lineage with turbo, which may
    favour turbo slightly on that column.

### Candidates and licenses (checked on each model card)

| Model | Source | License |
|---|---|---|
| `Systran/faster-whisper-small` | OpenAI Whisper small, ct2 | MIT |
| `Systran/faster-whisper-medium` | OpenAI Whisper medium, ct2 | MIT |
| `deepdml/faster-whisper-large-v3-turbo-ct2` | Whisper large-v3-turbo, ct2 | MIT |
| `watch-mcp/distil-whisper-large-v3-ptbr-ct2` | `freds0/distil-whisper-large-v3-ptbr` @3edf997 — Brazilian Portuguese (Common Voice 16 + a private pt-BR set), converted here | MIT |
| `watch-mcp/whisper-medium-portuguese-ct2` | `pierreguillou/whisper-medium-portuguese` @9ca8ecf — Portuguese Common Voice 11, converted here | Apache-2.0 |
| `Systran/faster-whisper-large-v3` | reference drafts only | MIT |

Ready-made ct2 copies of the freds0 model exist on the Hub, but without a
license tag, so it was converted here from the MIT-licensed original.

### Results

Prompt modes:
- `none`: no prompt.
- `ptbr`: Brazilian context ("Transcrição em português do Brasil… ônibus,
  celular, time, trem, registro…").
- `ptbr+terms`: the context plus the clip's proper names and jargon.

RAM caveat: the non-pieces runs were on the shared watch-whisper, whose
process keeps heap between models (4 GB limit hit). The pieces runs were on
the fresh watch-whisper-v2 with `MALLOC_ARENA_MAX=2`. Compare RAM within a
group only.

| Modelo | Prompt | WER TTS | WER real (legenda oficial) | WER real (ref. aproximada) | Números perdidos (TTS) | Nomes perdidos (TTS) | Termos perdidos (TTS) | s por min de áudio | RAM pico (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| deepdml/faster-whisper-large-v3-turbo-ct2 | none | 4.9% | 12.0% | 9.1% | 0/16 | 4/22 | 7/42 | 30.1 | 4038 |
| deepdml/faster-whisper-large-v3-turbo-ct2 | ptbr | 5.2% | 11.1% | 11.8% | 0/16 | 3/22 | 8/42 | 32.2 | 4038 |
| deepdml/faster-whisper-large-v3-turbo-ct2 | ptbr+terms | 2.7% | 11.3% | 9.2% | 0/16 | 2/22 | 3/42 | 33.5 | 4038 |
| deepdml/faster-whisper-large-v3-turbo-ct2 (pedaços de 30 s) | none | 4.9% | 12.0% | 10.4% | 0/16 | 4/22 | 7/42 | 34.4 | 2169 |
| deepdml/faster-whisper-large-v3-turbo-ct2 (pedaços de 30 s) | ptbr | 5.2% | 12.0% | 8.7% | 0/16 | 3/22 | 8/42 | 32.1 | 2169 |
| deepdml/faster-whisper-large-v3-turbo-ct2 (pedaços de 30 s) | ptbr+terms | 2.7% | 11.8% | 7.8% | 0/16 | 2/22 | 3/42 | 30.8 | 2169 |
| Systran/faster-whisper-medium | none | 7.0% | 9.8% | 13.3% | 1/16 | 5/22 | 6/42 | 33.2 | 3765 |
| Systran/faster-whisper-medium | ptbr | 7.3% | 9.9% | 12.8% | 1/16 | 5/22 | 7/42 | 34.0 | 3765 |
| Systran/faster-whisper-medium | ptbr+terms | 4.9% | 11.4% | 12.8% | 1/16 | 3/22 | 2/42 | 34.1 | 3765 |
| Systran/faster-whisper-small | none | 11.3% | 11.8% | 18.3% | 3/16 | 7/22 | 11/42 | 12.3 | 2461 |
| Systran/faster-whisper-small | ptbr | 10.7% | 6.4% | 19.7% | 2/16 | 8/22 | 10/42 | 12.0 | 2461 |
| Systran/faster-whisper-small | ptbr+terms | 9.5% | 7.6% | 57.3% | 3/16 | 3/22 | 6/42 | 11.4 | 2461 |
| watch-mcp/distil-whisper-large-v3-ptbr-ct2 | none | 8.5% | 91.5% | 66.7% | 3/16 | 3/22 | 12/42 | 21.6 | 3775 |
| watch-mcp/distil-whisper-large-v3-ptbr-ct2 | ptbr | 16.5% | 88.4% | 74.0% | 8/16 | 6/22 | 11/42 | 21.6 | 3775 |
| watch-mcp/distil-whisper-large-v3-ptbr-ct2 | ptbr+terms | 21.6% | 88.9% | 77.9% | 9/16 | 10/22 | 13/42 | 22.4 | 3775 |
| watch-mcp/distil-whisper-large-v3-ptbr-ct2 (pedaços de 30 s) | none | 8.5% | 19.7% | 14.6% | 3/16 | 3/22 | 12/42 | 23.4 | 1465 |
| watch-mcp/distil-whisper-large-v3-ptbr-ct2 (pedaços de 30 s) | ptbr | 16.5% | 61.2% | 33.2% | 8/16 | 6/22 | 11/42 | 22.3 | 1465 |
| watch-mcp/distil-whisper-large-v3-ptbr-ct2 (pedaços de 30 s) | ptbr+terms | 21.6% | 53.3% | 39.1% | 9/16 | 10/22 | 13/42 | 22.4 | 1465 |
| watch-mcp/whisper-medium-portuguese-ct2 | none | 20.1% | 7.5% | 48.9% | 4/16 | 9/22 | 11/42 | 28.7 | 4095 |
| watch-mcp/whisper-medium-portuguese-ct2 | ptbr | 9.5% | 9.3% | 26.0% | 3/16 | 6/22 | 9/42 | 33.8 | 4095 |
| watch-mcp/whisper-medium-portuguese-ct2 | ptbr+terms | 4.9% | 12.0% | 44.5% | 2/16 | 2/22 | 3/42 | 30.2 | 4095 |

### Observations

- **large-v3-turbo** is the most accurate on real Brazilian speech: reel 2.3%,
  Podpah 12%, approximate references 9.1% (7.8% in 30 s pieces with terms).
  It is also best on the TTS corpus with terms (2.7%, 0/16 numbers missed).
  At ~0.53× real time it is faster than medium.
- **small** runs at 0.2× real time. On the reel it gives 7.5% (13 s for 70 s),
  on Nerdologia 11.8%. It is weak on spontaneous crosstalk (Podpah 23%) and
  misses more numbers and names.
- **medium** is best on the official-caption narration (9.8%), behind turbo
  elsewhere, and slower.
- **distil-pt-BR (freds0)**, the only candidate trained mostly on Brazilian
  data, is excellent for ~30 s and then degenerates into ". . . ." under
  speaches' conditioned long-form decoding (Nerdologia 91%). In fresh 30 s
  pieces it recovers (19.7% / 14.6%) but stays behind turbo, and any prompt
  hurts it badly.
- **pierreguillou medium-pt** is very good on read narration (Nerdologia
  7.5%) and poor on everything else (reel 28%, Podpah 58%). That fits a
  Common Voice fine-tune.
- **Prompt:** the Brazilian context *alone* did not help consistently. Turbo
  got worse on the approximate references (9.1→11.8%); small got better on
  Nerdologia and worse on Podpah. With the clip's terms, names and jargon
  improved for every model: turbo missed names 4→2 and terms 7→3; medium
  missed terms 6→2.
  - Caveat: on the TTS corpus the terms are the very words scored, which is
    the intended use (the caller names the proper nouns), but it inflates
    that gain.
  - small + prompt fell into a repetition loop on Podpah (57%). The worker's
    loop guard covers that in production.
- **Repetition loops:** they happened to large-v3 (draft) and to small +
  prompt. This led to the worker's loop guard (re-transcribe a looping chunk
  in 30 s pieces).

### Decision (quality="auto")

- **Short media (≤ 15 min) and the direct reel path:** `Systran/faster-whisper-small`.
  It is the fastest with an acceptable WER on reels (6.6–7.5% on the reel, 2.6×
  faster than turbo). `quality="accurate"` in `analyze_video` gives turbo
  for a short video when accuracy matters more than time.
- **Long background jobs:** `deepdml/faster-whisper-large-v3-turbo-ct2`, the
  most accurate in pt-BR overall, at ~0.53× real time on 8 CPUs, so 4 h of
  audio is about 2 h of transcription.
- **Prompt:** sent only when the caller gives `terms` (Brazilian context +
  terms); no prompt otherwise. `language="pt"` is always explicit by default.
- **Isolation:** all v2 transcription runs on `watch-whisper-v2`, never on
  the v1 service (see the RAM note above and the compose comment).

## v2 — end-to-end results

All E2E runs below used the exact `watch-v2` arguments (`tests/e2e/desktop_entry.py::entry_v2`) and Claude Desktop's minimal environment, over real MCP stdio (protocol 2025-06-18), against the running `watch-worker` and `watch-whisper-v2`.

| Test | Result |
|---|---|
| Unit suite, Windows | 423 passed, 11 skipped (POSIX-only) |
| Unit suite, Linux inside `watch-mcp:v2` | **437 passed, 0 skipped** (fd/`O_NOFOLLOW`, symlinks, `killpg`) |
| Long synthetic inbox video (`e2e_v2_long.py`, 35 min mkv, one pt-BR TTS block per minute, "MINUTO n" + clock on screen) | OK in 122 s. `list_inbox` → `analyze_video` → `job_status` → `search_transcript("colheitadeira amarela")` found **[25:05–25:09]** "Bloco do minuto 25. A colheitadeira amarela chegou à fazenda de Uberlândia." → `watch_video(job_id, 25:02–25:25)`: 6 frames, every one minute 25 (colour checked); a frame shows "MINUTO 25 / 00:25:07.500". Test file removed from D:\Inbox. |
| Resume (same run, `--restart-at-chunk 3`) | `docker restart watch-mcp-worker` during chunk 4/7; the job resumed ("running → processing"), chunks 0000–0002 kept their mtimes (not retranscribed), finished `done` |
| Silence hallucinations (same job) | none; 31 segments for 35 blocks (some merged), one merged segment lost the words "Bloco do minuto zero/um" |
| Inbox attacks (`e2e_v2_inbox.py`) | 11 refusals with clear errors and no job created: `../x`, `..\x`, `/etc/passwd`, `C:\Windows\win.ini`, `sub/x`, `.txt`, text-as-mp4, HLS-as-mp4, ffconcat-as-mkv, missing file, `file://` |
| Symlink (`e2e_v2_symlink.py`; Windows can't create symlinks without privilege, so a throwaway Linux volume served as /inbox) | links to `/etc/passwd` and to a valid clip both refused ("links simbólicos não são aceitos"); `list_inbox` hides them; volume removed |
| Long video from a URL (`e2e_v2_url.py`, TEDx 16:58, YouTube) | first run: YouTube answered 403 for the media (metadata fine), the job failed with that error; same args worked in a probe → transient; the worker now retries once (8b89485). Second run OK in 750 s: download through netguard (IPv6), index, turbo; **the loop guard fired for real on chunk 3** and re-transcribed it in 30 s pieces; `search_transcript("produtividade")` → 6 hits; `watch_video` 6 frames, 87 KB |
| Compatibility: Instagram reel `DKw2J6TMZd7` via `watch_video(url)` on watch-v2 | OK, same as v1: 24 frames, 282 KB, Spanish auto-detected, 25 s (server 0.3.0) |
| v1 regression: same reel via the `watch` entry (`watch-mcp:latest`) | OK: 24 frames, 282 KB, 24.8 s (server 0.2.0) |
| Other containers | identical to the baseline taken at the start (IDs, start times); only watch-mcp-whisper's start time changed (two documented restarts) |

Known limitations, accepted:
- The ETA is updated at chunk boundaries (every ~5 min of audio), so it looks frozen during a chunk.
- Inbox files are hashed (SHA-256) when a job starts and on resume. That is a few seconds per GB through Docker Desktop's bind mount; the stage shows "verificando o arquivo".
- An in-place rewrite of the same inode by the host *during* processing is not detected. The fd and the hash protect against swaps and changes between runs.

## v2 — deviations from the brief (and why)

- **v2 does not reuse watch-mcp-whisper; it has `watch-whisper-v2`.** The
  brief allowed reuse, without requiring it.
  - Why: one large model drove the v1 service to its memory limit, and the
    explicit-unload deadlock showed how fragile sharing is.
  - Result: v1's service definition is untouched; v2 runs on the same image,
    with its own 8 GB.
- **Source of a job: a frame index instead of keeping the video or
  re-reading the origin.** See "Architecture".
- **The Brazilian prompt is sent only together with `terms`.** The benchmark
  showed the context alone did not help consistently ("mantenha só se
  melhorar").
- **`fine-tuned pt-BR` candidate:** freds0's distil model, converted here,
  was benchmarked in two modes. It was not chosen.
- **Approximate references (reel, Podpah):** they are large-v3 drafts
  reviewed by reading, not by listening. Marked as such above.
- **Codex:** three passes (each with three internal rounds), the brief's
  command each time. The third pass was relaunched with stdin closed: the
  first launch blocked on "Reading additional input from stdin" for two
  hours. Details in "Codex counter-review (v2)".

## v2 — delivery

- Claude Desktop: backup `%APPDATA%\Claude\claude_desktop_config.backup-20261018-054127.json`, then only `mcpServers["watch-v2"]` was added (generated by `entry_v2()`); `watch`, `filesystem`, `MCP_DOCKER` and every other key compared equal to the backup; the file parses with `ConvertFrom-Json`. Claude Desktop was not restarted.
- Docker (project watch-mcp): new `watch-mcp-whisper-v2`, `watch-mcp-worker`, volume `watch-mcp-jobs`, image `watch-mcp:v2`; `watch-mcp:latest` untouched (600e77791f74). Benchmark models added to `watch-mcp-models` (turbo, large-v3, the two converted pt models, Kokoro int8, Piper edresson).
- Outside the repo and the project: only the Desktop config (backup above). Test files placed in D:\Inbox were removed.
- First real use: put the `.mkv` in `D:\Inbox`, restart Claude Desktop, then in the chat: list_inbox → analyze_video("<nome>.mkv") → job_status → search_transcript / watch_video(job_id=…).
