# /watch

**Give Claude the ability to watch any video.**

Claude Code (recommended — auto-updates via marketplace):
```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

Codex, Cursor, Copilot, Gemini CLI, or any of 50+ [Agent Skills](https://agentskills.io) hosts:
```bash
npx skills add bradautomates/claude-video -g
```
(`-g` installs globally for your user, available across all projects. Drop it to scope per-project.)

More install options (claude.ai web, manual) in the [Install](#install) section below.

Zero config to start — `yt-dlp` and `ffmpeg` install on first run via `brew` on macOS (Linux/Windows print exact commands). Captions cover most public videos for free. Whisper API key is only needed when a video has no captions.

---

Claude can read a webpage, run a script, browse a repo. What it can't do, out of the box, is *watch a video*. You paste a YouTube link and it has to either guess from the title or pull a transcript that's missing 90% of what's on screen.

With Claude Video `/watch` you can paste a URL or a local path, ask a question, and Claude fetches captions first, downloads only what it needs, extracts frames (scene-aware, or fast keyframes at `efficient` detail), pulls a timestamped transcript (free captions when available, Whisper API as fallback), and `Read`s every frame as an image. By the time it answers, it has *seen* the video and *heard* the audio.

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
```

## What people actually use it for

**Analyze someone else's content.** `/watch https://youtu.be/<viral-video> what hook did they open with?` Claude looks at the first frames, reads the opening transcript, breaks down the structure. Same for ad creative, competitor launches, podcast intros, anything where the *how* matters as much as the *what*.

**Diagnose a bug from a video.** Someone sends you a screen recording of something broken. `/watch bug-repro.mov what's going wrong?` Claude watches the recording, finds the frame where the issue appears, describes what's on screen, often catches the cause without you ever opening the file.

**Summarize a video.** `/watch https://youtu.be/<long-thing> summarize this` does the obvious thing — pulls the structure, the key moments, what was actually said and shown. Faster than watching at 2x.

**Cut the hype out of an update video.** `/watch https://youtu.be/<launch-video> what's actually new — skip the hype` Strip a "game-changer" feature drop down to the few things that matter, so you get the substance without ten minutes of intro and overselling.

**Turn a playlist into notes.** `/watch https://youtu.be/<video> summarize this to a note` Run it across a series and file a per-video summary, so a channel or course becomes a searchable set of notes instead of hours you have to sit through.

## How it works

1. **You paste a video and a question.** URL (anything yt-dlp supports — YouTube, Loom, TikTok, X, Instagram, plus a few hundred more) or a local path (`.mp4`, `.mov`, `.mkv`, `.webm`).
2. **`yt-dlp` checks captions first.** At `transcript` detail, captioned URLs return without downloading video. Otherwise, or when Whisper needs audio, it downloads only what the run needs.
3. **`ffmpeg` extracts frames at the chosen detail.** `efficient` decodes keyframes only (near-instant); `balanced`/`token-burner` prefer scene-change frames and fall back to the duration-aware uniform sampler when they under-produce. JPEGs are 512px wide by default and clamped to 1998px tall for Claude Read compatibility.
4. **The transcript comes from one of two places.** First try: `yt-dlp` pulls native captions (manual or auto-generated) from the source. Free, instant, accurate-ish. Fallback: extract a mono 16 kHz 64 kbps mp3 audio clip (~480 kB/min) and ship it to Whisper — Groq's `whisper-large-v3` (preferred — cheaper and faster) or OpenAI's `whisper-1`.
5. **Frames + transcript are handed to Claude.** The script prints frame paths with `t=MM:SS` markers and the transcript with timestamps. Claude `Read`s each frame in parallel — JPEGs render directly as images in its context.
6. **Claude answers grounded in what's actually on screen and in the audio.** Not "based on the description" or "according to the title." It saw the frames. It heard the transcript. It answers the way someone who watched the video would.
7. **Cleanup.** The script prints a working directory at the end. If you're not asking follow-ups, Claude removes it.

## Frame budget — why it matters

Token cost is dominated by frames. Every frame is an image; image tokens add up fast. The script's auto-fps logic exists so you don't blow your context budget on a sparse scan of a 30-minute video that would have been better answered by a focused 30-second window.

| Duration | Default frame budget | What you get |
|----------|---------------------|--------------|
| ≤30 s | ~30 frames | Dense — basically every key moment |
| 30 s - 1 min | ~40 frames | Still dense |
| 1 - 3 min | ~60 frames | Comfortable |
| 3 - 10 min | ~80 frames | Sparse but workable |
| > 10 min | 100 frames (capped modes) | "Sparse scan" warning — re-run focused, or `--detail token-burner` for full uncapped coverage |

When the user names a moment ("around 2:30", "the last 30 seconds", "from 0:45 to 1:00"), pass `--start` / `--end`. Focused mode gets denser per-second budgets, capped at 2 fps. Far more useful than a sparse pass over the whole thing.

## Frame deduplication

Frame selection — keyframes (`efficient`), scene-change detection (`balanced`/`token-burner`), or the uniform sampler it falls back to — can still surface near-identical frames: a screen recording that holds one slide for 90 seconds produces a dozen, each billed as a separate image. A dedup pass drops them before frames reach Claude. It runs by default on every frame mode (`--no-dedup` turns it off):

1. One `ffmpeg` call scales each extracted JPEG to a 16×16 grayscale thumbnail. Everything after is pure-stdlib Python — no image libraries.
2. For each frame, compute the **mean absolute difference** against the *last frame that was kept* (average per-pixel brightness change, 0–255 scale).
3. If that difference is at or below the threshold (`2.0`), the frame is a near-duplicate and is dropped. Otherwise it's kept and becomes the new reference.
4. The frame-budget cap applies *after* dedup, so the budget is spent on distinct frames.

Comparing against the last *kept* frame (not the previous one) catches slow fades that never trip a frame-to-frame threshold. The threshold is deliberately low and measures absolute brightness rather than structure, so a one-line code diff, a terminal scrolling a row, or two differently-colored flat slides all survive.

The **Frames** line reports what was collapsed, e.g. `6 selected from 14 candidates (… 8 near-duplicates dropped …)`. On always-moving footage nothing is dropped and you pay what you would have anyway.

## Detail modes — measured

The `--detail` dial trades speed and token cost for visual fidelity. Numbers below are from a real run against a **49:08** YouTube video (1280×720, English auto-captions) — a long, mostly-static screen recording, the case that stresses the caps hardest. Extraction times are local CPU against a pre-downloaded copy; the one-time download was **~37 s** / 76 MB, shared by the three frame modes.

| Mode | Engine | Frames | Cap | Extraction time | Temporal coverage | Est. image tokens |
|------|--------|--------|-----|-----------------|-------------------|-------------------|
| `transcript` | none (captions) | 0 | — | **~4.5 s** (one yt-dlp call, no download) | full (text) | 0 (≈26.6k text tokens) |
| `efficient` | keyframe (`-skip_frame nokey`) | 50 | 50 | **~0.5 s** | 0:00 → 49:04 (full) | **~9.8k** |
| `balanced` | scene-change | 100 | 100 | **~20.9 s** | 0:00 → 48:38 (full) | **~19.7k** |
| `token-burner` | scene-change | 116 | uncapped | **~21.0 s** | 0:00 → 48:38 (full) | **~22.8k** |

- **Image tokens** use Anthropic's `(width × height) / 750` — at the default 512px width these 720p frames are 512×288, **≈197 tokens/frame**; `--resolution 1024` roughly 4×s that. The transcript is surfaced in every captioned mode and on long videos is often the larger cost.
- **One sampling rule across frame modes.** Each detects all candidates across the full range, then even-samples (first + last always kept) down to its cap. The modes differ only in candidate *source* (keyframes vs. scene cuts) and cap, never in how coverage is spread — so the last frame always lands at the end, not partway through.
- **`efficient` is the speed tier** (~0.5 s) — it only reconstructs keyframes, so it's ~40× faster than the scene modes, which decode every frame to find cuts. It can also return *more* frames than `balanced` on low-motion footage (keyframes outnumber scene cuts); "efficient" means fast extraction, not fewer frames.
- **`token-burner` only diverges from `balanced` past the cap.** This clip had 116 cuts, so `balanced` sampled 100 and `token-burner` kept all 116. On high-motion video with hundreds of cuts, `token-burner` keeps everything (and trips the >250-frame token warning) while `balanced` thins to 100.

End-to-end from a cold URL, `transcript` is the cheapest mode by far; the frame modes add the shared ~37 s download on top of the extraction times above.

## Install

| Surface | Install |
|---------|---------|
| **Claude Code** | `/plugin marketplace add bradautomates/claude-video` then `/plugin install watch@claude-video` |
| **Codex, Cursor, Copilot, Gemini CLI, +50 more** | `npx skills add bradautomates/claude-video -g` |
| **claude.ai** (web) | [Download `watch.skill`](https://github.com/bradautomates/claude-video/releases/latest) → Settings → Capabilities → Skills → `+` |
| **Manual / dev** | `git clone` then symlink `skills/watch` into your host's skills dir (see below) |

### Claude Code

```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

Update later with `/plugin update watch@claude-video`.

### Codex, Cursor, Copilot, Gemini CLI, and 50+ other hosts

The [Agent Skills](https://agentskills.io) CLI installs the skill into whatever agents it detects:

```bash
npx skills add bradautomates/claude-video -g
```

`-g` installs globally for your user (`~/.codex/skills`, `~/.cursor/skills`, etc.); drop it to install into the current project instead. Useful flags:

- `-a, --agent <names…>` — target specific hosts, e.g. `-a codex -a cursor`
- `-l, --list` — list the skills in this repo without installing
- `--copy` — copy files instead of symlinking (for filesystems without symlink support)

The CLI discovers the skill from `skills/watch/SKILL.md` and copies the whole folder — `SKILL.md` plus its `scripts/` runtime — as a self-contained unit. `SKILL.md` resolves its own scripts relative to wherever it was installed, so it works the same on every host.

Update later with `npx skills update watch -g`.

### claude.ai (web)

1. [Download `watch.skill`](https://github.com/bradautomates/claude-video/releases/latest) from the latest release.
2. Go to Settings → Capabilities → Skills.
3. Click `+` and drop the file in.

Enable "Code execution and file creation" under Capabilities first — the skill shells out to `ffmpeg` and `yt-dlp`, so it won't run without it.

### Manual (developer)

Clone the repo and symlink the self-contained skill folder into your host's skills directory — the symlink keeps the install in sync with your working tree as you edit:

```bash
git clone https://github.com/bradautomates/claude-video.git
ln -s "$(pwd)/claude-video/skills/watch" ~/.claude/skills/watch   # or ~/.codex/skills/watch
```

For claude.ai, build the `.skill` bundle from source: `bash skills/watch/scripts/build-skill.sh` produces `dist/watch.skill`.

## First run

On the first `/watch` call, the skill runs `scripts/setup.py --check`. If `ffmpeg` / `yt-dlp` aren't on your PATH, or no Whisper API key is set, it walks you through fixing it:

- **macOS** — auto-runs `brew install ffmpeg yt-dlp`.
- **Linux** — prints the exact `apt` / `dnf` / `pipx` commands.
- **Windows** — prints the `winget` / `pip` commands.
- **API key** — scaffolds `~/.config/watch/.env` (mode `0600`) with commented placeholders for `GROQ_API_KEY` (preferred) and `OPENAI_API_KEY`.

After setup, preflight is silent and `/watch` just works. The check is a sub-100ms lookup, so it doesn't slow you down on subsequent runs.

## Bring your own keys

Captions cover the majority of public videos for free. The Whisper fallback only kicks in when a video genuinely has no caption track — typically local files, TikToks, some Vimeos, and the occasional caption-less YouTube upload.

| Capability | What you need | Cost |
|------------|---------------|------|
| Download + native captions | `yt-dlp` + `ffmpeg` | Free |
| Whisper fallback (preferred) | [Groq API key](https://console.groq.com/keys) — `whisper-large-v3` | Cheap, fast |
| Whisper fallback (alt) | [OpenAI API key](https://platform.openai.com/api-keys) — `whisper-1` | Standard pricing |
| Disable Whisper entirely | `--no-whisper` | Free, frames-only when no captions |

## Usage

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
/watch https://www.tiktok.com/@user/video/123 summarize this
/watch ~/Movies/screen-recording.mp4 when does the UI break?
/watch https://vimeo.com/123 what tools does she mention?
```

Focused on a specific section — denser frame budget, lower token cost:
```
/watch https://youtu.be/abc --start 2:15 --end 2:45
/watch video.mp4 --start 50 --end 60
/watch "$URL" --start 1:12:00            # from 1h12m to end
```

Other knobs (passed to `scripts/watch.py`):

- `--detail transcript|efficient|balanced|token-burner` — fidelity/speed dial. `transcript` skips frames (transcript only); `efficient` uses fast keyframes (cap 50); `balanced` uses scene-aware frames (cap 100); `token-burner` is scene-aware and uncapped.
- `--timestamps T1,T2,…` — grab a frame at each absolute timestamp (`SS`/`MM:SS`/`HH:MM:SS`). Claude reads the transcript first, then targets the moments the presenter flags ("look here", "as you can see"). Added on top of the detail frames (reserved against the cap); out-of-window cues are dropped in focus mode; with `--detail transcript` these become the only frames.
- `--max-frames N` — lower the frame cap for a tighter token budget.
- `--resolution W` — bump frame width to 1024 px when Claude needs to read on-screen text (slides, terminals, code).
- `--fps F` — override the auto-fps calculation (still capped at 2 fps).
- `--whisper groq|openai` — force a specific Whisper backend.
- `--no-whisper` — disable transcription entirely; frames only.
- `--no-dedup` — keep near-duplicate frames. By default a frame-delta pass drops frames that are visually near-identical to the one before them (held slides, static screen recordings, paused video), so the frame budget is spent on distinct content; this flag turns that off.
- `--out-dir DIR` — keep working files somewhere specific (default: auto-generated tmp dir).

## Limits

- **Long-video accuracy depends on the detail mode.** On the capped modes (`efficient`, default `balanced`) coverage thins out past ~10 minutes — the frame cap spreads across the whole clip, so the script prints a "sparse scan" warning and you're better off re-running focused with `--start`/`--end`. `token-burner` lifts the cap and keeps *every* scene-change frame across the full video, so it stays complete on longer clips at the cost of more image tokens. The 10-minute mark is guidance for the capped modes, not a hard ceiling.
- **Detail is one dial.** Defaults are balanced: scene-aware frames, 2 fps max, 100-frame cap. Use `--detail efficient` for a fast 50-frame keyframe pass, or `--detail token-burner` for uncapped scene candidates. Set `WATCH_DETAIL` in `~/.config/watch/.env` to change the default.

## MCP local

Uma stack Docker isolada (projeto Compose `watch-mcp`) expõe o `/watch` como servidor MCP para o Claude Desktop: você cola um link de reel/vídeo (Instagram, TikTok, YouTube) e o Claude recebe metadados, legenda do post, transcrição com timestamps e frames. A transcrição roda num Whisper próprio (speaches, CPU) e o áudio não sai da máquina.

- `watch-whisper` (`docker-compose.yml`): speaches `0.8.3-cpu` com tag e digest fixos, sem portas publicadas, rede `watch-mcp-net`, modelos no volume `watch-mcp-models`, descarrega modelos ociosos após 120 s.
- `watch-mcp:latest` (`Dockerfile`, `mcp_server.py`): o servidor MCP (stdio). O Claude Desktop o inicia sob demanda com `docker run -i --rm` e ele fala com o `watch-whisper` pela rede.

### Subir a stack

```powershell
docker compose -p watch-mcp up -d          # sobe o watch-whisper
docker build -t watch-mcp:latest .         # imagem do servidor MCP
```

Na primeira vez, baixe os modelos pela API do próprio serviço:

```powershell
docker exec watch-mcp-whisper curl -s -X POST http://localhost:8000/v1/models/Systran/faster-whisper-small
```

A entrada `watch` em `%APPDATA%\Claude\claude_desktop_config.json` roda o `docker run` com `--network watch-mcp-net --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --memory 2g --pids-limit 256 --cpus 4`, sem portas publicadas (o gerador está em `tests/e2e/desktop_entry.py`). Reinicie o Claude Desktop depois de mudar a configuração.

### Ferramentas

- `watch_video(url, question?, start?, end?, detail="balanced", max_frames=24)`: devolve um texto com metadados, transcrição e a lista de frames (`t=MM:SS`), seguido dos frames em JPEG (lado maior de 512 px) em ordem temporal. Envia no máximo ~800 KB de imagens (em base64); se passar disso, reduz os frames uniformemente e avisa no texto.
- `transcribe_video(url)`: só o texto, sem frames.

### Rebuild quando o yt-dlp quebrar

Instagram, TikTok e YouTube mudam com frequência e quebram versões antigas do yt-dlp. O `yt-dlp` não tem versão fixada de propósito, então um rebuild sem cache baixa a versão mais nova:

```powershell
docker build --no-cache -t watch-mcp:latest .
```

### Cookies do Instagram (opcional)

Reels públicos costumam funcionar sem login. Quando o Instagram exigir login, a ferramenta responde explicando isso. Nesse caso, exporte os cookies de uma conta **secundária** (nunca a principal) em formato Netscape e salve em:

```
%USERPROFILE%\.config\watch\secrets\cookies.txt
```

A pasta é montada read-only no container como `/secrets` (montar a pasta, e não o arquivo, permite que o servidor suba sem cookies). O arquivo é lido a cada chamada, então não é preciso reiniciar nada. O servidor passa ao yt-dlp uma cópia privada na área temporária da chamada, apagada no fim, e nunca registra cookies em log nem na saída. O mesmo arquivo também ajuda no YouTube.

### Trocar o modelo do Whisper

O padrão é `Systran/faster-whisper-small`, porque o `medium` transcreve melhor em português mas é cerca de 3,5× mais lento nesta CPU. Para trocar:

1. Baixe o modelo: `docker exec watch-mcp-whisper curl -s -X POST http://localhost:8000/v1/models/<modelo>`
2. Mude `-e WATCH_WHISPER_MODEL=<modelo>` na entrada `watch` do `claude_desktop_config.json` e reinicie o Claude Desktop.

`Systran/faster-whisper-medium` já está no volume. CPU e memória do serviço ficam em `docker-compose.yml` (8 CPUs, 4 GB).

### Segurança

Só URLs `http`/`https` para endereços públicos são aceitas. Todo acesso do yt-dlp passa por um proxy local (`netguard.py`) que confere de novo cada destino, inclusive redirecionamentos e CDNs, e conecta no IP já verificado, o que impede DNS rebinding. Quando o yt-dlp delega o download ao ffmpeg, só protocolos que passam pelo proxy são aceitos. Os frames devolvidos são só os JPEGs gerados na pasta da própria chamada, e valores de cookies nunca aparecem em erros nem na saída.

Limites por chamada:

- **Duração:** 15 min, verificados antes do download quando o site informa a duração. O Instagram nem sempre informa; aí valem os outros limites.
- **Download:** 500 MB no total, com teto rígido contado pelo proxy (não depende do `Content-Length`).
- **Área temporária:** 1 GB.
- **Tempo:** 15 min por chamada, no total.

A variável `WATCH_MCP_TEST_ALLOW_HOSTS` existe só para os testes de ponta a ponta e fica vazia por padrão. Detalhes e decisões estão em [NOTES-mcp.md](NOTES-mcp.md).

### Testes de ponta a ponta

```powershell
.venv\Scripts\python tests\e2e\e2e_offline.py --work <pasta-temp>                 # vídeo sintético pt-BR, rede interna
.venv\Scripts\python tests\e2e\e2e_mcp_stdio.py --url <URL> --out <pasta-temp>    # protocolo MCP real via stdio
```

## MCP local v2

A v2 (branch `mcp-v2`, imagem `watch-mcp:v2`, entrada `watch-v2` no Claude Desktop) acrescenta três coisas à v1, que continua intacta (`watch-mcp:latest`, entrada `watch`):

1. **Caixa de entrada local:** arquivos em `D:\Inbox` (montada somente leitura).
2. **Vídeos longos:** até 4 h, processados em segundo plano, sem o limite de 15 min.
3. **Transcrição mais precisa em português do Brasil:** `language="pt"` explícito, o modelo escolhido pelo benchmark abaixo e, quando você informa `terms`, um prompt com contexto brasileiro e esses termos.

### Como funciona

O Claude Desktop inicia o servidor MCP a cada sessão e uma chamada de ferramenta não comporta horas de processamento. Por isso os trabalhos longos rodam no serviço **`watch-worker`** (compose, sempre ligado, mesma imagem), que processa uma fila em disco no volume **`watch-mcp-jobs`**. O servidor stdio só valida, enfileira e lê os resultados.

A v2 transcreve no seu próprio Whisper, **`watch-whisper-v2`**: mesma imagem fixada, modelos do mesmo volume em somente leitura, 8 GB. No benchmark, um modelo grande deixou o `watch-whisper` da v1 no limite de memória (4 GB). Com serviços separados, um trabalho longo nunca derruba as transcrições da v1.

Para cada trabalho, o worker:

1. baixa a URL (pelo mesmo proxy `netguard` da v1) ou abre o arquivo da caixa de entrada;
2. extrai o áudio e monta um índice de frames (cortes de cena mais uma grade de tempo, JPEG com lado maior de 512 px);
3. **apaga o vídeo**;
4. transcreve em blocos de ~5 min, cortados em silêncios, gravando cada bloco assim que termina. Se o worker reiniciar, ele retoma do bloco seguinte, sem refazer o que já estava pronto.

### Subir

```powershell
docker build -t watch-mcp:v2 .
docker compose -p watch-mcp up -d          # watch-whisper (v1, inalterado) + watch-whisper-v2 + watch-worker
```

A entrada `watch-v2` do `claude_desktop_config.json` (gerada por `entry_v2()` em `tests/e2e/desktop_entry.py`) usa os mesmos limites de container da v1 e monta também `watch-mcp-jobs` em `/jobs` e `D:/Inbox` em `/inbox` (somente leitura). Para usar outra pasta, defina `WATCH_INBOX_DIR` antes do `docker compose up` e ajuste o `--mount` da entrada.

### Fluxo de uso

1. `list_inbox()`: lista os vídeos e áudios da caixa de entrada (mp4, mkv, mov, webm, m4v, mp3, m4a, wav, ogg, opus).
2. `analyze_video(source, language="pt", quality="auto", terms=None)`: `source` é uma URL ou só o **nome** de um arquivo da caixa de entrada. Devolve um `job_id` na hora. `terms` são nomes próprios e jargões que aparecem no vídeo, para ajudar a transcrição. Pedir de novo o mesmo vídeo com os mesmos parâmetros reaproveita o trabalho existente.
3. `job_status(job_id)`: estado, etapa, progresso em % e estimativa. Em caso de falha, traz o erro em português.
4. `search_transcript(job_id, query)`: trechos com timestamps. A busca ignora acentos e maiúsculas.
5. `watch_video(job_id=..., start=..., end=..., max_frames=24, detail="balanced")`: frames e transcrição do trecho, na hora, a partir do cache, sem novo download nem retranscrição. Mantém o orçamento de ~800 KB de imagens.
6. `list_jobs()`: os trabalhos recentes.

`watch_video(url=...)` e `transcribe_video(url)` continuam iguais à v1 para vídeos de até 15 min. Um vídeo mais longo recebe a sugestão de usar `analyze_video`.

### Retenção do cache

Ficam em `watch-mcp-jobs`, por trabalho:
- `job.json`;
- `meta.json`, só com os campos de exibição;
- `transcript.json`, com os segmentos;
- `index.json` e `frames/`.

O vídeo e o áudio são apagados assim que deixam de ser necessários. Trabalhos concluídos ou com falha são apagados **7 dias** depois da última atualização. Se o cache passar de **20 GB**, os mais antigos saem primeiro. A limpeza roda a cada 10 min e nunca toca em trabalhos na fila ou em andamento. Tudo é configurável no serviço `watch-worker` do `docker-compose.yml`:

| Variável | Padrão | O quê |
|---|---|---|
| `WATCH_JOB_MAX_DURATION` | 14400 (4 h) | duração máxima da mídia |
| `WATCH_JOB_MAX_DOWNLOAD_BYTES` | 4 GiB | teto rígido de download **por tentativa** (contado pelo proxy). Um trabalho interrompido é retomado no máximo 3 vezes; o disco continua limitado por `WATCH_JOBS_MAX_BYTES` |
| `WATCH_JOB_TIMEOUT` | 28800 (8 h) | tempo total por trabalho |
| `WATCH_JOB_TTL` | 604800 (7 dias) | retenção depois da última atualização |
| `WATCH_JOBS_MAX_BYTES` | 20 GiB | teto do cache inteiro: conferido antes de cada etapa e vigiado durante o download, a extração de áudio e o índice (a etapa é interrompida e o trabalho falha com "sem espaço") |
| `WATCH_MODEL_FAST` / `WATCH_MODEL_ACCURATE` | ver benchmark | modelos do `quality="auto"` |

### Benchmark pt-BR

Todas as transcrições usam `language="pt"`. O áudio é sempre brasileiro:
- **TTS:** 20 frases com números, datas, valores em reais, nomes, gírias, TI e agro, em 5 vozes brasileiras (Piper e Kokoro).
- **Fala real:** um reel narrado, um episódio do Nerdologia com legenda oficial e 3 min de conversa espontânea do Podpah.

WER normalizado (números por extenso dos dois lados), tempo nesta CPU (Xeon E5-2680 v4, 8 CPUs):

| Modelo | WER TTS (com termos) | WER fala real (legenda oficial / ref. aproximada) | s por min de áudio |
|---|---|---|---|
| **large-v3-turbo** (deepdml, MIT) | **2,7%** | 11,3% / **9,2%** | 33 |
| medium (Systran, MIT) | 4,9% | **9,8%** / 13,3% | 34 |
| **small** (Systran, MIT) | 9,5% | 7,6% / 18,3% | **12** |
| distil-large-v3 pt-BR (freds0, MIT; em pedaços de 30 s) | 8,5% | 19,7% / 14,6% | 23 |
| medium pt (pierreguillou, Apache-2.0) | 4,9% | 7,5% / 48,9% | 29 |

Decisão do `quality="auto"`:
- **Curtos e reels:** `small`, o mais rápido com WER aceitável (7% no reel).
- **Trabalhos longos:** `large-v3-turbo`, o mais preciso em pt-BR no geral, a ~0,5× o tempo real.
- **Prompt brasileiro:** sozinho não melhorou de forma consistente; com `terms`, nomes e jargões melhoraram. Por isso ele só é enviado quando há `terms`.

A tabela completa (3 modos de prompt, números, nomes e termos perdidos, RAM) está em [tests/bench/RESULTS.md](tests/bench/RESULTS.md); método, referências e licenças, na seção "v2" do [NOTES-mcp.md](NOTES-mcp.md). Para reproduzir:

```powershell
.venv\Scripts\python tests\bench\fetch_models.py <pasta>      # modelos (revisões fixas), no host
#   instalar no volume watch-mcp-models: tests/bench/install_models.py num container descartável (ver NOTES)
.venv\Scripts\python tests\bench\fetch_real.py                # clipes reais
#   run_bench.py tts | refs | run <modelos> | run-pieces 30 <modelos>, num container na watch-mcp-net
.venv\Scripts\python tests\bench\report.py                    # tabela
```

### Promover a v2 para substituir a v1

Quando decidir:

1. **Copie a entrada:** em `%APPDATA%\Claude\claude_desktop_config.json`, troque os `args` da entrada `watch` pelos da `watch-v2` (ou apague a `watch` e renomeie `watch-v2` para `watch`). Faça um backup antes.
2. **Reinicie o Claude Desktop.**
3. **Opcional, para manter um único nome de imagem:** `docker tag watch-mcp:v2 watch-mcp:latest`. A imagem antiga continua acessível pelo ID enquanto não for removida, e dá para voltar com `docker tag <id-antigo> watch-mcp:latest`.
4. **Branches:** faça o merge do `mcp-v2` no branch que você usa como principal.

Para voltar atrás, restaure o backup do config e reinicie o Claude Desktop. O `watch-worker` pode continuar ligado; ele não interfere na v1.

### Testes da v2

```powershell
.venv\Scripts\python -m pytest -q                                                     # unitários (worker, fila, caixa de entrada, busca)
.venv\Scripts\python tests\e2e\e2e_v2_long.py --work <pasta-temp> --restart-at-chunk 3  # vídeo sintético longo + retomada
.venv\Scripts\python tests\e2e\e2e_mcp_stdio.py --entry v2 --url <reel> --out <pasta-temp>   # compatibilidade com reels
```

## Structure

```
.
├── skills/watch/                 # self-contained skill — copied as a unit by every installer
│   ├── SKILL.md                  # skill contract — the source of truth across all surfaces
│   └── scripts/
│       ├── watch.py              # entry point — orchestrates download → frames → transcript
│       ├── download.py           # yt-dlp wrapper
│       ├── frames.py             # ffmpeg frame extraction + auto-fps logic
│       ├── transcribe.py         # VTT parsing + dedupe + Whisper orchestration
│       ├── whisper.py            # Groq / OpenAI clients (pure stdlib)
│       ├── config.py             # shared config (~/.config/watch/.env)
│       ├── setup.py              # preflight + installer
│       └── build-skill.sh        # build dist/watch.skill for claude.ai upload (dev-only)
├── hooks/                        # SessionStart status hook (Claude Code only)
├── .claude-plugin/               # plugin.json + marketplace.json (Claude Code)
├── .codex-plugin/                # plugin.json — Codex/agents manifest ("skills": "./skills/")
├── .agents/plugins/              # marketplace.json — Agent Skills marketplace listing
├── AGENTS.md → CLAUDE.md         # generic-agent entry point
├── tests/                        # pytest suite (ffmpeg-synthesized clips, no network)
└── .github/workflows/            # release.yml — auto-builds watch.skill on tag push
```

## Develop

```bash
# Run the test suite (stdlib + pytest; ffmpeg required for frame tests):
python3 -m pytest -q

# Build the claude.ai upload bundle:
bash skills/watch/scripts/build-skill.sh      # → dist/watch.skill
```

Releasing: tag `vX.Y.Z`, push the tag. The workflow builds `dist/watch.skill` and attaches it to the GitHub release. Keep the version in sync across `skills/watch/SKILL.md`, `.claude-plugin/plugin.json`, and `.codex-plugin/plugin.json`.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Open source

MIT license.

Built on `yt-dlp`, `ffmpeg`, and Claude's multimodal `Read` tool. Whisper transcription via [Groq](https://groq.com) or [OpenAI](https://openai.com).

Built by Brad Bonanno — I make content about building with AI on [YouTube (@bradbonanno)](https://www.youtube.com/@bradbonanno), and build AI operating systems for businesses at [Solaris Automation](https://www.solarisautomation.io/). If `/watch` saves you from scrubbing through a video, come say hi on the channel.

## Star History

<a href="https://www.star-history.com/?repos=bradautomates%2Fclaude-video&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&legend=top-left" />
 </picture>
</a>

---

[github.com/bradautomates/claude-video](https://github.com/bradautomates/claude-video) · [@bradbonanno](https://www.youtube.com/@bradbonanno) · [Solaris Automation](https://www.solarisautomation.io/) · [LICENSE](LICENSE)
