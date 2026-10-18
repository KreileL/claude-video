# watch-mcp: MCP server (stdio) that downloads a video, extracts frames and
# transcribes it with the watch-whisper service. Built as watch-mcp:latest:
#   docker build -t watch-mcp:latest .
# Run on demand by Claude Desktop with --read-only, --tmpfs /tmp, --cap-drop ALL
# (see README, "MCP local").
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-mcp.txt /app/requirements-mcp.txt
RUN pip install --no-cache-dir -r /app/requirements-mcp.txt

COPY skills/watch /app/skills/watch
COPY mcp_server.py netguard.py /app/

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin watch

# The root filesystem is read-only at runtime; everything writable lives on the
# /tmp tmpfs (yt-dlp/deno caches, the per-call work dir).
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    DENO_DIR=/tmp/.deno \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 10001
WORKDIR /tmp
ENTRYPOINT ["python", "/app/mcp_server.py"]
