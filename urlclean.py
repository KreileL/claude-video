"""Drop tracking parameters from shared video links before they are stored.

The origin goes into job.json, the cache key and the Holocron note. Only known
link formats lose their whole query (the video id is in the path, or in YouTube's
v=); elsewhere only parameters that are tracking everywhere go. Kept parameters
stay byte for byte (no re-encoding: signed URLs keep working). Never changes the
scheme, host or path.
"""
from __future__ import annotations

import re
from urllib.parse import unquote_plus, urlsplit, urlunsplit

# Formats whose video is fully identified by the path: the whole query is tracking.
PATH_ID_FORMATS = (
    (("instagram.com",), re.compile(r"^/(reels?|p|tv)/[A-Za-z0-9_-]+/?$")),
    (("tiktok.com",), re.compile(r"^/@[^/]+/video/\d+/?$")),
    (("vm.tiktok.com", "vt.tiktok.com"), re.compile(r"^/[A-Za-z0-9]+/?$")),
    (("youtu.be",), re.compile(r"^/[A-Za-z0-9_-]{6,}/?$")),
    (("youtube.com",), re.compile(r"^/shorts/[A-Za-z0-9_-]{6,}/?$")),
)
YOUTUBE_HOSTS = ("youtube.com", "youtube-nocookie.com")
WATCH_KEEP = {"v", "list", "index"}  # youtube.com/watch: what identifies the video
# Tracking by name on the platforms above (other pages of them, e.g. embeds and lives).
PLATFORM_TRACKING = {"igsh", "igshid", "dlrf", "si", "pp", "feature", "is_from_webapp", "sender_device",
                     "sender_web_id", "_r", "_t", "tt_from", "u_code", "share_app_id", "share_link_id",
                     "share_item_id", "social_sharing"}
PLATFORMS = ("instagram.com", "tiktok.com", "youtube.com", "youtu.be", "youtube-nocookie.com")
# Tracking everywhere.
GLOBAL_TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def _host_is(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _keep(query: str, drop) -> str:
    """Filter raw `k=v` pieces, keeping the survivors exactly as they were written."""
    out = []
    for piece in query.split("&"):
        if not piece:
            continue
        name = unquote_plus(piece.split("=", 1)[0]).lower()
        if not drop(name):
            out.append(piece)
    return "&".join(out)


def _global(name: str) -> bool:
    return name in GLOBAL_TRACKING or name.startswith("utm_")


def clean_url(url: str) -> str:
    """The same URL without tracking parameters (and without a fragment)."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    query = parts.query
    if any(_host_is(host, hosts) and pattern.match(parts.path) for hosts, pattern in PATH_ID_FORMATS):
        query = ""
    elif _host_is(host, YOUTUBE_HOSTS) and parts.path in ("/watch", "/watch/"):
        query = _keep(query, lambda n: n not in WATCH_KEEP)
    elif _host_is(host, PLATFORMS):
        query = _keep(query, lambda n: n in PLATFORM_TRACKING or _global(n))
    else:
        query = _keep(query, _global)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))
