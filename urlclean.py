"""Drop tracking parameters from shared video links before they are stored.

The origin goes into job.json, the cache key and the Holocron note. Only links of
Instagram, YouTube and TikTok are touched: known formats lose their whole query
(the video id is in the path, or in YouTube's v=); their other pages lose only
tracking names. Any other site is left exactly as given (a signed URL must not
change by a byte). Kept pieces stay as written, empty ones included. Never
changes the scheme, host or path.
"""
from __future__ import annotations

import re
from urllib.parse import unquote_plus, urlsplit, urlunsplit

# Formats whose video is fully identified by the path: the whole query is tracking.
PATH_ID_FORMATS = (
    (("instagram.com",), re.compile(r"^/(reels?|p|tv)/[A-Za-z0-9_-]+/?$")),
    (("tiktok.com",), re.compile(r"^/@[^/]+/video/\d+/?$")),
    (("vm.tiktok.com", "vt.tiktok.com"), re.compile(r"^/[A-Za-z0-9]+/?$")),
    (("tiktok.com",), re.compile(r"^/t/[A-Za-z0-9]+/?$")),
    (("tiktokv.com",), re.compile(r"^/share/video/\d+/?$")),
    (("youtu.be",), re.compile(r"^/[A-Za-z0-9_-]{6,}/?$")),
    (("youtube.com",), re.compile(r"^/shorts/[A-Za-z0-9_-]{6,}/?$")),
)
YOUTUBE_HOSTS = ("youtube.com", "youtube-nocookie.com")
WATCH_KEEP = {"v", "list", "index"}  # youtube.com/watch: what identifies the video
# Tracking by name on the platforms above (other pages of them, e.g. embeds and lives).
PLATFORM_TRACKING = {"igsh", "igshid", "dlrf", "si", "pp", "feature", "is_from_webapp", "sender_device",
                     "sender_web_id", "_r", "_t", "tt_from", "u_code", "share_app_id", "share_link_id",
                     "share_item_id", "social_sharing"}
PLATFORMS = ("instagram.com", "tiktok.com", "tiktokv.com", "youtube.com", "youtu.be",
             "youtube-nocookie.com")
GLOBAL_TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid"}  # dropped on the platforms above


def _host_is(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _keep(query: str, drop) -> str:
    """Filter raw `k=v` pieces, keeping the survivors exactly as they were written
    (empty pieces too); with nothing dropped, the query is returned unchanged."""
    pieces = query.split("&")
    out = [p for p in pieces if not (p and drop(unquote_plus(p.split("=", 1)[0]).lower()))]
    return query if len(out) == len(pieces) else "&".join(out)


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
        return url  # other sites: not ours to rewrite
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))
