"""Drop tracking parameters from shared video links before they are stored.

The origin goes into job.json, the cache key and the Holocron note: only what
identifies the video stays (YouTube's v=, the path of an Instagram reel or a
TikTok video). Pure stdlib; never changes the host or the path.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Hosts whose video id is in the path: every query parameter is tracking.
PATH_ID_HOSTS = ("instagram.com", "tiktok.com")
# YouTube: keep only these (the video id; list/index identify a playlist entry).
YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
YOUTUBE_KEEP = {"v", "list", "index"}
# Anywhere else: parameters that are tracking by name.
TRACKING = {"igsh", "igshid", "dlrf", "si", "pp", "feature", "fbclid", "gclid", "mc_cid", "mc_eid",
            "ref", "ref_src", "_r", "_t", "is_from_webapp", "sender_device", "sender_web_id",
            "share_app_id", "share_link_id", "share_item_id", "social_sharing", "tt_from", "u_code",
            "checksum", "timestamp", "user_id", "web_id", "embed_source", "source", "refer"}


def _host_is(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _tracking(name: str) -> bool:
    n = name.lower()
    return n in TRACKING or n.startswith(("utm_", "share_"))


def clean_url(url: str) -> str:
    """The same URL without tracking parameters and without a fragment."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if _host_is(host, PATH_ID_HOSTS):
        query = []
    elif _host_is(host, YOUTUBE_HOSTS):
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k in YOUTUBE_KEEP]
    else:
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _tracking(k)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
