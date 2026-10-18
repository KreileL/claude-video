"""Tracking parameters are dropped before a link is stored (real links from the v3 acceptance)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urlclean import clean_url  # noqa: E402


@pytest.mark.parametrize("raw,clean", [
    # Acceptance 2 (the reel sent through Telegram) and its other share forms
    ("https://www.instagram.com/reel/DeX7nFXC_Hm/?dlrf=MXJ2OXVmYWx4MTl3aA==",
     "https://www.instagram.com/reel/DeX7nFXC_Hm/"),
    ("https://www.instagram.com/reel/DeX7nFXC_Hm/?igsh=MWQ1ZGUxMzBkMA==&utm_source=ig_web_copy_link",
     "https://www.instagram.com/reel/DeX7nFXC_Hm/"),
    ("https://www.instagram.com/p/AbC123/?img_index=1&igshid=x", "https://www.instagram.com/p/AbC123/"),
    # Acceptance 3 / e2e links, already clean: unchanged
    ("https://www.youtube.com/watch?v=GuPyhzLCUHE", "https://www.youtube.com/watch?v=GuPyhzLCUHE"),
    # YouTube share forms (the adjustments e2e used youtu.be/...?si=)
    ("https://youtu.be/x5SrLM1xJ3A?si=Ab12Cd34Ef56Gh78", "https://youtu.be/x5SrLM1xJ3A"),
    ("https://www.youtube.com/watch?v=9TdxDghfUEw&pp=ygUObWludXRvIGRhIHRlcnJh",
     "https://www.youtube.com/watch?v=9TdxDghfUEw"),
    ("https://www.youtube.com/watch?v=GuPyhzLCUHE&pp=ygU&si=x&t=42", "https://www.youtube.com/watch?v=GuPyhzLCUHE"),
    ("https://m.youtube.com/watch?feature=share&v=abc", "https://m.youtube.com/watch?v=abc"),
    ("https://www.youtube.com/shorts/Xyz987abc?si=q1w2e3", "https://www.youtube.com/shorts/Xyz987abc"),
    ("https://www.youtube.com/watch?v=abc&list=PL1&index=3&si=z", "https://www.youtube.com/watch?v=abc&list=PL1&index=3"),
    # TikTok
    ("https://www.tiktok.com/@canal/video/7345678901234567890?is_from_webapp=1&sender_device=pc&web_id=99",
     "https://www.tiktok.com/@canal/video/7345678901234567890"),
    ("https://vm.tiktok.com/ZMabc123/?_r=1&_t=8kXyz", "https://vm.tiktok.com/ZMabc123/"),
    # TikTok short forms recognized by yt-dlp (Codex v3 pass 7)
    ("https://www.tiktok.com/t/ZTRC5xgJp/?share_iid=1&sec_user_id=2&checksum=3&timestamp=4&user_id=5&web_id=6",
     "https://www.tiktok.com/t/ZTRC5xgJp/"),
    ("https://www.tiktokv.com/share/video/7345678901234567890/?_t=x&share_link_id=y",
     "https://www.tiktokv.com/share/video/7345678901234567890/"),
])
def test_tracking_parameters_are_dropped(raw, clean):
    assert clean_url(raw) == clean


# ── Codex v3 pass 6 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", [
    # parameters that may select content or authorize access on other sites stay
    "https://example.com/video.mp4?source=A",
    "https://example.com/v.mp4?checksum=1&timestamp=2&user_id=3&token=4",
    # kept parameters are not re-encoded (signed URLs)
    "https://cdn.example.com/v.mp4?name=a%20b&path=%2fx%2Fy&sig=AbC%3D",
    # other YouTube formats keep their functional parameters
    "https://www.youtube.com/embed/live_stream?channel=UC1234567890123456789012",
    "https://www.youtube.com/playlist?list=PL123",
])
def test_functional_parameters_and_their_encoding_survive(url):
    assert clean_url(url) == url


def test_only_tracking_names_go_on_other_pages_and_the_rest_is_verbatim():
    assert clean_url("https://www.youtube.com/embed/live_stream?channel=UC1&si=x") == \
        "https://www.youtube.com/embed/live_stream?channel=UC1"
    assert clean_url("https://www.instagram.com/stories/user/123/?igsh=x&story_media_id=5") == \
        "https://www.instagram.com/stories/user/123/?story_media_id=5"


def test_host_and_path_never_change():
    assert clean_url("https://www.instagram.com/reel/A_b-1/?igsh=1").startswith("https://www.instagram.com/reel/A_b-1/")
    assert clean_url("https://notyoutube.com/watch?v=1&si=2") == "https://notyoutube.com/watch?v=1&si=2"


# ── Codex v3 pass 7 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", [
    "https://example.com/v.mp4?resource=a%20b&&signature=abc&",
    "https://example.com/v.mp4?fbclid=1&name=a%20b&utm_x=2",  # other sites: not touched at all
    "https://vimeo.com/123?utm_campaign=x&h=abc#t=10",
])
def test_other_sites_are_left_byte_for_byte(url):
    assert clean_url(url) == url


def test_empty_pieces_survive_when_a_tracking_piece_goes():
    assert clean_url("https://www.youtube.com/embed/live_stream?channel=UC1&&si=x&") ==         "https://www.youtube.com/embed/live_stream?channel=UC1&&"
