"""Tracking parameters are dropped before a link is stored (real links from the v3 acceptance)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urlclean import clean_url  # noqa: E402


@pytest.mark.parametrize("raw,clean", [
    # Acceptance 2 (the reel sent through Telegram)
    ("https://www.instagram.com/reel/DeX7nFXC_Hm/?dlrf=MXJ2OXVmYWx4MTl3aA==",
     "https://www.instagram.com/reel/DeX7nFXC_Hm/"),
    ("https://www.instagram.com/reel/DeX7nFXC_Hm/?igsh=MWQ1ZGUxMzBkMA==&utm_source=ig_web_copy_link",
     "https://www.instagram.com/reel/DeX7nFXC_Hm/"),
    ("https://www.instagram.com/p/AbC123/?img_index=1&igshid=x", "https://www.instagram.com/p/AbC123/"),
    # Acceptance 3 / e2e links, already clean: unchanged
    ("https://www.youtube.com/watch?v=GuPyhzLCUHE", "https://www.youtube.com/watch?v=GuPyhzLCUHE"),
    ("https://www.youtube.com/watch?v=TQlYHUhCWSc", "https://www.youtube.com/watch?v=TQlYHUhCWSc"),
    # YouTube share forms
    ("https://youtu.be/GuPyhzLCUHE?si=Ab12Cd34Ef56", "https://youtu.be/GuPyhzLCUHE"),
    ("https://www.youtube.com/watch?v=GuPyhzLCUHE&pp=ygUMbWludXRvIGRhIHRlcnJh&si=x&t=42",
     "https://www.youtube.com/watch?v=GuPyhzLCUHE"),
    ("https://m.youtube.com/watch?feature=share&v=abc", "https://m.youtube.com/watch?v=abc"),
    ("https://www.youtube.com/shorts/Xyz987?si=q1w2e3", "https://www.youtube.com/shorts/Xyz987"),
    ("https://www.youtube.com/watch?v=abc&list=PL1&index=3&si=z", "https://www.youtube.com/watch?v=abc&list=PL1&index=3"),
    # TikTok
    ("https://www.tiktok.com/@canal/video/7345678901234567890?is_from_webapp=1&sender_device=pc&web_id=99",
     "https://www.tiktok.com/@canal/video/7345678901234567890"),
    ("https://vm.tiktok.com/ZMabc123/?_r=1&_t=8kXyz", "https://vm.tiktok.com/ZMabc123/"),
    # Other sites: only tracking names go, the rest stays; fragments go
    ("https://vimeo.com/123?utm_campaign=x&h=abc#t=10", "https://vimeo.com/123?h=abc"),
    ("https://example.com/v.mp4?fbclid=1&token=keep", "https://example.com/v.mp4?token=keep"),
])
def test_tracking_parameters_are_dropped(raw, clean):
    assert clean_url(raw) == clean


def test_host_and_path_never_change():
    url = "https://www.instagram.com/reel/A_b-1/?igsh=1"
    assert clean_url(url).startswith("https://www.instagram.com/reel/A_b-1/")
    assert clean_url("https://notyoutube.com/watch?v=1&si=2") == "https://notyoutube.com/watch?v=1"
