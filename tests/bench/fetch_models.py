"""Download the pinned model files of the benchmark on the host.

Plain HTTPS with the host's own trust store, so the install container needs no
network for model files:

    python tests/bench/fetch_models.py <dest-dir>
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from sources import CONVERT, CONVERT_FILES, DIRECT


def download(repo: str, revision: str, names: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = out / name
        if target.exists():
            continue
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(f"{target}.part", "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # not every fine-tune ships every file
                continue
            raise
        Path(f"{target}.part").rename(target)
        print(f"{repo}: {name} {target.stat().st_size} bytes", flush=True)


def main(dest: Path) -> None:
    for repo, revision, _name, _license in CONVERT:
        download(repo, revision, CONVERT_FILES, dest / repo.replace("/", "--"))
    for repo, revision, _license, names in DIRECT:
        download(repo, revision, names, dest / repo.replace("/", "--"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
