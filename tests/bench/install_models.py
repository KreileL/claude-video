"""Install the benchmark models into the watch-whisper model cache (the
watch-mcp-models volume), where speaches lists them as local models.

- CONVERT fine-tunes are converted to CTranslate2 (float16) and installed as
  watch-mcp/<name>, with a model card that speaches accepts
  (library_name: ctranslate2, pipeline_tag: automatic-speech-recognition).
- DIRECT models are copied as they are, under their own id.

Inputs come from fetch_models.py (host), mounted read-only at /src; this runs
offline in a throwaway container (see tests/bench/README.md). ctranslate2
matches the version inside speaches 0.8.3, so the converted model.bin loads.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from sources import CONVERT, DIRECT

CACHE = Path("/models")
SRC = Path("/src")
WORK = Path("/work")


def place(repo_id: str, revision: str, files_dir: Path) -> None:
    """Write files_dir as the cached snapshot of repo_id@revision (HF cache layout)."""
    target = CACHE / f"models--{repo_id.replace('/', '--')}"
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "refs").mkdir(parents=True)
    (tmp / "refs" / "main").write_text(revision)
    shutil.copytree(files_dir, tmp / "snapshots" / revision)
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    subprocess.run(["chown", "-R", "1000:1000", str(target)], check=True)
    print(f"installed {repo_id}: {sorted(p.name for p in files_dir.iterdir())}", flush=True)


def installed(repo_id: str, revision: str) -> bool:
    return (CACHE / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision).is_dir()


def convert(repo: str, revision: str, name: str, license_id: str) -> None:
    import ctranslate2
    from transformers import WhisperTokenizerFast

    repo_id = f"watch-mcp/{name}"
    if installed(repo_id, revision):
        print(f"{repo_id}: already installed")
        return
    src = WORK / "src" / name
    shutil.copytree(SRC / repo.replace("/", "--"), src, dirs_exist_ok=True)
    if not (src / "tokenizer.json").exists():
        # Older fine-tunes ship only the slow tokenizer files; faster-whisper needs tokenizer.json.
        WhisperTokenizerFast.from_pretrained(src).save_pretrained(WORK / f"{name}-tok")
        shutil.copy(WORK / f"{name}-tok" / "tokenizer.json", src / "tokenizer.json")
    out = WORK / name
    subprocess.run(
        [
            "ct2-transformers-converter", "--model", str(src), "--output_dir", str(out),
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--quantization", "float16", "--force",
        ],
        check=True,
    )
    (out / "README.md").write_text(
        "---\n"
        "library_name: ctranslate2\n"
        "pipeline_tag: automatic-speech-recognition\n"
        f"license: {license_id}\n"
        "language:\n- pt\n"
        "tags:\n- automatic-speech-recognition\n- ctranslate2\n"
        f"base_model: {repo}\n"
        "---\n\n"
        f"CTranslate2 conversion of [{repo}](https://huggingface.co/{repo}) at revision "
        f"`{revision}`, made locally with ctranslate2 {ctranslate2.__version__} (float16 weights). "
        f"License: {license_id}, inherited from the source model.\n",
        encoding="utf-8",
    )
    place(repo_id, revision, out)


if __name__ == "__main__":
    for repo, revision, _license, _names in DIRECT:
        if not installed(repo, revision):
            place(repo, revision, SRC / repo.replace("/", "--"))
    for model in CONVERT:
        convert(*model)
