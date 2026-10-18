"""Every repo-root module the entrypoints import is copied into the image."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINTS = ("mcp_server.py", "worker.py", "telegram_bot.py")


def local_imports(path: Path, seen: set[str]) -> set[str]:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
        for name in names:
            top = name.split(".")[0]
            if (ROOT / f"{top}.py").exists() and top not in seen:
                seen.add(top)
                local_imports(ROOT / f"{top}.py", seen)
    return seen


def test_every_local_module_is_copied_into_the_image():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("\r\n", "\n")
    joined = dockerfile.replace("\\\n", " ")  # line continuations
    copied = set(re.findall(r"([a-z_]+)\.py", " ".join(
        line for line in joined.splitlines() if line.startswith("COPY"))))
    needed = set()
    for entry in ENTRYPOINTS:
        needed |= local_imports(ROOT / entry, {entry[:-3]})
    assert needed - copied == set(), f"missing from the Dockerfile COPY: {sorted(needed - copied)}"
