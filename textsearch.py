"""Accent- and case-insensitive transcript search for the MCP server."""
from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """Fold to lowercase ASCII-ish words: no accents, punctuation as single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    folded = stripped.casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in folded).split())


def _has(haystack: str, needle: str) -> bool:
    """Whole-word substring test on already-normalized text."""
    return f" {needle} " in f" {haystack} "


def search(segments: list[dict], query: str, limit: int = 20) -> tuple[list[dict], int]:
    """Return (hits[:limit], total_hits) in time order; ValueError on empty query."""
    q = normalize(query)
    if not q:
        raise ValueError("busca vazia")
    norms = [normalize(s["text"]) for s in segments]
    n = len(segments)
    found: list[tuple[int, int]] = []  # (first index, last index)
    for i in range(n):
        if _has(norms[i], q):
            found.append((i, i))
        elif i + 1 < n and not _has(norms[i + 1], q) and _has(f"{norms[i]} {norms[i + 1]}", q):
            found.append((i, i + 1))
    if not found:
        words = [w for w in q.split() if len(w) >= 2]
        if words:
            for i in range(n):
                toks = set(norms[i].split())
                if all(w in toks for w in words):
                    found.append((i, i))
    hits = []
    for i, j in found:
        text = segments[i]["text"] if i == j else f"{segments[i]['text']} {segments[j]['text']}"
        hits.append({
            "start": segments[i]["start"],
            "end": segments[j]["end"],
            "text": text,
            "before": segments[i - 1]["text"] if i > 0 else "",
            "after": segments[j + 1]["text"] if j + 1 < n else "",
        })
    return hits[:limit], len(hits)
