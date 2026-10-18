"""Tests for textsearch: accent/case-insensitive transcript search."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textsearch import normalize, search  # noqa: E402


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


SEGS = [
    seg(0.0, 2.0, "Bom dia a todos"),
    seg(2.0, 4.0, "Hoje vamos falar de ação e reação"),
    seg(4.0, 6.0, "O ônibus de São Paulo saiu"),
    seg(6.0, 8.0, "Entrei em contato com ele"),
    seg(8.0, 10.0, "Fim do vídeo"),
]


def test_normalize():
    assert normalize("Ônibus São-Paulo, AÇÃO!") == "onibus sao paulo acao"
    assert normalize("ç") == "c"
    assert normalize("  a \t b\n") == "a b"
    assert normalize("!!!") == ""


def test_accent_and_case_insensitive():
    hits, total = search(SEGS, "ACAO")
    assert total == 1 and hits[0]["start"] == 2.0
    hits, _ = search(SEGS, "onibus sao paulo")
    assert hits[0]["start"] == 4.0


def test_punctuation_in_query():
    hits, total = search(SEGS, "são-paulo!")
    assert total == 1 and hits[0]["start"] == 4.0


def test_word_boundary():
    hits, total = search(SEGS, "ato")
    assert (hits, total) == ([], 0)
    hits, total = search(SEGS, "contato")
    assert total == 1


def test_hit_shape_and_context():
    hits, _ = search(SEGS, "ônibus")
    h = hits[0]
    assert h == {
        "start": 4.0, "end": 6.0, "text": "O ônibus de São Paulo saiu",
        "before": "Hoje vamos falar de ação e reação",
        "after": "Entrei em contato com ele",
    }
    first, _ = search(SEGS, "bom dia")
    assert first[0]["before"] == ""
    last, _ = search(SEGS, "fim")
    assert last[0]["after"] == ""


def test_spanning_hit():
    segs = [seg(0, 2, "uma frase que"), seg(2, 4, "continua aqui"), seg(4, 6, "outra")]
    hits, total = search(segs, "que continua")
    assert total == 1
    h = hits[0]
    assert h["start"] == 0 and h["end"] == 4
    assert h["text"] == "uma frase que continua aqui"
    assert h["before"] == "" and h["after"] == "outra"


def test_spanning_not_duplicated_when_next_matches_alone():
    segs = [seg(0, 2, "x casa"), seg(2, 4, "casa verde")]
    hits, total = search(segs, "casa verde")
    assert total == 1 and hits[0]["start"] == 2


def test_fallback_all_words():
    hits, total = search(SEGS, "reação hoje")
    assert total == 1 and hits[0]["start"] == 2.0
    assert search(SEGS, "reação ônibus") == ([], 0)


def test_fallback_ignores_short_words_and_only_when_no_phrase():
    segs = [seg(0, 1, "a casa azul"), seg(1, 2, "azul a casa")]
    hits, total = search(segs, "casa azul")
    assert total == 1 and hits[0]["start"] == 0  # phrase wins, no fallback
    hits, total = search(segs, "x casa")  # 'x' has len 1, ignored in fallback
    assert total == 2


def test_limit_and_total():
    segs = [seg(i, i + 1, "gato preto") for i in range(5)]
    hits, total = search(segs, "gato", limit=2)
    assert total == 5 and [h["start"] for h in hits] == [0, 1]


def test_empty_query_error():
    with pytest.raises(ValueError, match="busca vazia"):
        search(SEGS, "  ?! ")


def test_empty_segments():
    assert search([], "algo") == ([], 0)
