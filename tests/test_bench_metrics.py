"""Tests for the pt-BR benchmark metrics (number words, normalization, WER)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
import metrics  # noqa: E402


@pytest.mark.parametrize("n,words", [
    (0, "zero"), (1, "um"), (10, "dez"), (14, "catorze"), (16, "dezesseis"),
    (21, "vinte e um"), (100, "cem"), (101, "cento e um"), (110, "cento e dez"),
    (200, "duzentos"), (999, "novecentos e noventa e nove"), (1000, "mil"),
    (1001, "mil e um"), (1050, "mil e cinquenta"), (1100, "mil e cem"),
    (1500, "mil e quinhentos"), (1984, "mil novecentos e oitenta e quatro"),
    (2024, "dois mil e vinte e quatro"), (10000, "dez mil"),
    (21000, "vinte e um mil"), (100000, "cem mil"), (1000000, "um milhão"),
    (2000000, "dois milhões"), (1500000, "um milhão e quinhentos mil"),
    (1000000000, "um bilhão"), (2000000000, "dois bilhões"),
    (999_999_999_999, "novecentos e noventa e nove bilhões novecentos e noventa e nove milhões novecentos e noventa e nove mil novecentos e noventa e nove"),
])
def test_number_to_words(n, words):
    assert metrics.number_to_words(n) == words


def test_number_to_words_negative():
    with pytest.raises(ValueError):
        metrics.number_to_words(-1)


def test_money():
    assert metrics.normalize_ptbr("R$ 1.500,00") == "mil e quinhentos reais"
    assert metrics.normalize_ptbr("R$1.500") == "mil e quinhentos reais"
    assert metrics.normalize_ptbr("R$ 2,50") == "dois reais e cinquenta centavos"
    assert metrics.normalize_ptbr("R$ 1") == "um real"
    assert metrics.normalize_ptbr("R$ 1,01") == "um real e um centavo"


def test_dates():
    assert metrics.normalize_ptbr("15/03/2024") == "quinze de março de dois mil e vinte e quatro"
    assert metrics.normalize_ptbr("1/5") == "primeiro de maio"


def test_time():
    assert metrics.normalize_ptbr("14h30") == "catorze horas e trinta"
    assert metrics.normalize_ptbr("14h") == "catorze horas"
    assert metrics.normalize_ptbr("14:30") == "catorze horas e trinta"
    assert metrics.normalize_ptbr("1h") == "uma hora"


def test_percent_decimal_ordinal():
    assert metrics.normalize_ptbr("3,5%") == "três vírgula cinco por cento"
    assert metrics.normalize_ptbr("10%") == "dez por cento"
    assert metrics.normalize_ptbr("2,5") == "dois vírgula cinco"
    assert metrics.normalize_ptbr("1º") == "primeiro"
    assert metrics.normalize_ptbr("1ª") == "primeira"
    assert metrics.normalize_ptbr("2º") == "segundo"


def test_integers_and_cleanup():
    assert metrics.normalize_ptbr("1.500 itens") == "mil e quinhentos itens"
    assert metrics.normalize_ptbr("Quatorze, guarda-chuva!") == "catorze guarda chuva"
    assert metrics.normalize_ptbr("Ação  é  AÇÃO") == "ação é ação"


def test_wer():
    assert metrics.wer("a casa é azul", "a casa é azul") == 0.0
    assert metrics.wer("a casa é azul", "a casa é verde") == 0.25
    assert metrics.wer("Olá, mundo!", "olá mundo") == 0.0
    assert metrics.wer("custa R$ 1.500,00", "custa mil e quinhentos reais") == 0.0
    assert metrics.wer("", "") == 0.0
    assert metrics.wer("", "algo") == 1.0


def test_wer_details():
    d = metrics.wer_details("a b c d", "a x c")
    assert d == {"wer": 0.5, "substitutions": 1, "deletions": 1,
                 "insertions": 0, "ref_words": 4}
    d = metrics.wer_details("a b", "a b c")
    assert d["insertions"] == 1 and d["wer"] == 0.5


def test_missing_terms():
    hyp = "O Doutor Silva chegou às 14h30 em São Paulo"
    assert metrics.missing_terms(["Silva", "são paulo", "Rio", "14h30", "Sil"], hyp) == ["Rio", "Sil"]


def test_scoreline_x_reads_as_a():
    assert metrics.normalize_ptbr("venceu por 2x1") == metrics.normalize_ptbr("venceu por dois a um")
