"""pt-BR text normalization and WER for the Whisper benchmark."""
from __future__ import annotations

import re

_UNITS = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito",
          "nove", "dez", "onze", "doze", "treze", "catorze", "quinze", "dezesseis",
          "dezessete", "dezoito", "dezenove"]
_TENS = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta",
         "oitenta", "noventa"]
_HUNDREDS = ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
             "seiscentos", "setecentos", "oitocentos", "novecentos"]
_SCALES = [(10**9, "um bilhão", "bilhões"), (10**6, "um milhão", "milhões"),
           (1000, "mil", "mil")]
_MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
           "agosto", "setembro", "outubro", "novembro", "dezembro"]
_ORD_M = ["primeiro", "segundo", "terceiro", "quarto", "quinto", "sexto",
          "sétimo", "oitavo", "nono", "décimo"]
_ORD_F = ["primeira", "segunda", "terceira", "quarta", "quinta", "sexta",
          "sétima", "oitava", "nona", "décima"]
_MAX = 999_999_999_999


def _below_1000(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    if n == 100:
        return "cem"
    h, r = divmod(n, 100)
    parts = [_HUNDREDS[h]] if h else []
    if r:
        t, u = divmod(r, 10)
        parts.append(_UNITS[r] if r < 20 else _TENS[t] + (f" e {_UNITS[u]}" if u else ""))
    return " e ".join(parts)


def number_to_words(n: int) -> str:
    """Brazilian Portuguese cardinal for 0..999_999_999_999."""
    if n < 0 or n > _MAX:
        raise ValueError(f"out of range: {n}")
    if n < 1000:
        return _below_1000(n)
    for scale, one, many in _SCALES:
        if n >= scale:
            q, rest = divmod(n, scale)
            head = one if q == 1 else f"{number_to_words(q)} {many}"
            if not rest:
                return head
            sep = " e " if rest < 100 or rest % 100 == 0 else " "
            return head + sep + number_to_words(rest)
    raise AssertionError("unreachable")


_NUM = r"\d{1,3}(?:\.\d{3})+|\d+"


def _int(s: str) -> int:
    return int(s.replace(".", ""))


def _words(s: str) -> str:
    v = _int(s)
    return number_to_words(v) if v <= _MAX else s


def _fem(words: str) -> str:
    """Feminine form of a leading 'um'/'dois' (for hours)."""
    return re.sub(r"^um\b", "uma", re.sub(r"^dois\b", "duas", words))


def _money(m: re.Match) -> str:
    reais = _int(m.group(1))
    out = f"{number_to_words(reais)} {'real' if reais == 1 else 'reais'}"
    if m.group(2):
        cents = int(m.group(2).ljust(2, "0"))
        if cents:
            out += f" e {number_to_words(cents)} {'centavo' if cents == 1 else 'centavos'}"
    return out


def _date(m: re.Match) -> str:
    day, month = int(m.group(1)), int(m.group(2))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    out = f"{'primeiro' if day == 1 else number_to_words(day)} de {_MONTHS[month - 1]}"
    if m.group(3):
        out += f" de {number_to_words(int(m.group(3)))}"
    return out


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), m.group(2)
    if h > 23 or (mi and int(mi) > 59):
        return m.group(0)
    out = f"{_fem(number_to_words(h))} {'hora' if h == 1 else 'horas'}"
    if mi and int(mi):
        out += f" e {number_to_words(int(mi))}"
    return out


def _decimal_words(whole: str, frac: str) -> str:
    zeros = len(frac) - len(frac.lstrip("0"))
    rest = frac.lstrip("0")
    fw = " ".join(["zero"] * zeros + ([number_to_words(int(rest))] if rest else []))
    return f"{_words(whole)} vírgula {fw}"


def _percent(m: re.Match) -> str:
    base = _decimal_words(m.group(1), m.group(2)) if m.group(2) else _words(m.group(1))
    return base + " por cento"


def _ordinal(m: re.Match) -> str:
    n = int(m.group(1))
    if 1 <= n <= 10:
        return (_ORD_F if m.group(2) == "ª" else _ORD_M)[n - 1]
    return number_to_words(n)


def normalize_ptbr(text: str) -> str:
    """Casefold and spell out numbers/money/dates/etc.; drop punctuation, keep accents."""
    t = text.casefold()
    t = re.sub(rf"r\$\s*({_NUM})(?:,(\d{{1,2}}))?(?!\d)", _money, t)
    t = re.sub(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{4}))?(?![\d/])", _date, t)
    t = re.sub(r"(?<!\d)(\d{1,2})h(\d{2})?(?!\w)", _time, t)
    t = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", _time, t)
    t = re.sub(rf"({_NUM})(?:,(\d+))?\s*%", _percent, t)
    t = re.sub(r"(\d+)\s*([ºª])", _ordinal, t)
    t = re.sub(rf"({_NUM}),(\d+)", lambda m: _decimal_words(m.group(1), m.group(2)), t)
    t = re.sub(_NUM, lambda m: _words(m.group(0)), t)
    t = re.sub(r"\bquatorze\b", "catorze", t)
    t = re.sub(r"[^\w\s]|_", " ", t)
    return " ".join(t.split())


def wer_details(ref: str, hyp: str) -> dict:
    """Word-level Levenshtein alignment counts between normalized texts."""
    r, h = normalize_ptbr(ref).split(), normalize_ptbr(hyp).split()
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (r[i - 1] != h[j - 1]),
                          d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, s, dl, ins = n, m, 0, 0, 0
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            s += r[i - 1] != h[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    score = (0.0 if not m else 1.0) if not n else d[n][m] / n
    return {"wer": score, "substitutions": s, "deletions": dl,
            "insertions": ins, "ref_words": n}


def wer(ref: str, hyp: str) -> float:
    """Word error rate of hyp against ref (both normalized)."""
    return wer_details(ref, hyp)["wer"]


def missing_terms(terms: list[str], hyp: str) -> list[str]:
    """Terms not present in hyp as a whole-word sequence."""
    padded = f" {normalize_ptbr(hyp)} "
    return [t for t in terms if f" {normalize_ptbr(t)} " not in padded]
