"""Writer for the Obsidian "00 - Entrada" folder (bind-mounted at /entrada).

Everything written here derives from hostile data (video titles, uploaders,
transcripts, LLM output). Rules: create-only (never overwrite, never delete
anything except our own staging dir .watch-tmp/<job_id>), confined (the realpath
of every target stays inside the mount root), atomic publish (stage in
<root>/.watch-tmp/<job_id>/, rename once to the final folder name).

Pure stdlib.
"""
from __future__ import annotations

import errno
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ENTRADA_DIR = Path(os.environ.get("WATCH_ENTRADA_DIR", "/entrada"))
TMP_NAME = ".watch-tmp"
TITLE_LIMIT = 80
KEY_FRAMES_MIN, KEY_FRAMES_MAX = 10, 20
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SRC_FRAME_RE = re.compile(r"^f_\d{5}\.jpg$")
OUT_FRAME_RE = re.compile(r"^frame-\d{2}-\d{2}-\d{2}(-\d+)?\.jpg$")
_BAD_CATS = {"Cc", "Cf", "Cs", "Co", "Cn"}
_RESERVED_RE = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])$")
_MAX_SUFFIX = 99


class EntradaError(Exception):
    """Failure writing to the vault folder; message in Portuguese, no file contents."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else ENTRADA_DIR


def _is_link(p: object) -> bool:
    return os.path.islink(p) or bool(getattr(os.path, "isjunction", lambda _: False)(p))


def _drop_bad(s: str, keep: str = "") -> str:
    return "".join(c for c in s if c in keep or unicodedata.category(c) not in _BAD_CATS)


def _one_line(raw: object) -> str:
    """Single-line text without control/format chars (newlines become spaces)."""
    s = re.sub(r"[\r\n\t]", " ", str(raw))
    return " ".join(_drop_bad(s).split())


def _component_ok(name: object) -> bool:
    return (isinstance(name, str) and bool(name) and len(name) <= 200
            and not any(c in name for c in "/\\\0") and name not in (".", "..")
            and not name.startswith("."))


def _directly_inside(child: object, parent: object) -> bool:
    return os.path.dirname(os.path.realpath(child)) == os.path.realpath(parent)


# ---------- names ----------

def sanitize_title(raw: object, limit: int = TITLE_LIMIT) -> str:
    """Safe single path component built from untrusted text."""
    s = unicodedata.normalize("NFC", str(raw))
    s = re.sub(r"[\r\n\t]", " ", s)
    s = _drop_bad(s)
    s = re.sub(r'[/\\:*?"<>|]', " ", s)
    s = re.sub(r"\.{2,}", ".", s)
    s = " ".join(s.split()).strip(" .")
    s = s[:limit].strip(" .")
    if _RESERVED_RE.match(s.split(".")[0].strip().upper()):
        s = "_" + s
    if not s:
        return "Vídeo"
    if s.startswith("."):
        s = "_" + s
    return s[:limit].strip(" .") or "Vídeo"


def index_filename(title: str) -> str:
    return sanitize_title(title) + ".md"


def folder_name(date: str, title: str, job_id: str) -> str:
    if not isinstance(date, str) or not DATE_RE.fullmatch(date):
        raise EntradaError("data inválida para o nome da pasta")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise EntradaError("data inválida para o nome da pasta") from None
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise EntradaError("job_id inválido")
    name = f"{date} - {sanitize_title(title)} ({job_id[:6]})"
    if not _component_ok(name) or len(name) > 150:
        raise EntradaError("nome de pasta inválido")
    return name


# ---------- filesystem ----------

def available(root: Path | None = None) -> bool:
    try:
        r = _root(root)
        return r.is_dir() and not _is_link(r) and os.access(r, os.W_OK)
    except (OSError, ValueError):
        return False


def _tmp_dir(root: Path) -> Path | None:
    """<root>/.watch-tmp when it is a real directory, else None."""
    tmp = root / TMP_NAME
    if _is_link(tmp) or not tmp.is_dir():
        return None
    return tmp


def _remove_ours(p: Path) -> None:
    """Remove one entry of our staging area without ever following links."""
    if _is_link(p):
        try:
            os.unlink(p)
        except OSError:
            os.rmdir(p)  # Windows junction
    elif p.is_dir():
        shutil.rmtree(p)
    else:
        os.unlink(p)


def clean_orphans(root: Path | None = None, *, max_age: float = 3600.0,
                  now: float | None = None) -> list[str]:
    """Remove stale <root>/.watch-tmp/<job_id> dirs left by a crashed worker."""
    tmp = _tmp_dir(_root(root))
    if tmp is None:
        return []
    now = time.time() if now is None else now
    removed: list[str] = []
    try:
        names = os.listdir(tmp)
    except OSError:
        return []
    for name in names:
        if not JOB_ID_RE.fullmatch(name):
            continue
        p = tmp / name
        try:
            if not _is_link(p) and not p.is_dir():
                continue
            if now - os.lstat(p).st_mtime <= max_age:
                continue
            _remove_ours(p)
            removed.append(name)
        except OSError:
            continue
    return removed


def staging(root: Path | None, job_id: str) -> Path:
    """Fresh, empty <root>/.watch-tmp/<job_id>."""
    r = _root(root)
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise EntradaError("job_id inválido")
    if not r.is_dir() or _is_link(r):
        raise EntradaError("a pasta de entrada não está disponível")
    tmp = r / TMP_NAME
    try:
        if not os.path.lexists(tmp):
            tmp.mkdir(mode=0o755)
        if _tmp_dir(r) is None:
            raise EntradaError("pasta temporária inválida na entrada")
        job = tmp / job_id
        if os.path.lexists(job):
            _remove_ours(job)
        job.mkdir(mode=0o755)
    except OSError:
        raise EntradaError("não foi possível preparar a pasta temporária") from None
    return job


def create_file(path: Path, data: bytes | str) -> None:
    """Create a new file exclusively; FileExistsError if it already exists."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        try:
            os.unlink(path)  # our own half-written file
        except OSError:
            pass
        raise


def _rename_noreplace(src: str, dst: str) -> None:
    """Rename that never replaces dst (FileExistsError if it exists)."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            fn = ctypes.CDLL(None, use_errno=True).renameat2
        except (OSError, AttributeError):
            fn = None
        if fn is not None:
            # AT_FDCWD = -100, RENAME_NOREPLACE = 1
            if fn(-100, os.fsencode(src), -100, os.fsencode(dst), 1) == 0:
                return
            err = ctypes.get_errno()
            if err in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(err, os.strerror(err), dst)
            if err not in (errno.EINVAL, errno.ENOSYS):
                raise OSError(err, os.strerror(err), dst)
    # Fallback: check then rename (POSIX rename could replace an empty dir; Windows never does).
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, "exists", dst)
    os.rename(src, dst)


def publish(root: Path | None, staged: Path, base_name: str) -> str:
    """Atomically move the staged dir to <root>/<base_name>[ (n)]; return the name used."""
    r = _root(root)
    if not _component_ok(base_name):
        raise EntradaError("nome de pasta inválido")
    tmp = r / TMP_NAME
    staged = Path(staged)
    if (_tmp_dir(r) is None or not _directly_inside(staged, tmp) or _is_link(staged)
            or not staged.is_dir()):
        raise EntradaError("pasta temporária fora da área permitida")
    for n in range(1, _MAX_SUFFIX + 1):
        candidate = base_name if n == 1 else f"{base_name} ({n})"
        target = r / candidate
        if os.path.realpath(target.parent) != os.path.realpath(r):
            raise EntradaError("destino fora da pasta de entrada")
        if os.path.lexists(target):
            continue
        try:
            _rename_noreplace(str(staged), str(target))
        except FileExistsError:
            continue
        except OSError:
            raise EntradaError("não foi possível publicar a pasta na entrada") from None
        if not _directly_inside(target, r):
            raise EntradaError("destino fora da pasta de entrada")
        return candidate
    raise EntradaError("nomes de pasta esgotados na entrada")


def add_file(root: Path | None, folder: str, name: str, data: bytes | str) -> str | None:
    """Add a new .md to an already published folder; None when the folder is unusable."""
    r = _root(root)
    if not _component_ok(folder):
        return None
    if not _component_ok(name) or not name.endswith(".md") or len(name) <= 3:
        raise EntradaError("nome de arquivo inválido")
    d = r / folder
    if not os.path.lexists(d) or _is_link(d) or not d.is_dir() or not _directly_inside(d, r):
        return None
    stem = name[:-3]
    for n in range(1, _MAX_SUFFIX + 1):
        candidate = name if n == 1 else f"{stem} ({n}).md"
        try:
            create_file(d / candidate, data)
        except FileExistsError:
            continue
        except OSError:
            raise EntradaError("não foi possível gravar o arquivo na entrada") from None
        return candidate
    raise EntradaError("nomes de arquivo esgotados na entrada")


# ---------- hostile text -> Markdown ----------

_BLOCK_RE = re.compile(r"<\s*(script|style|iframe)\b.*?<\s*/\s*\1\s*>", re.I | re.S)
_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]*)\)")
_SCHEME_RE = re.compile(r"\b(file|javascript|data|obsidian):", re.I)
_FENCE_RE = re.compile(r"^((?:\s|>)*)(```|~~~)", re.M)
_REFDEF_RE = re.compile(r"^(\s*)\[([^\]]*)\]:", re.M)


def _link(m: re.Match) -> str:
    bang, text, target = m.group(1), m.group(2), m.group(3)
    t = target.strip().lower().replace("<", "").replace(">", "").replace("&lt;", "").replace("&gt;", "")
    if bang or not t.startswith(("http://", "https://")):
        return text  # no remote images, no local/other-scheme targets
    return m.group(0)


def sanitize_markdown(text: object) -> str:
    """Neutralize hostile text before it goes into a note body."""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    s = _drop_bad(s, keep="\n\t")
    s = _BLOCK_RE.sub("", s)
    s = s.replace("<", "&lt;")
    s = s.replace("[[", "[ [").replace("]]", "] ]")
    s = _LINK_RE.sub(_link, s)
    s = _REFDEF_RE.sub(r"\1\\[\2]:", s)
    s = _SCHEME_RE.sub(lambda m: m.group(1) + " :", s)
    s = _FENCE_RE.sub(lambda m: m.group(1) + "'''", s)
    s = re.sub(r"`(\$?=)", r"` \1", s)  # Dataview inline queries
    return re.sub(r"\A-{3,}", "—", s)


# ---------- frames ----------

def _num(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        return None
    return float(v)


def hms(t: float) -> str:
    v = _num(t)
    s = int(v) if v is not None else 0
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def frame_filename(t: float) -> str:
    return "frame-" + hms(t).replace(":", "-") + ".jpg"


def pick_key_frames(frames: list[dict], n_min: int = KEY_FRAMES_MIN,
                    n_max: int = KEY_FRAMES_MAX) -> list[dict]:
    """Spread n_min..n_max frames over the whole duration, preferring scene cuts."""
    valid = []
    for f in frames if isinstance(frames, list) else []:
        if not isinstance(f, dict):
            continue
        t = _num(f.get("t"))
        if t is not None and isinstance(f.get("file"), str) and SRC_FRAME_RE.fullmatch(f["file"]):
            valid.append((t, f))
    valid.sort(key=lambda x: x[0])
    if len(valid) <= n_min:
        return [f for _, f in valid]
    scenes = sum(1 for _, f in valid if f.get("kind") == "scene")
    count = max(n_min, min(n_max, scenes // 50 + 10))
    tmax = valid[-1][0]
    width = tmax / count if tmax > 0 else 1.0
    buckets: list[list[tuple[float, dict]]] = [[] for _ in range(count)]
    for item in valid:
        buckets[min(int(item[0] / width), count - 1)].append(item)
    chosen: dict[int, tuple[float, dict]] = {}
    for i, items in enumerate(buckets):
        if not items:
            continue
        center = (i + 0.5) * width
        pool = [x for x in items if x[1].get("kind") == "scene"] or items
        best = min(pool, key=lambda x: abs(x[0] - center))
        chosen[id(best[1])] = best
    # Top up with the frame farthest from the ones already picked.
    rest = [x for x in valid if id(x[1]) not in chosen]
    while len(chosen) < n_min and rest:
        far = max(rest, key=lambda x: min(abs(x[0] - c[0]) for c in chosen.values()) if chosen else 0)
        chosen[id(far[1])] = far
        rest.remove(far)
    return [f for _, f in sorted(chosen.values(), key=lambda x: x[0])]


# ---------- notes ----------

def _md_line(raw: object) -> str:
    return " ".join(sanitize_markdown(raw).split())


def _q(v: object) -> str:
    """Safe YAML scalar: JSON string of single-line clean text."""
    return json.dumps(_one_line(v), ensure_ascii=False)


def render_transcript(title: str, segments: list[dict]) -> str:
    lines = []
    for seg in segments if isinstance(segments, list) else []:
        if not isinstance(seg, dict) or not isinstance(seg.get("text"), str):
            continue
        start = _num(seg.get("start"))
        text = _md_line(seg["text"])
        if start is None or not text:
            continue
        lines.append(f"[{hms(start)}] {text}")
    body = "\n".join(lines) if lines else "_(sem fala reconhecida)_"
    return f"# Transcrição — {_md_line(title)}\n\n{body}\n"


def _check_info(info: dict) -> tuple[str, str]:
    job_id, date = info.get("job_id"), info.get("date")
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise EntradaError("job_id inválido")
    if not isinstance(date, str) or not DATE_RE.fullmatch(date):
        raise EntradaError("data inválida")
    return job_id, date


def render_index(info: dict) -> str:
    job_id, date = _check_info(info)
    status = info.get("resumo_status")
    if status not in ("ok", "pendente", "desligado"):
        raise EntradaError("estado do resumo inválido")
    title = info.get("title", "")
    summary = info.get("summary")
    dur = _num(info.get("duration"))
    autor = _one_line(info.get("autor") or info.get("canal") or "")
    modelo = _one_line(info.get("modelo") or "")
    idioma = _one_line(info.get("idioma") or "")
    origem = info.get("origem", "")
    fm = ["---", f"title: {_q(title)}", "tipo: entrada", "subtipo: video", "status: rascunho",
          f"origem: {_q(origem)}", f"job_id: {_q(job_id)}",
          f"data_processamento: {date}", f"data_criacao: {date}"]
    if dur is not None:
        fm.append(f"duracao: {_q(hms(dur))}")
    if autor:
        fm.append(f"autor: {_q(autor)}")
    fm += [f"modelo_transcricao: {_q(modelo)}", f"idioma: {_q(idioma)}", f"resumo: {status}",
           "metodo_ingestao: transcricao", "tags:", "  - entrada", "  - video", "---", ""]

    out = [f"# {_md_line(title)}", "", "## Resumo"]
    if status == "ok" and isinstance(summary, str) and summary.strip():
        out.append(sanitize_markdown(summary).strip())
    elif status == "desligado":
        out.append("> [!info] Resumo desligado para esta origem.")
    else:
        out.append("> [!info] Resumo pendente\n> O resumo automático ainda não foi gerado; "
                   "o worker tenta de novo e, quando conseguir, cria `Resumo.md` nesta pasta.")
    out += ["", "## Pontos principais"]
    if isinstance(summary, str) and summary.strip():
        out.append("Ver o resumo acima.")
    else:
        pts = []
        for h in info.get("highlights") or []:
            t = _num(h.get("t")) if isinstance(h, dict) else None
            text = _md_line(h.get("text", "")) if isinstance(h, dict) else ""
            if t is not None and text:
                pts.append(f"- [{hms(t)}] {text}")
        out.append("\n".join(pts) if pts else "_(ver a transcrição)_")
    out += ["", "## Fonte", f"- Origem: {_md_line(origem)}",
            f"- Duração: {hms(dur) if dur is not None else 'desconhecida'}"]
    if autor:
        out.append(f"- Autor/canal: {_md_line(autor)}")
    out.append(f"- Transcrição: modelo {_md_line(modelo)}, idioma {_md_line(idioma)}")
    frames = []
    for f in info.get("frames") or []:
        t = _num(f.get("t")) if isinstance(f, dict) else None
        if t is not None and isinstance(f.get("file"), str) and OUT_FRAME_RE.fullmatch(f["file"]):
            frames.append(f"![{hms(t)}](frames/{f['file']})")
    out += ["", "## Arquivos", "- [Transcrição](Transcrição.md)", f"- Frames-chave ({len(frames)}):"]
    out += frames
    out += ["", "## Como reprocessar", "Peça a uma LLM com o MCP `watch` conectado:",
            f'- `search_transcript(job_id="{job_id}", query="...")` para achar trechos;',
            f'- `watch_video(job_id="{job_id}", start="HH:MM:SS", end="HH:MM:SS")` para ver frames '
            "e a transcrição de um trecho.",
            "O processamento fica arquivado no worker (sem o vídeo); não é preciso baixar de novo."]
    return "\n".join(fm + out) + "\n"


def render_resumo(info: dict, summary: str) -> str:
    job_id, date = _check_info(info)
    index_file = info.get("index_file")
    if not _component_ok(index_file) or not index_file.endswith(".md"):
        raise EntradaError("nome da nota índice inválido")
    title = info.get("title", "")
    fm = ["---", f"title: {_q('Resumo — ' + _one_line(title))}", "tipo: entrada",
          "subtipo: resumo-video", "status: rascunho", f"job_id: {_q(job_id)}",
          f"data_criacao: {date}", "resumo: ok", "metodo_ingestao: transcricao", "---", ""]
    body = [f"# Resumo — {_md_line(title)}", "", sanitize_markdown(summary).strip(), "",
            f"[Nota índice]({quote(index_file, safe='')})"]
    return "\n".join(fm + body) + "\n"
