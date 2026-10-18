"""Tests for jobstore: on-disk job queue/cache (tmp_path only, never /jobs)."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jobstore  # noqa: E402
from jobstore import JobError  # noqa: E402


def backdate(root, job_id, **fields):
    """Rewrite job.json fields directly (simulating the worker at an earlier time)."""
    p = root / job_id / "job.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d.update(fields)
    p.write_text(json.dumps(d), encoding="utf-8")


def make(root, state="queued", **fields):
    job = jobstore.create_job(fields, root=root)
    if state != "queued":
        jobstore.update_job(job["id"], root=root, state=state)
    return job["id"]


# ---------- ids and paths ----------

def test_new_job_id_format_and_uniqueness():
    ids = {jobstore.new_job_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(jobstore.JOB_ID_RE.fullmatch(i) for i in ids)


@pytest.mark.parametrize("bad", [
    "../x", "A" * 32, "a" * 31, "a" * 33, "", "a" * 32 + "\n", "..", "a/b" + "0" * 29,
    "a" * 31 + "\\", None, 123, b"a" * 32,
])
def test_job_path_rejects_bad_ids(tmp_path, bad):
    with pytest.raises(JobError, match="job_id inválido"):
        jobstore.job_path(bad, root=tmp_path)


def test_job_path_ok(tmp_path):
    jid = "0123456789abcdef" * 2
    assert jobstore.job_path(jid, root=tmp_path) == tmp_path / jid


# ---------- json io ----------

def test_write_json_atomic_roundtrip_no_tmp_left(tmp_path):
    p = tmp_path / "x.json"
    jobstore.write_json_atomic(p, {"a": "ção", "n": 1})
    assert "ção" in p.read_text(encoding="utf-8")  # ensure_ascii=False
    assert jobstore.read_json(p) == {"a": "ção", "n": 1}
    jobstore.write_json_atomic(p, {"a": 2})
    assert jobstore.read_json(p) == {"a": 2}
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_write_json_atomic_failure_cleans_tmp(tmp_path):
    p = tmp_path / "x.json"
    jobstore.write_json_atomic(p, {"ok": True})
    with pytest.raises(TypeError):
        jobstore.write_json_atomic(p, {"bad": object()})
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]
    assert jobstore.read_json(p) == {"ok": True}


def test_read_json_missing_and_corrupt(tmp_path):
    assert jobstore.read_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert jobstore.read_json(bad) is None
    bad.write_bytes(b"\xff\xfe\x00")
    assert jobstore.read_json(bad) is None


# ---------- create / load / update ----------

def test_create_job_fields_and_dir(tmp_path):
    job = jobstore.create_job({"source": "url", "cache_key": "k"}, root=tmp_path)
    assert jobstore.JOB_ID_RE.fullmatch(job["id"])
    assert job["state"] == "queued" and job["progress"] == 0.0
    assert job["stage"] == "na fila" and job["error"] is None
    assert job["source"] == "url" and job["cache_key"] == "k"
    assert isinstance(job["created"], float) and job["updated"] == job["created"]
    d = tmp_path / job["id"]
    assert d.is_dir()
    if os.name == "posix":
        assert (d.stat().st_mode & 0o777) == 0o700
    assert jobstore.load_job(job["id"], root=tmp_path) == job


def test_load_job_missing(tmp_path):
    with pytest.raises(JobError, match="trabalho não encontrado"):
        jobstore.load_job("0" * 32, root=tmp_path)


def test_update_job(tmp_path):
    jid = make(tmp_path)
    before = jobstore.load_job(jid, root=tmp_path)
    after = jobstore.update_job(jid, root=tmp_path, state="running", progress=0.5)
    assert after["state"] == "running" and after["progress"] == 0.5
    assert after["updated"] >= before["updated"]
    assert jobstore.load_job(jid, root=tmp_path) == after


def test_update_job_missing(tmp_path):
    with pytest.raises(JobError, match="trabalho não encontrado"):
        jobstore.update_job("0" * 32, root=tmp_path, state="done")


# ---------- listing ----------

def test_list_jobs_order_and_skips(tmp_path):
    a = make(tmp_path)
    b = make(tmp_path)
    backdate(tmp_path, a, created=100.0)
    backdate(tmp_path, b, created=200.0)
    # foreign / corrupt / mismatched / non-dict / not-a-dir
    (tmp_path / "lost+found").mkdir()
    (tmp_path / ("c" * 32)).mkdir()
    (tmp_path / ("c" * 32) / "job.json").write_text("{oops", encoding="utf-8")
    (tmp_path / ("d" * 32)).mkdir()
    (tmp_path / ("d" * 32) / "job.json").write_text("[1]", encoding="utf-8")
    (tmp_path / ("e" * 32)).mkdir()
    (tmp_path / ("e" * 32) / "job.json").write_text(json.dumps({"id": "f" * 32}), encoding="utf-8")
    (tmp_path / ("9" * 32)).write_text("file, not dir", encoding="utf-8")
    (tmp_path / ("8" * 32)).mkdir()  # no job.json
    assert [j["id"] for j in jobstore.list_jobs(root=tmp_path)] == [b, a]


def test_list_jobs_missing_root(tmp_path):
    assert jobstore.list_jobs(root=tmp_path / "nope") == []


def test_list_jobs_skips_symlink(tmp_path):
    real = make(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link_id = "b" * 32
    (outside / "job.json").write_text(json.dumps({"id": link_id, "state": "queued"}), encoding="utf-8")
    try:
        os.symlink(outside, tmp_path / link_id, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert [j["id"] for j in jobstore.list_jobs(root=tmp_path)] == [real]


# ---------- cache / queue ----------

def test_find_cached(tmp_path):
    old = make(tmp_path, cache_key="k")
    new = make(tmp_path, cache_key="k")
    failed = make(tmp_path, state="failed", cache_key="k")
    other = make(tmp_path, cache_key="z")
    backdate(tmp_path, old, created=1.0)
    backdate(tmp_path, new, created=2.0)
    backdate(tmp_path, failed, created=3.0)
    assert jobstore.find_cached("k", root=tmp_path)["id"] == new
    assert jobstore.find_cached("z", root=tmp_path)["id"] == other
    assert jobstore.find_cached("none", root=tmp_path) is None
    jobstore.update_job(new, root=tmp_path, state="failed")
    jobstore.update_job(old, root=tmp_path, state="done")
    assert jobstore.find_cached("k", root=tmp_path)["id"] == old


def test_next_job_prefers_running_then_oldest_queued(tmp_path):
    assert jobstore.next_job(root=tmp_path) is None
    q1, q2 = make(tmp_path), make(tmp_path)
    backdate(tmp_path, q1, created=10.0)
    backdate(tmp_path, q2, created=5.0)
    assert jobstore.next_job(root=tmp_path)["id"] == q2
    r1, r2 = make(tmp_path, state="running"), make(tmp_path, state="running")
    backdate(tmp_path, r1, created=50.0)
    backdate(tmp_path, r2, created=40.0)
    assert jobstore.next_job(root=tmp_path)["id"] == r2
    make(tmp_path, state="done")
    for j in (r1, r2):
        jobstore.update_job(j, root=tmp_path, state="done")
    assert jobstore.next_job(root=tmp_path)["id"] == q2


# ---------- dir_size ----------

def test_dir_size(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "a").write_bytes(b"x" * 100)
    (d / "sub" / "b").write_bytes(b"y" * 50)
    assert jobstore.dir_size(d) == 150
    assert jobstore.dir_size(tmp_path / "missing") == 0


def test_dir_size_ignores_symlinks(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"z" * 1000)
    d = tmp_path / "d"
    d.mkdir()
    (d / "a").write_bytes(b"x" * 10)
    try:
        os.symlink(big, d / "link")
        os.symlink(tmp_path, d / "dirlink", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert jobstore.dir_size(d) == 10


# ---------- cleanup ----------

def test_cleanup_ttl(tmp_path):
    old_done = make(tmp_path, state="done")
    old_failed = make(tmp_path, state="failed")
    fresh_done = make(tmp_path, state="done")
    old_queued = make(tmp_path)
    old_running = make(tmp_path, state="running")
    for j in (old_done, old_failed, old_queued, old_running):
        backdate(tmp_path, j, updated=1000.0)
    backdate(tmp_path, fresh_done, updated=9500.0)
    removed = jobstore.cleanup(root=tmp_path, ttl_seconds=1000, max_bytes=10**9, now=10000.0)
    assert sorted(removed) == sorted([old_done, old_failed])
    left = {j["id"] for j in jobstore.list_jobs(root=tmp_path)}
    assert left == {fresh_done, old_queued, old_running}


def test_cleanup_cap_removes_oldest_finished_only(tmp_path):
    ids = [make(tmp_path, state="done") for _ in range(3)]
    active = make(tmp_path, state="running")
    for i, j in enumerate(ids):
        backdate(tmp_path, j, updated=100.0 + i)
        (tmp_path / j / "blob").write_bytes(b"x" * 10_000)
    (tmp_path / active / "blob").write_bytes(b"x" * 10_000)
    one = jobstore.dir_size(tmp_path / ids[0])
    # keep room for the active job plus roughly one finished job
    cap = jobstore.dir_size(tmp_path / active) + one + 100
    removed = jobstore.cleanup(root=tmp_path, ttl_seconds=10**9, max_bytes=cap, now=200.0)
    assert removed == [ids[0], ids[1]]
    assert (tmp_path / active).exists() and (tmp_path / ids[2]).exists()


def test_cleanup_cap_never_deletes_active_even_if_over(tmp_path):
    q = make(tmp_path)
    r = make(tmp_path, state="running")
    for j in (q, r):
        (tmp_path / j / "blob").write_bytes(b"x" * 10_000)
    assert jobstore.cleanup(root=tmp_path, ttl_seconds=0, max_bytes=1, now=10**10) == []
    assert (tmp_path / q).exists() and (tmp_path / r).exists()


def test_cleanup_orphans(tmp_path):
    orphan_old = "a" * 32
    orphan_new = "b" * 32
    orphan_corrupt = "c" * 32
    foreign = tmp_path / "keepme"
    for n in (orphan_old, orphan_new, orphan_corrupt):
        (tmp_path / n).mkdir()
        (tmp_path / n / "stuff").write_bytes(b"x")
    (tmp_path / orphan_corrupt / "job.json").write_text("{bad", encoding="utf-8")
    foreign.mkdir()
    now = 10_000_000.0
    for n in (orphan_old, orphan_corrupt):
        os.utime(tmp_path / n, (now - 7200, now - 7200))
    os.utime(tmp_path / orphan_new, (now - 60, now - 60))
    os.utime(foreign, (now - 7200, now - 7200))
    removed = jobstore.cleanup(root=tmp_path, ttl_seconds=10**9, max_bytes=10**9, now=now)
    assert sorted(removed) == sorted([orphan_old, orphan_corrupt])
    assert (tmp_path / orphan_new).exists() and foreign.exists()


def test_cleanup_does_not_follow_symlinks(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious").write_text("x", encoding="utf-8")
    root = tmp_path / "jobs"
    root.mkdir()
    link = "d" * 32
    try:
        os.symlink(victim, root / link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    removed = jobstore.cleanup(root=root, ttl_seconds=0, max_bytes=0, now=10**10)
    assert removed == []
    assert (victim / "precious").exists()


def test_cleanup_missing_root(tmp_path):
    assert jobstore.cleanup(root=tmp_path / "nope", ttl_seconds=1, max_bytes=1) == []
