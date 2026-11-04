"""GMIN and OPTIM build from the latest Wales snapshot, and every run names its bytes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

from optbench.campaign import list_templates
from optbench.wales import LATEST, header, same_build, snapshot

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "codes/checkout.sh",
    "codes/gmin/build.sh",
    "codes/gmin/build-bh.sh",
    "codes/gmin/driver.f90",
    "codes/optim/build.sh",
)
ROWS = ("gmin", "gmin-bh", "optim")
SHA = "a" * 64
OTHER_SHA = "b" * 64
REVISION = "1" * 40


def test_the_wales_rows_pin_no_tarball():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in registry["code"]}
    for code in ROWS:
        assert rows[code]["repository"] == rows["gmin"]["repository"], code
        assert rows[code]["ref"] == "svn.tar.bz2", code
        assert rows[code]["commit"] == LATEST, code
        assert "sha256" not in rows[code] and "bytes" not in rows[code], code
    assert "wales_sha256" in rows["gmin"]["note"]


def test_every_wales_template_takes_the_latest():
    seen = set()
    for template in list_templates(ROOT):
        if str(template.get("code") or "") not in {"GMIN", "optim"}:
            continue
        engine = str(template.get("engine") or "")
        if engine not in {"gmin", "noise", "gmin-bh", "optim"}:
            continue
        assert template["commit"] == LATEST, template["id"]
        assert template["code_version"] == LATEST, template["id"]
        seen.add(engine)
    assert seen == {"gmin", "noise", "gmin-bh", "optim"}


@pytest.mark.parametrize("relative", SCRIPTS)
def test_build_files_hold_no_pin(relative: str):
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert not re.search(r"\b[0-9a-f]{40}\b", text)
    assert not re.search(r"\b[0-9a-f]{64}\b", text)


def _shell(script: str, **env: str) -> subprocess.CompletedProcess:
    command = f'source "{ROOT}/codes/checkout.sh"\n{script}'
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env})


def _tarball(path: Path, revision: str) -> Path:
    source = path.with_suffix(".tree")
    (source / "wales" / "GMIN" / "source").mkdir(parents=True)
    (source / "wales" / "GMIN" / "source" / "SVNREV").write_text(revision + "\n", encoding="utf-8")
    with tarfile.open(path, "w:bz2") as archive:
        archive.add(source / "wales", arcname="./wales")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wales_tarball_reads_the_local_copy_and_prints_its_sha256(tmp_path: Path):
    tarball = _tarball(tmp_path / "copy.tar.bz2", REVISION)
    proc = _shell(f'wales_tarball "{tmp_path / "unused"}"', WALES_TARBALL=str(tarball))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(tarball)
    assert f"svn.tar.bz2 sha256 {_sha256(tarball)}" in proc.stderr
    assert not (tmp_path / "unused").exists()
    missing = _shell(f'wales_tarball "{tmp_path}"', WALES_TARBALL=str(tmp_path / "absent.tar.bz2"))
    assert missing.returncode != 0
    assert "is not a file" in missing.stderr


def test_wales_extract_never_mixes_two_snapshots(tmp_path: Path):
    first = _tarball(tmp_path / "first.tar.bz2", REVISION)
    second = _tarball(tmp_path / "second.tar.bz2", "2" * 40)
    dest = tmp_path / "dest"

    def extract(tarball: Path) -> str:
        proc = _shell(f'wales_extract "{tarball}" "{dest}" && wales_svnrev "{dest}" GMIN')
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    assert extract(first) == REVISION
    marker = dest / "wales" / "marker"
    marker.write_text("kept\n", encoding="utf-8")
    assert extract(first) == REVISION
    assert marker.is_file()
    assert extract(second) == "2" * 40
    assert not marker.exists()
    assert (dest / "svn.tar.bz2.sha256").read_text(encoding="utf-8").split()[0] == _sha256(second)


def test_snapshot_refuses_what_names_no_bytes():
    assert snapshot(SHA, REVISION + "\n") == {"wales_sha256": SHA, "wales_svnrev": REVISION}
    with pytest.raises(RuntimeError, match="64 hexadecimal"):
        snapshot("f" * 12, REVISION)
    with pytest.raises(RuntimeError, match="SVNREV"):
        snapshot(SHA, " ")


def _fake_gmin(folder: Path, revision: str, sha256: str | None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "GMIN"
    binary.write_text(f'#!/bin/sh\necho "SVN revision: {revision}" >&2\n', encoding="utf-8")
    binary.chmod(0o755)
    if sha256 is not None:
        (folder / "svn.tar.bz2.sha256").write_text(f"{sha256}  svn.tar.bz2\n", encoding="utf-8")
    return binary


def test_gmin_binary_build_is_gmin_v_and_the_stamp(tmp_path: Path):
    from optbench.runners.gmin_go import binary_build, binary_revision

    binary = _fake_gmin(tmp_path / "good", REVISION, SHA)
    assert binary_revision(binary) == REVISION
    assert binary_build(binary) == {"wales_sha256": SHA, "wales_svnrev": REVISION}
    with pytest.raises(RuntimeError, match="svn.tar.bz2.sha256 is missing"):
        binary_build(_fake_gmin(tmp_path / "unstamped", REVISION, None))
    with pytest.raises(RuntimeError, match="SVNREV"):
        binary_build(_fake_gmin(tmp_path / "empty", "", SHA))


def test_gmin_bh_record_from_another_build_is_run_again(monkeypatch, tmp_path: Path):
    import benchmarks.bench_gmin_bh as bench

    monkeypatch.setenv("OPTBENCH_GMIN", str(_fake_gmin(tmp_path / "bin", REVISION, SHA)))
    monkeypatch.setenv("OPTBENCH_ASV_CACHE", str(tmp_path / "cache"))
    spec = bench.GminLj38Bh.spec
    monkeypatch.setattr(bench, "_compute", lambda spec, seed, start: {"rerun": True})
    path = bench._cache_path(spec, "0")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"wales_sha256": OTHER_SHA, "wales_svnrev": REVISION}), encoding="utf-8")
    assert bench._load(spec, "0", "0") == {"rerun": True}
    path.write_text(json.dumps({"wales_sha256": SHA, "wales_svnrev": REVISION, "force_calls": 7}), encoding="utf-8")
    assert bench._load(spec, "0", "0")["force_calls"] == 7


def _fake_optim(folder: Path, sha256: str | None, revision: str | None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "OPTIM"
    binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    binary.chmod(0o755)
    if revision is not None:
        (folder / "SVNREV").write_text(revision + "\n", encoding="utf-8")
    if sha256 is not None:
        (folder / "svn.tar.bz2.sha256").write_text(f"{sha256}  svn.tar.bz2\n", encoding="utf-8")
    return binary


def test_optim_stamp_names_the_build(tmp_path: Path):
    from optbench.engines import optim_search

    assert optim_search.build_stamp(_fake_optim(tmp_path / "good", SHA, REVISION)) == {
        "wales_sha256": SHA,
        "wales_svnrev": REVISION,
    }
    for name, sha256, revision in (("sha", None, REVISION), ("rev", SHA, None), ("short", "f" * 12, REVISION)):
        with pytest.raises(RuntimeError):
            optim_search.build_stamp(_fake_optim(tmp_path / name, sha256, revision))


def test_optim_record_from_another_build_is_run_again(monkeypatch, tmp_path: Path):
    from optbench import asv_engines

    monkeypatch.setenv("OPTIM_BIN", str(_fake_optim(tmp_path / "bin", SHA, REVISION)))
    monkeypatch.setenv("OPTBENCH_ASV_CACHE", str(tmp_path / "cache"))
    cell = next(cell for cell in asv_engines.cells(ROOT, "optim"))
    monkeypatch.setattr(asv_engines, "compute", lambda cell, seed, start: {"rerun": True})
    path = asv_engines.record_path(cell, "0", "start")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"wales_sha256": OTHER_SHA, "wales_svnrev": REVISION}), encoding="utf-8")
    assert asv_engines.load(cell, "0", "start") == {"rerun": True}
    path.write_text(json.dumps({"wales_sha256": SHA, "wales_svnrev": REVISION, "force_calls": 7}), encoding="utf-8")
    assert asv_engines.load(cell, "0", "start")["force_calls"] == 7


def test_raw_logs_open_with_the_build():
    from optbench.runners import gmin_go, gmin_min

    found = snapshot(SHA, REVISION)
    template = {"id": "gmin/lj38-bh", "problem": "globalopt/lj38", "algorithm": "basin-hopping"}
    lines = gmin_go._dat({}, template, {}, False, found).splitlines()
    assert lines[:2] == header(found)
    assert f"commit {SHA}" in lines
    assert f"code_version {REVISION[:8]}" in lines
    template = {"id": "gmin/lj38-lbfgs", "problem": "minimization/lj38", "code": "GMIN"}
    lines = gmin_min._dat({}, template, {}, REVISION, REVISION[:12], found, False).splitlines()
    assert lines[:2] == header(found)
    assert same_build({**found, "force_calls": 1}, found)
    assert not same_build({"wales_sha256": OTHER_SHA, "wales_svnrev": REVISION}, found)


def test_library_build_comes_from_the_tree():
    from optbench.gmin_lbfgs import available, build, revision

    if not available():
        pytest.skip("libgminlbfgs.so is not built")
    found = build()
    assert re.fullmatch(r"[0-9a-f]{64}", found["wales_sha256"])
    assert found["wales_svnrev"] == revision()
