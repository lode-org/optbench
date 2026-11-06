"""Builds and rewrites of shared files survive concurrent callers."""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from optbench import fastpot


def _together(call, workers: int = 6) -> list:
    gate = threading.Barrier(workers)

    def run(_index):
        gate.wait()
        return call()

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(run, range(workers)))


def _read_while_building(build, workers: int = 12) -> set:
    """Each caller builds, then reads its path until every caller has built."""
    lock = threading.Lock()
    building = [workers]

    def call():
        path = build()
        with lock:
            building[0] -= 1
        seen = {(path, path.read_bytes())}
        while building[0]:
            seen.add((path, path.read_bytes()))
        return seen

    return set().union(*_together(call, workers))


@pytest.fixture
def gcc():
    if shutil.which("gcc") is None:
        pytest.skip("gcc is not installed")


def test_concurrent_fastpot_compiles_publish_one_library(gcc, tmp_path):
    dest = tmp_path / "kernels.so"
    assert all(_together(lambda: fastpot._compile(fastpot._KERNEL_SRC, dest, ["-fPIC", "-shared"])))
    assert [entry.name for entry in tmp_path.iterdir()] == [dest.name]
    assert ctypes.CDLL(str(dest)).optbench_lj_cut


def test_concurrent_fastpot_compiles_publish_one_bridge(gcc, tmp_path):
    dest = tmp_path / "extpot-bridge"
    assert all(_together(lambda: fastpot._compile(fastpot._BRIDGE_SRC, dest, [])))
    assert [entry.name for entry in tmp_path.iterdir()] == [dest.name]
    assert os.access(dest, os.X_OK)


def test_concurrent_gmin_builds_share_one_library(gcc, tmp_path, monkeypatch):
    from optbench.gmin_pot import _lj40_path

    monkeypatch.setenv("OPTBENCH_FASTPOT_DIR", str(tmp_path))
    builds = _read_while_building(_lj40_path)
    assert len(builds) == 1
    ((library, _bytes),) = builds
    assert [entry.name for entry in tmp_path.iterdir()] == [library.name]
    assert ctypes.CDLL(str(library)).optbench_lj40


def test_concurrent_surface_builds_share_one_library(gcc, tmp_path, monkeypatch):
    from optbench import surfaces

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = Path(surfaces.__file__).with_name("morse_pair.c")
    builds = _read_while_building(lambda: surfaces._morse_path(source))
    assert len(builds) == 1
    ((library, _bytes),) = builds
    assert [entry.name for entry in tmp_path.iterdir()] == [library.name]
    assert ctypes.CDLL(str(library)).morse_pair_energy


def test_concurrent_eon_svn_bridge_builds_share_one_binary(gcc, tmp_path, monkeypatch):
    from benchmarks import bench_eon_svn

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(bench_eon_svn, "_BRIDGE", None)
    paths = set(_together(bench_eon_svn._bridge))
    assert paths == {tmp_path / "optbench-eon-svn-extpot-bridge"}
    assert [entry.name for entry in tmp_path.iterdir()] == ["optbench-eon-svn-extpot-bridge"]
    assert os.access(tmp_path / "optbench-eon-svn-extpot-bridge", os.X_OK)


def test_concurrent_eon_svn_morse_builds_share_one_library(gcc, tmp_path, monkeypatch):
    from benchmarks import bench_eon_svn

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("OPTBENCH_FASTPOT_DIR", str(tmp_path))
    monkeypatch.setattr(bench_eon_svn, "_MORSE_LIB", None)
    libraries = _together(bench_eon_svn._morse_lib)
    assert all(library.morse_eon_flat for library in libraries)
    assert [entry.name for entry in tmp_path.iterdir()] == [bench_eon_svn._morse_path().name]


def test_concurrent_catalog_writes_keep_the_registry_whole(tmp_path):
    from optbench.ase_catalog import write_all

    root = Path(__file__).resolve().parents[1]
    for name in ("serial", "shared"):
        (tmp_path / name / "codes").mkdir(parents=True)
        shutil.copy(root / "codes" / "registry.toml", tmp_path / name / "codes" / "registry.toml")
    write_all(tmp_path / "serial")
    shared = tmp_path / "shared"
    registry = shared / "codes" / "registry.toml"
    samples = [shared / "templates" / "ase" / "lj38-lbfgs-default.toml", registry]
    writing = threading.Event()
    writing.set()

    def read():
        reads = 0
        while writing.is_set() or reads == 0:
            for path in samples:
                if path.is_file():
                    assert tomllib.loads(path.read_text(encoding="utf-8"))
            assert any(row["id"] == "eon" for row in tomllib.loads(registry.read_text(encoding="utf-8"))["code"])
            reads += 1
        return reads

    def write():
        write_all(shared)
        return 0

    with ThreadPoolExecutor(2) as readers:
        pending = [readers.submit(read) for _ in range(2)]
        try:
            _together(write, workers=4)
        finally:
            writing.clear()
        assert all(future.result() for future in pending)
    assert registry.read_text(encoding="utf-8") == (tmp_path / "serial" / "codes" / "registry.toml").read_text(
        encoding="utf-8"
    )
    leftovers = [path for path in shared.rglob("*") if path.name.startswith(".")]
    assert leftovers == []
