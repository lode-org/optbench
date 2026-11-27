"""The head-to-head eon arm records provenance on that arm's checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

CC = "gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n"
CXX = "g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n"
FC = "GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n"
OPTIONS = [
    "-Dbuildtype=release",
    "-Doptimization=3",
    "-Dprefix=/opt/eon",
    "-Dlibdir=lib",
]


def _load_head2head_module():
    path = Path(__file__).resolve().parents[1] / "results" / "campaign" / "rgsaddle_head2head.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_head2head_eon_provenance", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _facts() -> dict:
    return {
        "version_stdout": "3.5.0 (d9cc6c5be)",
        "cc_stdout": CC,
        "cxx_stdout": CXX,
        "fc_stdout": FC,
        "cc": "gcc",
        "cxx": "g++",
        "fc": "gfortran",
        "builddir": "builddir",
        "options": list(OPTIONS),
        "meson": "1.12.1",
        "ninja": "1.13.2",
        "source": {"commit": "d9cc6c5bee432d625eb6c56d7ea74b723a37ee3b", "tag": "v3.5.0"},
        "run": {"communicator_jobs": 1, "core": 0, "threads": {"OMP_NUM_THREADS": "1"}},
        "argv_origin": "Published command lines use the token eonclient.",
    }


def test_head2head_eon_arm_writes_provenance_on_its_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    head2head = _load_head2head_module()
    monkeypatch.setattr(head2head, "ROOT", tmp_path)
    monkeypatch.setattr(head2head, "_single_thread", lambda _cpu: None)
    monkeypatch.setattr(head2head.catalog, "_single_thread", lambda _cpu: None)
    shared = tmp_path / "shared-eon-provenance.json"
    monkeypatch.setattr(head2head.catalog, "PROVENANCE_PATH", shared)
    seen: dict[str, Path] = {}

    def stop(_client: Path) -> str:
        seen["provenance_path"] = Path(head2head.catalog.PROVENANCE_PATH)
        raise RuntimeError("stop-after-provenance")

    monkeypatch.setattr(head2head.catalog, "_client_version", stop)
    lib = tmp_path / "librgsaddle.so"
    lib.write_bytes(b"library")
    monkeypatch.setenv("RGSADDLE_LIB", str(lib))
    monkeypatch.delenv("RGSADDLE_SRC", raising=False)
    client = tmp_path / "eonclient"
    client.write_bytes(b"eonclient")
    with pytest.raises(RuntimeError, match="stop-after-provenance"):
        head2head.run_eon("old", 0, client, _facts())
    arm_provenance = tmp_path / "build" / "rgsaddle-h2h" / "eon-classic" / "old" / "eon-provenance.json"
    assert seen["provenance_path"] == arm_provenance
    written = json.loads(arm_provenance.read_text(encoding="utf-8"))
    assert written["binary"]["sha256"] == hashlib.sha256(b"eonclient").hexdigest()
    assert written["build"]["compilers"]["fc"] == FC.strip()
    assert "-Doptimization=3" in written["build"]["commands"][0]
    assert head2head.catalog.PROVENANCE_PATH == shared
    assert not shared.is_file()
