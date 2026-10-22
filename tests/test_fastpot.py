"""The native kernels and ext_pot bridge match the Python path exactly."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from benchmarks.bench_eon_develop import (
    MORSE_CUTOFF,
    _lj_cut_python,
    _morse_python,
    lj_cut,
    morse_cluster,
)
from optbench.eon_jobs import _stored
from optbench.extpot import respond
from optbench.fastpot import bridge_executable, kernels
from optbench.lj import _lj_cluster_python, lj_cluster
from optbench.potserver import PotServer


def _require_gcc():
    if shutil.which("gcc") is None:
        pytest.skip("gcc is not installed")


def test_native_kernels_match_the_python_loops():
    _require_gcc()
    assert kernels() is not None
    rng = random.Random(20261016)
    positions = [rng.uniform(-4.0, 4.0) for _ in range(38 * 3)]
    assert lj_cluster(positions) == _lj_cluster_python(positions)
    assert lj_cut(positions) == _lj_cut_python(positions)
    assert morse_cluster(positions) == _morse_python(positions)
    assert lj_cluster([0.0, 0.0, 0.0]) == _lj_cluster_python([0.0, 0.0, 0.0])
    assert morse_cluster([]) == _morse_python([])

    # Equality at the Morse cutoff is inside the kernel; Lennard-Jones leaves it out.
    at_cut = [0.0, 0.0, 0.0, MORSE_CUTOFF, 0.0, 0.0]
    assert morse_cluster(at_cut) == _morse_python(at_cut)
    assert lj_cluster([0.0, 0.0, 0.0, 15.0, 0.0, 0.0]) == (0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert lj_cut([0.0, 0.0, 0.0, 40.0, 0.0, 0.0]) == (0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    side = 12.0
    box = [side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side]
    periodic = [rng.uniform(0.0, side) for _ in range(40 * 3)]
    assert morse_cluster(periodic, box=box) == _morse_python(periodic, box=box)
    half = [0.0, 0.0, 0.0, side / 2.0, 0.1, -0.2]
    assert morse_cluster(half, box=box) == _morse_python(half, box=box)

    root = Path(__file__).resolve().parents[1]
    archive = _stored(root, "website/tsopt/pt-island/pt-island-con.tgz")
    with tarfile.open(archive) as packed:
        text = packed.extractfile("pt-island-con/reactant.con").read().decode()
    lines = [line for line in text.splitlines() if line.strip()]
    lengths = [float(part) for part in lines[2].split()]
    cell = [lengths[0], 0.0, 0.0, 0.0, lengths[1], 0.0, 0.0, 0.0, lengths[2]]
    start = next(index for index, line in enumerate(lines) if line.lower().startswith("coordinates"))
    reactant: list[float] = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            break
        reactant.extend(float(part) for part in parts[:3])
    assert morse_cluster(reactant, box=cell) == _morse_python(reactant, box=cell)
    nudged = [value + rng.uniform(-0.2, 0.2) for value in reactant]
    assert morse_cluster(nudged, box=cell) == _morse_python(nudged, box=cell)


def _numbers(text: str) -> list[float]:
    return [float(part) for part in text.split()]


def test_native_bridge_matches_python_extpot(tmp_path):
    _require_gcc()
    binary = bridge_executable()
    assert binary is not None and os.access(binary, os.X_OK)
    request = "\n".join(
        [
            "19.2088\t0.0\t0.0",
            "0.0\t19.0118\t0.0",
            "0.0\t0.0\t30.0",
            "78\t0.1\t0.2\t0.3",
            "78\t2.4\t0.2\t0.25",
            "78\t9.0\t1.0\t1.0",
            "",
        ]
    )

    def run(directory: Path, command) -> str:
        directory.mkdir()
        (directory / "from_eon_to_extpot").write_text(request, encoding="utf-8")
        env = os.environ.copy()
        env["OPTBENCH_POTSERVER"] = server.endpoint()
        if command is None:
            respond(directory, server.endpoint())
        else:
            subprocess.run(command, cwd=directory, env=env, check=True, capture_output=True, text=True)
        return (directory / "from_extpot_to_eon").read_text(encoding="utf-8")

    server = PotServer(morse_cluster).start()
    try:
        native = run(tmp_path / "native", [str(binary)])
        python = run(tmp_path / "python", None)
    finally:
        server.stop()
    assert _numbers(native) == _numbers(python)


def test_native_bridge_keeps_a_long_morse_atom_line(tmp_path):
    _require_gcc()
    binary = bridge_executable()
    assert binary is not None and os.access(binary, os.X_OK)
    # Seventeen significant digits make one atom line longer than 64 characters.
    request = "\n".join(
        [
            "16.387907\t0.0\t0.0",
            "0.0\t16.387907\t0.0",
            "0.0\t0.0\t16.387907",
            "78\t0.04410100000000089\t0.01000400000000155\t0.024467999999999115",
            "78\t2.1045110000000005\t2.0951769999999983\t16.363474999999998",
            "",
        ]
    )
    server = PotServer(morse_cluster).start()
    try:
        native_dir = tmp_path / "native"
        python_dir = tmp_path / "python"
        native_dir.mkdir()
        python_dir.mkdir()
        (native_dir / "from_eon_to_extpot").write_text(request, encoding="utf-8")
        (python_dir / "from_eon_to_extpot").write_text(request, encoding="utf-8")
        env = os.environ.copy()
        env["OPTBENCH_POTSERVER"] = server.endpoint()
        subprocess.run([str(binary)], cwd=native_dir, env=env, check=True, capture_output=True, text=True)
        respond(python_dir, server.endpoint())
        native = (native_dir / "from_extpot_to_eon").read_text(encoding="utf-8")
        python = (python_dir / "from_extpot_to_eon").read_text(encoding="utf-8")
    finally:
        server.stop()
    assert _numbers(native) == _numbers(python)
    assert len(native.splitlines()) == 3
