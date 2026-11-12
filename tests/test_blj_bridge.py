"""The binary LJ ext_pot force is minus the derivative of its own energy.

EON steps along the vector this bridge returns. The opposite sign is the
gradient, and a minimiser then climbs. Geometry 0 is the first archived
start: 66 atoms of the first species and 34 of the second.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "cells" / "eon" / "lj100-binary-basinhopping" / "sources"
_Z = {"H": 1, "C": 6}


def _compile(tmp: Path, source: Path, *args: str) -> Path:
    binary = tmp / source.stem
    subprocess.run(["gcc", "-O2", "-o", str(binary), str(source), *args], check=True)
    return binary


def _geometry_zero() -> tuple[list[float], list[tuple[int, float, float, float]]]:
    matches = sorted((ROOT / "inputs" / "sha256").glob("*/lj100-binary-con.tgz"))
    assert len(matches) == 1
    with tarfile.open(matches[0]) as bundle:
        text = bundle.extractfile("lj100-binary-con/0.con").read().decode()
    lines = [line.strip() for line in text.splitlines()]
    box = None
    for line in lines:
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            box = [float(part) for part in parts]
        except ValueError:
            continue
        break
    assert box == [100.0, 100.0, 100.0]
    atoms: list[tuple[int, float, float, float]] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("Coordinates of Component"):
            index += 1
            continue
        symbol = lines[index - 1]
        atomic_number = _Z[symbol]
        index += 1
        while index < len(lines) and lines[index] and lines[index] not in _Z:
            parts = lines[index].split()
            atoms.append((atomic_number, float(parts[0]), float(parts[1]), float(parts[2])))
            index += 1
    assert len(atoms) == 100
    assert sum(atom[0] == 1 for atom in atoms) == 66
    assert sum(atom[0] == 6 for atom in atoms) == 34
    return box, atoms


def _write(directory: Path, box: list[float], atoms: list[tuple[int, float, float, float]]) -> None:
    rows = [f"{box[0]} 0 0", f"0 {box[1]} 0", f"0 0 {box[2]}"]
    for atomic_number, x, y, z in atoms:
        rows.append(f"{atomic_number} {x:.16e} {y:.16e} {z:.16e}")
    reply = directory / "from_extpot_to_eon"
    if reply.exists():
        reply.unlink()
    (directory / "from_eon_to_extpot").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_reply(directory: Path) -> tuple[float, list[list[float]]]:
    lines = (directory / "from_extpot_to_eon").read_text(encoding="utf-8").splitlines()
    energy = float(lines[0])
    forces = [[float(value) for value in line.split()] for line in lines[1:]]
    return energy, forces


def _bridge_eval(binary: Path, directory: Path, box, atoms):
    _write(directory, box, atoms)
    env = os.environ.copy()
    env.pop("OPTBENCH_CALLS", None)
    env.pop("OPTBENCH_CAP", None)
    env.pop("LD_PRELOAD", None)
    subprocess.run([binary], cwd=directory, check=True, env=env)
    return _read_reply(directory)


def _preload_eval(library: Path, driver: Path, directory: Path, box, atoms) -> str:
    _write(directory, box, atoms)
    env = os.environ.copy()
    env.pop("OPTBENCH_CALLS", None)
    env.pop("OPTBENCH_CAP", None)
    env["LD_PRELOAD"] = str(library)
    command = f"cd {directory} && blj/bridge"
    completed = subprocess.run([driver, command], cwd=directory, check=True, env=env, capture_output=True)
    assert completed.returncode == 0
    return (directory / "from_extpot_to_eon").read_text(encoding="utf-8")


def _central(binary: Path, directory: Path, box, atoms, step: float = 1e-5):
    gradient = [[0.0, 0.0, 0.0] for _ in atoms]
    for index, atom in enumerate(atoms):
        for axis in range(3):
            plus = [tuple(item) for item in atoms]
            minus = [tuple(item) for item in atoms]
            moved = list(atom)
            moved[1 + axis] += step
            plus[index] = tuple(moved)
            moved = list(atom)
            moved[1 + axis] -= step
            minus[index] = tuple(moved)
            high, _forces = _bridge_eval(binary, directory, box, plus)
            low, _forces = _bridge_eval(binary, directory, box, minus)
            gradient[index][axis] = (high - low) / (2.0 * step)
    return gradient


def _flat(rows: list[list[float]]) -> list[float]:
    return [value for row in rows for value in row]


def test_bridge_force_matches_central_differences(tmp_path: Path):
    bridge = _compile(tmp_path, SOURCES / "bridge.c", "-lm")
    library = tmp_path / "preload.so"
    subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", str(library), str(SOURCES / "preload.c"), "-ldl", "-lm"],
        check=True,
    )
    driver_source = tmp_path / "driver.c"
    driver_source.write_text(
        "#include <stdlib.h>\nint main(int argc, char **argv) { return system(argv[1]); }\n",
        encoding="utf-8",
    )
    driver = _compile(tmp_path, driver_source)

    box, geometry = _geometry_zero()
    cases = {
        "pair": (box, [(1, 0.0, 0.0, 0.0), (1, 2.0, 0.0, 0.0)]),
        "geometry 0": (box, geometry),
    }
    for name, (lengths, atoms) in cases.items():
        work = tmp_path / name.replace(" ", "-")
        work.mkdir()
        _energy, force = _bridge_eval(bridge, work, lengths, atoms)
        reply = (work / "from_extpot_to_eon").read_text(encoding="utf-8")
        gradient = _central(bridge, work, lengths, atoms)
        negative = [[-value for value in row] for row in gradient]
        assert _flat(force) == pytest.approx(_flat(negative), rel=1e-4, abs=1e-7), name
        assert max(abs(value) for value in _flat(force)) > 1e-4
        assert _preload_eval(library, driver, work, lengths, atoms) == reply
