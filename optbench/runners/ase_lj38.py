"""Minimize the archived Lennard-Jones 38 starts with ASE.

The stopping rule is the minimization/lj38 contract: the L2 norm of
the force at or below 0.01, and at most 10000 calculator calls.
The starts are the geometries in lj38-clusters.tgz, not a new sample.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

ARCHIVE = (
    "inputs/sha256/86bf774955b906c486d37282c48e42628e8fe4af3579222f6e07fcf5913157b1/"
    "lj38-clusters.tgz"
)
FORCE_NORM = 0.01
MAX_CALLS = 10000
CONTRIBUTOR = "Rohit Goswami"


def read_con(text: str) -> np.ndarray:
    lines = [line for line in text.splitlines() if line.strip()]
    count = int(lines[0].split()[0])
    coords = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(coords) != count:
        raise ValueError(f"expected {count} atoms, read {len(coords)}")
    return np.asarray(coords, dtype=float)


def minimize_one(positions: np.ndarray):
    from ase import Atoms
    from ase.optimize import FIRE

    from optbench.ase_lj import LennardJones
    from optbench.counter import attach_ase_counter

    atoms = Atoms(symbols=["H"] * len(positions), positions=positions, pbc=False)
    calc = attach_ase_counter(LennardJones(epsilon=1.0, sigma=1.0, rc=40.0), cap=MAX_CALLS)
    atoms.calc = calc
    try:
        FIRE(atoms, logfile=None).run(fmax=FORCE_NORM, steps=MAX_CALLS)
        failed = float(np.linalg.norm(atoms.get_forces())) > FORCE_NORM
    except RuntimeError:
        failed = True
    return calc.optbench_counter["calls"], failed


def run(root: Path) -> dict[str, str]:
    calls = []
    failed = 0
    with tarfile.open(root / ARCHIVE, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for member in members:
            raw = archive.extractfile(member).read().decode("utf-8", errors="replace")
            if not raw.lstrip()[:2].isdigit() and not raw.startswith("38"):
                continue
            try:
                positions = read_con(raw)
            except (ValueError, IndexError):
                continue
            if len(positions) != 38:
                continue
            n_calls, did_fail = minimize_one(positions)
            if did_fail or n_calls >= MAX_CALLS:
                failed += 1
                continue
            calls.append(n_calls)
    if not calls:
        raise RuntimeError("no Lennard-Jones 38 starts were minimized")
    values = np.asarray(calls, dtype=float)
    return {
        "force_calls": f"{round(values.mean()):d}",
        "force_calls_max": f"{int(values.max())}",
        "force_calls_min": f"{int(values.min())}",
        "nfailed": f"{failed}",
        "code": "ASE",
        "code_version": "3.29.0",
        "algorithm": "FIRE",
        "contributor": CONTRIBUTOR,
    }


def write_dat(path: Path, fields: dict[str, str]) -> None:
    order = [
        "force_calls",
        "force_calls_max",
        "force_calls_min",
        "nfailed",
        "algorithm",
        "code",
        "code_version",
        "contributor",
    ]
    path.write_text("".join(f"{key} {fields[key]}\n" for key in order), encoding="utf-8")
