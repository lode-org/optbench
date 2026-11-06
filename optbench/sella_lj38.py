#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase>=3.26",
#     "cyclopts>=3.24",
#     "numpy",
#     "sella==2.6.0",
# ]
# ///
"""Sella on the archived LJ38 transition-state geometries.

The stop is the Euclidean norm of the force at 1e-3, or 1000 steps.
The pair cutoff is 40. The trust-region numbers are the archived ones.
"""

from __future__ import annotations

import os
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import cyclopts
import numpy as np
from ase.io import read
from sella import Sella

# The script's own environment does not install this repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbench.ase_lj import LennardJones
from optbench.counter import attach_ase_counter

MAXITER = 1000
GRADTOL = 1e-3
HYPER = dict(
    eta=1e-6,
    gamma=0.4,
    rho_inc=1.035,
    rho_dec=5.0,
    sigma_inc=1.15,
    sigma_dec=0.65,
    delta0=1.3e-3,
)


def run_one(path: str) -> tuple[int, int]:
    atoms = read(path, format="eon")
    atoms.calc = attach_ase_counter(LennardJones(epsilon=1.0, sigma=1.0, rc=40.0))
    dyn = Sella(atoms, logfile=os.devnull, **HYPER)
    for _ in dyn.irun(0.0, MAXITER):
        if np.linalg.norm(atoms.get_forces()) < GRADTOL:
            break
    failed = int(dyn.nsteps == MAXITER)
    harness_calls = int(atoms.calc.optbench_counter["calls"])
    if harness_calls == 0 and int(dyn.pes.neval) > 0:
        raise RuntimeError("the harness counter did not see the Sella potential")
    return harness_calls, failed


app = cyclopts.App(
    name="sella-lj38",
    help="Sella on the 200 archived LJ38 transition states.",
)


@app.default
def main(coords: Path, output: Path, workers: int = 1, version: str = "2.6.0") -> None:
    """Write benchmark.dat for every geometry in the coordinate archive."""
    work = output.parent / "sella-geometries"
    if not work.is_dir():
        work.mkdir(parents=True)
        with tarfile.open(coords) as packed:
            packed.extractall(work, filter="data")
    paths = [str(work / "coords-con" / f"{index:04d}.con") for index in range(200)]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    with Pool(workers) as pool:
        rows = pool.map(run_one, paths)
    calls = np.asarray([row[0] for row in rows], dtype=float)
    failed = sum(row[1] for row in rows)
    lines = [
        f"force_calls {calls.mean():.0f}",
        f"force_calls_min {int(calls.min())}",
        f"force_calls_median {float(np.median(calls))}",
        f"force_calls_max {int(calls.max())}",
        f"nfailed {failed}",
        "code Sella",
        f"code_version {version}",
        "template sella-lj38",
    ]
    text = "\n".join(lines) + "\n"
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    app()
