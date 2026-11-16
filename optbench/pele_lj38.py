#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cyclopts>=3.24",
#     "numpy",
#     "pele>=0.1.2",
# ]
# ///
"""pele L-BFGS on the Lennard-Jones 38 starts.

The stop is the Euclidean norm of the force at 0.01, or 10000 gradient
calls. The potential is unshifted Lennard-Jones. Every pair in these
clusters is inside the problem cutoff of 40.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import cyclopts
import numpy as np

# The script's own environment does not install this repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbench.counter import PeleCounter

FORCE = 0.01
CAP = 10000
VARIANTS = (
    ("M100-maxstep0.2", 100, 0.2),
    ("M20-maxstep0.1", 20, 0.1),
    ("M10-maxstep0.1", 10, 0.1),
    ("M4-maxstep0.2", 4, 0.2),
    ("M4-maxstep0.1", 4, 0.1),
    ("M4-maxstep0.05", 4, 0.05),
    ("M1-maxstep0.1", 1, 0.1),
)


def read_con(text: str) -> np.ndarray:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("coordinate")) + 1
    xyz = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    return np.asarray(xyz, dtype=float).ravel()


def _stop(energy, gradient, tol, coords) -> bool:
    return float(np.linalg.norm(gradient)) < FORCE


def run_one(job: tuple[str, int, float]) -> int:
    from pele.optimize._lbfgs_py import LBFGS
    from pele.potentials import LJ

    path, memory, maxstep = job
    pot = PeleCounter(LJ(eps=1.0, sig=1.0))
    coords = read_con(Path(path).read_text(encoding="utf-8"))
    quench = LBFGS(
        coords,
        pot,
        M=memory,
        maxstep=maxstep,
        nsteps=CAP,
        alternate_stop_criterion=_stop,
        iprint=-1,
    )
    result = quench.run()
    # The published 0.1.2 files used result.nfev. A new run is charged
    # for entries to the wrapped potential.
    calls = int(pot.calls)
    if (not result.success) or calls >= CAP:
        return CAP
    return calls


def write_dat(calls: list[int], memory: int, maxstep: float, template: str, output: Path) -> None:
    kept = [value for value in calls if value < CAP]
    failed = len(calls) - len(kept)
    if not kept:
        text = f"nfailed {failed}\ncode pele\ncode_version 0.1.2\ntemplate {template}\n"
    else:
        values = np.asarray(kept, dtype=float)
        text = (
            f"force_calls {values.mean():.0f}\n"
            f"force_calls_min {int(values.min())}\n"
            f"force_calls_max {int(values.max())}\n"
            f"nfailed {failed}\n"
            f"code pele\n"
            f"code_version 0.1.2\n"
            f"algorithm L-BFGS M={memory} maxstep={maxstep}\n"
            f"template {template}\n"
        )
    output.write_text(text, encoding="utf-8")
    print(text, end="")


app = cyclopts.App(name="pele-lj38", help="pele L-BFGS on the 1000 Lennard-Jones 38 starts.")


@app.default
def main(geometries: Path, output_dir: Path, workers: int = 1) -> None:
    """Write one benchmark.dat for each archived memory and step cap."""
    paths = sorted(str(path) for path in geometries.glob("*.con"))
    if len(paths) != 1000:
        raise SystemExit(f"{geometries} has {len(paths)} con files, the problem has 1000")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, memory, maxstep in VARIANTS:
        jobs = [(path, memory, maxstep) for path in paths]
        print(f"BEGIN pele {name}", flush=True)
        with Pool(workers) as pool:
            calls = list(pool.map(run_one, jobs))
        write_dat(calls, memory, maxstep, f"pele-lbfgs-{name}", output_dir / f"pele-{name}.dat")
        print(f"DONE pele {name}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    app()
