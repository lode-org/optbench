"""Global optimization of Lennard-Jones clusters with OGOLEM.

One force call is one harness evaluation of ``lennard_jones``: epsilon 1,
sigma 1, cutoff 40, no shift. OGOLEM stores cartesians in bohr and its
collision radii assume the argon Lennard-Jones sigma. Seeds are written
so one reduced sigma is that many bohr. The backend divides by the same
sigma. A repeated geometry is one call. The refused call at the force
cap is not counted.

The search is the Ar38 Sweden example: pool 100, 19900 iterations, LBFGS
at the OGOLEM defaults. One archived geometry is centered on the origin,
because ``setRandomCOM`` draws the rest of the pool about the origin, and
written as one xyz seed. The RNG seed is the start index plus one.

A start succeeds when the problem states ``energy_above_minimum`` and a
reference energy, the lowest harness energy is at or below that sum, and
the call count is under the cap. The reference is the problem's Cambridge
energy when that block is present. The cap is ``max_force_calls`` when the
problem states one, and the template ``failure_cap`` otherwise. Without
that rule the start is not a success. A finite finish under the cap is
``wrong_stationary``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import socketserver
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

COMMIT = "cce35a376d929852ab1e80e98105c0d13d784b8d"
ANGTOBOHR = 1.889725989
# AtomicProperties.ljSigmaTable Ar. Radii in that table are bohr; Ar is 1.85.
# A packed reduced cluster at Constants.ANGTOBOHR sits inside the collision
# threshold 1.2*(1.85+1.85) and every Sweden child is discarded.
AR_SIGMA_BOHR = 6.434516991
XYZ_SCALE = AR_SIGMA_BOHR / ANGTOBOHR
# The jar's version line is PRINTED_VERSION. The row uses the year.
CODE_VERSION = "2016"
PRINTED_VERSION = "2016.CURRENT"
# Archive eon-basinhopping stop. reference_energy + 0.001 lands here.
LJ38_STOP_ENERGY = -173.927427
GLOB_ALGO = (
    "cluster{xover(multiple:75%sweden:cutstyle=2|25%sweden:cutstyle=1)"
    "mutation(multiple:75%montecarlo:mode=some|25%montecarlo:mode=one)}"
)
_NUMBER = re.compile(r"(\d+)")


def ogolem_home() -> Path:
    override = os.environ.get("OGOLEM_HOME")
    if override:
        return Path(override)
    return Path.home() / "opt" / "ogolem" / COMMIT


def java_bin() -> Path:
    override = os.environ.get("OGOLEM_JAVA")
    if override:
        return Path(override)
    bundled = Path.home() / "opt" / "jdk-25" / "bin" / "java"
    if bundled.is_file():
        return bundled
    home = os.environ.get("JAVA_HOME")
    if home:
        return Path(home) / "bin" / "java"
    return Path("java")


def jars_ready(home: Path | None = None) -> bool:
    root = home or ogolem_home()
    backend = root / "optbench-backend.jar"
    snapshot = root / "ogolem-snapshot.jar"
    stamp = root / "COMMIT"
    if not backend.is_file() or not snapshot.is_file() or not stamp.is_file():
        return False
    return stamp.read_text(encoding="utf-8").strip() == COMMIT


def _ensure_exploded(home: Path) -> Path:
    classes = home / "exploded" / "BOOT-INF" / "classes"
    libs = home / "exploded" / "BOOT-INF" / "lib"
    if classes.is_dir() and libs.is_dir():
        return home / "exploded"
    if not jars_ready(home):
        raise RuntimeError(f"{home} has no OGOLEM jars for commit {COMMIT}")
    exploded = home / "exploded"
    exploded.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(Path(java_bin()).with_name("jar")), "xf", str(home / "ogolem-snapshot.jar"), "BOOT-INF"],
        cwd=exploded,
        check=True,
    )
    if not classes.is_dir():
        raise RuntimeError(f"{home} snapshot has no BOOT-INF/classes")
    return exploded


def _classpath(home: Path) -> str:
    exploded = _ensure_exploded(home)
    return os.pathsep.join(
        [
            str(exploded / "BOOT-INF" / "classes"),
            str(exploded / "BOOT-INF" / "lib" / "*"),
            str(home / "optbench-backend.jar"),
        ]
    )


def classify_start(
    *,
    exit_code: int,
    calls: int,
    energy: float | None,
    cap: int,
    threshold: float | None,
    hit_cap: bool,
) -> tuple[bool, str | None, list[str]]:
    """Success bit, primary failure, and the reason list."""
    finite = energy is not None and math.isfinite(energy)
    reasons: list[str] = []
    if exit_code not in (0, 3):
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif exit_code == 3 or hit_cap or calls >= cap:
        reasons.append("cap")
    elif threshold is not None and energy is not None and energy <= threshold and calls < cap:
        pass
    else:
        # classify_failure maps an energy rejection to wrong_stationary.
        reasons.append("energy")
    failure = classify_failure(reasons)
    return failure is None, failure, reasons


def center_cluster(positions) -> "object":
    """Translate a cluster so its centroid is the origin."""
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
    pos -= pos.mean(axis=0)
    return pos


def _sort_key(name: str) -> tuple[int, str]:
    match = _NUMBER.search(name)
    return (int(match.group(1)) if match else -1, name)


def _num(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def ogo_text(
    *,
    atoms: int,
    pool: int,
    iterations: int,
    cell: float,
    seed: int,
    threshold: float | None,
) -> str:
    """One cluster input. Pool size must be at least 2."""
    if pool < 2:
        raise ValueError("OGOLEM rejects a pool smaller than 2")
    cell_text = _num(cell)
    lines = [
        "###OGOLEM###",
        "<GEOMETRY>",
        f"NumberOfParticles={atoms}",
        "<MOLECULE>",
        f"MoleculeRepetitions={atoms}",
        "Ar;0.0;0.0;0.0",
        "</MOLECULE>",
        "</GEOMETRY>",
        "<CLUSTERBACKEND>",
        "BackendTag=lj",
        "Backend=xyz:optbench",
        "</CLUSTERBACKEND>",
        f"CellSize={cell_text};{cell_text};{cell_text}",
        f"NumberOfGlobIterations={iterations}",
        f"PoolSize={pool}",
        "GeneticRecordsToSerial=1000000",
        "GeometriesToSerial=100000",
        "SerializeAfterNewBest=false",
        "LocOptAlgo=lbfgs:backend=lj",
        f"GlobOptAlgo={GLOB_ALGO}",
        "SeedingPath=seeds",
        f"RandomNumberGenerator=javarng:seed={seed}",
    ]
    if threshold is not None:
        lines.append(f"AcceptableFitness={threshold:.16g}")
    return "\n".join(lines) + "\n"


def write_xyz(path: Path, positions) -> None:
    """Write a seed. OGOLEM multiplies the file by ``ANGTOBOHR`` on read."""
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3) * XYZ_SCALE
    lines = [str(pos.shape[0]), "optbench seed"]
    for row in pos:
        lines.append(f"Ar {row[0]:.16e} {row[1]:.16e} {row[2]:.16e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
