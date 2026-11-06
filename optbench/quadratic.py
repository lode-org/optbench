"""A deterministic quadratic sample of the v2 score.

This is not a campaign. The bowl and the saddle are analytic, the
optimizers are a few lines, and the force calls are harness counts.
Lennard-Jones campaigns stay on ``workstation-a``.
"""

from __future__ import annotations

import json
import math
from importlib import metadata
from pathlib import Path

from .v2 import collect_provenance, record_start, summarise_run

FORCE_NORM = 1e-6
ENERGY_TOL = 1e-10
RMSD_TOL = 1e-5
REFERENCE_MINIMUM = [[1.0, -2.0, 0.0]]
REFERENCE_SADDLE = [[0.0, 0.0, 0.0]]


def bowl(coords):
    x, y = coords
    energy = (x - 1.0) ** 2 + 10.0 * (y + 2.0) ** 2
    return energy, (2.0 * (x - 1.0), 20.0 * (y + 2.0))


def saddle(coords):
    x, y = coords
    return -(x ** 2) + y ** 2, (-2.0 * x, 2.0 * y)


def _norm(gradient) -> float:
    return math.sqrt(sum(value * value for value in gradient))


def gradient_descent(x0, step: float, cap: int, force_norm: float = FORCE_NORM):
    def optimize(potential):
        point = [float(value) for value in x0]
        last = None
        for _ in range(cap):
            energy, gradient = potential.energy_forces(point)
            last = (energy, gradient, point[:])
            size = _norm(gradient)
            if size <= force_norm:
                return {"converged": True, "energy": energy, "x": point[:], "force_norm": size}
            point = [value - step * component for value, component in zip(point, gradient, strict=True)]
        energy, gradient, point = last
        return {"converged": False, "energy": energy, "x": point, "force_norm": _norm(gradient)}

    return optimize


def newton_to_saddle(x0):
    """One Newton step on ``-x^2 + y^2``, then a confirming evaluation."""

    def optimize(potential):
        potential.energy_forces(x0)
        landed = [0.0, 0.0]
        energy, gradient = potential.energy_forces(landed)
        return {"converged": True, "energy": energy, "x": landed, "force_norm": _norm(gradient)}

    return optimize


def _run(entry, algorithm, tuning, step, cap, evaluate, optimize, names, kind, energy, coords, commit, machine):
    starts = [
        record_start(
            name,
            evaluate,
            optimize if not isinstance(optimize, list) else optimize[index],
            cap=cap,
            kind=kind,
            reference_energy=energy,
            reference_coords=coords,
            force_norm=FORCE_NORM,
            energy_tol=ENERGY_TOL,
            rmsd_tol=RMSD_TOL,
        )
        for index, name in enumerate(names)
    ]
    parameters = {"step": step, "force_norm": FORCE_NORM, "cap": cap, "kind": kind}
    version = metadata.version("optbench")
    provenance = collect_provenance(
        code="optbench",
        code_version=version,
        commit=commit,
        parameters=parameters,
        machine=machine,
    )
    metrics = summarise_run(starts, cap)
    return {
        "entry": entry,
        "code": "optbench",
        "algorithm": algorithm,
        "code_version": version,
        "commit": commit,
        "tuning": tuning,
        "present_source": False,
        "cap": cap,
        "parameters": parameters,
        "provenance": provenance,
        "metrics": metrics,
        "starts": starts,
    }


def build_smoke(commit: str = "", machine: dict | str | None = None) -> dict:
    """Four runs: a default minimizer, an adjusted one, a capped failure, and two saddle checks."""
    origin = gradient_descent((0.0, 0.0), 0.05, 5000)
    other = gradient_descent((2.0, -1.0), 0.05, 5000)
    default = _run(
        "gd-step-0.05",
        "gradient descent",
        "default",
        0.05,
        5000,
        bowl,
        [origin, other],
        ["origin", "other"],
        "minimum",
        0.0,
        REFERENCE_MINIMUM,
        commit,
        machine,
    )
    tuned = _run(
        "gd-step-0.01",
        "gradient descent",
        "adjusted",
        0.01,
        20000,
        bowl,
        gradient_descent((0.0, 0.0), 0.01, 20000),
        ["origin"],
        "minimum",
        0.0,
        REFERENCE_MINIMUM,
        commit,
        machine,
    )
    failed = _run(
        "gd-step-0.001-cap-200",
        "gradient descent",
        "adjusted",
        0.001,
        200,
        bowl,
        gradient_descent((0.0, 0.0), 0.001, 200),
        ["origin"],
        "minimum",
        0.0,
        REFERENCE_MINIMUM,
        commit,
        machine,
    )
    good_saddle = _run(
        "newton-saddle",
        "newton",
        "default",
        None,
        100,
        saddle,
        newton_to_saddle((0.4, -0.3)),
        ["offset"],
        "saddle",
        0.0,
        REFERENCE_SADDLE,
        commit,
        machine,
    )
    false_saddle = _run(
        "bowl-reported-as-saddle",
        "gradient descent",
        "adjusted",
        0.05,
        5000,
        bowl,
        gradient_descent((0.0, 0.0), 0.05, 5000),
        ["origin"],
        "saddle",
        0.0,
        REFERENCE_SADDLE,
        commit,
        machine,
    )
    return {
        "schema": "optbench.result/v2",
        "status": "quadratic",
        "problem": "quadratic/bowl",
        "scoring": "par2-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Quadratic sample of the harness counter and the v2 score. "
            "Not a Lennard-Jones campaign."
        ),
        "runs": [default, tuned, failed, good_saddle, false_saddle],
    }


def write_smoke(path: Path, commit: str = "", machine: dict | str | None = None) -> dict:
    document = build_smoke(commit=commit, machine=machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document
