"""Deterministic v2 sample: seeds, a performance profile, noise, and failures.

This is not a Lennard-Jones campaign. The bowl and the saddle are the
quadratic bowl. Every number in the document is produced by running
these optimizers.
"""

from __future__ import annotations

import math
import random
from importlib import metadata

from .noise import KAPPA, SIGMA_LEVELS, WINDOW, noise_floor, noisy_descent, stop_threshold
from .profile import performance_profile, profile_cost
from .score import score_v2
from .seeds import descend, dimer_search, displace, random_orientation
from .quadratic import (
    ENERGY_TOL,
    FORCE_NORM,
    REFERENCE_MINIMUM,
    REFERENCE_SADDLE,
    RMSD_TOL,
    bowl,
    gradient_descent,
    saddle,
)
from .v2 import collect_provenance, record_start

SEED = 20261016
DISPLACEMENT_SAMPLES = 8
DISPLACEMENT_SIGMA = 0.02
DIMER_SIGMA = 0.05
DIMER_SAMPLES = 8
PROFILE_SEED = 11
PROFILE_STARTS = 5
PROFILE_RESAMPLES = 200
NOISE_RESAMPLES = 200
FAILURE_ROW_RESAMPLES = 50
COMBINED_RESAMPLES = 200
DISPLACEMENT_CAP = 5000
DISPLACEMENT_STEP = 0.05
DIMER_CAP = 500
NOISE_CAP = 2000
NOISE_STEP = 0.05
KEPT_CAP = 200
KEPT_STEP = 0.05
FAILURE_ORDER = ("cap", "wrong_stationary", "divergence", "crash")


def _problem_points() -> list[tuple[float, float]]:
    points = [(0.0, 0.0)]
    rng = random.Random(PROFILE_SEED)
    for _ in range(PROFILE_STARTS - 1):
        points.append((rng.gauss(0.0, 0.05), rng.gauss(0.0, 0.05)))
    return points


def displacement_start(seed: int) -> dict:
    """One bowl descent. ``seed`` is ``SEED + index``, the same stream as ``run_seeded``."""
    index = seed - SEED
    rng = random.Random(seed)
    origin = displace((0.0, 0.0), rng, DISPLACEMENT_SIGMA)

    def optimize(potential, origin=origin):
        return descend(potential, origin, DISPLACEMENT_STEP, DISPLACEMENT_CAP, FORCE_NORM)

    return _one(
        bowl,
        optimize,
        f"seed-{index}",
        DISPLACEMENT_CAP,
        "minimum",
        0.0,
        REFERENCE_MINIMUM,
        FORCE_NORM,
        ENERGY_TOL,
        RMSD_TOL,
    )


def dimer_start(seed: int) -> dict:
    """One dimer search. ``seed`` is ``SEED + index``."""
    index = seed - SEED
    rng = random.Random(seed)
    origin = displace((0.4, -0.3), rng, DIMER_SIGMA)
    orientation = random_orientation(rng, len(origin))

    def optimize(potential, origin=origin, orientation=orientation):
        return dimer_search(potential, origin, orientation, force_norm=FORCE_NORM)

    return _one(
        saddle,
        optimize,
        f"seed-{index}",
        DIMER_CAP,
        "saddle",
        0.0,
        REFERENCE_SADDLE,
        FORCE_NORM,
        ENERGY_TOL,
        RMSD_TOL,
    )


def profile_start(index: int, step: float, cap: int) -> dict:
    """Gradient descent from problem ``index`` of the profile sample."""
    point = _problem_points()[index]

    def optimize(potential, point=point, step=step, cap=cap):
        return descend(potential, point, step, cap, FORCE_NORM)

    return _one(
        bowl,
        optimize,
        f"problem-{index}",
        cap,
        "minimum",
        0.0,
        REFERENCE_MINIMUM,
        FORCE_NORM,
        ENERGY_TOL,
        RMSD_TOL,
    )


def noise_start(sigma: float) -> dict:
    """One noisy descent of the bowl at this sigma. The seed is ``SEED``."""
    rng = random.Random(SEED)
    components = 2
    threshold = stop_threshold(FORCE_NORM, sigma, components, KAPPA)
    energy_tol = max(ENERGY_TOL, KAPPA * sigma)
    rmsd_tol = max(RMSD_TOL, KAPPA * sigma)
    optimizer = noisy_descent(
        bowl,
        (0.0, 0.0),
        NOISE_STEP,
        NOISE_CAP,
        sigma=sigma,
        rng=rng,
        clean_tol=FORCE_NORM,
        energy_tol=energy_tol,
        kappa=KAPPA,
        window=WINDOW,
    )
    return record_start(
        f"sigma-{sigma}",
        optimizer.evaluate,
        optimizer,
        cap=NOISE_CAP,
        kind="minimum",
        reference_energy=0.0,
        reference_coords=REFERENCE_MINIMUM,
        force_norm=threshold,
        energy_tol=energy_tol,
        rmsd_tol=rmsd_tol,
    )


def failure_start(kind: str) -> tuple[dict, int]:
    """One of the four failure classes. ``kind`` is the class name."""

    def crash(potential):
        potential.energy_forces((0.0, 0.0))
        raise RuntimeError("optimizer aborted")

    def diverge(potential):
        energy, gradient = potential.energy_forces((0.0, 0.0))
        point = [0.0, 0.0]
        # Step up the gradient so the force grows, then stop under the cap.
        for _ in range(3):
            energy, gradient = potential.energy_forces(point)
            point = [value + 0.05 * component for value, component in zip(point, gradient, strict=True)]
        size = math.sqrt(sum(value * value for value in gradient))
        return {"converged": False, "diverged": True, "energy": energy, "x": point, "force_norm": size}

    if kind == "cap":
        return (
            _one(
                bowl,
                gradient_descent((0.0, 0.0), 0.001, 200),
                "cap-hit",
                200,
                "minimum",
                0.0,
                REFERENCE_MINIMUM,
                FORCE_NORM,
                ENERGY_TOL,
                RMSD_TOL,
            ),
            200,
        )
    if kind == "wrong_stationary":
        return (
            _one(
                bowl,
                gradient_descent((0.0, 0.0), 0.05, 5000),
                "wrong-stationary",
                5000,
                "saddle",
                0.0,
                REFERENCE_SADDLE,
                FORCE_NORM,
                ENERGY_TOL,
                RMSD_TOL,
            ),
            5000,
        )
    if kind == "divergence":
        return (
            _one(bowl, diverge, "divergence", 100, "minimum", 0.0, REFERENCE_MINIMUM, FORCE_NORM, ENERGY_TOL, RMSD_TOL),
            100,
        )
    if kind == "crash":
        return (
            _one(bowl, crash, "crash", 100, "minimum", 0.0, REFERENCE_MINIMUM, FORCE_NORM, ENERGY_TOL, RMSD_TOL),
            100,
        )
    raise ValueError("kind must be a failure class")


def kept_start() -> dict:
    """A success under cap 200, paired with the cap-hit in the combined mean."""
    return _one(
        bowl,
        gradient_descent((0.0, 0.0), KEPT_STEP, KEPT_CAP),
        "finished-under-cap",
        KEPT_CAP,
        "minimum",
        0.0,
        REFERENCE_MINIMUM,
        FORCE_NORM,
        ENERGY_TOL,
        RMSD_TOL,
    )


def _one(evaluate, optimize, name, cap, kind, energy, coords, force_norm, energy_tol, rmsd_tol) -> dict:
    return record_start(
        name,
        evaluate,
        optimize,
        cap=cap,
        kind=kind,
        reference_energy=energy,
        reference_coords=coords,
        force_norm=force_norm,
        energy_tol=energy_tol,
        rmsd_tol=rmsd_tol,
    )


def _code_runs(step: float, cap: int) -> list[dict]:
    return [profile_start(index, step, cap) for index in range(PROFILE_STARTS)]


def _failure_examples() -> list[tuple[dict, int]]:
    return [failure_start(kind) for kind in FAILURE_ORDER]


def _noise_runs() -> list[dict]:
    runs = []
    for sigma in SIGMA_LEVELS:
        components = 2
        threshold = stop_threshold(FORCE_NORM, sigma, components, KAPPA)
        energy_tol = max(ENERGY_TOL, KAPPA * sigma)
        rmsd_tol = max(RMSD_TOL, KAPPA * sigma)
        start = noise_start(sigma)
        runs.append(
            {
                "sigma": sigma,
                "kappa": KAPPA,
                "window": WINDOW,
                "components": components,
                "noise_floor": noise_floor(sigma, components, KAPPA),
                "force_threshold": threshold,
                "energy_tol": energy_tol,
                "rmsd_tol": rmsd_tol,
                "start": start,
                "metrics": score_v2(
                    [{"force_calls": start["force_calls"], "success": start["success"], "failure": start["failure"]}],
                    NOISE_CAP,
                    resamples=NOISE_RESAMPLES,
                    seed=SEED,
                ),
            }
        )
    return runs


def build_statistics(commit: str = "", machine: dict | str | None = "validation") -> dict:
    """Run the seed, profile, noise, and failure samples."""
    from .v2 import summarise_run

    displacement = [displacement_start(SEED + index) for index in range(DISPLACEMENT_SAMPLES)]
    displacement_metrics = summarise_run(displacement, DISPLACEMENT_CAP)
    dimer = [dimer_start(SEED + index) for index in range(DIMER_SAMPLES)]
    dimer_metrics = summarise_run(dimer, DIMER_CAP)
    codes = {
        "gd-step-0.05": _code_runs(0.05, 5000),
        "gd-step-0.01": _code_runs(0.01, 20000),
        "gd-step-0.001-cap-30": _code_runs(0.001, 30),
    }
    # The capped code's budget is 30. The other codes are scored with their
    # own caps. A cross-code PAR-2 table uses each code's cap.
    par2 = {
        "gd-step-0.05": score_v2(
            [{"force_calls": start["force_calls"], "success": start["success"], "failure": start["failure"]} for start in codes["gd-step-0.05"]],
            5000,
            resamples=PROFILE_RESAMPLES,
            seed=SEED,
        ),
        "gd-step-0.01": score_v2(
            [{"force_calls": start["force_calls"], "success": start["success"], "failure": start["failure"]} for start in codes["gd-step-0.01"]],
            20000,
            resamples=PROFILE_RESAMPLES,
            seed=SEED,
        ),
        "gd-step-0.001-cap-30": score_v2(
            [
                {"force_calls": start["force_calls"], "success": start["success"], "failure": start["failure"]}
                for start in codes["gd-step-0.001-cap-30"]
            ],
            30,
            resamples=PROFILE_RESAMPLES,
            seed=SEED,
        ),
    }
    # The profile compares costs on the same problems. Unsolved starts are
    # infinite, so they remain in the denominator.
    profile_costs = {code: [profile_cost(start) for start in starts] for code, starts in codes.items()}
    profile = performance_profile(profile_costs)
    examples = _failure_examples()
    # Same cap on a success and a cap-hit, so the PAR-2 mean has somewhere
    # to put the failure. The success-only mean leaves it out.
    kept = kept_start()
    combined_records = [
        {"force_calls": item["force_calls"], "success": item["success"], "failure": item["failure"]}
        for item in (kept, examples[0][0])
    ]
    failure_metrics = score_v2(combined_records, KEPT_CAP, resamples=COMBINED_RESAMPLES, seed=SEED)
    provenance = collect_provenance(
        code="optbench",
        code_version=metadata.version("optbench"),
        commit=commit,
        parameters={
            "displacement_sigma": DISPLACEMENT_SIGMA,
            "dimer_sigma": DIMER_SIGMA,
            "seed": SEED,
            "samples": DISPLACEMENT_SAMPLES,
            "noise_sigma": list(SIGMA_LEVELS),
            "kappa": KAPPA,
            "window": WINDOW,
        },
        machine=machine,
    )
    return {
        "schema": "optbench.validation/v2",
        "status": "quadratic",
        "problem": "quadratic/bowl",
        "note": (
            "Seeded displacements, dimer orientations, a Dolan-Moré profile, "
            "noisy forces, and one of each failure class. Not a Lennard-Jones campaign."
        ),
        "provenance": provenance,
        "displacement": {
            "sigma": DISPLACEMENT_SIGMA,
            "samples": DISPLACEMENT_SAMPLES,
            "seed": SEED,
            "step": 0.05,
            "starts": displacement,
            "metrics": displacement_metrics,
        },
        "dimer": {
            "sigma": DIMER_SIGMA,
            "samples": DIMER_SAMPLES,
            "seed": SEED,
            "x0": [0.4, -0.3],
            "starts": dimer,
            "metrics": dimer_metrics,
        },
        "profile": {
            "problems": [list(point) for point in _problem_points()],
            "calls": {code: [start["force_calls"] for start in starts] for code, starts in codes.items()},
            "success": {code: [start["success"] for start in starts] for code, starts in codes.items()},
            "par2": par2,
            "dolan_more": {
                "taus": profile["taus"],
                "rho": profile["rho"],
                "solved": profile["solved"],
                "area_log10": profile["area_log10"],
                "reference": profile["reference"],
            },
        },
        "noise": {"sigma_levels": list(SIGMA_LEVELS), "kappa": KAPPA, "window": WINDOW, "runs": _noise_runs()},
        "failures": {
            "classes": [
                {"cap": cap, "start": start, "metrics": score_v2(
                    [{"force_calls": start["force_calls"], "success": start["success"], "failure": start["failure"]}],
                    cap,
                    resamples=FAILURE_ROW_RESAMPLES,
                    seed=SEED,
                )}
                for start, cap in examples
            ],
            "combined_cap": 200,
            "combined_starts": [kept, examples[0][0]],
            "metrics": failure_metrics,
        },
    }


def write_statistics(path, commit: str = "", machine: dict | str | None = "validation") -> dict:
    import json
    from pathlib import Path

    document = build_statistics(commit=commit, machine=machine)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document
