"""The lj38-noise contract and the campaign catalog."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from optbench.campaign import list_templates
from optbench.noise import SIGMA_LEVELS, window_energy_tolerance
from optbench.noise_catalog import HESSIAN_METHODS, allows_method, sigma_slug, specs
from optbench.problems import load_problem

ROOT = Path(__file__).resolve().parents[1]


def test_lj38_noise_does_not_invent_a_reference():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38-noise.toml")
    success = problem["scoring"]["v2"]["success"]
    assert problem["convergence"]["energy_tol"] == 0.01
    assert "per-start reference" in success
    assert "not applied" in success
    assert "against the reference" not in success
    assert "rigid" in success or "rigid modes" in problem["noise"]["rule"] or "vibrational" in success
    assert "Newton and dogleg" in problem["noise"]["rule"]
    assert math.isclose(window_energy_tolerance(0.01, 0.0), 0.01)
    assert math.isclose(window_energy_tolerance(0.01, 1e-2), 0.03)
    assert math.isclose(window_energy_tolerance(0.01, 1e-4), 0.01)


def test_noise_catalog_covers_the_campaign_and_skips_exact_hessians():
    catalog = specs()
    assert len(catalog) == 70
    on_disk = [row for row in list_templates(ROOT) if row.get("engine") == "noise"]
    assert {row["id"] for row in on_disk} == {row["id"] for row in catalog}
    by_id = {row["id"]: row for row in on_disk}
    for row in catalog:
        stored = by_id[row["id"]]
        assert stored["problem"] == "minimization/lj38-noise"
        assert stored["noise"]["sigma"] == row["noise"]["sigma"]
        assert stored["noise"]["seed"] == 20261016
        assert stored["commit"] == row["commit"]
        method = str((stored.get("solver") or {}).get("method") or "")
        sigma = float(stored["noise"]["sigma"])
        if method in HESSIAN_METHODS:
            assert sigma == 0.0
        assert allows_method(method, sigma)
    force = {"lbfgs", "lbfgs-m100", "fire", "cg"}
    for name in force:
        for sigma in SIGMA_LEVELS:
            assert f"noise/rgmin-{name}-{sigma_slug(sigma)}" in by_id
    for name in ("dogleg", "newton"):
        assert f"noise/rgmin-{name}-s0" in by_id
        for sigma in SIGMA_LEVELS:
            if sigma == 0.0:
                continue
            assert f"noise/rgmin-{name}-{sigma_slug(sigma)}" not in by_id
            assert allows_method(name, sigma) is False
    for name in ("bfgs", "lbfgs", "fire"):
        for sigma in SIGMA_LEVELS:
            assert f"noise/ase-{name}-{sigma_slug(sigma)}" in by_id
    for sigma in SIGMA_LEVELS:
        assert f"noise/sella-min-{sigma_slug(sigma)}" in by_id
        assert f"noise/gmin-lbfgs-{sigma_slug(sigma)}" in by_id
        assert by_id[f"noise/gmin-lbfgs-{sigma_slug(sigma)}"]["code"] == "GMIN"
        assert by_id[f"noise/gmin-lbfgs-{sigma_slug(sigma)}"]["solver"]["m"] == 4
    for method in ("sqnm", "ase"):
        for sigma in SIGMA_LEVELS:
            ident = f"noise/vcsqnm-{method}-{sigma_slug(sigma)}"
            assert ident in by_id
            assert by_id[ident]["solver"]["family"] == "vcsqnm"
            assert by_id[ident]["solver"]["initial_step_size"] == -1e-7
            assert by_id[ident]["code"] == "vc-SQNM"
    for name in ("lbfgs", "mylbfgs", "fire", "sd"):
        for sigma in SIGMA_LEVELS:
            assert f"noise/pele-{name}-{sigma_slug(sigma)}" in by_id
    for coordsys in ("cart", "tric"):
        for sigma in SIGMA_LEVELS:
            assert f"noise/geometric-{coordsys}-{sigma_slug(sigma)}" in by_id


def test_exact_hessian_job_is_refused():
    from optbench.runners.noise_min import _minimize_job

    job = {
        "index": 0,
        "start": "cluster_0000",
        "coords": [float(i) for i in range(114)],
        "family": "rgmin",
        "method": "newton",
        "optimizer": "",
        "options": {"maxstep": 0.2},
        "sigma": 1e-3,
        "seed": 20261016,
        "clean_tol": 0.01,
        "energy_tol": 0.01,
        "cap": 10,
    }
    with pytest.raises(ValueError, match="exact Hessian"):
        _minimize_job(job)


def test_exception_text_keeps_a_single_type_prefix():
    from optbench.counter import ForceCallCap
    from optbench.runners.noise_min import _exception_text

    assert _exception_text(RuntimeError("RuntimeError: Not bracketed")) == "RuntimeError: Not bracketed"
    assert _exception_text(RuntimeError("coordinate rebuild did not settle")) == (
        "RuntimeError: coordinate rebuild did not settle"
    )
    assert _exception_text(ForceCallCap("ForceCallCap: force-call cap")) == "ForceCallCap: force-call cap"
