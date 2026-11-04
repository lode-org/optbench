"""GMIN LBFGS templates and the MYMYLBFGS callback."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.gmin_lbfgs import DEFAULTS
from optbench.wales import LATEST

ROOT = Path(__file__).resolve().parents[1]


def test_registry_takes_the_latest_snapshot():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(row for row in registry["code"] if row["id"] == "gmin")
    assert row["name"] == "GMIN"
    assert row["commit"] == LATEST
    assert "sha256" not in row
    assert "mymylbfgs" in row["note"]


def test_templates_use_the_keyword_defaults():
    found = [row for row in list_templates(ROOT) if row.get("engine") == "gmin"]
    assert {row["id"] for row in found} == {"gmin/lj38-lbfgs", "gmin/morse-lbfgs"}
    for template in found:
        assert template["code"] == "GMIN"
        assert template["code_version"] == LATEST
        assert template["commit"] == LATEST
        assert template["tuning"] == "default"
        assert template["gmin"]["m"] == DEFAULTS["m"]
        assert template["gmin"]["maxbfgs"] == DEFAULTS["maxbfgs"]
        assert template["gmin"]["dguess"] == DEFAULTS["dguess"]
        assert template["gmin"]["maxerise"] == DEFAULTS["maxerise"]
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        assert cap == int(problem["convergence"]["max_force_calls"])


def test_gmin_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["limit"] = kwargs["limit"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 1,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.gmin_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/lj38",
        "gmin/lj38-lbfgs",
        output=tmp_path / "benchmark.dat",
        workers=2,
        limit=3,
    )
    assert captured == {
        "template": "gmin/lj38-lbfgs",
        "problem": "minimization/lj38",
        "workers": 2,
        "limit": 3,
    }
    assert summary["template"] == "gmin/lj38-lbfgs"


def test_quadratic_quench_counts_harness_calls():
    pytest.importorskip("numpy")
    from optbench.gmin_lbfgs import available

    if not available():
        pytest.skip("libgminlbfgs.so is not installed")
    import numpy as np

    from optbench.counter import HarnessCounter
    from optbench.gmin_lbfgs import build, minimize, revision
    from optbench.potserver import RgpotCounter

    assert revision() == build()["wales_svnrev"]
    counter = HarnessCounter(cache=True, cap=1000)

    def evaluate_raw(positions, _numbers=None, _box=None):
        coords = np.asarray(positions, dtype=np.float64)
        energy = 0.5 * float(np.dot(coords, coords))
        return energy, coords.copy()

    potential = RgpotCounter(evaluate_raw, counter)

    def evaluate(coords):
        energy, gradient = potential.calculate(coords)
        return energy, gradient

    result = minimize(np.ones(6), evaluate, gtol=1e-6, cap=1000)
    assert result.mflag
    assert result.info == 0
    assert counter.calls < 1000
    assert counter.calls >= 1
    with counter.verification():
        _energy, gradient = potential.calculate(result.coords)
    assert float(np.linalg.norm(gradient)) <= 1e-6
