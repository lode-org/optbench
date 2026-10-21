import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.runners.pele_min import PELE_COMMIT, PELE_VERSION, merged_options

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lbfgs", "mylbfgs", "fire", "sd")


def _pele():
    pytest.importorskip("pele")


def test_pele_templates_cover_the_minimization_problems():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pele")
    assert pinned["commit"] == PELE_COMMIT
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("pele/")]
    assert len(found) == 8
    seen = set()
    for template in found:
        assert template["engine"] == "pele"
        assert template["commit"] == PELE_COMMIT
        assert template["code_version"] == PELE_VERSION
        assert template["code"] == "pele"
        assert template["tuning"] == "default"
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        method = template["pele"]["method"]
        assert method in METHODS
        merged_options(method, {key: value for key, value in template["pele"].items() if key != "method"})
        seen.add((template["problem"], method, template["tuning"]))
    for problem in ("minimization/lj38", "minimization/morse-bulk"):
        for method in METHODS:
            assert (problem, method, "default") in seen


def test_unknown_pele_option_is_rejected():
    with pytest.raises(ValueError, match="no option"):
        merged_options("lbfgs", {"memory": 4})


def test_pele_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["limit"] = kwargs["limit"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 0,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.pele_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/morse-bulk",
        "pele/morse-lbfgs",
        output=tmp_path / "benchmark.dat",
        workers=2,
        limit=3,
    )
    assert captured == {
        "template": "pele/morse-lbfgs",
        "problem": "minimization/morse-bulk",
        "workers": 2,
        "limit": 3,
    }
    assert summary["template"] == "pele/morse-lbfgs"


def test_steepest_step_matches_the_library():
    _pele()
    import numpy as np
    from pele.optimize import steepest_descent

    from optbench.counter import HarnessCounter
    from optbench.runners.pele_min import CountedPelePotential, _steepest

    def evaluate(coords):
        x = np.asarray(coords, dtype=float)
        return float(np.dot(x, x)), 2.0 * x

    start = np.array([0.4, -0.2, 0.1, 0.05], dtype=float)
    library_pot = CountedPelePotential(evaluate, HarnessCounter(cache=True, cap=100))
    library = steepest_descent(
        start.copy(),
        library_pot,
        dx=1e-4,
        nsteps=1,
        tol=0.0,
        maxstep=-1.0,
        iprint=-1,
    )
    harness_pot = CountedPelePotential(evaluate, HarnessCounter(cache=True, cap=100))
    point, stopped = _steepest(
        start,
        harness_pot,
        dx=1e-4,
        maxstep=-1.0,
        cap=2,
        stop=lambda **kwargs: False,
    )
    assert stopped is False
    assert harness_pot.counter.calls == library_pot.counter.calls == 2
    assert np.allclose(point, library.coords)


def test_one_lj38_lbfgs_start_is_a_harness_minimum():
    _pele()
    from optbench.runners.pele_min import _minimize_job, merged_options

    problem = resolve_problem(ROOT, "minimization/lj38")
    from optbench.runners.pele_min import _load_geometries

    loaded = _load_geometries(ROOT, problem, None, Path("/tmp/optbench-pele-test"), Path("/tmp/optbench-pele-test/unused.dat"))
    name, coords, _box = loaded[0]
    record = _minimize_job(
        {
            "index": 0,
            "start": name,
            "coords": coords.tolist(),
            "box": None,
            "kind": "lj",
            "method": "lbfgs",
            "options": merged_options("lbfgs", {}),
            "gtol": 0.01,
            "cap": 10000,
        }
    )
    assert record["counter"] == "harness"
    assert record["start"] == name
    assert record["force_calls"] < 10000
    assert record["force_norm"] <= 0.01
    assert record["success"] is True
    assert record["hessian_index"] == 0
    assert record["failure"] is None
