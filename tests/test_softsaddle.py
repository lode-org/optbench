"""SoftSaddle templates and the compiled eigen-solvers."""

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.softsaddle_run import hypersphere, library_path, run_mode_point

ROOT = Path(__file__).resolve().parents[1]


def test_softsaddle_templates_match_the_registry_pin():
    rows = [row for row in list_templates(ROOT) if str(row["id"]).startswith("softsaddle/")]
    assert {row["id"] for row in rows} == {
        "softsaddle/mode-lanczos",
        "softsaddle/mode-davidson",
        "softsaddle/saddle-davidson-sm50",
        "softsaddle/saddle-lanczos-sm0",
        "softsaddle/saddle-lanczos-sm50",
    }
    for row in rows:
        assert row["engine"] == "softsaddle"
        assert row["code"] == "softsaddle"
        assert row["commit"] == "5a00aeede06fd822ef302624ec4c9bc7889d21fc0b5804b701a5785695c68d16"
        _check_stop(resolve_problem(ROOT, row["problem"]), row)


def test_hypersphere_is_the_sorted_5000_point_set():
    vectors, order = hypersphere()
    assert vectors.shape == (5000, 21)
    assert order.shape == (5000,)
    assert len(np.unique(order)) == 5000
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)
    pair = vectors[int(order[0])] - vectors[int(order[1])]
    assert float(np.linalg.norm(pair)) == pytest.approx(1.924429334848153, rel=1e-9)


@pytest.mark.parametrize("solver", ["lanczos", "davidson"])
def test_quadratic_lowest_mode(tmp_path: Path, solver: str):
    if library_path(solver) is None:
        pytest.skip(f"{solver} library is not built")

    def evaluate(flat):
        x, y, z = np.asarray(flat, dtype=np.float64)
        energy = float(-x * x + 2.0 * y * y + 3.0 * z * z)
        gradient = np.array([-2.0 * x, 4.0 * y, 6.0 * z], dtype=np.float64)
        return energy, gradient

    record = run_mode_point(
        {
            "name": "quadratic",
            "evaluate": evaluate,
            "x": np.array([0.2, 0.3, -0.1], dtype=np.float64),
            "reference": np.array([1.0, 0.0, 0.0]),
            "cap": 10000,
            "overlap": 0.99,
            "solver": solver,
            "seed": 20261015,
            "work": str(tmp_path / solver),
            "settings": {
                "solver": solver,
                "seed_mode": "gradient",
                "iteration_limit": 8,
                "tolerance": 1e-8,
                "finite": 1e-6,
            },
        }
    )
    assert record["success"] is True
    assert record["overlap"] >= 0.99
    assert record["curvature"] == pytest.approx(-2.0, rel=1e-3, abs=1e-3)
    assert record["force_calls"] < 10000
    assert record["failure"] is None
