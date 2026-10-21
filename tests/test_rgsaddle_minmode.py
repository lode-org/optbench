"""Surfaces and the rgsaddle saddle driver.

The library test loads librgsaddle.so from RGSADDLE_LIB. CI does not
build that library, so the test skips when the variable is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.surfaces import lennard_jones, morse_smooth_linear, read_con

ROOT = Path(__file__).resolve().parents[1]


def test_island_saddle_success_is_a_cartesian_connection():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    success = problem["scoring"]["v2"]["success"]
    assert "Cartesian RMSD" in success
    assert "Kabsch" not in success
    assert "max_force_calls" not in (problem.get("convergence") or {})


def test_lennard_jones_minimum_is_minus_one():
    distance = 2.0 ** (1.0 / 6.0)
    energy, gradient = lennard_jones(np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]))
    assert energy == pytest.approx(-1.0, abs=1e-10)
    assert np.linalg.norm(gradient) == pytest.approx(0.0, abs=1e-8)


def test_morse_smooth_linear_is_flat_at_the_cutoff():
    energy, gradient = morse_smooth_linear(
        np.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]]),
        depth=0.7102,
        alpha=1.6047,
        r0=2.897,
        cutoff=9.5,
    )
    assert energy == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(gradient) == pytest.approx(0.0, abs=1e-10)


def test_morse_pair_loop_matches_the_numpy_sum():
    from optbench.surfaces import _morse_numpy

    positions = np.array(
        [
            [0.2, 0.1, 0.0],
            [2.4, -0.3, 0.4],
            [10.1, 1.2, 0.5],
            [0.4, 9.6, 1.1],
        ]
    )
    kwargs = dict(
        depth=0.7102,
        alpha=1.6047,
        r0=2.897,
        cutoff=9.5,
        cell=np.array([12.0, 11.0, 8.0]),
        pbc=(True, True, False),
    )
    energy, gradient = morse_smooth_linear(positions, **kwargs)
    reference_energy, reference_gradient = _morse_numpy(positions, **kwargs)
    assert energy == pytest.approx(reference_energy, abs=1e-9, rel=1e-12)
    assert gradient == pytest.approx(reference_gradient, abs=1e-9)
    shifted = positions.copy()
    shifted[1] += np.array([0.05, -0.02, 0.01])
    energy, gradient = morse_smooth_linear(shifted, **kwargs)
    reference_energy, reference_gradient = _morse_numpy(shifted, **kwargs)
    assert energy == pytest.approx(reference_energy, abs=1e-9, rel=1e-12)
    assert gradient == pytest.approx(reference_gradient, abs=1e-9)
    shifted[0] += np.array([1.4, -0.8, 0.6])
    energy, gradient = morse_smooth_linear(shifted, **kwargs)
    reference_energy, reference_gradient = _morse_numpy(shifted, **kwargs)
    assert energy == pytest.approx(reference_energy, abs=1e-9, rel=1e-12)
    assert gradient == pytest.approx(reference_gradient, abs=1e-9)


def test_read_con_marks_fixed_atoms_and_the_box():
    text = "\n".join(
        [
            "0",
            "19.2 19.0 30.0",
            "90 90 90",
            "0 0",
            "0 0",
            "1",
            "2",
            "1.0",
            "Pt",
            "Coordinates",
            "0 0 0 0 0",
            "1 2 3 1 1",
        ]
    )
    positions, fixed, cell = read_con(text)
    assert positions.shape == (2, 3)
    assert fixed.tolist() == [False, True]
    assert cell.tolist() == pytest.approx([19.2, 19.0, 30.0])


def test_vibrational_index_drops_only_a_numerical_rigid_mode():
    from optbench.rgsaddle_minmode import _rigid_basis, _vibrational_eigenvalues
    from optbench.stationary import negative_index

    point = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 1.1, 0.0],
            [0.1, 0.2, 1.3],
        ],
        dtype=np.float64,
    )
    flat = point.reshape(-1)
    basis = _rigid_basis(flat)
    extra = np.random.default_rng(0).normal(size=(flat.size, flat.size))
    frame, _r = np.linalg.qr(np.concatenate([basis, extra], axis=1))
    rigid = basis.shape[1]
    vibrational = rigid

    numerical = [1.0] * flat.size
    numerical[0] = -1e-5
    numerical[vibrational] = -3.0
    adjusted, removed = _vibrational_eigenvalues(flat, numerical, frame)
    assert removed == 1
    assert adjusted[vibrational] == -3.0
    assert adjusted[0] == 0.0
    assert negative_index(adjusted, 1e-6) == 1

    large = [1.0] * flat.size
    large[0] = -0.5
    large[vibrational] = -3.0
    adjusted_large, removed_large = _vibrational_eigenvalues(flat, large, frame)
    assert removed_large == 0
    assert negative_index(adjusted_large, 1e-6) == 2

    soft = [1.0] * flat.size
    soft[vibrational] = -1e-5
    adjusted_soft, removed_soft = _vibrational_eigenvalues(flat, soft, frame)
    assert removed_soft == 0
    assert negative_index(adjusted_soft, 1e-6) == 1

    single, removed_single = _vibrational_eigenvalues(np.zeros(3), [-1e-5, -3.0, 0.1], np.eye(3))
    assert removed_single == 0
    assert negative_index(single, 1e-6) == 2


def test_rgsaddle_templates_match_their_problem_stop():
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "rgsaddle":
            continue
        found.append(template["id"])
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        assert cap >= 1000
        assert template["commit"] == "38c3137f62ce1436d662bb37188a8d7cd2d789b8"
        assert template["engine"] == "rgsaddle"
    assert "rgsaddle/dimer-lbfgs" in found
    assert "rgsaddle/prfo" in found
    assert "rgsaddle/prfo-tuned" not in found
    assert "rgsaddle/mode-lanczos" in found
    assert "rgsaddle/mode-dimer-tuned" not in found
    assert "rgsaddle/mode-lanczos-tuned" not in found
    assert "rgsaddle/saddle-dimer-lbfgs" in found
    assert "rgsaddle/pt-dimer-lbfgs" in found


@pytest.mark.skipif(not os.environ.get("RGSADDLE_LIB"), reason="RGSADDLE_LIB is not set")
def test_dimer_finds_the_quadratic_saddle():
    from optbench.rgsaddle_minmode import _dispatch

    def evaluate(coords):
        x, y, z = (float(value) for value in coords)
        return -x * x + y * y + 2.0 * z * z, (-2.0 * x, 2.0 * y, 4.0 * z)

    spec = {
        "name": "quadratic",
        "positions": np.array([[0.4, -0.3, 0.2]], dtype=np.float64),
        "fixed": np.array([False]),
        "cell": np.zeros(3),
        "evaluate": evaluate,
        "potential": {},
        "session": "minmode",
        "kind": "dimer",
        "method": "lbfgs",
        "mode_seed": "gradient",
        "task": "saddle",
        "saddle": {"dr": 1e-3, "rotation_tol": 1e-6, "max_rotations": 20, "krylov_dim": 4},
        "optimizer": {"max_move": 0.2},
        "sella": {},
        "cap": 200,
        "force_tol": 1e-3,
        "reactant": None,
        "file_mode": None,
        "reference_mode": None,
        "overlap": 0.99,
    }
    record = _dispatch(spec)
    assert record["success"], record
    assert record["hessian_index"] == 1
    assert record["counter"] == "harness"
    assert record["force_calls"] < 200
    assert np.linalg.norm(record["x"]) < 1e-2


@pytest.mark.skipif(not os.environ.get("RGSADDLE_LIB"), reason="RGSADDLE_LIB is not set")
def test_prfo_finds_the_quadratic_saddle():
    from optbench.rgsaddle_minmode import _dispatch

    def evaluate(coords):
        x, y, z = (float(value) for value in coords)
        return -x * x + y * y + 2.0 * z * z, (-2.0 * x, 2.0 * y, 4.0 * z)

    spec = {
        "name": "quadratic-prfo",
        "positions": np.array([[0.4, -0.3, 0.2]], dtype=np.float64),
        "fixed": np.array([False]),
        "cell": np.zeros(3),
        "evaluate": evaluate,
        "potential": {},
        "session": "prfo",
        "kind": "dimer",
        "method": "lbfgs",
        "mode_seed": "gradient",
        "task": "saddle",
        "saddle": {},
        "optimizer": {},
        "sella": {"delta": 0.5, "hess_update": "bfgs", "expand": "lanczos", "restricted": "trust-region", "order": 1},
        "cap": 100,
        "force_tol": 1e-3,
        "reactant": None,
        "file_mode": None,
        "reference_mode": None,
        "overlap": 0.99,
    }
    record = _dispatch(spec)
    assert record["success"], record
    assert record["hessian_index"] == 1
    assert np.linalg.norm(record["x"]) < 1e-2
