"""How a point the driver reached is judged, and how it minimizes and checkpoints.

Three surfaces carry the answers, and each one's stationary points are known
from its form. A pinned pair around one free atom, the pair of Lennard-Jones
clusters below, and a three-atom chain: two springs that hold the 1-2 and 2-3
distances at 1, and a double well in the 1-3 distance whose barrier top is at
1.4. At the chain's isosceles geometry the force vanishes and the Hessian is
the three internal curvatures carried to Cartesians: six null modes, two stiff
modes and exactly one negative one, so it is an index-1 saddle on a free
cluster with every rigid motion in the null space.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from optbench.rgsaddle_minmode import (
    HarnessSurface,
    _append_partial,
    _checkpoint_paths,
    _json_default,
    _lbfgs_minimize,
    _load_partial,
    _rigid_basis,
    _rng_restore,
    _rng_state,
    _saddle_record,
)
from optbench.surfaces import lennard_jones

SPRING = 10.0
PAIR = 2.0 ** (1.0 / 6.0)


def _well(distance):
    return (distance - 1.0) ** 2 * (distance - 1.8) ** 2


def _well_slope(distance):
    return 4.0 * (distance - 1.0) * (distance - 1.8) * (distance - 1.4)


def _chain(flat):
    positions = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    energy = 0.0
    gradient = np.zeros((3, 3))
    for (left, right), rest in (((0, 1), 1.0), ((1, 2), 1.0)):
        delta = positions[left] - positions[right]
        length = float(np.linalg.norm(delta))
        energy += SPRING * (length - rest) ** 2
        pull = 2.0 * SPRING * (length - rest) * delta / length
        gradient[left] += pull
        gradient[right] -= pull
    delta = positions[0] - positions[2]
    length = float(np.linalg.norm(delta))
    energy += _well(length)
    pull = _well_slope(length) * delta / length
    gradient[0] += pull
    gradient[2] -= pull
    return float(energy), gradient.reshape(-1)


APEX = float(np.sqrt(1.0 - 0.7**2))
SADDLE = np.array([[-0.7, 0.0, 0.0], [0.0, APEX, 0.0], [0.7, 0.0, 0.0]]).reshape(-1)
MINIMUM = np.array([[-0.5, 0.0, 0.0], [0.0, np.sqrt(0.75), 0.0], [0.5, 0.0, 0.0]]).reshape(-1)


def _tilted_well(tilt):
    """One free atom on a double well along x, held at rest across y and z, with a constant force of ``tilt`` along y."""

    def evaluate(x):
        x = np.asarray(x, dtype=np.float64)
        energy = (x[0] ** 2 - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2 + tilt * x[1]
        gradient = np.array([4.0 * x[0] * (x[0] ** 2 - 1.0), 2.0 * x[1] + tilt, 2.0 * x[2]])
        return float(energy), gradient

    return evaluate


def _judge(point, evaluate=_chain, **overrides):
    surface = HarnessSurface(evaluate)
    arguments = {
        "cap": 10000,
        "force_tol": 1e-3,
        "library_converged": True,
        "rotations": 3,
        "library_evals": 7,
        "steps": 5,
        "project_rigid": True,
    }
    arguments.update(overrides)
    return _saddle_record("bend", surface, np.asarray(point, dtype=np.float64), **arguments), surface


def _lj(flat):
    energy, gradient = lennard_jones(np.asarray(flat, dtype=np.float64).reshape(-1, 3))
    return float(energy), np.asarray(gradient, dtype=np.float64).reshape(-1)


# Minimizing from a displaced geometry


def test_a_stiff_and_a_soft_direction_are_both_minimized_to_the_force_stop():
    curvature = np.array([1.0, 4.0, 9.0, 16.0])

    def bowl(x):
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * float(np.sum(curvature * x * x)), curvature * x

    surface = HarnessSurface(bowl)
    x, converged = _lbfgs_minimize(surface, np.array([0.3, -0.2, 0.1, 0.25]), 1e-6, 500)
    assert converged
    assert np.linalg.norm(curvature * x) <= 1e-6
    assert surface.calls < 40


def test_a_displaced_dimer_relaxes_to_the_pair_distance():
    surface = HarnessSurface(_lj)
    x, converged = _lbfgs_minimize(surface, np.array([0.0, 0.0, 0.0, 1.4, 0.05, -0.03]), 1e-8, 200)
    assert converged
    assert np.linalg.norm(x[3:] - x[:3]) == pytest.approx(PAIR, abs=1e-7)
    assert _lj(x)[0] == pytest.approx(-1.0, abs=1e-10)


def test_a_displaced_tetrahedron_relaxes_to_six_pair_bonds():
    ideal = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) * (PAIR / (2.0 * np.sqrt(2.0)))
    shake = np.array([[0.05, -0.03, 0.04], [-0.06, 0.02, 0.03], [0.04, 0.05, -0.05], [-0.02, -0.06, 0.01]])
    surface = HarnessSurface(_lj)
    x, converged = _lbfgs_minimize(surface, (ideal + shake).reshape(-1), 1e-7, 400)
    assert converged
    energy, gradient = _lj(x)
    assert energy == pytest.approx(-6.0, abs=1e-9)
    assert np.linalg.norm(gradient) <= 1e-7
    assert surface.calls < 100


def test_a_start_already_under_the_force_stop_costs_one_call_and_is_returned_as_a_copy():
    surface = HarnessSurface(lambda x: (float(np.sum(np.asarray(x) ** 2)), 2.0 * np.asarray(x)))
    start = np.array([1e-9, 0.0, 0.0])
    x, converged = _lbfgs_minimize(surface, start, 1e-3, 100)
    assert converged
    assert surface.calls == 1
    assert np.array_equal(x, start)
    x[0] = 7.0
    assert start[0] == 1e-9


def test_the_budget_counts_the_calls_of_this_minimization_only():
    surface = HarnessSurface(lambda x: (float(np.sum(np.asarray(x) ** 2)), 2.0 * np.asarray(x)))
    for step in range(5):
        surface.energy_forces(np.array([float(step), 0.0, 0.0, 0.0]))
    x, converged = _lbfgs_minimize(surface, np.array([5.0, -4.0, 3.0, 2.0]), 1e-12, 4)
    assert not converged
    assert surface.calls == 5 + 4


def test_a_budget_of_one_spends_it_on_the_start_and_stops():
    surface = HarnessSurface(lambda x: (float(np.sum(np.asarray(x) ** 2)), 2.0 * np.asarray(x)))
    start = np.array([1.0, 2.0])
    x, converged = _lbfgs_minimize(surface, start, 1e-9, 1)
    assert not converged
    assert surface.calls == 1
    assert np.array_equal(x, start)


def test_the_first_trial_is_a_step_of_at_most_two_tenths_down_the_force():
    seen = []

    def bowl(x):
        x = np.asarray(x, dtype=np.float64)
        seen.append(x.copy())
        return 0.5 * float(np.sum(x * x)), x.copy()

    _lbfgs_minimize(HarnessSurface(bowl), np.array([5.0, 0.0, 0.0, 0.0]), 1e-3, 3)
    assert [row.tolist() for row in seen] == [[5.0, 0.0, 0.0, 0.0], [4.8, 0.0, 0.0, 0.0], [4.6, 0.0, 0.0, 0.0]]


def _stiff(seen):
    def stiff(x):
        x = np.asarray(x, dtype=np.float64)
        seen.append(x.tolist())
        return 50.0 * float(np.sum(x * x)), 100.0 * x

    return stiff


def test_a_trial_that_does_not_lower_the_energy_is_halved_until_one_does():
    seen = []
    surface = HarnessSurface(_stiff(seen))
    x, converged = _lbfgs_minimize(surface, np.array([0.1]), 1e-6, 50)
    assert seen == [[0.1], [-0.1], [0.0]]
    assert converged
    assert x.tolist() == [0.0]


def test_the_budget_stops_the_halving_too():
    seen = []
    surface = HarnessSurface(_stiff(seen))
    x, converged = _lbfgs_minimize(surface, np.array([0.1]), 1e-6, 2)
    assert seen == [[0.1], [-0.1]]
    assert not converged
    assert x.tolist() == [0.1]


def test_a_surface_whose_force_lies_about_its_energy_stops_the_minimization_unconverged():
    def lying(x):
        x = np.asarray(x, dtype=np.float64)
        return 10.0 + 1000.0 * float(np.sum((x - 1.0) ** 2)), np.ones_like(x)

    surface = HarnessSurface(lying)
    x, converged = _lbfgs_minimize(surface, np.array([1.0, 1.0]), 1e-3, 100)
    assert not converged
    assert x.tolist() == [1.0, 1.0]
    assert surface.calls == 1 + 10


def test_a_long_narrow_valley_is_minimized_in_fewer_calls_than_steps_down_it_one_by_one():
    curvature = np.geomspace(1.0, 400.0, 30)

    def bowl(x):
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * float(np.sum(curvature * x * x)), curvature * x

    surface = HarnessSurface(bowl)
    x, converged = _lbfgs_minimize(surface, np.full(30, 0.05), 1e-8, 2000)
    assert converged
    assert np.linalg.norm(curvature * x) <= 1e-8
    assert surface.calls < 1200


# Judging a point as a saddle


def test_the_chain_has_a_vanishing_force_at_its_barrier_top_and_at_its_minimum():
    assert np.linalg.norm(_chain(SADDLE)[1]) < 1e-12
    assert np.linalg.norm(_chain(MINIMUM)[1]) < 1e-12
    assert _chain(MINIMUM)[0] == pytest.approx(0.0, abs=1e-12)
    assert _chain(SADDLE)[0] > 0.0


def test_a_converged_index_one_point_is_a_success_and_its_record_is_filled_in():
    record, surface = _judge(SADDLE)
    assert record["schema"] == "optbench.start/v2"
    assert record["start"] == "bend"
    assert (record["counter"], record["comparable"], record["reference"]) == ("harness", True, "converged-point")
    assert record["success"] is True
    assert record["failure"] is None
    assert record["reasons"] == []
    assert (record["hessian_index"], record["expected_index"]) == (1, 1)
    assert record["converged"] is True
    assert (record["rotations"], record["library_evaluations"], record["steps"]) == (3, 7, 5)
    assert record["energy"] == pytest.approx(_chain(SADDLE)[0], abs=1e-12)
    assert record["force_norm"] < 1e-12
    assert record["rmsd"] == 0.0
    assert record["x"] == SADDLE.tolist()


def test_the_hessian_is_taken_as_verification_and_costs_no_search_call():
    record, surface = _judge(SADDLE)
    assert record["force_calls"] == surface.calls == 1
    assert record["verify_calls"] == surface.verify_calls == 2 * SADDLE.size


def test_the_record_lists_the_eight_lowest_eigenvalues_and_the_saddle_mode_is_the_first():
    record, _surface = _judge(SADDLE)
    values = record["lowest_eigenvalues"]
    assert len(values) == 8
    assert values == sorted(values)
    assert record["lowest_eigenvalue"] == values[0] < -0.5
    assert values[1:7] == [0.0] * 6
    assert values[7] > 1.0


def test_rigid_modes_are_removed_from_a_free_cluster_and_kept_when_asked_not_to():
    projected, _ = _judge(SADDLE)
    assert projected["rigid_modes_removed"] == 6
    assert projected["hessian_index"] == projected["cartesian_index"] == 1
    kept, _ = _judge(SADDLE, project_rigid=False)
    assert kept["rigid_modes_removed"] == 0
    assert kept["hessian_index"] == kept["cartesian_index"] == 1
    assert kept["success"] is True


def test_a_point_with_a_small_residual_force_is_index_one_only_once_its_rotations_are_removed():
    point = SADDLE.copy()
    point[0] -= 1e-5
    projected, _ = _judge(point)
    assert 1e-4 < projected["force_norm"] < 1e-3
    assert projected["hessian_index"] == 1
    assert projected["cartesian_index"] == 2
    assert projected["rigid_modes_removed"] == 6
    assert projected["success"] is True
    kept, _ = _judge(point, project_rigid=False)
    assert kept["hessian_index"] == kept["cartesian_index"] == 2
    assert kept["reasons"] == ["hessian_index"]
    assert kept["success"] is False


def test_one_free_atom_has_no_rigid_subspace_to_remove():
    record, _surface = _judge(np.zeros(3), _tilted_well(0.0))
    assert record["rigid_modes_removed"] == 0
    assert record["hessian_index"] == 1
    assert record["success"] is True


def test_a_minimum_is_a_wrong_stationary_point_not_a_saddle():
    record, _surface = _judge(MINIMUM)
    assert record["success"] is False
    assert record["failure"] == "wrong_stationary"
    assert record["reasons"] == ["hessian_index"]
    assert record["hessian_index"] == 0
    assert record["expected_index"] == 1
    assert record["lowest_eigenvalue"] == 0.0


def test_a_force_above_the_stop_is_a_divergence_even_at_index_one():
    record, _surface = _judge(np.zeros(3), _tilted_well(5e-3))
    assert record["hessian_index"] == 1
    assert record["reasons"] == ["force_norm"]
    assert record["failure"] == "divergence"
    assert record["success"] is False
    assert record["converged"] is False
    assert record["force_norm"] == pytest.approx(5e-3, abs=1e-15)


def test_the_force_stop_allows_a_rounding_margin_of_a_tenth_of_a_nanounit_and_no_more():
    inside, _ = _judge(np.zeros(3), _tilted_well(1e-3 + 5e-11))
    assert inside["reasons"] == []
    assert inside["success"] is True
    outside, _ = _judge(np.zeros(3), _tilted_well(1e-3 + 2e-10))
    assert outside["reasons"] == ["force_norm"]
    assert outside["success"] is False


def test_a_search_that_spent_the_cap_is_a_cap_failure_even_at_a_saddle():
    spent, _ = _judge(SADDLE, cap=1)
    assert spent["reasons"] == ["cap"]
    assert spent["failure"] == "cap"
    assert spent["success"] is False
    assert spent["hessian_index"] == 1
    inside, _ = _judge(SADDLE, cap=2)
    assert inside["success"] is True


def test_the_cap_outranks_the_force_in_naming_the_failure_and_both_reasons_are_kept():
    record, _surface = _judge(np.zeros(3), _tilted_well(5e-3), cap=1)
    assert record["reasons"] == ["cap", "force_norm"]
    assert record["failure"] == "cap"


def test_the_library_status_alone_does_not_reject_a_point_under_the_stop_with_index_one():
    record, _surface = _judge(SADDLE, library_converged=False)
    assert record["success"] is True
    assert record["failure"] is None
    assert record["converged"] is False


# The rigid motions of a free cluster


def test_a_single_atom_has_no_rigid_basis():
    assert _rigid_basis(np.zeros(3)).shape == (3, 0)


@pytest.mark.parametrize(
    ("points", "modes"),
    [
        ([[0, 0, 0], [1.4, 0, 0]], 5),
        ([[0, 0, 0], [0, 1.4, 0]], 5),
        ([[0, 0, 0], [0, 0, 1.4]], 5),
        ([[0.0, 0.1, 0.2], [0.7, 0.9, 0.6]], 5),
        ([[0, 0, 0], [1, 0, 0], [0.2, 1.1, 0]], 6),
        ([[0, 0, 0], [1, 0, 0], [0.2, 1.1, 0], [0.1, 0.2, 1.3]], 6),
        ([[-1.1, 0, 0], [0, 0, 0], [1.1, 0, 0]], 5),
    ],
    ids=["pair-x", "pair-y", "pair-z", "pair-oblique", "triangle", "tetrahedron", "line"],
)
def test_a_free_cluster_of_two_atoms_or_more_has_an_orthonormal_basis_of_its_rigid_motions(points, modes):
    positions = np.asarray(points, dtype=float)
    basis = _rigid_basis(positions.reshape(-1))
    assert basis.shape == (positions.size, modes)
    assert np.allclose(basis.T @ basis, np.eye(modes), atol=1e-12)
    centered = positions - positions.mean(axis=0)
    motions = [np.tile(axis, (len(positions), 1)) for axis in np.eye(3)]
    motions += [np.cross(axis, centered) for axis in np.eye(3)]
    for motion in motions:
        flat = motion.reshape(-1)
        assert np.linalg.norm(flat - basis @ (basis.T @ flat)) < 1e-10


def _pair_barrier(flat):
    """Two free atoms joined by the double well, whose top is at a distance of 1.4."""
    positions = np.asarray(flat, dtype=np.float64).reshape(2, 3)
    delta = positions[0] - positions[1]
    length = float(np.linalg.norm(delta))
    pull = _well_slope(length) * delta / length
    return _well(length), np.concatenate([pull, -pull])


def test_a_pair_just_past_its_barrier_top_is_index_one_once_its_rotations_are_removed():
    start = np.array([0.1, 0.2, 0.3])
    direction = np.array([1.0, 2.0, 2.0]) / 3.0
    point = np.concatenate([start, start + (1.4 + 1e-4) * direction])
    projected, _ = _judge(point, _pair_barrier)
    assert projected["force_norm"] < 1e-3
    assert projected["cartesian_index"] == 3
    assert projected["hessian_index"] == 1
    assert projected["rigid_modes_removed"] == 5
    assert projected["success"] is True
    kept, _ = _judge(point, _pair_barrier, project_rigid=False)
    assert kept["hessian_index"] == 3
    assert kept["reasons"] == ["hessian_index"]


# The checkpoint files a catalogue keeps beside its output


def test_the_header_and_the_partial_rows_sit_beside_the_output_under_its_name(tmp_path):
    header, partial = _checkpoint_paths(tmp_path / "out" / "catalogue.dat")
    assert header == tmp_path / "out" / "catalogue.header.json"
    assert partial == tmp_path / "out" / "catalogue.partial.jsonl"


def test_a_missing_partial_file_has_no_rows_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "rows.partial.jsonl"
    assert _load_partial(path) == []
    path.write_text('{"search": 1}\n\n   \n{"search": 2}\n', encoding="utf-8")
    assert _load_partial(path) == [{"search": 1}, {"search": 2}]


def test_rows_are_appended_one_per_line_in_order_and_numpy_numbers_are_written_as_numbers(tmp_path):
    path = tmp_path / "deep" / "rows.partial.jsonl"
    _append_partial(path, {"search": np.int64(1), "barrier": np.float64(0.25), "ok": np.bool_(True)})
    _append_partial(path, {"search": 2, "barrier": 0.5, "ok": False})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"search": 1, "barrier": 0.25, "ok": True}
    assert lines[1] == '{"barrier": 0.5, "ok": false, "search": 2}'
    assert _load_partial(path)[1] == {"search": 2, "barrier": 0.5, "ok": False}


def test_a_value_json_cannot_write_is_named_by_its_type_and_the_rows_before_it_stay(tmp_path):
    path = tmp_path / "rows.partial.jsonl"
    _append_partial(path, {"search": 1})
    with pytest.raises(TypeError, match="set"):
        _append_partial(path, {"search": 2, "seen": {1, 2}})
    assert _load_partial(path) == [{"search": 1}]
    with pytest.raises(TypeError, match="complex"):
        _json_default(1j)
    assert _json_default(np.float32(0.5)) == 0.5


def test_a_saved_random_state_continues_the_stream_where_it_stopped():
    rng = np.random.default_rng(20261016)
    rng.normal(size=5)
    saved = json.loads(json.dumps(_rng_state(rng)))
    expected = rng.normal(size=4)
    assert np.array_equal(_rng_restore(saved).normal(size=4), expected)
