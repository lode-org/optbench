"""The minimum-mode walk of the SoftSaddle campaign and how a finished walk is judged.

The walk is an L-BFGS descent on a surface where the gradient is reflected
through the lowest curvature mode. The 2017 eigen-solvers are shared objects
that this environment does not build, so their place is taken by an exact
solve on a surface whose Hessian is known: one atom in a double well with its
minima at x = -1 and x = +1 and an index-one saddle at the origin, barrier
1. Everything else is the code under test: the memory, the step rules, the
mode reuse, the judging of the end point, and the checkpoint files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from optbench import softsaddle_run
from optbench.connection import ProcessTable
from optbench.rgsaddle_minmode import HarnessSurface
from optbench.softsaddle_run import (
    INITIAL_GRADIENT,
    INITIAL_PREVIOUS,
    INITIAL_USER,
    _LBFGS,
    _analyze,
    _append_partial,
    _apply_novelty,
    _checkpoint_paths,
    _continues,
    _displace,
    _effective,
    _jsonable,
    _judge_saddle,
    _lambda_zero,
    _load_partial,
    _max_motion,
    _mode_at,
    _radius_update,
    _replay_radius,
    _settings,
    _sign,
    climb,
)


def _well(x):
    x = np.asarray(x, dtype=np.float64)
    energy = (x[0] ** 2 - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2
    gradient = np.array([4.0 * x[0] * (x[0] ** 2 - 1.0), 2.0 * x[1], 2.0 * x[2]])
    return float(energy), gradient


def _lowest_mode(x):
    diagonal = np.array([12.0 * x[0] ** 2 - 4.0, 2.0, 2.0])
    index = int(np.argmin(diagonal))
    mode = np.zeros(3)
    mode[index] = 1.0
    return float(diagonal[index]), mode


@pytest.fixture
def exact_mode(monkeypatch):
    """Stand in for the shared object: the exact lowest mode of the double well."""
    seen: list[dict] = []

    def solve(lib, potential, x, *, initial, iteration_limit, finite, tolerance, seed_vector=None):
        seen.append({"x": np.array(x), "initial": initial, "seed": seed_vector})
        energy, gradient = potential.energy_forces(x)
        eigenvalue, mode = _lowest_mode(np.asarray(x))
        return eigenvalue, mode, gradient, 3, 0, energy

    monkeypatch.setattr(softsaddle_run, "_minimum_mode", solve)
    return seen


TEMPLATE = {
    "ini": {
        "Optimizer": {"max_step": 0.5, "memory_size": 60, "max_iterations": 200, "cg_err": 1e-3},
        "SaddleSearch": {"lanc_iter": 8},
    }
}


def _climb(start, *, potential=None, force_tol=1e-3, minimum_energy=0.0, budget=1000, **override):
    settings = _settings(TEMPLATE)
    settings.update(override)
    surface = potential or HarnessSurface(_well)
    result = climb(
        None,
        surface,
        np.array(start, dtype=np.float64),
        settings=settings,
        force_tol=force_tol,
        minimum_energy=minimum_energy,
        climb_budget=budget,
    )
    return result, surface


def test_the_gradient_is_reflected_through_the_mode_when_the_curvature_is_negative():
    gradient = np.array([3.0, 4.0, 0.0])
    reflected = _effective(gradient, -1.0, np.array([1.0, 0.0, 0.0]))
    assert reflected == pytest.approx([-3.0, 4.0, 0.0])


def test_only_the_mode_part_of_the_gradient_survives_when_the_curvature_is_not_negative():
    gradient = np.array([3.0, 4.0, 0.0])
    mode = np.array([2.0, 0.0, 0.0])
    assert _effective(gradient, 0.5, mode) == pytest.approx([-3.0, 0.0, 0.0])
    assert _effective(gradient, 0.0, mode) == pytest.approx([-3.0, 0.0, 0.0])
    assert _effective(gradient, -1e-12, mode) == pytest.approx([-3.0, 4.0, 0.0])


def test_a_step_longer_than_the_limit_is_scaled_to_it_and_a_shorter_one_is_kept():
    long = np.array([3.0, 4.0])
    assert _max_motion(long, 1.0) == pytest.approx([0.6, 0.8])
    assert _max_motion(long, 5.0) is long
    assert _max_motion(long, 10.0) is long
    assert _max_motion(long, 0.0) is long


def test_sign_leaves_zero_at_zero_even_for_negative_zero():
    assert (_sign(2.5), _sign(-0.1), _sign(0.0), _sign(-0.0)) == (1.0, -1.0, 0.0, 0.0)
    assert math.copysign(1.0, _sign(-0.0)) == 1.0


def test_the_curvature_zero_is_found_by_linear_interpolation_between_two_points():
    crossing = _lambda_zero(np.array([0.0, 0.0]), np.array([4.0, 0.0]), 1.0, -3.0)
    assert crossing == pytest.approx([1.0, 0.0])
    diagonal = _lambda_zero(np.array([1.0, 1.0]), np.array([4.0, 5.0]), 2.0, -2.0)
    assert diagonal == pytest.approx([1.0 + 0.5 * 3.0, 1.0 + 0.5 * 4.0])


@pytest.mark.parametrize(
    ("positive", "negative", "up", "down"),
    [
        ([1.0, 1.0], [1.0, 1.0], 1.0, -1.0),
        ([0.0, 0.0], [2.0, 0.0], 1.0, 1.0),
        ([0.0, 0.0], [2.0, 0.0], float("inf"), -1.0),
        ([0.0, 0.0], [2.0, 0.0], 1.0, float("nan")),
    ],
)
def test_there_is_no_crossing_between_the_same_point_or_equal_or_unbounded_values(positive, negative, up, down):
    assert _lambda_zero(np.array(positive), np.array(negative), up, down) is None


def test_a_saved_mode_is_kept_while_the_steps_since_it_was_found_stay_inside_the_tolerance():
    mode, vector = -2.0, np.array([1.0, 0.0])
    kept = _analyze(mode, vector, np.array([0.1, 0.0]), None, 1, 0.25)
    assert kept[0] == mode and kept[1] is vector and kept[3] == 1
    assert kept[2] == pytest.approx([0.1, 0.0])
    again = _analyze(mode, vector, np.array([0.1, 0.0]), kept[2], 2, 0.25)
    assert again[0] == mode and again[3] == 2
    assert again[2] == pytest.approx([0.2, 0.0])
    dropped = _analyze(mode, vector, np.array([0.1, 0.0]), again[2], 3, 0.25)
    assert dropped[0] is None and dropped[1] is None and dropped[3] == 0
    assert dropped[2] == pytest.approx([0.0, 0.0])


def test_a_saved_mode_is_kept_when_the_accumulated_step_is_exactly_the_tolerance():
    mode, vector = -2.0, np.array([1.0, 0.0])
    at_limit = _analyze(mode, vector, np.array([0.25, 0.0]), None, 1, 0.25)
    assert at_limit[0] == mode
    past_limit = _analyze(mode, vector, np.array([0.2500001, 0.0]), None, 1, 0.25)
    assert past_limit[0] is None


def test_the_settings_take_the_template_values_and_otherwise_the_archived_defaults():
    defaults = _settings({})
    assert (defaults["memory"], defaults["max_step"], defaults["batch"], defaults["hyper_rad0"]) == (60, 0.5, 25, 0.1)
    assert (defaults["solver"], defaults["seed_mode"], defaults["iteration_limit"]) == ("lanczos", "previous", 8)
    assert defaults["reuse_tolerance"] == pytest.approx(0.01 + (0.5 - 0.01) * 50.0 / 100.0)
    assert defaults["cg_iter_max"] == softsaddle_run.CLIMB_BUDGET
    template = {
        "ini": {
            "Main": {"solver": "davidson"},
            "Optimizer": {"max_step": 0.2, "ls_err": 0.02, "memory_size": 10, "finite_diff": 2e-4},
            "SaddleSearch": {"mode_seed": "gradient", "save_lanczos_factor": 10.0, "lanc_iter": 4, "save_lanczos_call": False},
            "Search": {"batch": 5, "mix_prev_hyper_rad": False, "keep_efficiency": 0.5},
        }
    }
    custom = _settings(template)
    assert (custom["solver"], custom["seed_mode"], custom["iteration_limit"]) == ("davidson", "gradient", 4)
    assert (custom["memory"], custom["max_step"], custom["batch"]) == (10, 0.2, 5)
    assert custom["finite"] == pytest.approx(2e-4)
    assert custom["reuse_tolerance"] == pytest.approx(0.02 + (0.2 - 0.02) * 0.1)
    assert custom["save_lanczos"] is False
    assert custom["mix_radius"] is False
    assert custom["keep_efficiency"] == 0.5


def _pair(memory: _LBFGS, x_new, x_old, g_new, g_old):
    memory.update(np.asarray(x_new, float), np.asarray(x_old, float), np.asarray(g_new, float), np.asarray(g_old, float))


def test_the_memory_keeps_the_newest_pairs_and_drops_the_oldest_when_it_is_full():
    memory = _LBFGS(1, 2)
    for k in range(1, 4):
        _pair(memory, [float(k)], [0.0], [float(2 * k)], [0.0])
    assert memory.count == 2
    assert memory.s[0].tolist() == [2.0, 3.0]
    assert memory.y[0].tolist() == [4.0, 6.0]
    assert memory.rho.tolist() == pytest.approx([1.0 / 8.0, 1.0 / 18.0])


def test_a_pair_with_no_curvature_has_no_weight():
    memory = _LBFGS(2, 3)
    _pair(memory, [1.0, 0.0], [0.0, 0.0], [0.0, 3.0], [0.0, 0.0])
    assert memory.count == 1
    assert memory.rho[0] == 0.0


def _bowl(hessian):
    hessian = np.asarray(hessian, dtype=np.float64)
    return HarnessSurface(lambda x: (0.5 * float(np.asarray(x) @ hessian @ np.asarray(x)), hessian @ np.asarray(x)))


def _step(memory, x, g, surface, *, iteration, max_step=100.0, eigenvalue=None, mode=None, bandera=False):
    return memory.step(
        np.asarray(x, float),
        np.asarray(g, float),
        iteration=iteration,
        finite=1e-4,
        max_step=max_step,
        potential=surface,
        eigenvalue=eigenvalue,
        mode=mode,
        bandera=bandera,
    )


def test_two_conjugate_pairs_on_a_quadratic_give_the_newton_step():
    hessian = np.array([[3.0, 1.0], [1.0, 2.0]])
    surface = _bowl(hessian)
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])
    second = second - (first @ hessian @ second) / (first @ hessian @ first) * first
    start = np.array([0.7, -0.4])
    memory = _LBFGS(2, 5)
    middle = start + first
    end = middle + second
    _pair(memory, middle, start, hessian @ middle, hessian @ start)
    _pair(memory, end, middle, hessian @ end, hessian @ middle)
    step, forced = _step(memory, end, hessian @ end, surface, iteration=2)
    assert forced is False
    assert step == pytest.approx(-np.linalg.solve(hessian, hessian @ end), abs=1e-12)
    assert memory.count == 2


def test_the_step_is_the_explicit_bfgs_inverse_applied_to_the_gradient_for_pairs_that_are_not_conjugate():
    rng = np.random.default_rng(7)
    root = rng.normal(size=(3, 3))
    hessian = root @ root.T + 3.0 * np.eye(3)
    surface = _bowl(hessian)
    memory = _LBFGS(3, 5)
    point = rng.normal(size=3)
    for _ in range(3):
        move = rng.normal(size=3)
        _pair(memory, point + move, point, hessian @ (point + move), hessian @ point)
        point = point + move
    gradient = rng.normal(size=3)
    last = memory.count - 1
    scale = float(memory.s[:, last] @ memory.y[:, last]) / float(memory.y[:, last] @ memory.y[:, last])
    inverse = scale * np.eye(3)
    for index in range(memory.count):
        s, y = memory.s[:, index], memory.y[:, index]
        rho = 1.0 / float(y @ s)
        left = np.eye(3) - rho * np.outer(s, y)
        inverse = left @ inverse @ left.T + rho * np.outer(s, s)
    step, _ = _step(memory, point, gradient, surface, iteration=3)
    assert step == pytest.approx(-inverse @ gradient, rel=1e-9, abs=1e-12)
    assert memory.count == 3


def test_one_pair_in_one_dimension_gives_the_newton_step():
    surface = _bowl([[5.0]])
    memory = _LBFGS(1, 3)
    _pair(memory, [1.5], [2.0], [7.5], [10.0])
    step, _ = _step(memory, [1.5], [7.5], surface, iteration=1)
    assert step == pytest.approx([-1.5])


def test_a_newton_step_longer_than_the_limit_is_replaced_by_the_short_steepest_descent_step():
    surface = _bowl([[5.0]])
    memory = _LBFGS(1, 3)
    _pair(memory, [1.5], [2.0], [7.5], [10.0])
    step, _ = _step(memory, [1.5], [7.5], surface, iteration=1, max_step=0.5)
    assert step == pytest.approx([-0.5])
    assert memory.count == 0


def test_the_first_step_is_steepest_descent_scaled_by_the_curvature_along_it():
    hessian = np.array([[3.0, 1.0], [1.0, 2.0]])
    surface = _bowl(hessian)
    x = np.array([1.0, 1.0])
    gradient = hessian @ x
    direction = -gradient / np.linalg.norm(gradient)
    curvature = direction @ hessian @ direction
    step, forced = _step(_LBFGS(2, 5), x, gradient, surface, iteration=0)
    assert forced is False
    assert step == pytest.approx(-gradient / curvature, rel=1e-6)
    clamped, _ = _step(_LBFGS(2, 5), x, gradient, surface, iteration=0, max_step=0.5)
    assert np.linalg.norm(clamped) == pytest.approx(0.5)
    assert clamped == pytest.approx(direction * 0.5, rel=1e-6)


def test_the_first_step_measures_curvature_on_the_reflected_surface_when_a_mode_is_known():
    saddle = HarnessSurface(lambda x: (0.0, np.array([-x[0], x[1]])))
    x = np.array([0.5, 0.5])
    effective = _effective(np.array([-0.5, 0.5]), -1.0, np.array([1.0, 0.0]))
    assert effective == pytest.approx([0.5, 0.5])
    step, _ = _step(
        _LBFGS(2, 5), x, effective, saddle, iteration=0, eigenvalue=-1.0, mode=np.array([1.0, 0.0])
    )
    assert step == pytest.approx([-0.5, -0.5], abs=1e-6)


def test_negative_curvature_on_the_first_step_sends_a_long_downhill_step_cut_to_the_limit():
    inverted = HarnessSurface(lambda x: (-float(np.sum(np.asarray(x) ** 2)), -2.0 * np.asarray(x)))
    x = np.array([0.3, 0.4])
    memory = _LBFGS(2, 5)
    memory.count = 3
    step, forced = _step(memory, x, -2.0 * x, inverted, iteration=0, max_step=0.5)
    assert forced is False
    assert step == pytest.approx([0.3, 0.4])
    assert memory.count == 0


def test_a_first_step_from_a_flat_point_is_a_kick_along_the_first_coordinate():
    surface = _bowl(np.eye(3))
    step, _ = _step(_LBFGS(3, 5), np.zeros(3), np.zeros(3), surface, iteration=0, max_step=0.5)
    assert step == pytest.approx([0.5, 0.0, 0.0])
    unit, _ = _step(_LBFGS(3, 5), np.zeros(3), np.zeros(3), surface, iteration=0, max_step=5.0)
    assert unit == pytest.approx([1.0, 0.0, 0.0])


def test_the_forced_flag_and_an_empty_memory_both_restart_from_steepest_descent():
    surface = _bowl(np.eye(2))
    x = np.array([3.0, 4.0])
    for flagged, count in ((True, 1), (False, 0)):
        memory = _LBFGS(2, 5)
        if count:
            _pair(memory, x, x - 1.0, x, x - 1.0)
        step, forced = _step(memory, x, x, surface, iteration=3, max_step=100.0, bandera=flagged)
        assert forced is False
        assert step == pytest.approx([-60.0, -80.0])
        assert memory.count == 0


def test_a_last_pair_with_negative_curvature_empties_the_memory():
    surface = _bowl(np.eye(1))
    memory = _LBFGS(1, 2)
    _pair(memory, [1.0], [0.0], [-1.0], [0.0])
    step, _ = _step(memory, [1.0], [-1.0], surface, iteration=1, max_step=0.5)
    assert step == pytest.approx([0.5])
    assert memory.count == 0


def test_a_memory_that_proposes_an_uphill_step_is_dropped_for_scaled_steepest_descent():
    surface = _bowl(np.eye(2))
    memory = _LBFGS(2, 4)
    origin = np.zeros(2)
    _pair(memory, [-2.0, -2.0], origin, [-1.0, 2.0], origin)
    _pair(memory, [-2.0, -1.0], [-2.0, -2.0], [2.0, 1.0], origin)
    assert memory.count == 2
    gradient = np.array([1.0, 0.0])
    step, _ = _step(memory, origin, gradient, surface, iteration=2)
    assert step == pytest.approx([-0.2, 0.0])
    assert memory.count == 0


@pytest.mark.parametrize(
    ("radius", "rows", "lam", "sad", "expected", "lam_after", "sad_after"),
    [
        (0.1, [], [], [], 0.1, [], []),
        (0.1, [{"success": False}], [], [], 0.1, [], []),
        (0.1, [{"success": True, "lambda_distance": 0.6, "saddle_distance": 0.9}] * 2, [], [], 0.1 * 0.5 + 0.3, [0.6, 0.6], [0.9, 0.9]),
        (0.1, [{"success": True}], [], [1.0], 0.1, [], [1.0]),
        (0.1, [{"success": True, "lambda_distance": 0.0, "saddle_distance": 0.0}], [0.5], [], 0.3, [0.5], []),
        (0.1, [{"success": False}], [0.5], [0.8], 0.3, [0.5], [0.8]),
    ],
)
def test_the_radius_moves_to_the_mean_lambda_distance_when_few_searches_connect(
    radius, rows, lam, sad, expected, lam_after, sad_after
):
    settings = {"batch": 25, "mix_radius": True, "keep_efficiency": 0.2}
    lam, sad = list(lam), list(sad)
    assert _radius_update(radius, rows, lam, sad, settings) == pytest.approx(expected)
    assert lam == lam_after
    assert sad == sad_after


def test_the_radius_is_kept_when_enough_searches_connect_or_the_proposal_is_unusable():
    rows = [{"success": True, "lambda_distance": 0.6}] * 5
    keep = {"batch": 25, "mix_radius": False, "keep_efficiency": 0.2}
    assert _radius_update(0.1, rows, [], [], keep) == 0.1
    move = {"batch": 25, "mix_radius": False, "keep_efficiency": 0.5}
    assert _radius_update(0.1, rows, [], [], move) == pytest.approx(0.6)
    assert _radius_update(0.1, rows, [float("nan")], [], move) == 0.1
    assert _radius_update(0.1, [{"success": True}], [-1.0, -1.0], [], move) == 0.1


def test_replaying_the_radius_uses_whole_batches_only_and_returns_the_distances_it_saw():
    settings = {"batch": 2, "hyper_rad0": 0.2, "mix_radius": False, "keep_efficiency": 0.9}
    good = {"success": True, "lambda_distance": 0.5, "saddle_distance": 0.7}
    bad = {"success": False}
    radius, lam, sad = _replay_radius([good, bad, good, good, good], settings)
    assert radius == pytest.approx(0.5)
    assert lam == [0.5, 0.5, 0.5]
    assert sad == [0.7, 0.7, 0.7]
    assert _replay_radius([good], settings) == (0.2, [], [])
    assert _replay_radius([], settings) == (0.2, [], [])


def test_a_displacement_moves_only_the_first_atoms_and_leaves_its_input_alone():
    minimum = np.arange(9, dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    moved = _displace(minimum, direction, 0.5, 2)
    assert moved.tolist() == [0.5, 1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0, 8.0]
    assert minimum.tolist() == list(range(9))


def test_the_checkpoint_files_sit_beside_the_output_and_take_its_stem(tmp_path: Path):
    header, partial = _checkpoint_paths(tmp_path / "out" / "softsaddle-x.dat")
    assert header == tmp_path / "out" / "softsaddle-x.header.json"
    assert partial == tmp_path / "out" / "softsaddle-x.partial.jsonl"


def test_partial_results_are_appended_one_per_line_and_read_back_in_order(tmp_path: Path):
    path = tmp_path / "deep" / "run.partial.jsonl"
    assert _load_partial(path) == []
    _append_partial(path, {"b": 2, "a": 1})
    _append_partial(path, {"start": "rep1-search2", "success": False})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"a": 1, "b": 2}'
    path.write_text("\n".join([lines[0], "", "   ", lines[1]]) + "\n", encoding="utf-8")
    assert _load_partial(path) == [{"a": 1, "b": 2}, {"start": "rep1-search2", "success": False}]


def test_numpy_values_become_plain_json_values():
    record = {"x": np.array([1.0, 2.0]), "n": np.int64(3), "flag": np.bool_(True), "name": "a", "none": None}
    plain = _jsonable(record)
    assert plain == {"x": [1.0, 2.0], "n": 3, "flag": True, "name": "a", "none": None}
    assert json.dumps(plain)
    assert type(plain["n"]) is int


def test_a_checkpoint_is_continued_only_when_asked_and_only_for_the_same_template():
    assert _continues({"template": "softsaddle/a"}, "softsaddle/a", True) is True
    assert _continues({"template": "softsaddle/a"}, "softsaddle/a", False) is False
    assert _continues({"template": "softsaddle/b"}, "softsaddle/a", True) is False
    assert _continues(None, "softsaddle/a", True) is False
    assert _continues({}, "softsaddle/a", True) is False


def test_a_climb_from_inside_the_barrier_ends_on_the_saddle(exact_mode):
    result, surface = _climb([-0.4, 0.15, -0.1])
    assert result["converged"] is True
    assert result["error"] is None
    assert result["x"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    assert result["eigenvalue"] == pytest.approx(-4.0, abs=1e-6)
    assert result["force_norm"] <= 1e-3
    assert result["energy"] == pytest.approx(1.0, abs=1e-6)
    assert result["mode"].tolist() == [1.0, 0.0, 0.0]
    assert 0 < result["iterations"] < 10
    assert surface.calls < 20


def test_a_climb_gets_to_the_same_saddle_from_other_displacements(exact_mode):
    for start in ([-0.3, 0.0, 0.0], [-0.55, 0.2, 0.2], [0.35, -0.1, 0.1]):
        result, _ = _climb(start)
        assert result["converged"] is True
        assert result["x"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_a_climb_stops_at_its_iteration_limit_without_converging(exact_mode):
    result, _ = _climb([-0.4, 0.15, -0.1], cg_iter_max=2)
    assert result["converged"] is False
    assert result["iterations"] == 2
    assert result["force_norm"] > 1e-3
    assert result["error"] is None


def test_a_climb_in_a_convex_basin_follows_the_soft_mode_until_the_energy_ceiling(exact_mode):
    result, _ = _climb([-0.9, 0.1, 0.1])
    assert result["converged"] is False
    assert result["energy"] >= 480.0
    assert result["iterations"] < 200
    assert result["x_lambda"] is None


def test_a_climb_stops_after_the_step_that_crosses_the_energy_ceiling(exact_mode):
    result, _ = _climb([-0.4, 0.15, -0.1], dE_max=0.5)
    assert result["converged"] is False
    assert result["iterations"] == 1
    assert result["energy"] - 0.0 >= 0.5


def test_a_climb_stops_when_the_force_call_budget_is_used(exact_mode):
    result, surface = _climb([-0.4, 0.15, -0.1], budget=3)
    assert result["converged"] is False
    assert surface.calls >= 3
    assert result["iterations"] < 4


def test_a_saved_mode_is_reused_so_the_solver_is_called_less_often(exact_mode):
    saved, _ = _climb([-0.4, 0.15, -0.1], save_lanczos=True)
    solves_saved = len(exact_mode)
    exact_mode.clear()
    fresh, _ = _climb([-0.4, 0.15, -0.1], save_lanczos=False)
    solves_fresh = len(exact_mode)
    assert solves_saved < solves_fresh
    assert saved["converged"] and fresh["converged"]
    assert saved["x"] == pytest.approx(fresh["x"], abs=1e-3)


def test_a_converged_walk_is_checked_once_more_with_a_fresh_mode_and_refused_if_that_is_not_negative(monkeypatch):
    def solve(lib, potential, x, **_):
        energy, gradient = potential.energy_forces(x)
        eigenvalue, mode = _lowest_mode(np.asarray(x))
        if np.linalg.norm(x) < 1e-2:
            eigenvalue = 1.0
        return eigenvalue, mode, gradient, 3, 0, energy

    monkeypatch.setattr(softsaddle_run, "_minimum_mode", solve)
    result, _ = _climb([-0.4, 0.15, -0.1])
    assert result["converged"] is False
    assert result["eigenvalue"] == 1.0
    assert result["error"] is None


def test_a_mode_that_is_not_finite_ends_the_climb_with_an_error_and_the_point_it_had(monkeypatch):
    def solve(lib, potential, x, **_):
        energy, gradient = potential.energy_forces(x)
        return float("nan"), np.array([1.0, 0.0, 0.0]), gradient, 1, 0, energy

    monkeypatch.setattr(softsaddle_run, "_minimum_mode", solve)
    start = [-0.4, 0.15, -0.1]
    result, _ = _climb(start)
    assert result["error"] == "softsaddle mode returned a non-finite value"
    assert result["converged"] is False
    assert result["iterations"] == 0
    assert result["x"].tolist() == start


def test_a_walk_that_crosses_from_positive_to_negative_curvature_records_the_crossing_between_its_iterates(exact_mode):
    result, _ = _climb([-0.7, 0.0, 0.0])
    assert result["converged"] is True
    crossing = result["x_lambda"]
    assert crossing.shape == (3,)
    assert -0.7 < crossing[0] < 0.0
    assert crossing[1:].tolist() == [0.0, 0.0]


def test_only_the_first_sign_change_of_the_curvature_is_recorded(monkeypatch):
    script = [2.0, -1.0, 2.0, -1.0, 2.0, -1.0]
    seen = []

    def solve(lib, potential, x, **_):
        energy, gradient = potential.energy_forces(x)
        seen.append(np.array(x))
        eigenvalue = script[min(len(seen) - 1, len(script) - 1)]
        return eigenvalue, np.array([1.0, 0.0, 0.0]), gradient, 3, 0, energy

    monkeypatch.setattr(softsaddle_run, "_minimum_mode", solve)
    result, _ = _climb([-0.4, 0.15, -0.1], force_tol=1e-12, save_lanczos=False, cg_iter_max=6)
    assert result["iterations"] == 6
    assert result["x_lambda"] == pytest.approx(_lambda_zero(seen[1], seen[2], 2.0, -1.0))


def test_a_walk_that_starts_where_the_curvature_is_already_negative_records_no_crossing(exact_mode):
    result, _ = _climb([-0.4, 0.15, -0.1])
    assert result["x_lambda"] is None


def test_the_first_mode_of_a_walk_is_seeded_from_the_previous_mode_by_default(exact_mode):
    surface = HarnessSurface(_well)
    settings = _settings(TEMPLATE)
    eigenvalue, mode, gradient, energy = _mode_at(None, surface, np.array([-0.3, 0.0, 0.0]), settings, None, None)
    assert exact_mode[0]["initial"] == INITIAL_PREVIOUS
    assert exact_mode[0]["seed"] is None
    assert eigenvalue == pytest.approx(12.0 * 0.09 - 4.0)
    assert mode.tolist() == [1.0, 0.0, 0.0]
    assert gradient == pytest.approx(_well([-0.3, 0.0, 0.0])[1])
    assert energy == pytest.approx(_well([-0.3, 0.0, 0.0])[0])


def test_a_walk_seeded_from_the_gradient_falls_back_to_a_unit_vector_at_a_stationary_point(exact_mode):
    surface = HarnessSurface(_well)
    settings = _settings(TEMPLATE)
    settings["seed_mode"] = "gradient"
    _mode_at(None, surface, np.array([-0.3, 0.2, 0.0]), settings, None, None)
    assert exact_mode[-1]["initial"] == INITIAL_GRADIENT
    assert exact_mode[-1]["seed"] is None
    _mode_at(None, surface, np.array([0.0, 0.0, 0.0]), settings, None, None)
    assert exact_mode[-1]["initial"] == INITIAL_USER
    assert exact_mode[-1]["seed"].tolist() == [1.0, 0.0, 0.0]


def test_a_known_mode_is_kept_and_only_the_energy_and_gradient_are_evaluated(exact_mode):
    surface = HarnessSurface(_well)
    settings = _settings(TEMPLATE)
    point = np.array([0.1, 0.2, 0.3])
    known = np.array([0.0, 1.0, 0.0])
    eigenvalue, mode, gradient, energy = _mode_at(None, surface, point, settings, -0.5, known)
    assert exact_mode == []
    assert (eigenvalue, mode.tolist()) == (-0.5, known.tolist())
    assert (energy, gradient.tolist()) == (pytest.approx(_well(point)[0]), pytest.approx(_well(point)[1].tolist()))
    assert surface.calls == 1


def test_a_point_with_a_non_finite_gradient_is_refused_by_the_mode_step():
    surface = HarnessSurface(lambda x: (0.0, np.array([float("inf"), 0.0, 0.0])))
    with pytest.raises(RuntimeError, match="^softsaddle mode returned a non-finite value$"):
        _mode_at(None, surface, np.zeros(3), _settings(TEMPLATE), -1.0, np.array([1.0, 0.0, 0.0]))


def _payload(**override) -> dict:
    payload = {
        "force_tol": 1e-3,
        "minimum": [-1.0, 0.0, 0.0],
        "min_energy": 0.0,
        "distance_difference": 0.3,
        "lengths": [0.0, 0.0, 0.0],
        "barrier_max": 1.5,
        "search_cap": 100000,
        "climb_budget": 1000,
    }
    payload.update(override)
    return payload


def _climbed(**override) -> dict:
    climbed = {
        "converged": True,
        "eigenvalue": -4.0,
        "iterations": 4,
        "error": None,
        "x": np.array([0.0, 0.0, 0.0]),
        "force_norm": 1e-5,
        "mode": np.array([1.0, 0.0, 0.0]),
        "energy": 1.0,
        "x_lambda": np.array([-0.3, 0.0, 0.0]),
    }
    climbed.update(override)
    return climbed


def _judge(*, payload=None, climbed=None, before=10):
    surface = HarnessSurface(_well)
    record = _judge_saddle(_payload(**(payload or {})), _climbed(**(climbed or {})), surface, before)
    return record, surface


def test_a_converged_index_one_saddle_joined_to_the_minimum_is_a_success():
    record, surface = _judge()
    assert record["success"] is True
    assert record["failure"] is None
    assert record["reasons"] == []
    assert record["hessian_index"] == 1
    assert record["barrier"] == pytest.approx(1.0)
    assert record["novel_candidate"] is True
    assert record["saddle_distance"] == pytest.approx(1.0)
    assert record["lambda_distance"] == pytest.approx(0.7)
    assert record["saddle_x"].tolist() == [0.0, 0.0, 0.0]
    assert sorted((round(record["per_atom_plus"], 3), round(record["per_atom_minus"], 3))) == [0.0, 2.0]
    assert record["lowest_eigenvalue"] == pytest.approx(-4.0, abs=1e-6)
    assert record["force_calls"] == surface.calls
    assert record["verify_calls"] == surface.verify_calls
    assert record["force_norm"] == 1e-5
    assert record["energy"] == 1.0


def test_a_force_just_above_the_stop_by_rounding_is_accepted_and_one_clearly_above_is_not():
    near, _ = _judge(climbed={"force_norm": 1e-3 + 5e-11})
    assert near["success"] is True
    above, _ = _judge(climbed={"force_norm": 1e-3 + 1e-9})
    assert above["reasons"] == ["force_norm"]
    assert above["success"] is False


def test_the_endpoints_are_minimized_from_a_tenth_of_an_angstrom_either_side_of_the_saddle_along_the_mode():
    visited = []

    def logged(x):
        visited.append(np.array(x))
        return _well(x)

    surface = HarnessSurface(logged)
    _judge_saddle(_payload(), _climbed(), surface, 10)
    starts = [point.tolist() for point in visited]
    assert [0.1, 0.0, 0.0] in starts
    assert [-0.1, 0.0, 0.0] in starts


def test_a_barrier_equal_to_the_limit_is_too_high_and_is_measured_from_the_minimum_energy():
    exact, _ = _judge(payload={"barrier_max": 1.0})
    assert exact["reasons"] == ["barrier"]
    shifted, _ = _judge(payload={"min_energy": 0.25, "barrier_max": 1.0})
    assert shifted["barrier"] == pytest.approx(0.75)
    assert shifted["success"] is True


def test_a_rejected_saddle_that_also_used_the_call_budget_is_a_cap():
    record, _ = _judge(payload={"search_cap": 5, "barrier_max": 0.5})
    assert record["reasons"] == ["barrier", "cap"]
    assert record["failure"] == "cap"
    assert record["success"] is False


def test_a_saddle_that_is_connected_but_too_high_is_a_barrier_rejection():
    record, _ = _judge(payload={"barrier_max": 0.5})
    assert record["reasons"] == ["barrier"]
    assert record["success"] is False
    assert record["failure"] == "divergence"
    assert record["barrier"] == pytest.approx(1.0)
    assert record["novel_candidate"] is True


def test_a_saddle_between_two_other_minima_is_not_connected_to_the_reactant():
    record, _ = _judge(payload={"minimum": [3.0, 0.0, 0.0]})
    assert record["reasons"] == ["not_connected"]
    assert record["success"] is False
    assert record["novel_candidate"] is False
    assert record["hessian_index"] == 1


def test_a_converged_point_that_is_a_minimum_is_a_wrong_stationary_point():
    record, surface = _judge(climbed={"x": np.array([1.0, 0.0, 0.0]), "energy": 0.0})
    assert record["hessian_index"] == 0
    assert record["reasons"] == ["hessian_index"]
    assert record["failure"] == "wrong_stationary"
    assert record["barrier"] is None
    assert record["saddle_x"] is None
    assert surface.calls == 0


def test_a_saddle_with_a_force_above_the_stop_is_not_followed_to_its_endpoints():
    record, surface = _judge(climbed={"force_norm": 0.5})
    assert record["hessian_index"] == 1
    assert record["reasons"] == ["force_norm"]
    assert record["failure"] == "divergence"
    assert record["barrier"] is None
    assert surface.calls == 0


def test_a_walk_that_raised_is_a_crash_whatever_else_it_returned():
    record, _ = _judge(climbed={"error": "callback failed", "converged": False})
    assert record["reasons"] == ["crash"]
    assert record["failure"] == "crash"
    assert record["error"] == "callback failed"


def test_a_walk_that_left_the_finite_numbers_is_a_divergence():
    record, _ = _judge(climbed={"x": np.array([float("nan"), 0.0, 0.0]), "converged": False})
    assert record["reasons"] == ["divergence"]
    assert record["failure"] == "divergence"


def test_an_unconverged_walk_that_used_its_budget_is_a_cap_and_one_that_did_not_is_not():
    spent, _ = _judge(climbed={"converged": False, "force_norm": 3.0}, before=1000)
    assert spent["reasons"] == ["cap", "force_norm"]
    assert spent["failure"] == "cap"
    early, _ = _judge(climbed={"converged": False, "force_norm": 3.0}, before=10)
    assert early["reasons"] == ["force_norm"]
    assert early["failure"] == "divergence"


def test_a_search_that_used_the_whole_call_budget_is_a_cap_even_if_it_found_the_saddle():
    record, _ = _judge(payload={"search_cap": 5})
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert record["reasons"] == ["cap"]
    assert record["hessian_index"] == 1


def test_a_positive_curvature_end_point_is_not_judged_as_a_saddle():
    record, surface = _judge(climbed={"eigenvalue": 2.0, "x": np.array([0.5, 0.0, 0.0])})
    assert record["hessian_index"] is None
    assert record["reasons"] == ["force_norm"]
    assert surface.calls == 0


def test_a_lambda_point_on_the_minimum_gives_no_lambda_distance_and_none_gives_none():
    on_minimum, _ = _judge(climbed={"x_lambda": np.array([-1.0, 0.0, 0.0])})
    assert on_minimum["lambda_distance"] is None
    absent, _ = _judge(climbed={"x_lambda": None})
    assert absent["lambda_distance"] is None
    assert absent["success"] is True


def test_a_missing_mode_still_finds_both_endpoints_and_an_infinite_force_is_stored_as_none():
    record, _ = _judge(climbed={"mode": None})
    assert record["success"] is True
    far, _ = _judge(climbed={"converged": False, "force_norm": float("inf"), "energy": None})
    assert far["force_norm"] is None
    assert far["energy"] is None


def _table() -> ProcessTable:
    return ProcessTable([0.0, 0.0, 0.0], 0.3, 0.05)


def _judged(**override) -> dict:
    judged = {
        "reasons": [],
        "success": True,
        "novel_candidate": True,
        "saddle_x": [0.0, 0.0, 0.0],
        "barrier": 1.0,
    }
    judged.update(override)
    return judged


def test_a_new_saddle_is_accepted_and_a_second_visit_to_it_is_a_duplicate():
    found = _table()
    first = _apply_novelty(_judged(), found)
    assert first["success"] is True
    assert first["failure"] is None
    assert first["reasons"] == []
    assert first["accepted_novel"] is True
    assert len(found) == 1
    second = _apply_novelty(_judged(), found)
    assert second["success"] is False
    assert second["reasons"] == ["duplicate"]
    assert second["failure"] == "divergence"
    assert "accepted_novel" not in second
    assert len(found) == 1


def test_the_same_geometry_at_a_different_barrier_or_another_geometry_is_new():
    found = _table()
    _apply_novelty(_judged(), found)
    other_barrier = _apply_novelty(_judged(barrier=1.2), found)
    assert other_barrier["success"] is True
    moved = _apply_novelty(_judged(saddle_x=[0.0, 0.5, 0.0]), found)
    assert moved["success"] is True
    assert len(found) == 3


def test_a_search_that_is_not_a_novel_candidate_does_not_touch_the_table():
    found = _table()
    judged = _judged(success=False, reasons=["not_connected"], novel_candidate=False)
    applied = _apply_novelty(judged, found)
    assert applied["success"] is False
    assert applied["reasons"] == ["not_connected"]
    assert applied["failure"] == "divergence"
    assert len(found) == 0
    assert "accepted_novel" not in applied


def test_novelty_returns_a_new_record_and_leaves_the_judged_one_as_it_was():
    found = _table()
    judged = _judged()
    applied = _apply_novelty(judged, found)
    assert applied is not judged
    assert judged["reasons"] == []
    assert judged["success"] is True
    again = _apply_novelty(judged, found)
    assert judged["reasons"] == []
    assert again["reasons"] == ["duplicate"]
