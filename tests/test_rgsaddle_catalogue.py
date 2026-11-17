"""How a Pt-island catalogue run judges each displacement search and keeps its checkpoint.

The library that climbs is not built in CI, so every search is given a scripted
climb: the point it ended at, its mode and its counts. Everything around the
climb is the driver's own and runs for real, on a surface whose answers are
known. Two atoms are pinned 3.0 apart on x and one atom is free between them,
on Lennard-Jones. The free atom has a well beside each pinned atom, at
x = -0.3723 and x = +0.3723, and an index-1 saddle at the origin: the Hessian
there is -8.84 along x and 1.54 across, and the barrier is 0.449. A climb that
ends at the origin along x links the two wells. Along y both endpoint
minimizations come straight back to the saddle, so neither is the reactant.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import optbench.rgsaddle_minmode as driver
from optbench.campaign import load_template, resolve_problem
from optbench.rgsaddle_minmode import CATALOG_SEED, HarnessSurface, _lbfgs_minimize
from optbench.surfaces import lennard_jones

ROOT = Path(__file__).resolve().parents[1]
PINNED = np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
WELL = -0.3723258
CLIMB_STEPS = 20
STEP_CEILING = 12
CLIMB_CAP = CLIMB_STEPS * STEP_CEILING
ENDPOINT_BUDGET = 100
SEARCH_CAP = CLIMB_CAP + 2 * ENDPOINT_BUDGET

DISPLACED = (-0.6, 0.1, 0.0)


def _con(start=(WELL, 0.0, 0.0)):
    x, y, z = start
    return "\n".join(
        [
            "island",
            "0 0 0",
            "30 30 30",
            "90 90 90",
            "0 0",
            "0 0 0",
            "1",
            "3",
            "39.9",
            "Ar",
            "Coordinates of Component 1",
            "-1.5 0 0 1 0",
            "1.5 0 0 1 1",
            f"{x} {y} {z} 0 2",
            "",
        ]
    )


TEMPLATE = {
    "id": "rgsaddle/toy-dimer",
    "code": "rgsaddle",
    "code_version": "0.2.2",
    "commit": "",
    "algorithm": "Dimer L-BFGS",
    "tuning": "default",
    "ini": {
        "Main": {"session": "minmode"},
        "Optimizer": {"opt_method": "lbfgs", "converged_force": 0.001, "max_move": 0.2, "max_iterations": ENDPOINT_BUDGET},
        "SaddleSearch": {
            "min_mode_method": "dimer",
            "max_iterations": CLIMB_STEPS,
            "max_rotations": 3,
            "difference": "forward",
        },
        "Search": {
            "repetitions": 1,
            "displace_atoms": "2",
            "displace_magnitude": 0.1,
            "displace_radius": 3.3,
            "max_searches": 20000,
        },
    },
}

PROBLEM = {
    "id": "saddle-search/pt-island",
    "potential": {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0},
    "goal": {"connected_saddles": 1, "barrier_below_eV": 1.5},
    "connection": {"distance_difference": 0.1, "energy_difference": 0.01},
    "input": {"geometries": ["toy/island.con"]},
}


def _surface(free, epsilon=1.0):
    positions = np.vstack([PINNED, np.asarray(free, dtype=np.float64).reshape(1, 3)])
    energy, gradient = lennard_jones(positions, epsilon=epsilon)
    return float(energy), np.asarray(gradient, dtype=np.float64)[2]


def _pair_curvature(distance):
    return 4.0 * (156.0 * distance**-14 - 42.0 * distance**-8)


def _climb(x=(0.0, 0.0, 0.0), mode=(1.0, 0.0, 0.0), epsilon=1.0, **changes):
    """What a library climb returns when it defers the Hessian to the driver.

    The force norm and the energy are the surface's at ``x``. A test that
    needs the driver to take a point as converged states the climb's own
    ``force_norm``, which is what the driver reads.
    """
    point = np.array(x, dtype=np.float64)
    if np.all(np.isfinite(point)):
        energy, gradient = _surface(point, epsilon)
        force_norm = float(np.linalg.norm(gradient))
    else:
        energy = force_norm = float("nan")
    outcome = {
        "name": "scripted",
        "x": point,
        "mode": None if mode is None else np.array(mode, dtype=np.float64),
        "free": np.array([2]),
        "calls": 12,
        "verify_calls": 0,
        "energy": energy,
        "force_norm": force_norm,
        "converged": True,
        "rotations": 4,
        "library_evaluations": 30,
        "steps": 3,
        "step_calls_max": 4,
        "curvature": -8.8,
        "solver_rc": 0,
    }
    outcome.update(changes)
    return outcome


class Killed(Exception):
    pass


def _killed(spec):
    raise Killed(spec["name"])


class Climbs:
    """Stands in for the library: the n-th search gets the n-th scripted outcome."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.specs = []

    def __call__(self, spec):
        index = len(self.specs)
        self.specs.append(spec)
        assert index < len(self.outcomes), f"{spec['name']} was not scripted"
        outcome = self.outcomes[index]
        return outcome(spec) if callable(outcome) else copy.deepcopy(outcome)


def _template(**sections):
    template = copy.deepcopy(TEMPLATE)
    for name, values in sections.items():
        section = template["ini"].setdefault(name, {})
        for key, value in values.items():
            if value is None:
                section.pop(key, None)
            else:
                section[key] = value
    return template


def _problem(**changes):
    problem = copy.deepcopy(PROBLEM)
    for name, values in changes.items():
        if isinstance(values, dict):
            problem.setdefault(name, {}).update(values)
        else:
            problem[name] = values
    return problem


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "island.con").write_text(_con(), encoding="utf-8")
    manifest = {"path": "toy/island.con", "stored": "inputs/island.con"}
    (root / "inputs" / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def search(monkeypatch, tmp_path):
    """Run the catalogue on the toy island with scripted climbs."""
    root = _repo(tmp_path)

    def run(*outcomes, template=None, problem=None, output=None, force_tol=1e-3, start=None, **options):
        options.setdefault("limit", len(outcomes))
        (root / "inputs" / "island.con").write_text(_con(start or (WELL, 0.0, 0.0)), encoding="utf-8")
        climbs = Climbs(*outcomes)
        monkeypatch.setattr(driver, "_dispatch", climbs)
        written = output or tmp_path / "out" / "toy.dat"
        result = driver._run_saddle_search(
            root,
            problem or _problem(),
            template or _template(),
            output=written,
            force_tol=force_tol,
            work=tmp_path / "work",
            **options,
        )
        return result, climbs, written

    return run


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]


def _fields(path: Path) -> dict[str, str]:
    return {line.split(" ", 1)[0]: line.split(" ", 1)[1] if " " in line else "" for line in path.read_text(encoding="utf-8").splitlines()}


def _calls(start, budget=ENDPOINT_BUDGET):
    surface = HarnessSurface(_surface)
    _lbfgs_minimize(surface, np.array(start, dtype=np.float64), 1e-3, budget)
    return surface.calls


def _minimum():
    x, converged = _lbfgs_minimize(HarnessSurface(_surface), np.array([-0.4, 0.05, 0.0]), 1e-10, 500)
    assert converged
    return x


REACTANT = np.array([WELL, 0.0, 0.0])


# A search that ends at a connected saddle


def test_a_climb_that_ends_at_a_saddle_linking_the_wells_is_a_success_that_ends_the_run(search):
    result, climbs, _written = search(_climb())
    assert len(climbs.specs) == 1
    (record,) = result["records"]
    assert record["success"] is True
    assert record["failure"] is None
    assert record["reasons"] == []
    assert record["hessian_index"] == record["cartesian_index"] == 1
    assert record["rigid_modes_removed"] == 0
    assert record["lowest_eigenvalue"] == pytest.approx(2.0 * _pair_curvature(1.5), rel=1e-6)
    well = _minimum()
    assert record["barrier"] == pytest.approx(_surface((0.0, 0.0, 0.0))[0] - _surface(well)[0], abs=1e-6)
    assert record["per_atom_minus"] < 1e-3
    assert record["per_atom_plus"] == pytest.approx(2.0 * abs(well[0]), abs=1e-3)
    assert result["repetitions"][0]["reached"] is True
    assert result["repetitions"][0]["connected"] == 1


def test_a_searchs_cost_is_its_climb_and_its_two_endpoint_minimizations_and_the_run_adds_the_first_minimization(search):
    result, _climbs, written = search(_climb())
    (record,) = result["records"]
    spent = _calls((-0.2, 0.0, 0.0)) + _calls((0.2, 0.0, 0.0))
    assert record["endpoint_calls"] == spent
    assert record["force_calls"] == 12 + spent
    summary = result["repetitions"][0]
    assert summary["minimize_calls"] == 1
    assert summary["minimum_converged"] is True
    assert summary["force_calls"] == 1 + 12 + spent
    assert result["force_calls"] == 1 + 12 + spent
    assert _fields(written)["force_calls"] == str(1 + 12 + spent) + ".0"


def test_the_hessian_probes_of_a_search_are_not_charged_to_it(search):
    result, _climbs, _written = search(_climb())
    (record,) = result["records"]
    assert record["verify_calls"] == 2 * 3
    assert record["force_calls"] == 12 + record["endpoint_calls"]


def test_each_search_counts_the_hessian_probes_it_made_and_not_those_of_the_searches_before(search):
    problem = _problem(goal={"connected_saddles": 5})
    result, _climbs, _written = search(_climb(), _climb(mode=(0.0, 1.0, 0.0)), _climb(x=(4e-5, 0.0, 0.0)), problem=problem)
    assert [record["verify_calls"] for record in result["records"]] == [2 * 3, 2 * 3, 2 * 3]


def test_the_record_of_a_search_keeps_what_the_climb_reported(search):
    result, _climbs, _written = search(_climb(rotations=7, library_evaluations=41, steps=5, step_calls_max=9, curvature=-8.5))
    (record,) = result["records"]
    assert record["schema"] == "optbench.start/v2"
    assert record["start"] == "rep1-search1"
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["converged"] is True
    assert record["rotations"] == 7
    assert record["library_evaluations"] == 41
    assert record["steps"] == 5
    assert record["step_calls_max"] == 9
    assert record["curvature"] == -8.5
    assert record["epicenter"] == 2
    assert record["displaced_atoms"] == 1


# What each climb is asked to do


def test_each_climb_is_asked_for_a_displaced_start_and_a_unit_mode_under_the_templates_bounds(search):
    _result, climbs, _written = search(_climb(), _climb(), problem=_problem(goal={"connected_saddles": 5}), limit=2)
    first, second = climbs.specs
    assert (first["name"], second["name"]) == ("rep1-search1", "rep1-search2")
    for spec in (first, second):
        assert np.array_equal(spec["positions"][:2], PINNED)
        assert 0.0 < np.linalg.norm(spec["positions"][2] - REACTANT) < 1.0
        assert spec["fixed"].tolist() == [True, True, False]
        assert spec["cell"].tolist() == [30.0, 30.0, 30.0]
        assert spec["file_mode"].shape == (3,)
        assert np.linalg.norm(spec["file_mode"]) == pytest.approx(1.0, abs=1e-12)
        assert spec["mode_seed"] == "file"
        assert spec["task"] == "saddle"
        assert spec["defer_hessian"] is True
        assert spec["reactant"] is None
        assert spec["reference_mode"] is None
        assert spec["overlap"] == 0.99
        assert spec["session"] == "minmode"
        assert spec["kind"] == "dimer"
        assert spec["method"] == "lbfgs"
        assert spec["cap"] == CLIMB_CAP
        assert spec["max_steps"] == CLIMB_STEPS
        assert spec["force_tol"] == 1e-3
        assert spec["potential"] == PROBLEM["potential"]
    assert not np.array_equal(first["positions"][2], second["positions"][2])
    assert not np.array_equal(first["file_mode"], second["file_mode"])


def test_the_displacement_scales_with_the_magnitude_the_template_states(search):
    _result, large, _ = search(_climb())
    _result, small, _ = search(_climb(), template=_template(Search={"displace_magnitude": 0.001}))
    wide = large.specs[0]["positions"][2] - REACTANT
    narrow = small.specs[0]["positions"][2] - REACTANT
    assert np.allclose(wide, 100.0 * narrow, rtol=1e-9, atol=1e-12)
    assert np.linalg.norm(narrow) < 0.02


@pytest.mark.parametrize("listed", ["2", " 2, ", 2])
def test_the_listed_atoms_are_read_from_a_text_or_a_number(search, listed):
    result, climbs, _written = search(_climb(), template=_template(Search={"displace_atoms": listed}))
    assert result["records"][0]["epicenter"] == 2
    assert len(climbs.specs) == 1


def test_a_climb_reported_at_the_ceiling_is_accepted_and_one_past_it_is_refused(search):
    result, _climbs, _written = search(_climb(step_calls_max=STEP_CEILING, calls=CLIMB_CAP))
    assert result["records"][0]["success"] is True
    with pytest.raises(RuntimeError, match=f"climbed past the step ceiling {STEP_CEILING} or the cap {CLIMB_CAP}"):
        search(_climb(step_calls_max=STEP_CEILING + 1))
    with pytest.raises(RuntimeError, match="rep1-search1 climbed past the step ceiling"):
        search(_climb(calls=CLIMB_CAP + 1))


# The ways a search is judged a failure


@pytest.mark.parametrize(
    ("outcome", "reasons", "failure"),
    [
        (_climb(converged=False, steps=3, force_norm=0.4), ["force_norm"], "divergence"),
        (_climb(converged=False, steps=CLIMB_STEPS - 1, force_norm=0.4), ["force_norm"], "divergence"),
        (_climb(converged=False, steps=CLIMB_STEPS, force_norm=0.4), ["cap", "force_norm"], "cap"),
        (_climb(converged=False, x=(np.nan, 0.0, 0.0)), ["divergence"], "divergence"),
        (_climb(converged=True, x=(np.inf, 0.0, 0.0)), ["divergence"], "divergence"),
        (_climb(curvature=1.0), ["force_norm"], "divergence"),
        (_climb(curvature=None), ["force_norm"], "divergence"),
        (_climb(curvature=float("nan")), ["force_norm"], "divergence"),
        (_climb(curvature=0.0), ["force_norm"], "divergence"),
        (_climb(curvature=-np.inf), ["force_norm"], "divergence"),
        ({"start": "scripted", "force_calls": 7, "reasons": ["crash"], "converged": False}, ["crash"], "crash"),
        ({"start": "scripted", "force_calls": 7, "reasons": ["solver"], "converged": False}, ["solver"], "divergence"),
        ({"start": "scripted", "force_calls": 7}, ["solver"], "divergence"),
    ],
    ids=[
        "still-climbing",
        "one-step-short-of-the-cap",
        "out-of-steps",
        "not-finite",
        "converged-at-an-infinite-point",
        "positive-curvature",
        "no-curvature",
        "nan-curvature",
        "zero-curvature",
        "infinite-curvature",
        "library-crash",
        "library-solver-error",
        "library-error-without-a-reason",
    ],
)
def test_a_climb_that_does_not_reach_a_saddle_is_a_failure_with_its_reasons(search, outcome, reasons, failure):
    result, _climbs, _written = search(outcome)
    (record,) = result["records"]
    assert record["success"] is False
    assert record["reasons"] == reasons
    assert record["failure"] == failure
    assert record["hessian_index"] is None
    assert "barrier" not in record
    assert record["endpoint_calls"] == 0
    assert record["force_calls"] == int(outcome.get("calls", outcome.get("force_calls")))
    assert record["converged"] is bool(outcome.get("converged", False))
    assert result["repetitions"][0]["connected"] == 0
    assert result["repetitions"][0]["unique_saddles"] == 0


def test_a_converged_point_that_is_a_minimum_is_the_wrong_stationary_point(search):
    result, _climbs, _written = search(_climb(x=(WELL, 0.0, 0.0), curvature=-0.5))
    (record,) = result["records"]
    assert record["hessian_index"] == 0
    assert record["reasons"] == ["hessian_index"]
    assert record["failure"] == "wrong_stationary"
    assert record["lowest_eigenvalue"] > 0.0
    assert record["verify_calls"] == 6
    assert record["endpoint_calls"] == 0


def test_an_index_one_point_still_above_the_force_stop_is_not_taken_for_a_saddle(search):
    result, _climbs, _written = search(_climb(x=(0.1, 0.0, 0.0)))
    (record,) = result["records"]
    assert record["hessian_index"] == 1
    assert record["reasons"] == ["force_norm"]
    assert record["failure"] == "divergence"
    assert "barrier" not in record
    assert record["endpoint_calls"] == 0


def test_a_force_just_over_the_stop_is_refused_and_one_inside_the_slack_is_taken(search):
    over, _climbs, _written = search(_climb(force_norm=1e-3 + 2e-10))
    assert over["records"][0]["reasons"] == ["force_norm"]
    inside, _climbs, _written = search(_climb(force_norm=1e-3 + 5e-11))
    assert inside["records"][0]["success"] is True


def test_a_saddle_whose_endpoints_both_stay_at_the_saddle_is_not_connected(search):
    result, _climbs, _written = search(_climb(mode=(0.0, 1.0, 0.0)))
    (record,) = result["records"]
    assert record["success"] is False
    assert record["reasons"] == ["not_connected"]
    assert record["failure"] == "divergence"
    assert record["per_atom_plus"] == pytest.approx(abs(WELL), abs=1e-6)
    assert record["per_atom_minus"] == pytest.approx(abs(WELL), abs=1e-6)
    assert record["hessian_index"] == 1
    assert record["barrier"] > 0.0
    summary = result["repetitions"][0]
    assert summary["connected"] == 0
    assert summary["unique_saddles"] == 0
    assert summary["fraction_connected"] == 0.0


def test_a_saddle_whose_endpoints_both_fall_into_the_reactants_well_is_not_connected(search):
    offset = {"Optimizer": {"max_move": 0.05}}
    result, _climbs, _written = search(_climb(x=(-0.1, 0.0, 0.0), force_norm=1e-5), template=_template(**offset))
    (record,) = result["records"]
    assert record["hessian_index"] == 1
    assert record["reasons"] == ["not_connected"]
    assert record["per_atom_plus"] < 1e-3
    assert record["per_atom_minus"] < 1e-3


def test_an_endpoint_that_does_not_converge_is_bad_minima_and_the_other_is_not_tried(search):
    budget = {"Optimizer": {"max_iterations": 5}}
    result, _climbs, _written = search(_climb(), template=_template(**budget))
    (record,) = result["records"]
    assert record["reasons"] == ["bad_minima"]
    assert record["failure"] == "divergence"
    assert record["endpoint_calls"] == 5
    assert record["force_calls"] == 12 + 5
    assert "per_atom_plus" not in record
    assert "per_atom_minus" not in record
    assert record["barrier"] > 0.0
    assert result["repetitions"][0]["unique_saddles"] == 0
    assert result["repetitions"][0]["fraction_connected"] == 1.0


def test_a_new_connected_saddle_over_the_barrier_limit_fails_for_its_barrier_and_still_enters_the_table(search):
    problem = _problem(goal={"connected_saddles": 5, "barrier_below_eV": 0.3})
    result, _climbs, _written = search(_climb(), _climb(), problem=problem, limit=2)
    first, second = result["records"]
    assert first["reasons"] == ["barrier"]
    assert first["success"] is False
    assert second["reasons"] == ["duplicate", "barrier"]
    summary = result["repetitions"][0]
    assert summary["connected"] == 0
    assert summary["unique_saddles"] == 1
    assert summary["fraction_connected"] == 1.0


def test_a_barrier_equal_to_the_limit_is_over_it(search):
    barrier = _surface((0.0, 0.0, 0.0))[0] - _surface(_minimum())[0]
    below, _climbs, _written = search(_climb(), problem=_problem(goal={"barrier_below_eV": barrier + 1e-6}))
    assert below["records"][0]["success"] is True
    above, _climbs, _written = search(_climb(), problem=_problem(goal={"barrier_below_eV": barrier - 1e-6}))
    assert above["records"][0]["reasons"] == ["barrier"]


def test_a_saddle_that_is_not_connected_and_over_the_barrier_limit_is_both(search):
    problem = _problem(goal={"connected_saddles": 5, "barrier_below_eV": 0.3})
    result, _climbs, _written = search(_climb(mode=(0.0, 1.0, 0.0)), problem=problem)
    assert result["records"][0]["reasons"] == ["not_connected", "barrier"]
    assert result["repetitions"][0]["fraction_connected"] == 0.0


def test_a_saddle_that_was_not_connected_does_not_enter_the_table(search):
    problem = _problem(goal={"connected_saddles": 5})
    result, _climbs, _written = search(_climb(mode=(0.0, 1.0, 0.0)), _climb(), problem=problem)
    first, second = result["records"]
    assert first["reasons"] == ["not_connected"]
    assert second["success"] is True
    assert result["repetitions"][0]["unique_saddles"] == 1


def test_a_climb_that_reports_no_mode_is_followed_along_the_displacement_seed(search):
    result, climbs, _written = search(_climb(mode=None), problem=_problem(goal={"connected_saddles": 5}))
    axis = climbs.specs[0]["file_mode"]
    axis = axis / np.linalg.norm(axis)
    saddle = np.zeros(3)
    ends = {}
    for name, sign in (("minus", -1.0), ("plus", 1.0)):
        end, converged = _lbfgs_minimize(HarnessSurface(_surface), saddle + sign * 0.2 * axis, 1e-3, ENDPOINT_BUDGET)
        assert converged
        ends[name] = float(np.linalg.norm(end - REACTANT))
    record = result["records"][0]
    assert record["per_atom_minus"] == pytest.approx(ends["minus"], abs=1e-9)
    assert record["per_atom_plus"] == pytest.approx(ends["plus"], abs=1e-9)


def test_the_templates_search_bound_ends_a_run_that_never_reaches_its_goal(search):
    template = _template(Search={"max_searches": 2})
    _result, climbs, _written = search(*[_climb(mode=(0.0, 1.0, 0.0))] * 3, template=template, problem=_problem(goal={"connected_saddles": 5}))
    assert len(climbs.specs) == 2


def test_a_climb_reporting_a_non_unit_mode_is_followed_a_max_move_along_its_direction(search):
    unit, _climbs, _written = search(_climb())
    long, _climbs, _written = search(_climb(mode=(0.5, 0.0, 0.0)))
    assert long["records"][0]["endpoint_calls"] == unit["records"][0]["endpoint_calls"]


def test_each_endpoint_starts_a_max_move_from_the_saddle_and_the_minus_side_follows_the_same_move(search):
    template = _template(Optimizer={"max_move": 0.05})
    result, _climbs, _written = search(_climb(x=(0.1, 0.0, 0.0), force_norm=1e-5), template=template)
    (record,) = result["records"]
    assert record["reasons"] == ["not_connected"]
    assert record["per_atom_minus"] == pytest.approx(2.0 * abs(WELL), abs=1e-3)
    assert record["per_atom_plus"] == pytest.approx(2.0 * abs(WELL), abs=1e-3)


def test_an_endpoint_that_converges_on_one_side_only_is_bad_minima_and_both_were_tried(search):
    template = _template(Optimizer={"max_iterations": 7})
    result, _climbs, _written = search(_climb(x=(0.02, 0.0, 0.0), force_norm=5e-4), template=template)
    (record,) = result["records"]
    assert _calls((-0.18, 0.0, 0.0), 7) == 7
    assert _calls((0.22, 0.0, 0.0), 7) == 7
    assert record["reasons"] == ["bad_minima"]
    assert record["endpoint_calls"] == 14
    assert "per_atom_plus" not in record


def test_the_default_barrier_limit_is_one_and_a_half_electron_volts(search):
    stiff = {"potential": {**PROBLEM["potential"], "epsilon": 4.0}}
    problem = _problem(**stiff)
    del problem["goal"]["barrier_below_eV"]
    saddle = _climb(epsilon=4.0)
    result, _climbs, _written = search(saddle, problem=problem)
    assert result["records"][0]["barrier"] == pytest.approx(1.797, abs=1e-3)
    assert result["records"][0]["reasons"] == ["barrier"]
    allowed = _problem(**stiff, goal={"barrier_below_eV": 2.0})
    result, _climbs, _written = search(saddle, problem=allowed)
    assert result["records"][0]["success"] is True


def test_the_problems_goal_is_read_before_the_templates(search):
    result, _climbs, _written = search(_climb(), template=_template(Search={"connected": 50}))
    assert result["repetitions"][0]["goal"] == 1
    assert result["repetitions"][0]["reached"] is True


def test_a_template_that_leaves_its_search_settings_out_gets_one_repetition_and_the_defaults(search):
    template = _template(
        Search={"repetitions": None, "max_searches": None, "displace_magnitude": None},
        SaddleSearch={"min_mode_method": None},
        Optimizer={"opt_method": None},
    )
    problem = _problem(goal={"connected_saddles": 50})
    result, climbs, _written = search(*[_climb(mode=(0.0, 1.0, 0.0))] * 5, template=template, problem=problem)
    assert len(result["repetitions"]) == 1
    assert len(climbs.specs) == 5
    spec = climbs.specs[0]
    assert (spec["kind"], spec["method"]) == ("dimer", "lbfgs")
    _stated, stated, _written = search(_climb(mode=(0.0, 1.0, 0.0)), problem=problem)
    assert np.array_equal(spec["positions"], stated.specs[0]["positions"])


def test_a_failed_library_run_with_no_count_costs_nothing(search):
    result, _climbs, _written = search({"start": "scripted", "reasons": ["crash"]})
    (record,) = result["records"]
    assert record["force_calls"] == 0
    assert record["failure"] == "crash"


def test_a_saddle_found_again_is_a_duplicate_and_does_not_count_toward_the_goal(search):
    problem = _problem(goal={"connected_saddles": 2})
    result, _climbs, _written = search(_climb(), _climb(x=(4e-5, 0.0, 0.0)), problem=problem, limit=2)
    first, second = result["records"]
    assert first["success"] is True
    assert second["success"] is False
    assert second["reasons"] == ["duplicate"]
    assert second["failure"] == "divergence"
    assert second["hessian_index"] == 1
    assert second["per_atom_plus"] == pytest.approx(first["per_atom_plus"], abs=1e-3)
    summary = result["repetitions"][0]
    assert summary["connected"] == 1
    assert summary["unique_saddles"] == 1
    assert summary["fraction_connected"] == 1.0
    assert summary["reached"] is False


def test_a_saddle_that_differs_in_barrier_by_more_than_the_tolerance_is_not_a_repeat(search):
    problem = _problem(goal={"connected_saddles": 2}, connection={"energy_difference": 1e-9})
    result, _climbs, _written = search(_climb(), _climb(x=(4e-5, 0.0, 0.0)), problem=problem, limit=2)
    first, second = result["records"]
    assert second["reasons"] == []
    assert second["success"] is True
    assert result["repetitions"][0]["unique_saddles"] == 2


def test_a_saddle_that_moved_more_than_the_distance_tolerance_is_not_a_repeat(search):
    problem = _problem(goal={"connected_saddles": 2}, connection={"distance_difference": 5e-5, "energy_difference": 1.0})
    result, _climbs, _written = search(_climb(), _climb(x=(1e-4, 0.0, 0.0)), problem=problem, limit=2)
    assert [record["success"] for record in result["records"]] == [True, True]
    assert result["repetitions"][0]["unique_saddles"] == 2
    assert result["repetitions"][0]["reached"] is True


# What a run reports


def test_the_run_reports_the_share_of_searches_that_were_connected_and_scores_the_rest_as_failures(search):
    problem = _problem(goal={"connected_saddles": 5})
    outcomes = [
        _climb(),
        _climb(mode=(0.0, 1.0, 0.0)),
        _climb(x=(4e-5, 0.0, 0.0)),
        _climb(converged=False, steps=CLIMB_STEPS, force_norm=0.4),
    ]
    result, _climbs, written = search(*outcomes, problem=problem, limit=4)
    summary = result["repetitions"][0]
    assert [record["reasons"] for record in result["records"]] == [[], ["not_connected"], ["duplicate"], ["cap", "force_norm"]]
    assert summary["total_searches"] == 4
    assert summary["connected"] == 1
    assert summary["unique_saddles"] == 1
    assert summary["fraction_connected"] == 0.75
    assert summary["reached"] is False
    assert summary["goal"] == 5
    assert result["nfailed"] == 1
    assert result["starts"] == 4
    costs = [record["force_calls"] for record in result["records"]]
    fields = _fields(written)
    assert float(fields["par2_mean"]) == pytest.approx((costs[0] + 3 * 2 * SEARCH_CAP) / 4, abs=1e-9)
    assert fields["v2_nfailed"] == "3"
    assert fields["n"] == "4"
    assert fields["goal"] == "5"
    assert fields["nfailed"] == "1"
    assert fields["cap"] == str(SEARCH_CAP)


def test_the_goal_defaults_to_the_templates_count_and_then_to_twenty_seven(search):
    problem = _problem()
    del problem["goal"]
    result, _climbs, _written = search(_climb(), problem=problem, template=_template(Search={"connected": 1}))
    assert result["repetitions"][0]["goal"] == 1
    assert result["repetitions"][0]["reached"] is True
    result, climbs, _written = search(*[_climb(mode=(0.0, 1.0, 0.0))] * 3, problem=problem, limit=3)
    assert result["repetitions"][0]["goal"] == 27
    assert len(climbs.specs) == 3


def test_a_run_allowed_no_searches_reports_nothing_to_score(search):
    result, climbs, written = search(template=_template(Search={"max_searches": 0}))
    assert climbs.specs == []
    assert result["records"] == []
    assert result["starts"] == 0
    summary = result["repetitions"][0]
    assert summary["fraction_connected"] == 0.0
    assert summary["reached"] is False
    assert result["nfailed"] == 1
    fields = _fields(written)
    assert fields["n"] == "0"
    assert "par2_mean" not in fields
    assert not written.with_suffix(".v2.json").exists()
    assert written.with_suffix(".starts.jsonl").read_text(encoding="utf-8") == ""


def test_a_start_limit_ends_the_run_before_its_goal(search):
    problem = _problem(goal={"connected_saddles": 5})
    result, climbs, _written = search(*[_climb(mode=(0.0, 1.0, 0.0))] * 3, problem=problem, limit=2)
    assert len(climbs.specs) == 2
    assert result["repetitions"][0]["total_searches"] == 2
    assert result["repetitions"][0]["reached"] is False
    assert result["nfailed"] == 1


def test_the_text_lists_each_repetition_then_the_calls_over_them_and_names_the_code(search):
    template = _template(Search={"repetitions": 2})
    result, _climbs, written = search(_climb(calls=12), _climb(calls=20), template=template)
    spent = _calls((-0.2, 0.0, 0.0)) + _calls((0.2, 0.0, 0.0))
    first, second = 1 + 12 + spent, 1 + 20 + spent
    lines = written.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        f"# rep 1 force_calls {first} unique_saddles 1 fraction_connected 1.0 total_searches 1 "
        f"connected 1 minimize_calls 1 minimum_converged 1 seed {CATALOG_SEED + 1}"
    )
    assert lines[1] == (
        f"# rep 2 force_calls {second} unique_saddles 1 fraction_connected 1.0 total_searches 1 "
        f"connected 1 minimize_calls 1 minimum_converged 1 seed {CATALOG_SEED + 2}"
    )
    fields = _fields(written)
    assert float(fields["force_calls"]) == (first + second) / 2
    assert fields["force_calls_min"] == str(first)
    assert fields["force_calls_max"] == str(second)
    assert fields["code"] == "rgsaddle"
    assert fields["code_version"] == "0.2.2"
    assert fields["algorithm"] == "Dimer L-BFGS"
    assert fields["tuning"] == "default"
    assert fields["template"] == "rgsaddle/toy-dimer"
    assert fields["problem"] == "saddle-search/pt-island"
    assert fields["counter"] == "harness"
    assert fields["repetitions"] == "2"
    assert fields["nfailed"] == "0"
    assert fields["n"] == "2"
    assert result["force_calls"] == (first + second) / 2
    assert [summary["seed"] for summary in result["repetitions"]] == [CATALOG_SEED + 1, CATALOG_SEED + 2]


def test_each_repetition_draws_from_its_own_seed_and_starts_from_an_empty_table(search):
    template = _template(Search={"repetitions": 2})
    result, climbs, _written = search(_climb(), _climb(), template=template)
    assert [spec["name"] for spec in climbs.specs] == ["rep1-search1", "rep2-search1"]
    assert not np.array_equal(climbs.specs[0]["positions"][2], climbs.specs[1]["positions"][2])
    assert [record["success"] for record in result["records"]] == [True, True]
    assert [summary["unique_saddles"] for summary in result["repetitions"]] == [1, 1]


def test_one_repetition_can_be_selected_and_keeps_its_seed(search):
    template = _template(Search={"repetitions": 2})
    both, both_climbs, _written = search(_climb(), _climb(), template=template)
    only, only_climbs, written = search(_climb(), template=template, repetition=2)
    assert [spec["name"] for spec in only_climbs.specs] == ["rep2-search1"]
    assert np.array_equal(only_climbs.specs[0]["positions"], both_climbs.specs[1]["positions"])
    assert [summary["repetition"] for summary in only["repetitions"]] == [2]
    assert only["repetitions"][0]["seed"] == CATALOG_SEED + 2
    assert _fields(written)["repetitions"] == "2"
    assert _fields(written)["n"] == "1"


def test_only_the_repetitions_that_miss_their_goal_count_as_failed_in_the_text_and_the_result(search):
    template = _template(Search={"repetitions": 2})
    outcomes = [_climb(mode=(0.0, 1.0, 0.0)), _climb(), _climb(mode=(0.0, 1.0, 0.0)), _climb(mode=(0.0, 1.0, 0.0))]
    result, _climbs, written = search(*outcomes, template=template, limit=2)
    assert [summary["reached"] for summary in result["repetitions"]] == [True, False]
    assert [summary["total_searches"] for summary in result["repetitions"]] == [2, 2]
    assert result["nfailed"] == 1
    assert _fields(written)["nfailed"] == "1"


def test_a_run_that_reaches_its_goal_reports_the_first_connected_saddle_as_the_last_search(search):
    problem = _problem(goal={"connected_saddles": 1})
    result, climbs, _written = search(_climb(mode=(0.0, 1.0, 0.0)), _climb(), _climb(), problem=problem)
    assert len(climbs.specs) == 2
    assert [record["reasons"] for record in result["records"]] == [["not_connected"], []]
    assert result["repetitions"][0]["total_searches"] == 2
    assert result["repetitions"][0]["fraction_connected"] == 0.5


def test_the_written_files_agree_with_the_returned_records(search):
    result, _climbs, written = search(_climb())
    assert _rows(written) == json.loads(json.dumps(result["records"], default=float))
    document = json.loads(written.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["schema"] == "optbench.result/v2"
    assert document["status"] == "archive"
    assert document["problem"] == "saddle-search/pt-island"
    assert document["scoring"] == "par2-bootstrap"
    assert document["par_factor"] == 2
    (run,) = document["runs"]
    assert run["entry"] == "toy"
    assert run["cap"] == SEARCH_CAP
    assert run["code"] == "rgsaddle"
    assert run["commit"] == ""
    assert run["provenance"]["parameters"] == {"template": "rgsaddle/toy-dimer", "problem": "saddle-search/pt-island", "cap": SEARCH_CAP}
    assert run["starts"] == _rows(written)
    assert run["metrics"]["nfailed"] == 0
    assert run["metrics"]["par2_mean"] == float(result["records"][0]["force_calls"])


def test_a_finished_run_leaves_no_checkpoint(search):
    _result, _climbs, written = search(_climb())
    assert sorted(path.name for path in written.parent.iterdir()) == ["toy.dat", "toy.starts.jsonl", "toy.v2.json"]


def test_progress_is_printed_for_the_first_search_and_the_one_that_reaches_the_goal(search, capsys):
    search(_climb(mode=(0.0, 1.0, 0.0)), _climb())
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("rgsaddle/toy-dimer rep 1 searches 1 connected 0 calls ")
    assert printed[0].endswith("failure divergence")
    assert printed[1].startswith("rgsaddle/toy-dimer rep 1 searches 2 connected 1 calls ")
    assert printed[1].endswith("failure None")


def test_progress_is_printed_every_twenty_fifth_search(search, capsys):
    problem = _problem(goal={"connected_saddles": 50})
    search(*[_climb(mode=(0.0, 1.0, 0.0))] * 26, problem=problem)
    printed = capsys.readouterr().out.splitlines()
    assert [line.split(" calls ")[0] for line in printed] == [
        "rgsaddle/toy-dimer rep 1 searches 1 connected 0",
        "rgsaddle/toy-dimer rep 1 searches 25 connected 0",
    ]


# What the run refuses before it climbs


def test_the_prfo_session_is_refused_for_displacement_searches(search):
    template = _template(Main={"session": "prfo"})
    with pytest.raises(RuntimeError, match="saddle-search displacements use the min-mode session"):
        search(_climb(), template=template)


def test_displacing_a_frozen_atom_is_refused_and_the_frozen_ones_are_named(search):
    template = _template(Search={"displace_atoms": "0, 2, 1"})
    with pytest.raises(RuntimeError, match=r"displace atoms \[0, 1\] are frozen"):
        search(_climb(), template=template)


@pytest.mark.parametrize("repetition", [0, 3])
def test_a_repetition_outside_the_templates_range_is_refused_before_any_search(search, repetition):
    template = _template(Search={"repetitions": 2})
    with pytest.raises(RuntimeError, match="rgsaddle/toy-dimer has repetitions 1 to 2, not " + str(repetition)):
        search(_climb(), template=template, repetition=repetition)


# The reactant is the minimized start


def test_the_first_minimization_moves_the_start_to_its_well_and_every_search_is_judged_against_that_well(search):
    _result, at_well, _written = search(_climb())
    result, climbs, _written = search(_climb(), start=DISPLACED)
    (record,) = result["records"]
    summary = result["repetitions"][0]
    assert summary["minimize_calls"] == _calls(DISPLACED)
    assert summary["minimum_converged"] is True
    assert record["success"] is True
    assert record["per_atom_minus"] < 1e-3
    assert np.allclose(climbs.specs[0]["positions"][2], at_well.specs[0]["positions"][2], atol=1e-3)


def test_a_first_minimization_that_runs_out_of_calls_is_reported_and_the_run_goes_on(search):
    template = _template(Optimizer={"max_iterations": 7})
    result, climbs, written = search(_climb(), template=template, start=DISPLACED)
    summary = result["repetitions"][0]
    assert summary["minimum_converged"] is False
    assert summary["minimize_calls"] == 7
    assert len(climbs.specs) == 1
    assert " minimum_converged 0 " in written.read_text(encoding="utf-8")


def test_a_problem_that_lists_no_con_geometry_is_refused_by_name(search):
    poscar = _problem(input={"geometries": ["toy/island.POSCAR"]})
    with pytest.raises(ValueError, match="saddle-search/pt-island lists no .con geometry"):
        search(_climb(), problem=poscar)
    empty = _problem(input={"geometries": []})
    with pytest.raises(ValueError, match="saddle-search/pt-island lists no .con geometry"):
        search(_climb(), problem=empty)


# Stopping a run and picking it up again


def _checkpoint(written: Path) -> tuple[Path, Path]:
    return written.with_name("toy.header.json"), written.with_name("toy.partial.jsonl")


def _outputs(written: Path) -> dict:
    document = json.loads(written.with_suffix(".v2.json").read_text(encoding="utf-8"))
    return {"text": written.read_text(encoding="utf-8"), "starts": _rows(written), "document": document}


SEQUENCE = [_climb(), _climb(mode=(0.0, 1.0, 0.0)), _climb(x=(4e-5, 0.0, 0.0)), _climb(calls=30, x=(-4e-5, 0.0, 0.0))]
MANY = {"goal": {"connected_saddles": 5}}


def test_a_killed_run_leaves_its_minimum_and_one_row_per_finished_search(search, tmp_path):
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed, match="rep1-search3"):
        search(SEQUENCE[0], SEQUENCE[1], _killed, problem=_problem(**MANY), output=written, start=DISPLACED)
    header, partial = _checkpoint(written)
    assert not written.exists()
    kept = json.loads(header.read_text(encoding="utf-8"))
    assert kept["template"] == "rgsaddle/toy-dimer"
    assert kept["rep"] == 1
    assert kept["minimum"] == pytest.approx([WELL, 0.0, 0.0], abs=1e-3)
    assert kept["min_energy"] == pytest.approx(_surface(_minimum())[0], abs=1e-6)
    assert kept["minimize_calls"] == _calls(DISPLACED)
    assert kept["min_ok"] is True
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    assert [row["start"] for row in rows] == ["rep1-search1", "rep1-search2"]
    assert [row["climb_calls"] for row in rows] == [12, 12]
    assert rows[0]["saddle_x"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert "saddle_x" not in rows[1]
    assert all("rng_state" in row and "endpoint_calls" in row for row in rows)


def test_a_killed_run_resumes_to_the_files_an_uninterrupted_run_writes(search, tmp_path, capsys):
    problem = _problem(**MANY)
    straight, straight_climbs, straight_path = search(*SEQUENCE, problem=problem, output=tmp_path / "straight" / "toy.dat", start=DISPLACED)
    killed_path = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(SEQUENCE[0], SEQUENCE[1], _killed, problem=problem, limit=4, output=killed_path, start=DISPLACED)
    capsys.readouterr()
    resumed, resumed_climbs, _ = search(SEQUENCE[2], SEQUENCE[3], problem=problem, limit=4, output=killed_path, start=DISPLACED, resume=True)
    assert "rgsaddle/toy-dimer rep 1 resume searches 2 connected 1" in capsys.readouterr().out
    assert [spec["name"] for spec in resumed_climbs.specs] == ["rep1-search3", "rep1-search4"]
    for before, after in zip(straight_climbs.specs[2:], resumed_climbs.specs):
        assert np.array_equal(before["positions"], after["positions"])
        assert np.array_equal(before["file_mode"], after["file_mode"])
    assert _outputs(killed_path) == _outputs(straight_path)
    assert [record["reasons"] for record in resumed["records"]] == [[], ["not_connected"], ["duplicate"], ["duplicate"]]
    assert resumed["repetitions"] == straight["repetitions"]
    header, partial = _checkpoint(killed_path)
    assert not header.exists()
    assert not partial.exists()


def test_a_run_killed_in_its_first_search_resumes_from_the_seed(search, tmp_path):
    problem = _problem(**MANY)
    _straight, straight_climbs, straight_path = search(*SEQUENCE[:2], problem=problem, output=tmp_path / "straight" / "toy.dat", start=DISPLACED)
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed, match="rep1-search1"):
        search(_killed, problem=problem, output=written, limit=2, start=DISPLACED)
    header, partial = _checkpoint(written)
    assert header.is_file()
    assert not partial.exists() or partial.read_text(encoding="utf-8") == ""
    _resumed, climbs, _ = search(*SEQUENCE[:2], problem=problem, output=written, resume=True, start=DISPLACED)
    assert [spec["name"] for spec in climbs.specs] == ["rep1-search1", "rep1-search2"]
    assert np.array_equal(climbs.specs[0]["positions"], straight_climbs.specs[0]["positions"])
    assert _outputs(written) == _outputs(straight_path)


def test_a_saddle_found_before_the_kill_is_still_a_repeat_after_the_resume_even_when_its_barrier_failed(search, tmp_path):
    problem = _problem(goal={"connected_saddles": 5, "barrier_below_eV": 0.3})
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(_climb(), _killed, problem=problem, output=written, start=DISPLACED)
    result, _climbs, _ = search(_climb(), problem=problem, output=written, resume=True, limit=2, start=DISPLACED)
    assert result["records"][0]["reasons"] == ["barrier"]
    assert result["records"][1]["reasons"] == ["duplicate", "barrier"]
    assert result["repetitions"][0]["unique_saddles"] == 1


def test_a_killed_run_started_again_without_resume_begins_with_its_first_search(search, tmp_path):
    problem = _problem(**MANY)
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(SEQUENCE[0], SEQUENCE[1], _killed, problem=problem, output=written, start=DISPLACED)
    result, climbs, _ = search(*SEQUENCE, problem=problem, output=written, start=DISPLACED)
    assert [spec["name"] for spec in climbs.specs] == ["rep1-search1", "rep1-search2", "rep1-search3", "rep1-search4"]
    assert len(_rows(written)) == 4
    assert result["repetitions"][0]["total_searches"] == 4
    assert _checkpoint(written)[0].exists() is False


def test_a_fresh_run_discards_the_rows_of_the_killed_one_it_replaces(search, tmp_path):
    problem = _problem(**MANY)
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(SEQUENCE[0], SEQUENCE[1], _killed, problem=problem, output=written, start=DISPLACED)
    with pytest.raises(Killed):
        search(SEQUENCE[0], _killed, problem=problem, output=written, start=DISPLACED)
    rows = [json.loads(line) for line in _checkpoint(written)[1].read_text(encoding="utf-8").splitlines()]
    assert [row["start"] for row in rows] == ["rep1-search1"]
    _result, climbs, _ = search(SEQUENCE[1], problem=problem, output=written, resume=True, limit=2, start=DISPLACED)
    assert [spec["name"] for spec in climbs.specs] == ["rep1-search2"]


def test_a_resumed_run_still_reports_a_first_minimization_that_did_not_converge(search, tmp_path):
    problem = _problem(**MANY)
    template = _template(Optimizer={"max_iterations": 7})
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(SEQUENCE[0], _killed, template=template, problem=problem, output=written, start=DISPLACED)
    result, _climbs, _ = search(SEQUENCE[1], template=template, problem=problem, output=written, resume=True, limit=2, start=DISPLACED)
    summary = result["repetitions"][0]
    assert summary["minimum_converged"] is False
    assert summary["minimize_calls"] == 7
    assert " minimum_converged 0 " in written.read_text(encoding="utf-8")


def test_a_checkpoint_of_another_template_is_not_picked_up_when_resume_is_asked(search, tmp_path):
    problem = _problem(**MANY)
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(SEQUENCE[0], _killed, problem=problem, output=written, start=DISPLACED)
    other = _template()
    other["id"] = "rgsaddle/other-dimer"
    _result, climbs, _ = search(SEQUENCE[0], SEQUENCE[1], template=other, problem=problem, output=written, resume=True, start=DISPLACED)
    assert [spec["name"] for spec in climbs.specs] == ["rep1-search1", "rep1-search2"]
    assert len(_rows(written)) == 2


def test_a_checkpoint_of_another_repetition_is_not_picked_up_when_resume_is_asked(search, tmp_path):
    problem = _problem(**MANY)
    template = _template(Search={"repetitions": 2})
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(SEQUENCE[0], _killed, template=template, problem=problem, output=written, repetition=1)
    _result, climbs, _ = search(SEQUENCE[0], SEQUENCE[1], template=template, problem=problem, output=written, repetition=2, resume=True)
    assert [spec["name"] for spec in climbs.specs] == ["rep2-search1", "rep2-search2"]


def test_the_run_selected_by_repetition_resumes_its_own_checkpoint(search, tmp_path):
    problem = _problem(**MANY)
    template = _template(Search={"repetitions": 2})
    written = tmp_path / "killed" / "toy.dat"
    with pytest.raises(Killed):
        search(SEQUENCE[0], _killed, template=template, problem=problem, output=written, repetition=2)
    _result, climbs, _ = search(SEQUENCE[1], template=template, problem=problem, output=written, repetition=2, resume=True, limit=2)
    assert [spec["name"] for spec in climbs.specs] == ["rep2-search2"]


# The entry from run_template


def test_the_saddle_search_problem_is_handed_to_the_catalogue_with_the_runs_options(monkeypatch, tmp_path):
    seen = {}

    def catalogue(root, problem, template, **options):
        seen.update(root=root, problem=problem, template=template, options=options)
        return {"force_calls": None}

    monkeypatch.delenv("RGSADDLE_SRC", raising=False)
    monkeypatch.setattr(driver, "_run_saddle_search", catalogue)
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    template = load_template(ROOT, "rgsaddle/saddle-dimer-lbfgs")
    output = tmp_path / "out" / "catalogue.dat"
    result = driver.run_template(ROOT, problem, template, output=output, repetition=3, limit=7, resume=True)
    assert result == {"force_calls": None}
    assert seen["root"] == ROOT
    assert seen["problem"] is problem
    assert seen["template"] is template
    options = seen["options"]
    assert options["output"] == output
    assert options["force_tol"] == 0.001
    assert options["work"] == output.parent / "work" / "rgsaddle-saddle-dimer-lbfgs"
    assert options["work"].is_dir()
    assert (options["repetition"], options["limit"], options["resume"]) == (3, 7, True)


@pytest.mark.parametrize(
    ("templates", "problems", "expected"),
    [(0.0005, None, 0.0005), (None, 0.002, 0.002), (None, None, 0.001)],
    ids=["the-template-states-it", "the-problem-states-it", "neither-states-it"],
)
def test_the_force_stop_of_a_catalogue_run_is_the_templates_then_the_problems_then_a_thousandth(monkeypatch, tmp_path, templates, problems, expected):
    seen = []
    monkeypatch.delenv("RGSADDLE_SRC", raising=False)
    monkeypatch.setattr(driver, "_run_saddle_search", lambda root, problem, template, **options: seen.append(options["force_tol"]) or {})
    problem = copy.deepcopy(resolve_problem(ROOT, "saddle-search/pt-island"))
    template = copy.deepcopy(load_template(ROOT, "rgsaddle/saddle-dimer-lbfgs"))
    del problem["convergence"]["force_norm"]
    del template["ini"]["Optimizer"]["converged_force"]
    if templates is not None:
        template["ini"]["Optimizer"]["converged_force"] = templates
    if problems is not None:
        problem["convergence"]["force_norm"] = problems
    driver.run_template(ROOT, problem, template, output=tmp_path / "catalogue.dat")
    assert seen == [expected]
