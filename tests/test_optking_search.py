"""OptKing's transition-state search on a saddle with a closed-form answer.

OptKing and qcelemental are in the ``campaign`` group of the lockfile, so
this file runs under ``uv run --frozen --group campaign pytest`` and is
skipped in the default environment.

One free atom sits between two pinned atoms on the x axis, ``half_gap`` from
each. By symmetry the force on it is zero at the midpoint. Along the axis its
curvature is ``2 u''(a)``, negative past the inflection of
``u(r) = 4 (r**-12 - r**-6)`` at 1.2445. Across the axis it is ``2 u'(a)/a``,
positive past the pair minimum at 1.1225. A half gap of 1.5 is therefore a
first-order saddle, 1.2 is a minimum, and 1.0 is a second-order saddle.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import signal
import types
from pathlib import Path

import numpy as np
import pytest

# OptKing opens a log file in the working directory when it is imported, so
# the check looks for it without importing it.
for _name in ("optking", "qcelemental"):
    if importlib.util.find_spec(_name) is None:
        pytest.skip(f"{_name} is in the campaign group of the lockfile", allow_module_level=True)

from optbench.counter import ForceCallCap
from optbench.runners import optking_ts

LJ = {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "rc": 40.0}
MORSE = {"kind": "morse", "D": 0.5, "alpha": 1.5, "r0": 2.5, "rc": 5.0, "form": "smooth"}
HEAD = ["a label", "0.0000 TIME", "10.0 11.0 12.0", "90.0 90.0 90.0", "0 0", "0 0 0"]
NEAR = (0.02, 0.03, -0.01)
CAP = 100


def _u(r: float) -> float:
    return 4.0 * (r**-12 - r**-6)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


def _saddle_energy(half_gap: float) -> float:
    return 2.0 * _u(half_gap) + _u(2.0 * half_gap)


def _morse_terms(r: float) -> tuple[float, float]:
    """The Morse well and its first derivative, before the linear term that zeroes the slope at the cutoff."""
    d, a, r0 = MORSE["D"], MORSE["alpha"], MORSE["r0"]
    x = math.exp(-a * (r - r0))
    return d * (x * x - 2.0 * x), 2.0 * a * d * x * (1.0 - x)


def _morse_energy(r: float) -> float:
    well, _ = _morse_terms(r)
    cut_well, cut_slope = _morse_terms(MORSE["rc"])
    return well - cut_well - cut_slope * (r - MORSE["rc"])


def _morse_curvature(r: float) -> float:
    d, a, r0 = MORSE["D"], MORSE["alpha"], MORSE["r0"]
    x = math.exp(-a * (r - r0))
    return 2.0 * a * a * d * x * (2.0 * x - 1.0)


@contextlib.contextmanager
def _within(seconds: int):
    """Turn a search that never ends into a crash record rather than a hung run."""

    def expired(signum, frame):
        raise TimeoutError(f"still searching after {seconds} s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _job(tmp_path: Path, half_gap: float = 1.5, free_at=(0.0, 0.0, 0.0), **overrides) -> dict:
    job = {
        "index": 0,
        "start": "initial_0",
        "positions": [[-half_gap, 0.0, 0.0], list(free_at), [half_gap, 0.0, 0.0]],
        "fixed": [True, False, True],
        "symbol": "Ar",
        "symbols": ["Ar", "Ar", "Ar"],
        "box": [40.0, 40.0, 40.0],
        "pbc": [False, False, False],
        "potential": dict(LJ),
        "step_type": "RS_I_RFO",
        "full_hess_every": 0,
        "fd_step": 1.0e-5,
        "gtol": 1.0e-6,
        "cap": CAP,
        "scratch": str(tmp_path / "scratch"),
    }
    job.update(overrides)
    return job


@pytest.mark.parametrize(
    ("half_gap", "index", "failure"), [(1.5, 1, None), (1.2, 0, "wrong_stationary"), (1.0, 2, "wrong_stationary")]
)
def test_a_start_on_a_stationary_point_is_judged_by_its_hessian_index(tmp_path, half_gap, index, failure):
    record = optking_ts.search_start(_job(tmp_path, half_gap))
    along = 2.0 * _curvature(half_gap)
    across = 2.0 * _slope(half_gap) / half_gap
    assert record["failure"] == failure
    assert record["success"] is (failure is None)
    assert record["converged"] is True
    assert (record["force_calls"], record["steps"], record["hessian_calls"]) == (1, 0, 0)
    assert record["hessian_index"] == index
    assert record["lowest_eigenvalue"] == pytest.approx(min(along, across), rel=1e-9)
    assert record["energy"] == pytest.approx(_saddle_energy(half_gap), rel=1e-12)
    assert record["force_norm"] == pytest.approx(0.0, abs=1e-12)
    assert "error" not in record


@pytest.mark.parametrize("step_type", ["RS_I_RFO", "P_RFO"])
@pytest.mark.parametrize("free_at", [NEAR, (0.0, 0.1, 0.0), (0.1, -0.08, 0.06)])
def test_a_displaced_start_climbs_to_the_saddle_between_the_pinned_atoms(tmp_path, step_type, free_at):
    half_gap = 1.5
    record = optking_ts.search_start(_job(tmp_path, half_gap, free_at, step_type=step_type))
    assert record["success"] is True
    assert record["failure"] is None
    assert record["hessian_index"] == 1
    assert record["force_norm"] <= 1.0e-6
    assert record["energy"] == pytest.approx(_saddle_energy(half_gap), abs=1e-9)
    assert record["lowest_eigenvalue"] == pytest.approx(2.0 * _curvature(half_gap), rel=1e-4)
    assert record["steps"] >= 1
    assert record["hessian_calls"] == 1
    assert "error" not in record


@pytest.mark.parametrize("step_type", ["RS_I_RFO", "P_RFO"])
@pytest.mark.parametrize("every", [0, 1])
def test_a_hessian_costs_two_calls_per_free_coordinate_and_a_step_costs_one(tmp_path, step_type, every):
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR, step_type=step_type, full_hess_every=every))
    assert record["success"] is True
    assert record["force_calls"] == 1 + 2 * 3 * record["hessian_calls"] + record["steps"]
    assert record["hessian_calls"] == (1 if every == 0 else record["steps"])


@pytest.mark.parametrize("free_at", [(0.0, 0.0, 0.0), NEAR])
def test_reaching_the_saddle_on_the_call_that_spends_the_budget_is_a_cap_failure(tmp_path, free_at):
    spent = optking_ts.search_start(_job(tmp_path, 1.5, free_at))["force_calls"]
    for cap, success in ((spent + 1, True), (spent, False), (spent - 1, False)):
        if cap < 1:
            continue
        record = optking_ts.search_start(_job(tmp_path, 1.5, free_at, cap=cap))
        assert record["success"] is success
        assert record["converged"] is success
        assert record["force_calls"] == min(spent, cap)
        assert record["failure"] == (None if success else "cap")


def test_a_budget_smaller_than_the_first_hessian_ends_inside_its_stencil(tmp_path):
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR, cap=4))
    assert record["failure"] == "cap"
    assert record["reasons"] == ["cap"]
    assert (record["force_calls"], record["hessian_calls"], record["steps"]) == (4, 0, 0)
    assert record["hessian_index"] is None


def test_an_unknown_potential_and_a_start_with_no_free_atom_are_crashes_that_name_themselves(tmp_path):
    unknown = optking_ts.search_start(_job(tmp_path, potential={"kind": "eam"}))
    assert (unknown["failure"], unknown["error"]) == ("crash", "ValueError: eam")
    pinned = optking_ts.search_start(_job(tmp_path, fixed=[True, True, True]))
    assert (pinned["failure"], pinned["error"]) == ("crash", "ValueError: initial_0 has no free atoms")
    assert pinned["force_calls"] == 0


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_a_force_stop_the_search_cannot_reach_ends_in_divergence_long_before_the_budget(tmp_path):
    with _within(30):
        record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR, gtol=0.0, cap=1000))
    assert record["failure"] == "divergence"
    assert record["success"] is False
    assert record["converged"] is False
    assert record["steps"] >= optking_ts.STALL_STEPS
    assert record["force_calls"] < 100
    assert record["force_norm"] < 1.0e-8
    assert record["energy"] == pytest.approx(_saddle_energy(1.5), abs=1e-9)


def test_a_search_with_one_step_of_patience_still_climbs_because_its_steps_are_not_stalls(tmp_path, monkeypatch):
    monkeypatch.setattr(optking_ts, "STALL_STEPS", 1)
    record = optking_ts.search_start(_job(tmp_path, 1.5, (0.1, -0.08, 0.06)))
    assert record["success"] is True
    assert record["steps"] >= 2


def test_an_unknown_step_type_reaches_the_library_and_is_a_crash_that_names_it(tmp_path):
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR, step_type="NOT_A_STEP"))
    assert record["failure"] == "crash"
    assert record["success"] is False
    assert "NOT_A_STEP" in record["error"]
    assert record["force_calls"] == 0


def test_the_search_leaves_the_working_directory_as_it_found_it(tmp_path, monkeypatch):
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR))
    assert record["success"] is True
    assert Path.cwd() == here
    assert list(here.iterdir()) == []
    assert (tmp_path / "scratch").is_dir()


def test_the_morse_saddle_between_two_pinned_atoms_is_found_with_its_closed_form_curvature(tmp_path):
    half_gap = 3.5
    record = optking_ts.search_start(
        _job(tmp_path, half_gap, (0.03, -0.04, 0.02), potential=dict(MORSE), box=[40.0, 40.0, 40.0])
    )
    assert record["success"] is True
    assert record["hessian_index"] == 1
    assert record["force_norm"] <= 1.0e-6
    assert record["energy"] == pytest.approx(2.0 * _morse_energy(half_gap), abs=1e-9)
    assert record["lowest_eigenvalue"] == pytest.approx(2.0 * _morse_curvature(half_gap), rel=1e-4)


def _con(half_gap: float) -> str:
    rows = [(-half_gap, 0.0, 0.0, 1), (0.0, 0.0, 0.0, 0), (half_gap, 0.0, 0.0, 1)]
    atoms = [f"{x!r} {y!r} {z!r} {frozen} {index}" for index, (x, y, z, frozen) in enumerate(rows)]
    return "\n".join([*HEAD, "1", "3", "39.948", "Ar", "Coordinates of Component 1", *atoms, ""])


@pytest.mark.parametrize("workers", [1, 2])
def test_a_run_over_the_three_stationary_starts_scores_a_wrong_point_as_a_legacy_success(
    tmp_path, monkeypatch, workers
):
    if optking_ts.installed_version() != optking_ts.VERSION:
        pytest.skip("a run wants the OptKing revision the registry pins")
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    cons = tmp_path / "cons"
    cons.mkdir()
    for name, half_gap in (("initial_0", 1.5), ("initial_1", 1.2), ("initial_2", 1.0)):
        (cons / f"{name}.con").write_text(_con(half_gap), encoding="utf-8")
    problem = {
        "id": "tsopt/test",
        "potential": {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0, "pbc": [False] * 3},
        "convergence": {"force_norm": 1.0e-6, "max_force_calls": CAP},
    }
    template = {
        "id": "optking/test-rfo",
        "problem": "tsopt/test",
        "code": "optking",
        "algorithm": "RS-I-RFO",
        "code_version": optking_ts.VERSION,
        "commit": optking_ts.PIN,
        "potential": dict(LJ, pbc=[False] * 3),
        "optking": {"step_type": "RS_I_RFO"},
        "input": {"select": "initial"},
    }
    output = tmp_path / "out" / "run.dat"
    summary = optking_ts.run_template(
        tmp_path, problem, template, geometries=cons, output=output, workers=workers, work=tmp_path / "work"
    )
    assert summary["starts"] == 3
    assert summary["partial"] is False
    assert (summary["force_calls"], summary["nfailed"]) == (1, 0)
    assert summary["v2_nfailed"] == 2
    assert summary["par2_mean"] == pytest.approx((1 + 2 * 2 * CAP) / 3)
    assert list(here.iterdir()) == []
    rows = [json.loads(line) for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["start"], row["failure"], row["hessian_index"]) for row in rows] == [
        ("initial_0", None, 1),
        ("initial_1", "wrong_stationary", 0),
        ("initial_2", "wrong_stationary", 2),
    ]


@pytest.fixture
def library(tmp_path_factory, monkeypatch):
    """The classes of OptKing the search drives. Importing it opens a log file in the working directory."""
    monkeypatch.chdir(tmp_path_factory.mktemp("first_import"))
    from optking.exceptions import AlgError, OptError
    from optking.opt_helper import CustomHelper

    return types.SimpleNamespace(helper=CustomHelper, optimizer_error=OptError, algorithm_error=AlgError)


def _raising(monkeypatch, library, name: str, error: Exception) -> None:
    def raises(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(library.helper, name, raises)


def _near_energy(half_gap: float = 1.5) -> float:
    left = math.dist(NEAR, (-half_gap, 0.0, 0.0))
    right = math.dist(NEAR, (half_gap, 0.0, 0.0))
    return _u(left) + _u(right) + _u(2.0 * half_gap)


def test_a_potential_with_no_finite_energy_at_the_start_is_a_divergence_after_one_call(tmp_path):
    record = optking_ts.search_start(_job(tmp_path, 3.5, NEAR, potential=dict(MORSE, D=float("nan"))))
    assert (record["success"], record["failure"], record["reasons"]) == (False, "divergence", ["divergence"])
    assert (record["force_calls"], record["steps"], record["hessian_calls"]) == (1, 0, 0)
    assert record["energy"] is None and record["force_norm"] is None
    assert "error" not in record


def test_a_search_given_no_calls_spends_none_and_does_not_succeed(tmp_path):
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR, cap=0))
    assert record["success"] is False
    assert (record["force_calls"], record["steps"], record["hessian_calls"]) == (0, 0, 0)
    assert record["energy"] is None and record["hessian_index"] is None


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_a_long_search_leaves_a_note_of_where_it_is_every_25_steps(tmp_path, monkeypatch):
    short = optking_ts.search_start(_job(tmp_path / "short", 1.5, NEAR))
    assert short["success"] is True and short["steps"] < 25
    assert not (tmp_path / "short" / "scratch" / "progress.txt").exists()
    monkeypatch.setattr(optking_ts, "STALL_STEPS", 30)
    record = optking_ts.search_start(_job(tmp_path / "long", 1.5, NEAR, gtol=0.0))
    assert record["failure"] == "divergence" and 25 < record["steps"] < 50
    words = (tmp_path / "long" / "scratch" / "progress.txt").read_text(encoding="utf-8").split()
    assert words[:3] == ["steps", "25", "calls"] and words[4] == "force"
    assert int(words[3]) <= record["force_calls"]
    assert float(words[5]) == record["force_norm"] < 1.0e-8


@pytest.mark.parametrize(
    ("error", "named"),
    [
        ("optimizer_error", "OptError: the step failed"),
        ("algorithm_error", "AlgError: no rational function step"),
        ("value", "ValueError: the step is not a number"),
        ("linalg", "LinAlgError: singular matrix"),
    ],
)
def test_an_error_the_library_raises_on_a_step_is_a_crash_that_names_it_and_keeps_the_start(
    tmp_path, monkeypatch, library, error, named
):
    errors = {
        "optimizer_error": library.optimizer_error("the step failed"),
        "algorithm_error": library.algorithm_error("no rational function step"),
        "value": ValueError("the step is not a number"),
        "linalg": np.linalg.LinAlgError("singular matrix"),
    }
    _raising(monkeypatch, library, "take_step", errors[error])
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR))
    assert (record["success"], record["failure"], record["reasons"]) == (False, "crash", ["crash"])
    assert record["error"] == named
    assert (record["force_calls"], record["hessian_calls"], record["steps"]) == (1 + 2 * 3, 1, 0)
    assert record["energy"] == pytest.approx(_near_energy(), abs=1e-12)


def test_the_librarys_own_step_limit_is_a_cap_only_when_the_budget_is_spent(tmp_path, monkeypatch, library):
    _raising(monkeypatch, library, "take_step", library.optimizer_error("Maximum number of steps exceeded"))
    spent_by_the_stencil = 1 + 2 * 3
    roomy = optking_ts.search_start(_job(tmp_path / "roomy", 1.5, NEAR))
    assert (roomy["failure"], roomy["reasons"], roomy["force_calls"]) == ("divergence", ["divergence"], spent_by_the_stencil)
    exact = optking_ts.search_start(_job(tmp_path / "exact", 1.5, NEAR, cap=spent_by_the_stencil))
    assert (exact["failure"], exact["reasons"], exact["force_calls"]) == ("cap", ["cap"], spent_by_the_stencil)
    assert "error" not in roomy and "error" not in exact


@pytest.mark.parametrize(("name", "calls"), [("take_step", 1 + 2 * 3), ("calculations_needed", 1)])
def test_a_budget_error_raised_by_the_library_is_a_cap_whichever_call_raises_it(
    tmp_path, monkeypatch, library, name, calls
):
    _raising(monkeypatch, library, name, ForceCallCap("budget"))
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR))
    assert (record["success"], record["failure"], record["reasons"]) == (False, "cap", ["cap"])
    assert record["force_calls"] == calls
    assert record["energy"] == pytest.approx(_near_energy(), abs=1e-12)
    assert "error" not in record


@pytest.mark.parametrize("moved", ["shifted", "wrong size"])
def test_a_library_that_changes_the_input_before_the_first_step_is_a_crash_with_no_force_call(
    tmp_path, monkeypatch, library, moved
):
    opened = library.helper.geom
    shapes = {
        "shifted": lambda self: np.asarray(opened.fget(self), dtype=float) + 1.0e-3,
        "wrong size": lambda self: np.zeros(6),
    }
    monkeypatch.setattr(library.helper, "geom", property(shapes[moved]))
    record = optking_ts.search_start(_job(tmp_path, 1.5, NEAR))
    assert (record["success"], record["failure"]) == (False, "crash")
    assert record["error"] == "RuntimeError: OptKing moved the input coordinates before the first step"
    assert (record["force_calls"], record["steps"], record["hessian_calls"]) == (0, 0, 0)
    assert record["energy"] is None
