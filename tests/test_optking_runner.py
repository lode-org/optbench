"""What the OptKing runner reads, judges and writes, with no OptKing in the process.

The search itself needs the library and is covered in ``test_optking_search``.
Everything around it does not: the con reader, the Hessian the index is read
from, the evaluator the search calls, the record each ending becomes, the
files a run publishes, and the driver that limits, resumes and survives a
kill. A free atom between two pinned atoms has curvatures that follow from
the pair derivatives of ``u(r) = 4 (r**-12 - r**-6)``, written out here, so
no expected index or eigenvalue is read back from the code under test.
"""

from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest

from optbench.counter import HarnessCounter
from optbench.runners import optking_ts

RMIN = 2.0 ** (1.0 / 6.0)
STIFFNESS = 144.0 * 2.0 ** (-4.0 / 3.0)

LJ = {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "rc": 40.0}
MORSE = {"kind": "morse", "D": 0.5, "alpha": 1.5, "r0": 2.5, "rc": 5.0, "form": "smooth"}
PROBLEM_LJ = {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0, "pbc": [False] * 3}
PROBLEM_MORSE = {
    "kind": "morse",
    "pair_style": "morse/smooth/linear",
    "D": 0.5,
    "alpha": 1.5,
    "r0": 2.5,
    "cutoff": 5.0,
    "pbc": [True, True, False],
}
TEMPLATE_MORSE = {**{key: MORSE[key] for key in ("kind", "D", "alpha", "r0", "rc", "form")}, "pbc": [True, True, False]}

HEAD = ["a label", "0.0000 TIME", "10.0 11.0 12.0", "90.0 90.0 90.0", "0 0", "0 0 0"]


def _u(r: float) -> float:
    return 4.0 * (r**-12 - r**-6)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


def _morse_slope(r: float) -> float:
    """First derivative of the Morse curve with the linear term that zeroes the slope at the cutoff."""
    d, a, r0, rc = MORSE["D"], MORSE["alpha"], MORSE["r0"], MORSE["rc"]
    x, cut = math.exp(-a * (r - r0)), math.exp(-a * (rc - r0))
    return 2.0 * a * d * x * (1.0 - x) - 2.0 * a * d * cut * (1.0 - cut)


def _morse_curvature(r: float) -> float:
    d, a, r0 = MORSE["D"], MORSE["alpha"], MORSE["r0"]
    x = math.exp(-a * (r - r0))
    return 2.0 * a * a * d * x * (2.0 * x - 1.0)


def _con(rows, symbol: str = "Ar") -> str:
    atoms = [f"{x!r} {y!r} {z!r} {int(frozen)} {index}" for index, (x, y, z, frozen) in enumerate(rows)]
    return "\n".join([*HEAD, "1", str(len(rows)), "39.948", symbol, "Coordinates of Component 1", *atoms, ""])


def _job(tmp_path: Path, half_gap: float = 1.5, free_at=(0.0, 0.0, 0.0), potential=None, **overrides) -> dict:
    """One free atom between two pinned atoms, as ``search_start`` reads a job."""
    job = {
        "index": 0,
        "start": "initial_0",
        "positions": [[-half_gap, 0.0, 0.0], list(free_at), [half_gap, 0.0, 0.0]],
        "fixed": [True, False, True],
        "symbol": "Ar",
        "symbols": ["Ar", "Ar", "Ar"],
        "box": [40.0, 40.0, 40.0],
        "pbc": [False, False, False],
        "potential": dict(potential or LJ),
        "step_type": "RS_I_RFO",
        "full_hess_every": 0,
        "fd_step": 1.0e-5,
        "gtol": 1.0e-6,
        "cap": 100,
        "scratch": str(tmp_path / "scratch"),
    }
    job.update(overrides)
    return job


def test_installed_version_reports_the_distribution_or_says_it_is_missing(monkeypatch):
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "version", lambda name: "7.8.9")
    assert optking_ts.installed_version() == "7.8.9"

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    with pytest.raises(RuntimeError, match="optking is not installed"):
        optking_ts.installed_version()


def test_a_con_with_no_box_is_refused():
    with pytest.raises(ValueError, match="no box"):
        optking_ts.parse_con("label\nPt\nCoordinates of Component 1\n1.0 2.0 3.0 0 0\n")


def test_a_con_with_a_box_and_no_atoms_is_refused():
    text = "\n".join([*HEAD, "1", "0", "1.0", "Pt", "Coordinates of Component 1", ""])
    with pytest.raises(ValueError, match="no coordinates"):
        optking_ts.parse_con(text)


def test_each_atom_keeps_the_element_of_its_block_and_odd_rows_are_skipped():
    text = "\n".join(
        [
            *HEAD,
            "2",
            "2 1",
            "63.546 1.008",
            "Cu",
            "Coordinates of Component 1",
            "0.5 0.25 0.125 1 0",
            "1.0 0.0 0.0 0 1",
            "stray words in the block",
            "7.0 8.0",
            "H",
            "Coordinates of Component 2",
            "2.0 3.0 4.0 x 2",
            "5.0 6.0 7.0",
            "",
        ]
    )
    parsed = optking_ts.parse_con(text)
    assert parsed["box"] == [10.0, 11.0, 12.0]
    assert parsed["symbol"] == "Cu"
    assert parsed["symbols"] == ["Cu", "Cu", "H", "H"]
    assert parsed["fixed"] == [True, False, False, False]
    assert parsed["positions"] == [[0.5, 0.25, 0.125], [1.0, 0.0, 0.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]


def test_a_con_that_names_no_element_gets_the_placeholder_symbol():
    text = "\n".join([*HEAD, "1", "1", "1.0", "Coordinates of Component 1", "1.0 2.0 3.0 1.0 0", ""])
    parsed = optking_ts.parse_con(text)
    assert parsed["symbol"] == "X"
    assert parsed["symbols"] == ["X"]
    assert parsed["fixed"] == [True]


def test_a_repeated_geometry_is_one_call_and_returns_what_was_stored():
    counter = HarnessCounter(cache=True, cap=10)
    seen = []

    def evaluate(flat):
        seen.append(flat.copy())
        return float(flat @ flat), 2.0 * flat

    point = np.array([1.0, -2.0, 0.5])
    energy, gradient = optking_ts.counted_call(counter, evaluate, point)
    gradient[:] = 0.0
    again_energy, again_gradient = optking_ts.counted_call(counter, evaluate, point.copy())
    assert energy == pytest.approx(5.25)
    assert again_energy == pytest.approx(5.25)
    assert again_gradient == pytest.approx([2.0, -4.0, 1.0])
    assert counter.calls == 1
    assert len(seen) == 1
    optking_ts.counted_call(counter, evaluate, point + 1.0)
    assert counter.calls == 2


def test_the_difference_hessian_of_a_linear_force_is_the_symmetric_part_of_its_matrix():
    matrix = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, -1.0], [4.0, 0.0, 5.0]])

    def evaluate(point):
        return 0.0, matrix @ np.asarray(point)

    found = optking_ts.finite_difference_hessian(evaluate, np.array([0.3, -0.2, 0.1]), 1.0e-4)
    assert found == pytest.approx(0.5 * (matrix + matrix.T), abs=1e-8)


def test_a_morse_pair_has_the_curvatures_the_smooth_linear_form_gives():
    distance = 3.2
    found = optking_ts.morse_smooth_hessian(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], [40.0] * 3, [False] * 3, 0.5, 1.5, 2.5, 5.0
    )
    block = np.diag([_morse_curvature(distance), _morse_slope(distance) / distance, _morse_slope(distance) / distance])
    assert found[:3, :3] == pytest.approx(block, abs=1e-12)
    assert found[3:, 3:] == pytest.approx(block, abs=1e-12)
    assert found[:3, 3:] == pytest.approx(-block, abs=1e-12)
    assert found == pytest.approx(found.T, abs=0.0)
    assert found @ np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) == pytest.approx(np.zeros(6), abs=1e-12)


def test_the_morse_hessian_wraps_a_periodic_axis_and_leaves_the_others_alone():
    args = ([10.0] * 3, [True, False, False], 0.5, 1.5, 2.5, 5.0)
    wrapped = optking_ts.morse_smooth_hessian([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], *args)
    near = optking_ts.morse_smooth_hessian([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [10.0] * 3, [False] * 3, 0.5, 1.5, 2.5, 5.0)
    assert wrapped == pytest.approx(near, abs=1e-12)
    assert np.abs(wrapped).max() > 0.1
    across = optking_ts.morse_smooth_hessian([[0.0, 0.5, 0.0], [0.0, 9.5, 0.0]], *args)
    assert np.array_equal(across, np.zeros((6, 6)))


def test_the_free_block_keeps_the_rows_and_columns_of_the_free_atoms():
    matrix = np.arange(81, dtype=float).reshape(9, 9)
    assert optking_ts._free_block(matrix, [False, False, False]) is matrix
    keep = [0, 1, 2, 6, 7, 8]
    block = optking_ts._free_block(matrix, [False, True, False])
    assert np.array_equal(block, matrix[np.ix_(keep, keep)])


@pytest.mark.parametrize(("half_gap", "index"), [(1.5, 1), (1.2, 0), (1.0, 2)])
def test_the_index_of_a_free_atom_between_two_pinned_ones_follows_the_pair_curvatures(tmp_path, half_gap, index):
    job = _job(tmp_path, half_gap)
    along = 2.0 * _curvature(half_gap)
    across = 2.0 * _slope(half_gap) / half_gap
    found, lowest = optking_ts.index_at(job, job["positions"])
    assert found == index
    assert lowest == pytest.approx(min(along, across), rel=1e-9)


@pytest.mark.parametrize(("half_gap", "index"), [(3.5, 1), (2.7, 0), (2.5, 2)])
def test_a_free_atom_between_two_pinned_ones_on_the_smooth_morse_curve(tmp_path, half_gap, index):
    job = _job(tmp_path, half_gap, potential=MORSE)
    along = 2.0 * _morse_curvature(half_gap)
    across = 2.0 * _morse_slope(half_gap) / half_gap
    found, lowest = optking_ts.index_at(job, job["positions"])
    assert found == index
    assert lowest == pytest.approx(min(along, across), rel=1e-9)


def test_a_free_cluster_drops_its_rigid_modes_from_the_index(tmp_path):
    triangle = [[0.0, 0.0, 0.0], [RMIN, 0.0, 0.0], [RMIN / 2.0, RMIN * math.sqrt(3.0) / 2.0, 0.0]]
    tetrahedron = [*triangle, [RMIN / 2.0, RMIN * math.sqrt(3.0) / 6.0, RMIN * math.sqrt(2.0 / 3.0)]]
    for positions, lowest in ((triangle, 1.5 * STIFFNESS), (tetrahedron, STIFFNESS)):
        job = _job(tmp_path, positions=positions, fixed=[False] * len(positions))
        index, found = optking_ts.index_at(job, positions)
        assert index == 0
        assert found == pytest.approx(lowest, rel=1e-9)


def test_an_unknown_potential_kind_is_a_value_error_naming_it(tmp_path):
    job = _job(tmp_path, potential={"kind": "eam"})
    with pytest.raises(ValueError, match="eam"):
        optking_ts.index_at(job, job["positions"])
    with pytest.raises(ValueError, match="eam"):
        optking_ts._evaluate_factory(job)


def test_the_lennard_jones_evaluator_sees_the_pinned_atoms_but_moves_only_the_free_one(tmp_path):
    half_gap, drift = 1.5, 0.05
    evaluate, positions, free = optking_ts._evaluate_factory(_job(tmp_path, half_gap, free_at=(0.0, drift, 0.0)))
    assert free.tolist() == [1]
    assert positions.shape == (3, 3)
    energy, gradient = evaluate(np.array([0.0, drift, 0.0]))
    radius = math.hypot(half_gap, drift)
    assert energy == pytest.approx(2.0 * _u(radius) + _u(2.0 * half_gap), rel=1e-12)
    assert gradient == pytest.approx([0.0, 2.0 * _slope(radius) * drift / radius, 0.0], abs=1e-12)


@pytest.mark.parametrize(("potential", "half_gap"), [(LJ, 1.5), (MORSE, 3.5)])
def test_the_evaluator_gradient_is_the_derivative_of_its_energy(tmp_path, potential, half_gap):
    free_at = np.array([0.07, -0.04, 0.03])
    evaluate, _positions, _free = optking_ts._evaluate_factory(_job(tmp_path, half_gap, free_at=free_at, potential=potential))
    _energy, gradient = evaluate(free_at)
    for axis in range(3):
        up, down = free_at.copy(), free_at.copy()
        up[axis] += 1.0e-6
        down[axis] -= 1.0e-6
        slope = (evaluate(up)[0] - evaluate(down)[0]) / 2.0e-6
        assert gradient[axis] == pytest.approx(slope, rel=1e-6, abs=1e-8)


def test_a_start_with_every_atom_pinned_cannot_be_searched(tmp_path):
    with pytest.raises(ValueError, match="initial_0 has no free atoms"):
        optking_ts._evaluate_factory(_job(tmp_path, fixed=[True, True, True]))


def test_a_record_starts_as_a_crash_and_takes_what_it_is_given(tmp_path):
    record = optking_ts._record(_job(tmp_path), force_calls=7, success=True)
    assert record["schema"] == "optbench.start/v2"
    assert (record["start"], record["index"]) == ("initial_0", 0)
    assert (record["failure"], record["reasons"]) == ("crash", ["crash"])
    assert (record["force_calls"], record["success"]) == (7, True)
    assert record["expected_index"] == 1
    assert record["energy"] is None


def _finish(job, *, outcome="converged", calls=1, verify=0, energy=0.0, gradient=(0.0, 0.0, 0.0), error="", steps=0, hessians=0):
    counter = HarnessCounter(cache=True, cap=job["cap"])
    counter.calls = calls
    counter.verify_calls = verify
    last = (np.asarray(job["positions"], dtype=float), energy, np.asarray(gradient, dtype=float))
    return optking_ts._finish(job, counter, outcome, error, last, steps, hessians)


def test_a_search_that_stops_on_a_first_order_saddle_is_a_success(tmp_path):
    record = _finish(_job(tmp_path, 1.5), calls=9, verify=2, steps=3, hessians=1)
    assert record["success"] is True
    assert record["failure"] is None
    assert record["reasons"] == []
    assert record["converged"] is True
    assert record["hessian_index"] == 1
    assert record["lowest_eigenvalue"] == pytest.approx(2.0 * _curvature(1.5), rel=1e-9)
    assert (record["force_calls"], record["verify_calls"], record["steps"], record["hessian_calls"]) == (9, 2, 3, 1)
    assert record["error"] is None


def test_a_stop_on_a_minimum_or_a_second_order_saddle_is_the_wrong_stationary_point(tmp_path):
    for half_gap, index in ((1.2, 0), (1.0, 2)):
        record = _finish(_job(tmp_path, half_gap))
        assert record["converged"] is True
        assert record["hessian_index"] == index
        assert record["reasons"] == ["hessian_index"]
        assert record["failure"] == "wrong_stationary"
        assert record["success"] is False


def test_a_stop_above_the_force_stop_is_a_divergence_and_reads_no_index(tmp_path):
    record = _finish(_job(tmp_path), gradient=(0.0, 1.0e-3, 0.0))
    assert record["converged"] is False
    assert record["hessian_index"] is None
    assert record["reasons"] == ["divergence"]
    assert record["force_norm"] == pytest.approx(1.0e-3)


def test_a_search_that_spends_the_budget_is_a_cap_failure_even_at_the_force_stop(tmp_path):
    spent = _finish(_job(tmp_path), calls=100)
    assert spent["reasons"] == ["cap"]
    assert spent["failure"] == "cap"
    assert spent["converged"] is False
    early = _finish(_job(tmp_path), outcome="cap", calls=4, gradient=(0.0, 1.0, 0.0))
    assert early["reasons"] == ["cap"]
    assert early["force_calls"] == 4


def test_a_non_finite_energy_or_force_is_a_divergence_without_those_numbers(tmp_path):
    no_energy = _finish(_job(tmp_path), energy=float("nan"))
    assert no_energy["reasons"] == ["divergence"]
    assert no_energy["energy"] is None
    no_force = _finish(_job(tmp_path), gradient=(float("inf"), 0.0, 0.0))
    assert no_force["reasons"] == ["divergence"]
    assert no_force["force_norm"] is None


def test_a_crash_keeps_its_message_and_a_silent_one_has_none(tmp_path):
    loud = _finish(_job(tmp_path), outcome="crash", error="OptError: step failed")
    assert (loud["reasons"], loud["failure"], loud["error"]) == (["crash"], "crash", "OptError: step failed")
    silent = _finish(_job(tmp_path), outcome="crash")
    assert silent["failure"] == "crash"
    assert silent["error"] is None


def test_only_a_crash_carries_its_message_into_the_record(tmp_path):
    stalled = _finish(
        _job(tmp_path), outcome="divergence", gradient=(0.0, 1.0, 0.0), error="cartesian step stayed below 1e-8"
    )
    assert stalled["reasons"] == ["divergence"]
    assert stalled["error"] is None
    tainted = _finish(_job(tmp_path), error="late warning")
    assert tainted["converged"] is False
    assert tainted["reasons"] == ["divergence"]
    assert tainted["error"] is None


def test_an_index_that_cannot_be_read_turns_a_converged_stop_into_a_crash(tmp_path):
    record = _finish(_job(tmp_path, potential={"kind": "eam"}))
    assert record["failure"] == "crash"
    assert record["error"] == "ValueError: eam"
    assert record["converged"] is False


def test_a_search_that_never_started_reports_no_calls_and_no_point(tmp_path):
    record = optking_ts._finish(_job(tmp_path), None, "crash", "could not start", None, 0, 0)
    assert (record["force_calls"], record["verify_calls"]) == (0, 0)
    assert (record["energy"], record["force_norm"], record["hessian_index"]) == (None, None, None)
    assert record["error"] == "could not start"


def test_the_free_atoms_take_their_elements_from_the_list_or_repeat_the_first_species():
    job = {"symbol": "Pt", "symbols": ["Cu", "Cu", "H"]}
    assert optking_ts._free_symbols(job, [1, 2]) == ["Cu", "H"]
    assert optking_ts._free_symbols({"symbol": "Pt"}, [0, 3, 4]) == ["Pt", "Pt", "Pt"]
    assert optking_ts._free_symbols({"symbol": "Pt", "symbols": []}, [2]) == ["Pt"]


def test_a_template_potential_is_checked_against_the_problem_potential():
    optking_ts._check_potential({"potential": PROBLEM_LJ}, dict(LJ, pbc=[False] * 3))
    optking_ts._check_potential({"potential": PROBLEM_MORSE}, dict(TEMPLATE_MORSE))


@pytest.mark.parametrize(
    ("problem", "template", "message"),
    [
        (PROBLEM_LJ, dict(LJ, kind="morse"), "template potential kind is not the problem potential"),
        (PROBLEM_LJ, dict(LJ, epsilon=2.0, pbc=[False] * 3), "template epsilon is not the problem epsilon"),
        (PROBLEM_LJ, dict(LJ, sigma=1.1, pbc=[False] * 3), "template sigma is not the problem sigma"),
        (PROBLEM_LJ, dict(LJ, rc=30.0, pbc=[False] * 3), "template rc is not the problem cutoff"),
        (PROBLEM_LJ, dict(LJ, pbc=[True, False, False]), "template pbc is not the problem pbc"),
        (PROBLEM_MORSE, dict(TEMPLATE_MORSE, form="shift"), "template Morse form is not morse/smooth/linear"),
        (dict(PROBLEM_MORSE, pair_style="morse"), dict(TEMPLATE_MORSE), "template Morse form is not morse/smooth/linear"),
        (PROBLEM_MORSE, dict(TEMPLATE_MORSE, D=0.6), "template D is not the problem D"),
        (PROBLEM_MORSE, dict(TEMPLATE_MORSE, alpha=1.4), "template alpha is not the problem alpha"),
        (PROBLEM_MORSE, dict(TEMPLATE_MORSE, r0=2.6), "template r0 is not the problem r0"),
        (PROBLEM_MORSE, dict(TEMPLATE_MORSE, rc=6.0), "template rc is not the problem cutoff"),
        (PROBLEM_MORSE, dict(TEMPLATE_MORSE, pbc=[True, True, True]), "template pbc is not the problem pbc"),
        ({"kind": "eam"}, {"kind": "eam"}, "eam"),
    ],
)
def test_a_template_potential_that_differs_from_the_problem_is_refused(problem, template, message):
    with pytest.raises(ValueError, match=message):
        optking_ts._check_potential({"potential": problem}, template)


def test_the_input_select_names_the_files_a_run_reads():
    assert optking_ts._select(Path("initial_3.con"), "initial") is True
    assert optking_ts._select(Path("reactant.con"), "initial") is False
    assert optking_ts._select(Path("product.con"), "con") is True
    assert optking_ts._select(Path("reactant.con"), "con") is False
    assert optking_ts._select(Path("initial_3.xyz"), "con") is False
    with pytest.raises(ValueError, match="unsupported input select"):
        optking_ts._select(Path("a.con"), "xyz")


def _problem(**fields) -> dict:
    problem = {
        "id": "tsopt/test",
        "potential": dict(PROBLEM_LJ),
        "convergence": {"force_norm": 1.0e-6, "max_force_calls": 100},
        "input": {"archive_path": "tsopt/starts.tar"},
        "sample": {"starts": 2},
    }
    problem.update(fields)
    return problem


def _template(**fields) -> dict:
    template = {
        "id": "optking/test-rfo",
        "problem": "tsopt/test",
        "code": "optking",
        "algorithm": "RS-I-RFO",
        "tuning": "default",
        "code_version": optking_ts.VERSION,
        "commit": optking_ts.PIN,
        "potential": dict(LJ, pbc=[False] * 3),
        "optking": {"step_type": "RS_I_RFO"},
        "stop": {"force_norm": 1.0e-6, "max_force_calls": 100},
        "input": {"select": "initial"},
    }
    template.update(fields)
    return template


def _starts(folder: Path, names, half_gap: float = 1.5) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [(-half_gap, 0.0, 0.0, True), (0.0, 0.0, 0.0, False), (half_gap, 0.0, 0.0, True)]
    for name in names:
        (folder / f"{name}.con").write_text(_con(rows), encoding="utf-8")
    return folder


def _jobs(tmp_path: Path, folder: Path, problem=None, template=None, **kwargs):
    return optking_ts.load_jobs(
        tmp_path,
        problem or _problem(),
        template or _template(),
        geometries=folder,
        work=kwargs.pop("work", tmp_path / "work"),
        output=tmp_path / "out" / "run.dat",
    )


def test_jobs_follow_the_number_in_the_file_name_and_leave_the_reactant_out(tmp_path):
    folder = _starts(tmp_path / "cons", ["initial_10", "initial_2", "initial_1", "reactant", "product"])
    jobs = _jobs(tmp_path, folder)
    assert [job["start"] for job in jobs] == ["initial_1", "initial_2", "initial_10"]
    assert [job["index"] for job in jobs] == [0, 1, 2]
    first = jobs[0]
    assert first["fixed"] == [True, False, True]
    assert first["symbols"] == ["Ar", "Ar", "Ar"]
    assert first["box"] == [10.0, 11.0, 12.0]
    assert first["pbc"] == [False, False, False]
    assert (first["step_type"], first["full_hess_every"], first["fd_step"]) == ("RS_I_RFO", 0, optking_ts.FD_STEP)
    assert (first["gtol"], first["cap"]) == (1.0e-6, 100)
    assert Path(first["scratch"]) == tmp_path / "work" / "optking-test-rfo" / "initial_1"


def test_a_template_that_selects_every_con_reads_the_product_too(tmp_path):
    folder = _starts(tmp_path / "cons", ["initial_0", "reactant", "product"])
    jobs = _jobs(tmp_path, folder, template=_template(input={"select": "con"}))
    assert [job["start"] for job in jobs] == ["initial_0", "product"]
    default = _jobs(tmp_path, folder, template=_template(input=None))
    assert [job["start"] for job in default] == ["initial_0", "product"]


def test_the_optking_section_sets_the_hessian_schedule_and_the_difference_step(tmp_path):
    folder = _starts(tmp_path / "cons", ["initial_0"])
    section = {"step_type": "P_RFO", "full_hess_every": 3, "fd_step": 2.0e-5}
    job = _jobs(tmp_path, folder, template=_template(optking=section))[0]
    assert (job["step_type"], job["full_hess_every"], job["fd_step"]) == ("P_RFO", 3, 2.0e-5)


def test_the_scratch_folders_default_to_a_work_folder_beside_the_output(tmp_path):
    folder = _starts(tmp_path / "cons", ["initial_0"])
    jobs = optking_ts.load_jobs(
        tmp_path, _problem(), _template(), geometries=folder, work=None, output=tmp_path / "out" / "run.dat"
    )
    assert Path(jobs[0]["scratch"]) == tmp_path / "out" / "work" / "optking-test-rfo" / "initial_0"


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ({"optking": {"step_type": "RS_PRFO"}}, "step_type must be RS_I_RFO or P_RFO"),
        ({"optking": {}}, "step_type must be RS_I_RFO or P_RFO"),
        ({"stop": {"force_norm": 1.0e-4}}, "template force_norm is not the problem force norm"),
        ({"stop": {"max_force_calls": 99}}, "template max_force_calls is below the problem force-call cap"),
        ({"potential": dict(LJ, sigma=1.5, pbc=[False] * 3)}, "template sigma is not the problem sigma"),
    ],
)
def test_a_template_that_disagrees_with_its_problem_builds_no_jobs(tmp_path, template, message):
    folder = _starts(tmp_path / "cons", ["initial_0"])
    with pytest.raises(ValueError, match=message):
        _jobs(tmp_path, folder, template=_template(**template))


def test_a_template_with_no_stop_block_takes_the_problem_stop(tmp_path):
    folder = _starts(tmp_path / "cons", ["initial_0"])
    job = _jobs(tmp_path, folder, template=_template(stop={}))[0]
    assert (job["gtol"], job["cap"]) == (1.0e-6, 100)


def _archive_root(tmp_path: Path, names=("initial_0", "initial_1", "reactant")) -> Path:
    root = tmp_path / "root"
    (root / "inputs").mkdir(parents=True)
    source = _starts(tmp_path / "pack", names)
    with tarfile.open(root / "inputs" / "starts.tar", "w") as packed:
        for name in names:
            packed.add(source / f"{name}.con", arcname=f"starts/{name}.con")
    manifest = {"path": "tsopt/starts.tar", "stored": "inputs/starts.tar"}
    (root / "inputs" / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def _from_archive(root: Path, work: Path, problem=None):
    return optking_ts.load_jobs(
        root, problem or _problem(), _template(), geometries=None, work=work, output=work.parent / "run.dat"
    )


def test_a_problem_archive_is_extracted_once_and_its_starts_are_counted(tmp_path):
    root = _archive_root(tmp_path)
    work = tmp_path / "work"
    jobs = _from_archive(root, work)
    assert [job["start"] for job in jobs] == ["initial_0", "initial_1"]
    assert (work / "tsopt-test" / ".extracted").read_text(encoding="utf-8") == "starts.tar"
    again = _from_archive(root, work)
    assert [job["start"] for job in again] == ["initial_0", "initial_1"]


def test_a_folder_extracted_from_another_archive_is_refused(tmp_path):
    root = _archive_root(tmp_path)
    work = tmp_path / "work"
    _from_archive(root, work)
    (work / "tsopt-test" / ".extracted").write_text("older.tar", encoding="utf-8")
    with pytest.raises(RuntimeError, match="was extracted from another archive"):
        _from_archive(root, work)


def test_an_archive_with_the_wrong_number_of_starts_is_refused(tmp_path):
    root = _archive_root(tmp_path)
    with pytest.raises(ValueError, match="expected 3 starts, found 2"):
        _from_archive(root, tmp_path / "work", _problem(sample={"starts": 3}))


def test_fields_print_whole_floats_as_integers_and_others_to_twelve_digits():
    assert optking_ts._format_field(3.0) == "3"
    assert optking_ts._format_field(0.1 + 0.2) == "0.3"
    assert optking_ts._format_field(1.0 / 3.0) == "0.333333333333"
    assert optking_ts._format_field("abc") == "abc"
    assert optking_ts._format_field(7) == "7"


def test_the_dat_text_skips_empty_fields_and_marks_a_partial_run():
    summary = {"force_calls": 15, "force_calls_min": 10, "force_calls_max": 20, "nfailed": 1, "median": 15.0}
    metrics = {"par2_mean": 43.5, "n": 3, "successes": 2, "nfailed": 1, "n_cap": 1}
    template = _template(note="first line\nsecond line", tuning="")
    text = optking_ts._dat(summary, template, metrics, "abc", "1.2", partial=True)
    lines = text.splitlines()
    assert text.endswith("\n")
    assert "# first line second line" in lines
    assert "# partial true" in lines
    for line in ("force_calls 15", "median 15", "code optking", "code_version 1.2", "commit abc", "partial true"):
        assert line in lines
    for line in ("par2_mean 43.5", "v2_successes 2", "v2_nfailed 1", "n 3", "n_cap 1", "template optking/test-rfo"):
        assert line in lines
    assert not any(line.startswith(("tuning", "rate_low", "n_crash", "problem optking")) for line in lines)
    plain = optking_ts._dat(summary, _template(), metrics, "abc", "1.2", partial=False)
    assert "partial true" not in plain
    assert "# partial true" not in plain


def _row(index: int, calls: int, *, success: bool = True, failure=None, error=None) -> dict:
    return {
        "index": index,
        "start": f"initial_{index}",
        "force_calls": calls,
        "converged": success,
        "success": success,
        "failure": failure,
        "error": error,
    }


def _publish(tmp_path: Path, records, *, partial: bool = False, name: str = "run") -> dict:
    return optking_ts._publish(
        tmp_path / "out" / f"{name}.dat",
        records,
        _template(optking={"step_type": "RS_I_RFO", "full_hess_every": 2, "fd_step": 3.0e-5}),
        _problem(),
        version="1.2",
        commit="abc",
        cap=50,
        partial=partial,
    )


def test_a_published_run_orders_its_starts_and_scores_the_failure_at_twice_the_cap(tmp_path):
    records = [_row(2, 50, success=False, failure="cap", error="stopped"), _row(0, 10), _row(1, 20)]
    summary = _publish(tmp_path, records)
    assert summary["starts"] == 3
    assert summary["template"] == "optking/test-rfo"
    assert summary["problem"] == "tsopt/test"
    assert summary["force_calls"] == 15
    assert (summary["force_calls_min"], summary["force_calls_max"], summary["nfailed"]) == (10, 20, 1)
    assert summary["median"] == 15
    assert summary["par2_mean"] == pytest.approx((10 + 20 + 2 * 50) / 3)
    assert summary["v2_nfailed"] == 1
    assert summary["partial"] is False
    rows = [json.loads(line) for line in (tmp_path / "out" / "run.starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["start"] for row in rows] == ["initial_0", "initial_1", "initial_2"]
    assert all("index" not in row for row in rows)
    assert [row.get("error") for row in rows] == [None, None, "stopped"]
    assert "error" not in rows[0]


def test_a_published_run_states_its_counts_in_the_dat_and_its_settings_in_the_v2_file(tmp_path):
    _publish(tmp_path, [_row(0, 10), _row(1, 20), _row(2, 50, success=False, failure="cap")])
    lines = (tmp_path / "out" / "run.dat").read_text(encoding="utf-8").splitlines()
    for line in ("force_calls 15", "force_calls_min 10", "force_calls_max 20", "nfailed 1", "v2_successes 2", "n 3", "n_cap 1"):
        assert line in lines
    document = json.loads((tmp_path / "out" / "run.v2.json").read_text(encoding="utf-8"))
    assert document["status"] == "archive"
    assert document["problem"] == "tsopt/test"
    run = document["runs"][0]
    assert run["entry"] == "run"
    assert run["cap"] == 50
    assert run["parameters"] == {
        "opt_type": "TS",
        "step_type": "RS_I_RFO",
        "opt_coordinates": "CARTESIAN",
        "full_hess_every": 2,
        "fd_step": 3.0e-5,
        "gtol": 1.0e-6,
        "max_force_calls": 50,
    }
    assert run["code_version"] == "1.2"
    assert run["commit"] == "abc"
    assert run["legacy"]["force_calls"] == 15
    assert [row["start"] for row in run["starts"]] == ["initial_0", "initial_1", "initial_2"]


def test_a_partial_publication_says_so_in_every_file(tmp_path):
    summary = _publish(tmp_path, [_row(0, 10)], partial=True)
    assert summary["partial"] is True
    assert "partial true" in (tmp_path / "out" / "run.dat").read_text(encoding="utf-8").splitlines()
    document = json.loads((tmp_path / "out" / "run.v2.json").read_text(encoding="utf-8"))
    assert document["status"] == "partial"


def test_a_run_with_no_starts_publishes_empty_files_and_no_score(tmp_path):
    summary = _publish(tmp_path, [])
    assert summary["starts"] == 0
    assert summary["force_calls"] is None
    assert summary["median"] is None
    assert summary["par2_mean"] is None
    assert summary["v2_nfailed"] is None
    assert (tmp_path / "out" / "run.starts.jsonl").read_text(encoding="utf-8") == ""
    document = json.loads((tmp_path / "out" / "run.v2.json").read_text(encoding="utf-8"))
    assert document["runs"][0]["metrics"] == {}
    assert not any(line.startswith("force_calls") for line in (tmp_path / "out" / "run.dat").read_text(encoding="utf-8").splitlines())


def test_the_rows_a_run_kept_are_read_back_by_start_name(tmp_path):
    path = tmp_path / "run.starts.jsonl"
    assert optking_ts._load_done(path) == {}
    path.write_text(json.dumps({"start": "initial_3", "force_calls": 4}) + "\n\n" + json.dumps({"start": "initial_1"}) + "\n", encoding="utf-8")
    done = optking_ts._load_done(path)
    assert sorted(done) == ["initial_1", "initial_3"]
    assert done["initial_3"]["force_calls"] == 4


def _fake_search(seen: list, calls: dict | None = None, fail_at: int | None = None):
    def search(job):
        seen.append(job["start"])
        if fail_at is not None and len(seen) == fail_at:
            raise KeyboardInterrupt
        return optking_ts._record(
            job,
            force_calls=(calls or {}).get(job["start"], 10 + job["index"]),
            success=True,
            failure=None,
            reasons=[],
            converged=True,
            hessian_index=1,
            energy=-1.0,
            force_norm=0.0,
        )

    return search


@pytest.fixture
def pinned_version(monkeypatch):
    monkeypatch.setattr(optking_ts, "installed_version", lambda: optking_ts.VERSION)


def _run(tmp_path: Path, folder: Path, **kwargs) -> dict:
    options = {"geometries": folder, "output": tmp_path / "out" / "run.dat", "workers": 1, "work": tmp_path / "work"}
    options.update(kwargs)
    return optking_ts.run_template(tmp_path, _problem(), _template(), **options)


@pytest.mark.parametrize(
    ("version", "template", "problem_id", "error", "message"),
    [
        (optking_ts.VERSION, {"code_version": "9.9"}, "tsopt/test", RuntimeError, "does not match template 9.9"),
        ("0.5.0", {"code_version": ""}, "tsopt/test", RuntimeError, "is not pin"),
        (optking_ts.VERSION, {"commit": "0" * 40}, "tsopt/test", RuntimeError, "commit is not"),
        (optking_ts.VERSION, {"commit": ""}, "tsopt/test", RuntimeError, "commit is not"),
        (optking_ts.VERSION, {}, "tsopt/other", ValueError, "is for tsopt/test, not tsopt/other"),
    ],
)
def test_a_run_refuses_a_version_pin_or_problem_that_is_not_the_templates(
    tmp_path, monkeypatch, version, template, problem_id, error, message
):
    monkeypatch.setattr(optking_ts, "installed_version", lambda: version)
    folder = _starts(tmp_path / "cons", ["initial_0"])
    with pytest.raises(error, match=message):
        optking_ts.run_template(
            tmp_path,
            _problem(id=problem_id),
            _template(**template),
            geometries=folder,
            output=tmp_path / "out" / "run.dat",
            workers=1,
            work=tmp_path / "work",
        )


def test_a_limited_run_searches_the_first_starts_and_publishes_a_partial_result(tmp_path, monkeypatch, pinned_version):
    seen: list = []
    monkeypatch.setattr(optking_ts, "search_start", _fake_search(seen))
    folder = _starts(tmp_path / "cons", ["initial_0", "initial_1", "initial_2"])
    summary = _run(tmp_path, folder, limit=2)
    assert seen == ["initial_0", "initial_1"]
    assert summary["starts"] == 2
    assert summary["partial"] is True
    assert "partial true" in (tmp_path / "out" / "run.dat").read_text(encoding="utf-8").splitlines()


def test_a_resumed_run_searches_only_the_starts_that_are_missing(tmp_path, monkeypatch, pinned_version):
    seen: list = []
    monkeypatch.setattr(optking_ts, "search_start", _fake_search(seen, {"initial_0": 12, "initial_1": 30, "initial_2": 8}))
    folder = _starts(tmp_path / "cons", ["initial_0", "initial_1", "initial_2"])
    _run(tmp_path, folder, limit=2)
    seen.clear()
    summary = _run(tmp_path, folder, resume=True)
    assert seen == ["initial_2"]
    assert summary["starts"] == 3
    assert summary["partial"] is False
    rows = [json.loads(line) for line in (tmp_path / "out" / "run.starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["start"], row["force_calls"]) for row in rows] == [("initial_0", 12), ("initial_1", 30), ("initial_2", 8)]


def test_a_fresh_run_drops_the_rows_of_an_earlier_one(tmp_path, monkeypatch, pinned_version):
    seen: list = []
    monkeypatch.setattr(optking_ts, "search_start", _fake_search(seen, {"initial_0": 12}))
    folder = _starts(tmp_path / "cons", ["initial_0"])
    stale = tmp_path / "out" / "run.starts.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"start": "initial_0", "force_calls": 999}) + "\n", encoding="utf-8")
    _run(tmp_path, folder)
    assert seen == ["initial_0"]
    assert [json.loads(line)["force_calls"] for line in stale.read_text(encoding="utf-8").splitlines()] == [12]


def test_a_resumed_run_keeps_the_row_it_finds_instead_of_searching_again(tmp_path, monkeypatch, pinned_version):
    seen: list = []
    monkeypatch.setattr(optking_ts, "search_start", _fake_search(seen))
    folder = _starts(tmp_path / "cons", ["initial_0"])
    kept = tmp_path / "out" / "run.starts.jsonl"
    kept.parent.mkdir(parents=True)
    kept.write_text(json.dumps(_row(0, 777)) + "\n", encoding="utf-8")
    summary = _run(tmp_path, folder, resume=True)
    assert seen == []
    assert summary["starts"] == 1
    assert [json.loads(line)["force_calls"] for line in kept.read_text(encoding="utf-8").splitlines()] == [777]


def test_a_killed_run_leaves_every_ten_starts_it_had_finished(tmp_path, monkeypatch, pinned_version):
    seen: list = []
    monkeypatch.setattr(optking_ts, "search_start", _fake_search(seen, fail_at=12))
    folder = _starts(tmp_path / "cons", [f"initial_{index}" for index in range(15)])
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, folder)
    assert len(seen) == 12
    kept = (tmp_path / "out" / "run.starts.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["start"] for line in kept] == [f"initial_{index}" for index in range(10)]
    assert "partial true" in (tmp_path / "out" / "run.dat").read_text(encoding="utf-8").splitlines()
    document = json.loads((tmp_path / "out" / "run.v2.json").read_text(encoding="utf-8"))
    assert document["status"] == "partial"


def test_a_run_over_a_folder_with_no_starts_publishes_an_empty_result(tmp_path, monkeypatch, pinned_version):
    seen: list = []
    monkeypatch.setattr(optking_ts, "search_start", _fake_search(seen))
    folder = _starts(tmp_path / "cons", ["reactant"])
    summary = _run(tmp_path, folder)
    assert seen == []
    assert summary["starts"] == 0
    assert summary["force_calls"] is None
