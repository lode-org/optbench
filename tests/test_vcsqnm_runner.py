"""The vc-SQNM runner on clusters and a dimer with closed-form answers.

Four atoms at the corners of a regular tetrahedron at the pair minimum
``2**(1/6)`` of ``u(r) = 4 (r**-12 - r**-6)`` have energy -6 and vibrational
eigenvalues ``k, k, 2k, 2k, 2k, 4k`` with ``k = u''(r)``. Two Morse atoms a
distance ``r0`` apart in a large cell have no force and no lattice derivative,
and their one stretching curvature is ``4 alpha**2 D0``.

The SQNM package is not in the lockfile. Without it, the loop around the
package is driven by a stand-in with the same call signatures that takes
steepest-descent steps. The installed package is tried only when it can be
imported and is the pinned distribution.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import signal
import statistics
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from optbench.counter import ForceCallCap
from optbench.potentials import MORSE_ALPHA, MORSE_CUTOFF, MORSE_D0, MORSE_R0, lennard_jones, morse_bulk
from optbench.score import BOOTSTRAP_SEED
from optbench.runners import vcsqnm_min
from optbench.v2 import HOST_LABEL_ENV
from optbench.vcsqnm_catalog import CODE_VERSION, COMMIT, DIST_VERSION
from optbench.vcsqnm_pes import morse_cell, sqnm_lattice_derivative

R_MIN = 2.0 ** (1.0 / 6.0)
CAP = 400
CELL = 20.0
PARAMETERS = {
    "initial_step_size": -1.0e-7,
    "nhist_max": 10,
    "alpha_min": 1.0e-3,
    "eps_subsp": 1.0e-5,
    "lattice_weight": 2.0,
}


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _tetrahedron(edge: float = R_MIN, shake: float = 0.0) -> list[float]:
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) * edge / math.sqrt(8.0)
    return [float(value + shake * math.sin(1.7 * index + 0.3)) for index, value in enumerate(corners.reshape(-1))]


def _spectrum(edge: float) -> list[float]:
    stiffness, tension = _curvature(edge), _slope(edge) / edge
    return sorted([stiffness + 3.0 * tension] * 2 + [2.0 * stiffness + 2.0 * tension] * 3 + [4.0 * stiffness])


def _square(side: float) -> list[float]:
    half = side / 2.0
    return [-half, -half, 0.0, half, -half, 0.0, half, half, 0.0, -half, half, 0.0]


def _square_side() -> float:
    low, high = 1.0, 1.2
    for _ in range(200):
        middle = (low + high) / 2.0
        slope = 4.0 * _slope(middle) + 2.0 * math.sqrt(2.0) * _slope(math.sqrt(2.0) * middle)
        if slope < 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _lattice(spacing: float, repeats: int = 3) -> list[float]:
    """A simple cubic block that fills a periodic cell of ``repeats * spacing``."""
    return [
        spacing * value
        for i in range(repeats)
        for j in range(repeats)
        for k in range(repeats)
        for value in (i, j, k)
    ]


def _dimer(stretch: float = 0.0) -> list[float]:
    return [5.0, 5.0, 5.0, 5.0 + MORSE_R0 + stretch, 5.0, 5.0]


def _dimer_energy() -> float:
    cut = math.exp(-MORSE_ALPHA * (MORSE_CUTOFF - MORSE_R0))
    return -MORSE_D0 - MORSE_D0 * (cut * cut - 2.0 * cut)


@pytest.fixture(autouse=True)
def _label(monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")


@pytest.fixture(autouse=True)
def _watchdog():
    """A stepper that never meets a stop must fail the test, not hold the suite."""

    def expired(signum, frame):
        raise TimeoutError("the test ran past its time limit")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(vcsqnm_min, "_PIN_OK", True)


def _package(tmp_path: Path, files: dict[str, bytes]) -> types.ModuleType:
    folder = tmp_path / "sqnm"
    folder.mkdir(exist_ok=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    package = types.ModuleType("sqnm")
    package.__path__ = [str(folder)]
    return package


def test_a_pin_that_was_already_checked_is_not_checked_again(monkeypatch):
    monkeypatch.setattr(vcsqnm_min, "_PIN_OK", True)
    monkeypatch.setitem(sys.modules, "sqnm", None)
    vcsqnm_min.assert_pin()


def test_a_missing_package_stops_the_pin_check(monkeypatch):
    monkeypatch.setattr(vcsqnm_min, "_PIN_OK", False)
    monkeypatch.setitem(sys.modules, "sqnm", None)
    with pytest.raises(ImportError, match="sqnm"):
        vcsqnm_min.assert_pin()


def test_a_package_of_another_distribution_fails_the_pin_check(monkeypatch, tmp_path):
    monkeypatch.setattr(vcsqnm_min, "_PIN_OK", False)
    monkeypatch.setitem(sys.modules, "sqnm", _package(tmp_path, {}))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9")
    with pytest.raises(RuntimeError, match=f"sqnm 9.9 is not distribution {DIST_VERSION}"):
        vcsqnm_min.assert_pin()
    assert vcsqnm_min._PIN_OK is False


def test_a_missing_or_changed_source_file_fails_the_pin_check(monkeypatch, tmp_path):
    monkeypatch.setattr(vcsqnm_min, "_PIN_OK", False)
    monkeypatch.setitem(sys.modules, "sqnm", _package(tmp_path, {}))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: DIST_VERSION)
    monkeypatch.setattr(vcsqnm_min, "SOURCE_SHA256", {"one.py": hashlib.sha256(b"one").hexdigest()})
    with pytest.raises(RuntimeError, match="installed sqnm is missing one.py"):
        vcsqnm_min.assert_pin()
    (tmp_path / "sqnm" / "one.py").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="one.py sha256 .* is not the v1.3 pin"):
        vcsqnm_min.assert_pin()
    assert vcsqnm_min._PIN_OK is False


def test_sources_that_match_the_pin_pass_and_are_remembered(monkeypatch, tmp_path):
    monkeypatch.setattr(vcsqnm_min, "_PIN_OK", False)
    files = {"one.py": b"one", "two.py": b"two"}
    monkeypatch.setitem(sys.modules, "sqnm", _package(tmp_path, files))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: DIST_VERSION)
    monkeypatch.setattr(vcsqnm_min, "SOURCE_SHA256", {name: hashlib.sha256(body).hexdigest() for name, body in files.items()})
    vcsqnm_min.assert_pin()
    assert vcsqnm_min._PIN_OK is True


def test_a_problem_names_its_potential_or_is_refused():
    assert vcsqnm_min._kind("minimization/lj38") == "lj"
    assert vcsqnm_min._kind("minimization/lj38-noise") == "lj"
    assert vcsqnm_min._kind("minimization/morse-bulk") == "morse"
    with pytest.raises(ValueError, match="minimization/other has no vc-SQNM potential"):
        vcsqnm_min._kind("minimization/other")


def test_the_sqnm_section_is_read_as_numbers_and_must_be_complete():
    section = {**PARAMETERS, "method": "sqnm", "nhist_max": "10", "alpha_min": "0.001"}
    read = vcsqnm_min._parameters(section)
    assert read == {
        "method": "sqnm",
        "initial_step_size": -1.0e-7,
        "nhist_max": 10,
        "alpha_min": 0.001,
        "eps_subsp": 1.0e-5,
        "lattice_weight": 2.0,
    }
    assert isinstance(read["nhist_max"], int)
    assert all(isinstance(read[key], float) for key in ("initial_step_size", "alpha_min", "eps_subsp", "lattice_weight"))
    with pytest.raises(ValueError, match="sqnm section needs alpha_min, lattice_weight"):
        vcsqnm_min._parameters({"method": "sqnm", "initial_step_size": 1.0, "nhist_max": 5, "eps_subsp": 1.0})
    with pytest.raises(ValueError, match="unknown sqnm method lbfgs"):
        vcsqnm_min._parameters({**PARAMETERS, "method": "lbfgs"})


def test_coordinates_become_atom_columns_and_back_and_must_come_in_threes():
    flat = np.arange(12, dtype=float)
    columns = vcsqnm_min._columns(flat)
    assert columns.shape == (3, 4)
    assert columns[:, 1].tolist() == [3.0, 4.0, 5.0]
    assert columns.flags["C_CONTIGUOUS"]
    assert vcsqnm_min._flatten(columns).tolist() == flat.tolist()
    with pytest.raises(ValueError, match="multiple of 3"):
        vcsqnm_min._columns(np.arange(7, dtype=float))


def test_a_cell_is_usable_only_when_it_is_finite_and_not_flat():
    assert vcsqnm_min._cell_ok(np.diag([2.0, 3.0, 4.0])) is True
    assert vcsqnm_min._cell_ok(np.diag([2.0, 3.0, 0.0])) is False
    assert vcsqnm_min._cell_ok(np.diag([1.0e-7, 1.0e-7, 1.0e-7])) is False
    assert vcsqnm_min._cell_ok(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])) is False
    assert vcsqnm_min._cell_ok(np.diag([2.0, 3.0, -4.0])) is True
    assert vcsqnm_min._cell_ok(np.diag([2.0, math.nan, 4.0])) is False
    assert vcsqnm_min._cell_ok(np.diag([2.0, math.inf, 4.0])) is False


def test_the_surface_counts_one_call_for_a_repeated_point_and_refuses_one_past_the_cap():
    surface = vcsqnm_min._Surface("lj", None, 2, False)
    point = _tetrahedron(shake=0.02)
    energy, gradient = surface.evaluate(point)
    clean_energy, clean = lennard_jones(point)
    assert energy == clean_energy
    assert gradient == pytest.approx(clean)
    assert surface.evaluate(point)[0] == energy
    assert surface.counter.calls == 1
    surface.evaluate(_tetrahedron(shake=0.03))
    with pytest.raises(ForceCallCap):
        surface.evaluate(_tetrahedron(shake=0.04))
    assert surface.counter.calls == 2


def test_the_surface_reads_the_morse_dimer_at_its_minimum_with_no_force_and_no_lattice_derivative():
    box = [CELL, CELL, CELL]
    surface = vcsqnm_min._Surface("morse", box, 10, False)
    energy, gradient = surface.evaluate(_dimer())
    assert energy == pytest.approx(_dimer_energy(), abs=1e-12)
    assert np.linalg.norm(gradient) < 1.0e-12
    assert surface.last_deriv is None
    variable = vcsqnm_min._Surface("morse", box, 10, True)
    energy, gradient = variable.evaluate(_dimer(), np.diag(box))
    assert energy == pytest.approx(_dimer_energy(), abs=1e-12)
    assert np.max(np.abs(variable.last_deriv)) < 1.0e-12
    assert variable.last_deriv.shape == (3, 3)


def test_a_variable_cell_surface_needs_a_cell_and_keys_a_point_by_its_cell():
    box = [CELL, CELL, CELL]
    surface = vcsqnm_min._Surface("morse", box, 10, True)
    with pytest.raises(ValueError, match="needs a cell"):
        surface.evaluate(_dimer())
    surface.evaluate(_dimer(0.05), np.diag(box))
    stretched = surface.last_deriv.copy()
    assert np.max(np.abs(stretched)) > 1.0e-3
    surface.evaluate(_dimer(0.05), np.diag(box))
    assert surface.counter.calls == 1
    assert surface.last_deriv == pytest.approx(stretched)
    surface.evaluate(_dimer(0.05), np.diag([CELL + 1.0, CELL, CELL]))
    assert surface.counter.calls == 2


def test_the_seen_evaluation_keeps_the_last_norm_and_lattice_derivative():
    box = [CELL, CELL, CELL]
    seen = vcsqnm_min._Seen(vcsqnm_min._Surface("morse", box, 10, True))
    energy, gradient = seen(_dimer(0.05), np.diag(box))
    assert seen.energy == energy
    assert seen.norm == pytest.approx(float(np.linalg.norm(gradient)))
    assert seen.grad.shape == (6,)
    assert seen.lattice == pytest.approx(float(np.max(np.abs(seen.last_deriv))))
    flat = vcsqnm_min._Seen(vcsqnm_min._Surface("lj", None, 10, False))
    flat(_tetrahedron())
    assert flat.lattice is None


@pytest.mark.parametrize("edge", [1.2, R_MIN, 1.3, 1.6])
def test_the_lj_index_follows_the_closed_form_spectrum(edge):
    index, values = vcsqnm_min._index_at("lj", _tetrahedron(edge), None, None, 1.0e-6, False)
    expected = _spectrum(edge)
    assert values == pytest.approx(expected, rel=1e-9)
    assert index == sum(1 for value in expected if value < -1.0e-6)


def test_a_loose_force_stop_forgives_curvature_up_to_its_size():
    point = _tetrahedron(1.3)
    assert vcsqnm_min._index_at("lj", point, None, None, 1.0e-6, False)[0] == 4
    index, values = vcsqnm_min._index_at("lj", point, None, None, 100.0, False)
    assert index == 0
    assert values == pytest.approx(_spectrum(1.3), rel=1e-9)


def test_the_morse_dimer_has_one_stretching_curvature_and_five_free_motions():
    stretch = 4.0 * MORSE_ALPHA**2 * MORSE_D0
    box = [CELL, CELL, CELL]
    for variable, cell in ((False, None), (True, np.diag(box))):
        index, values = vcsqnm_min._index_at("morse", _dimer(), box, cell, 1.0e-6, variable)
        assert index == 0
        assert values == pytest.approx([0.0] * 5 + [stretch], abs=1e-8)


def test_a_dat_file_names_the_potential_and_the_moving_cell_when_they_apply():
    template = {"id": "vcsqnm/test", "code": "vc-SQNM", "algorithm": "SQNM", "problem": "minimization/lj38", "note": "a\nb"}
    text = vcsqnm_min._dat({"force_calls": 12.5, "nfailed": 0, "energy_min": -6.0}, template, {"n": 3}, "abc", "1.3")
    lines = text.splitlines()
    assert any(line.startswith("# LJ38 index removes the six rigid modes") for line in lines)
    assert not any("The cell is a coordinate" in line for line in lines)
    assert "# a b" in lines
    assert [line for line in lines if not line.startswith("#")] == [
        "force_calls 12.5",
        "nfailed 0",
        "code vc-SQNM",
        "code_version 1.3",
        "commit abc",
        "algorithm SQNM",
        "template vcsqnm/test",
        "counter harness",
        "n 3",
        "energy_min -6",
    ]
    moving = vcsqnm_min._dat({}, {**template, "problem": "minimization/morse-bulk", "sqnm": {"method": "ase-vc"}}, {}, "a", "b")
    assert any("The cell is a coordinate" in line for line in moving.splitlines())
    assert not any("LJ38 index" in line for line in moving.splitlines())


def _driver(point, cell=None, stopped=True, error="", diverged=False, probe=None):
    def drive(coords, method, parameters, evaluate, stop, calls, cap, matrix=None):
        if probe is not None:
            probe.update(method=method, parameters=parameters, cap=cap, matrix=matrix, before=stop())
        evaluate(coords, matrix)
        if probe is not None:
            probe["after"] = stop()
            probe["calls"] = calls()
        return {"point": list(point), "cell": cell, "stopped": stopped, "error": error, "diverged": diverged}

    return drive


def _job(**overrides) -> dict:
    job = {
        "index": 2,
        "start": "initial_2",
        "coords": _tetrahedron(),
        "box": None,
        "kind": "lj",
        "method": "sqnm",
        "parameters": dict(PARAMETERS),
        "gtol": 1.0e-6,
        "cap": CAP,
    }
    job.update(overrides)
    return job


def test_a_stopped_search_at_the_tetrahedron_is_a_success_with_the_closed_form_spectrum(pinned, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(_tetrahedron(), probe=seen))
    record = vcsqnm_min._minimize_job(_job())
    assert record["success"] is True
    assert (record["failure"], record["reasons"], record["converged"]) == (None, [], True)
    assert record["energy"] == pytest.approx(-6.0, abs=1e-12)
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx(_spectrum(R_MIN), rel=1e-6)
    assert (record["force_calls"], record["verify_calls"], record["hessian_calls"]) == (1, 0, 0)
    assert (record["start"], record["index"], record["method"], record["expected_index"]) == ("initial_2", 2, "sqnm", 0)
    assert (record["schema"], record["counter"], record["comparable"]) == ("optbench.start/v2", "harness", True)
    assert "error" not in record and "lattice_deriv_max" not in record
    assert (seen["method"], seen["cap"], seen["matrix"], seen["before"], seen["after"], seen["calls"]) == (
        "sqnm",
        CAP,
        None,
        False,
        True,
        1,
    )
    assert seen["parameters"] == PARAMETERS


def test_a_force_norm_exactly_at_the_stop_is_stopped_and_one_above_it_is_not(pinned, monkeypatch):
    point = _tetrahedron(shake=0.03)
    norm = float(np.linalg.norm(lennard_jones(point)[1]))
    seen: dict = {}
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point, stopped=False, probe=seen))
    vcsqnm_min._minimize_job(_job(coords=point, gtol=norm))
    assert seen["after"] is True
    vcsqnm_min._minimize_job(_job(coords=point, gtol=norm * (1.0 - 1.0e-9)))
    assert seen["after"] is False


def test_the_stop_waits_for_an_evaluation_and_follows_the_force_norm(pinned, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(_tetrahedron(shake=0.03), stopped=False, probe=seen))
    vcsqnm_min._minimize_job(_job(coords=_tetrahedron(shake=0.03)))
    assert (seen["before"], seen["after"]) == (False, False)


def test_a_search_still_above_the_force_stop_under_the_cap_is_a_divergence(pinned, monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point, stopped=False))
    record = vcsqnm_min._minimize_job(_job(coords=point))
    assert record["failure"] == "divergence"
    assert record["reasons"] == ["force_norm"]
    assert record["converged"] is False
    assert record["force_norm"] > 1.0e-6


def test_a_driver_that_claims_a_stop_away_from_the_force_stop_is_not_converged(pinned, monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point, stopped=True))
    record = vcsqnm_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["reasons"], record["converged"]) == ("divergence", ["force_norm"], False)
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(_tetrahedron(), stopped=True, error="ForceCallCap: force-call cap 4"))
    record = vcsqnm_min._minimize_job(_job())
    assert (record["failure"], record["converged"]) == ("cap", False)


def test_a_search_that_spent_the_budget_is_a_cap_failure_whatever_else_it_reports(pinned, monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point, stopped=False))
    record = vcsqnm_min._minimize_job(_job(coords=point, cap=1))
    assert record["failure"] == "cap"
    assert record["force_calls"] == 1
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point, stopped=False, error="ForceCallCap: force-call cap 4"))
    record = vcsqnm_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["error"]) == ("cap", "ForceCallCap: force-call cap 4")
    assert record["hessian_index"] is not None
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point, stopped=False, error="force-call cap 4"))
    assert vcsqnm_min._minimize_job(_job(coords=point))["failure"] == "cap"


def test_reaching_the_force_stop_on_the_last_permitted_call_is_a_cap_failure(pinned, monkeypatch):
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(_tetrahedron()))
    record = vcsqnm_min._minimize_job(_job(cap=1))
    assert record["force_calls"] == 1
    assert (record["failure"], record["reasons"], record["converged"]) == ("cap", ["cap"], False)
    assert record["force_norm"] < 1.0e-6
    assert vcsqnm_min._minimize_job(_job(cap=2))["success"] is True


def test_an_error_from_the_driver_is_a_crash_and_leaves_no_index(pinned, monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point, stopped=False, error="ValueError: lost"))
    record = vcsqnm_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["error"]) == ("crash", "ValueError: lost")
    assert record["hessian_index"] is None
    assert record["energy"] is not None


def test_a_diverged_search_is_not_verified_and_is_a_divergence(pinned, monkeypatch):
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(_tetrahedron(), stopped=False, diverged=True))
    record = vcsqnm_min._minimize_job(_job())
    assert record["failure"] == "divergence"
    assert record["reasons"] == ["divergence"]
    assert record["energy"] is None and record["force_norm"] is None
    assert record["verify_calls"] == 0 and "error" not in record
    assert record["hessian_index"] is None and record["min_eigenvalues"] == []


def test_a_point_that_is_not_finite_is_a_divergence(pinned, monkeypatch):
    monkeypatch.setattr(vcsqnm_min, "drive", _driver([math.nan] * 12))
    record = vcsqnm_min._minimize_job(_job())
    assert record["failure"] == "divergence"
    assert record["energy"] is None
    assert record["hessian_index"] is None


def test_a_stationary_square_is_a_wrong_stationary_point(pinned, monkeypatch):
    point = _square(_square_side())
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(point))
    record = vcsqnm_min._minimize_job(_job(coords=point))
    assert record["failure"] == "wrong_stationary"
    assert record["reasons"] == ["hessian_index"]
    assert record["converged"] is True
    assert record["hessian_index"] == 2


def test_a_point_the_potential_cannot_read_is_a_crash_from_the_verification(pinned, monkeypatch):
    monkeypatch.setattr(vcsqnm_min, "drive", _driver([0.0, 0.0], stopped=True))
    record = vcsqnm_min._minimize_job(_job(coords=_tetrahedron()))
    assert record["failure"] == "crash"
    assert record["error"].startswith("ValueError: ")


def test_a_cluster_has_no_cell_to_move(pinned):
    record = vcsqnm_min._minimize_job(_job(method="vc"))
    assert record["failure"] == "crash"
    assert record["error"] == "ValueError: Lennard-Jones 38 has no cell coordinate"
    assert record["force_calls"] == 0
    assert record["energy"] is None


def test_the_stop_of_a_moving_cell_needs_the_lattice_derivative_too(pinned, monkeypatch):
    box = [CELL, CELL, CELL]
    seen: dict = {}
    monkeypatch.setattr(
        vcsqnm_min, "drive", _driver(_dimer(), cell=np.diag(box).tolist(), probe=seen)
    )
    record = vcsqnm_min._minimize_job(_job(kind="morse", method="vc", coords=_dimer(), box=box))
    assert record["success"] is True
    assert (seen["before"], seen["after"]) == (False, True)
    assert seen["matrix"].tolist() == np.diag(box).tolist()
    assert record["lattice_deriv_max"] < 1.0e-12
    assert record["min_eigenvalues"][-1] == pytest.approx(4.0 * MORSE_ALPHA**2 * MORSE_D0, rel=1e-6)
    stretched = _dimer(0.05)
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(stretched, cell=np.diag(box).tolist(), stopped=False, probe=seen))
    held = vcsqnm_min._minimize_job(_job(kind="morse", method="vc", coords=stretched, box=box))
    assert seen["after"] is False
    assert held["lattice_deriv_max"] > 1.0e-3
    assert held["failure"] == "divergence"


def test_a_block_with_no_force_by_symmetry_still_has_a_lattice_derivative_to_settle(pinned, monkeypatch):
    spacing, length = 3.0, 9.0
    block = _lattice(spacing)
    box = [length, length, length]
    seen: dict = {}
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(block, probe=seen))
    fixed = vcsqnm_min._minimize_job(_job(kind="morse", method="sqnm", coords=block, box=box))
    assert seen["after"] is True
    assert fixed["force_norm"] < 1.0e-9
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(block, cell=np.diag(box).tolist(), probe=seen))
    moving = vcsqnm_min._minimize_job(_job(kind="morse", method="vc", coords=block, box=box))
    assert seen["after"] is False
    assert moving["force_norm"] < 1.0e-9
    assert moving["lattice_deriv_max"] > 0.1


def test_a_moving_cell_that_never_came_back_is_not_verified_and_is_a_crash(pinned, monkeypatch):
    box = [CELL, CELL, CELL]
    monkeypatch.setattr(vcsqnm_min, "drive", _driver(_dimer(), cell=None, stopped=False))
    record = vcsqnm_min._minimize_job(_job(kind="morse", method="vc", coords=_dimer(), box=box))
    assert record["energy"] is None and record["force_norm"] is None
    assert record["failure"] == "crash"
    assert "cell must be" in record["error"]


def _sqnm_standin(
    monkeypatch,
    alpha: float = 0.004,
    noise: str = "",
    jump: bool = False,
    log: list | None = None,
    scale: float = 1.0,
    build_noise: str = "",
):
    """The package interface the runner calls, with steepest descent behind it.

    ``scale`` multiplies the cell at every step of the cell-moving steppers.
    """

    class free_sqnm:
        def __init__(self, nat, step, nhist, alpha_min, eps_subsp):
            if build_noise:
                print(build_noise)
            if log is not None:
                log.append(("built", nat, step, nhist, alpha_min, eps_subsp))

        def optimizer_step(self, positions, energy, forces):
            if log is not None:
                log.append(("step", positions.copy(), float(energy), forces.copy()))
            if noise:
                print(noise)
            if jump:
                return positions * math.nan
            return positions + alpha * forces

    class periodic_sqnm:
        def __init__(self, nat, cell, step, nhist, weight, alpha_min, eps_subsp):
            if log is not None:
                log.append(("periodic", nat, cell.copy(), step, nhist, weight, alpha_min, eps_subsp))

        def optimizer_step(self, positions, cell, energy, forces, deriv):
            if log is not None:
                log.append(("vstep", positions.copy(), cell.copy(), deriv.copy()))
            return positions + alpha * forces, cell * scale

    class aseOptimizer:
        def __init__(self, atoms, **settings):
            self.settings = settings
            if build_noise:
                print(build_noise)
            if log is not None:
                log.append(("ase", settings))
            self.vc = settings["vc_relax"]
            atoms.get_forces()

        def step(self, atoms):
            if noise:
                print(noise)
            if jump:
                atoms.set_positions(atoms.positions * math.nan)
                return
            atoms.set_positions(atoms.positions + alpha * atoms.get_forces())
            if self.vc and scale != 1.0:
                atoms.set_cell(np.asarray(atoms.cell.array) * scale)

    package = types.ModuleType("sqnm")
    package.__path__ = []
    parts = {
        "sqnm.free_or_fixed_cell_sqnm": {"free_sqnm": free_sqnm},
        "sqnm.periodic_sqnm": {"periodic_sqnm": periodic_sqnm},
        "sqnm.vcsqnm_for_ase": {"aseOptimizer": aseOptimizer},
    }
    monkeypatch.setitem(sys.modules, "sqnm", package)
    for name, members in parts.items():
        module = types.ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_the_loop_settles_a_shaken_tetrahedron_through_the_package_interface(pinned, monkeypatch, method):
    _sqnm_standin(monkeypatch)
    record = vcsqnm_min._minimize_job(_job(method=method, coords=_tetrahedron(shake=0.03)))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(-6.0, abs=1e-9)
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx(_spectrum(R_MIN), rel=1e-4)
    assert 10 < record["force_calls"] < CAP
    assert record["force_norm"] <= 1.0e-6


def test_the_package_is_built_with_the_template_numbers_and_stepped_with_columns_and_forces(pinned, monkeypatch):
    log: list = []
    _sqnm_standin(monkeypatch, log=log)
    point = _tetrahedron(shake=0.03)
    vcsqnm_min._minimize_job(_job(coords=point))
    assert log[0] == ("built", 4, -1.0e-7, 10, 1.0e-3, 1.0e-5)
    _, positions, energy, forces = log[1]
    clean_energy, clean = lennard_jones(point)
    assert positions.shape == forces.shape == (3, 4)
    assert positions[:, 1].tolist() == pytest.approx(point[3:6])
    assert energy == pytest.approx(clean_energy)
    assert forces == pytest.approx(vcsqnm_min._columns(-clean))


def test_the_ase_wrapper_is_built_with_the_template_numbers(pinned, monkeypatch):
    log: list = []
    _sqnm_standin(monkeypatch, log=log)
    vcsqnm_min._minimize_job(_job(method="ase", coords=_tetrahedron(shake=0.03), cap=50))
    settings = log[0][1]
    assert settings == {
        "vc_relax": False,
        "force_tol": 0.0,
        "maximalSteps": 51,
        "initial_step_size": -1.0e-7,
        "nhist_max": 10,
        "lattice_weigth": 2.0,
        "alpha_min": 1.0e-3,
        "eps_subsp": 1.0e-5,
    }


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_a_slow_package_runs_into_the_budget_without_an_error(pinned, monkeypatch, method):
    _sqnm_standin(monkeypatch, alpha=1.0e-6)
    record = vcsqnm_min._minimize_job(_job(method=method, coords=_tetrahedron(shake=0.03), cap=7))
    assert record["failure"] == "cap"
    assert record["force_calls"] == 7
    assert "error" not in record


@pytest.mark.parametrize("method", ["vc", "ase-vc"])
def test_a_moving_cell_that_the_package_collapses_is_a_divergence(pinned, monkeypatch, method):
    _sqnm_standin(monkeypatch, scale=0.0)
    box = [CELL, CELL, CELL]
    record = vcsqnm_min._minimize_job(_job(kind="morse", method=method, coords=_dimer(0.05), box=box, cap=20))
    assert record["failure"] == "divergence"
    assert record["reasons"] == ["divergence"]
    assert "error" not in record
    assert record["energy"] is None
    assert record["hessian_index"] is None and record["min_eigenvalues"] == []


@pytest.mark.parametrize("method", ["vc", "ase-vc"])
def test_a_moving_cell_that_turns_to_nan_is_a_divergence_with_no_index(pinned, monkeypatch, method):
    _sqnm_standin(monkeypatch, scale=math.nan)
    box = [CELL, CELL, CELL]
    record = vcsqnm_min._minimize_job(_job(kind="morse", method=method, coords=_dimer(0.05), box=box, cap=20))
    assert record["failure"] == "divergence"
    assert "error" not in record
    assert record["energy"] is None
    assert record["hessian_index"] is None and record["min_eigenvalues"] == []


def test_a_flat_starting_cell_is_a_divergence_before_any_evaluation(pinned, monkeypatch):
    _sqnm_standin(monkeypatch)
    record = vcsqnm_min._minimize_job(
        _job(kind="morse", method="vc", coords=_dimer(0.05), box=[0.0, 0.0, 0.0], cap=20)
    )
    assert record["force_calls"] == 0
    assert record["failure"] == "divergence"
    assert "error" not in record


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_a_package_that_returns_nothing_finite_is_a_divergence(pinned, monkeypatch, method):
    _sqnm_standin(monkeypatch, jump=True)
    record = vcsqnm_min._minimize_job(_job(method=method, coords=_tetrahedron(shake=0.03)))
    assert record["failure"] == "divergence"
    assert record["force_calls"] == 1
    assert "error" not in record


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_a_package_that_says_it_was_given_other_positions_is_a_crash(pinned, monkeypatch, method):
    _sqnm_standin(monkeypatch, noise="SQNM was not called with positions it returned")
    record = vcsqnm_min._minimize_job(_job(method=method, coords=_tetrahedron(shake=0.03)))
    assert record["failure"] == "crash"
    assert record["error"] == "SQNM was not called with positions that were expected"


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_a_package_that_says_so_while_it_is_built_is_a_crash_before_any_step(pinned, monkeypatch, method):
    log: list = []
    _sqnm_standin(monkeypatch, build_noise="SQNM was not called with positions it returned", log=log)
    point = _tetrahedron(shake=0.03)
    record = vcsqnm_min._minimize_job(_job(method=method, coords=point))
    assert record["failure"] == "crash"
    assert record["error"] == "SQNM was not called with positions that were expected"
    assert record["energy"] == pytest.approx(lennard_jones(point)[0])
    assert not [row for row in log if row[0] == "step"]


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_a_budget_of_one_call_reads_the_start_and_takes_no_step(pinned, monkeypatch, method):
    log: list = []
    _sqnm_standin(monkeypatch, log=log)
    point = _tetrahedron(shake=0.03)
    record = vcsqnm_min._minimize_job(_job(method=method, coords=point, cap=1))
    assert record["failure"] == "cap"
    assert (record["force_calls"], record["verify_calls"]) == (1, 0)
    assert record["energy"] == pytest.approx(lennard_jones(point)[0])
    assert not [row for row in log if row[0] == "step"]


def test_a_budget_of_nothing_reads_nothing_and_verifies_the_start(pinned, monkeypatch):
    _sqnm_standin(monkeypatch)
    point = _tetrahedron(shake=0.03)
    record = vcsqnm_min._minimize_job(_job(coords=point, cap=0))
    assert record["failure"] == "cap"
    assert (record["force_calls"], record["verify_calls"]) == (0, 1)
    assert record["energy"] == pytest.approx(lennard_jones(point)[0])
    assert "error" not in record


def test_an_unknown_method_in_the_loop_is_a_crash_that_names_it(pinned, monkeypatch):
    _sqnm_standin(monkeypatch)
    record = vcsqnm_min._minimize_job(_job(method="nope"))
    assert record["failure"] == "crash"
    assert record["error"] == "ValueError: nope"


def test_the_budget_refusing_a_call_inside_the_loop_is_a_cap(pinned, monkeypatch):
    _sqnm_standin(monkeypatch, alpha=1.0e-6)

    class Tight(vcsqnm_min._Surface):
        def __init__(self, kind, box, cap, variable):
            super().__init__(kind, box, 3, variable)

    monkeypatch.setattr(vcsqnm_min, "_Surface", Tight)
    record = vcsqnm_min._minimize_job(_job(coords=_tetrahedron(shake=0.03), cap=1000))
    assert record["failure"] == "cap"
    assert record["error"] == "ForceCallCap: force-call cap 3"
    assert record["force_calls"] == 3
    assert record["hessian_index"] is not None


def test_a_moving_cell_is_stepped_with_the_lattice_derivative_the_package_expects(pinned, monkeypatch):
    log: list = []
    _sqnm_standin(monkeypatch, alpha=1.0e-3, log=log)
    box = [CELL, CELL, CELL]
    record = vcsqnm_min._minimize_job(
        _job(kind="morse", method="vc", coords=_dimer(0.05), box=box, cap=6, gtol=1.0e-8)
    )
    assert record["failure"] == "cap"
    assert log[0][0] == "periodic"
    assert log[0][2].tolist() == np.diag(box).tolist()
    steps = [row for row in log if row[0] == "vstep"]
    assert steps
    _, _, cell, deriv = steps[0]
    surface = vcsqnm_min._Surface("morse", box, 10, True)
    surface.evaluate(_dimer(0.05), np.diag(box))
    assert cell.tolist() == np.diag(box).tolist()
    assert deriv == pytest.approx(sqnm_lattice_derivative(surface.last_deriv))
    assert record["lattice_deriv_max"] > 0.0


def test_the_cell_moving_ase_wrapper_is_built_to_relax_the_cell_and_is_read_back(pinned, monkeypatch):
    log: list = []
    _sqnm_standin(monkeypatch, alpha=1.0e-3, log=log)
    box = [CELL, CELL, CELL]
    record = vcsqnm_min._minimize_job(
        _job(kind="morse", method="ase-vc", coords=_dimer(0.05), box=box, cap=8, gtol=1.0e-8)
    )
    assert log[0][1]["vc_relax"] is True
    assert log[0][1]["maximalSteps"] == 9
    assert record["failure"] == "cap"
    assert record["force_calls"] == 8
    assert record["lattice_deriv_max"] > 0.0
    assert "error" not in record


def test_the_cell_moving_package_is_built_with_the_cell_and_the_template_numbers(pinned, monkeypatch):
    log: list = []
    _sqnm_standin(monkeypatch, alpha=1.0e-3, log=log)
    box = [CELL, CELL, CELL]
    vcsqnm_min._minimize_job(_job(kind="morse", method="vc", coords=_dimer(0.05), box=box, cap=3, gtol=1.0e-8))
    name, count, cell, step, history, weight, alpha_min, eps = log[0]
    assert (name, count, step, history, weight, alpha_min, eps) == ("periodic", 2, -1.0e-7, 10, 2.0, 1.0e-3, 1.0e-5)
    assert cell.tolist() == np.diag(box).tolist()


def test_a_moving_cell_needs_its_starting_cell():
    with pytest.raises(ValueError, match="needs the starting cell"):
        vcsqnm_min.drive(_dimer(), "vc", dict(PARAMETERS), lambda *a: None, lambda: False, lambda: 0, 5, None)
    with pytest.raises(ValueError, match="needs the starting cell"):
        vcsqnm_min.drive(_dimer(), "ase-vc", dict(PARAMETERS), lambda *a: None, lambda: False, lambda: 0, 5, np.eye(2))


def test_the_ase_harness_fills_energy_and_forces_for_one_call(pinned):
    surface = vcsqnm_min._Surface("lj", None, 10, False)
    seen = vcsqnm_min._Seen(surface)
    point = np.asarray(_tetrahedron(shake=0.02))
    atoms = vcsqnm_min._atoms(point, None)
    atoms.calc = vcsqnm_min._Calculator(seen, False)
    clean_energy, clean = lennard_jones(point)
    assert atoms.get_potential_energy() == pytest.approx(clean_energy)
    assert atoms.get_forces().reshape(-1) == pytest.approx(-clean)
    assert surface.counter.calls == 1
    assert atoms.pbc.tolist() == [False, False, False]


def test_a_triclinic_cell_reaches_the_potential_as_columns_of_lattice_vectors():
    columns = np.array([[12.0, 1.0, 2.0], [0.0, 13.0, 3.0], [0.0, 0.0, 14.0]])
    atoms = vcsqnm_min._atoms(np.asarray(_dimer()), columns)
    assert atoms.cell.array[1].tolist() == columns[:, 1].tolist()
    received = []

    def evaluate(coords, matrix):
        received.append(matrix.copy())
        evaluate.last_deriv = np.zeros((3, 3))
        return 0.0, np.zeros(coords.size)

    atoms.calc = vcsqnm_min._Calculator(evaluate, True)
    atoms.get_potential_energy()
    assert received[0].tolist() == columns.tolist()


@pytest.mark.parametrize("stretch", [0.05, -0.1])
def test_the_stress_of_a_stretched_dimer_is_its_pair_virial_over_the_volume(stretch):
    box = [CELL, CELL, CELL]
    surface = vcsqnm_min._Surface("morse", box, 10, True)
    atoms = vcsqnm_min._atoms(np.asarray(_dimer(stretch)), np.diag(box))
    atoms.calc = vcsqnm_min._Calculator(vcsqnm_min._Seen(surface), True)
    stress = atoms.get_stress(voigt=False)
    distance = MORSE_R0 + stretch
    expo = math.exp(-MORSE_ALPHA * stretch)
    slope = 2.0 * MORSE_D0 * MORSE_ALPHA * expo * (1.0 - expo)
    assert stress[0, 0] == pytest.approx(slope * distance / CELL**3, rel=1e-9)
    assert np.max(np.abs(stress - np.diag([stress[0, 0], 0.0, 0.0]))) == 0.0
    cell = np.diag(box)
    derivative = -np.linalg.det(cell) * stress @ np.linalg.inv(cell).T
    assert derivative == pytest.approx(sqnm_lattice_derivative(surface.last_deriv), abs=1e-12)


def test_the_ase_harness_gives_a_moving_cell_a_stress_that_vanishes_at_the_dimer_minimum():
    box = [CELL, CELL, CELL]
    surface = vcsqnm_min._Surface("morse", box, 10, True)
    seen = vcsqnm_min._Seen(surface)
    atoms = vcsqnm_min._atoms(np.asarray(_dimer()), np.diag(box))
    atoms.calc = vcsqnm_min._Calculator(seen, True)
    assert atoms.pbc.tolist() == [True, True, True]
    assert atoms.cell.array.tolist() == np.diag(box).tolist()
    assert atoms.get_potential_energy() == pytest.approx(_dimer_energy(), abs=1e-12)
    assert np.max(np.abs(atoms.get_stress())) < 1.0e-12
    assert surface.counter.calls == 1


def _problem(**convergence) -> dict:
    return {"id": "minimization/lj38", "convergence": {"force_norm": 1.0e-6, "max_force_calls": CAP, **convergence}}


def _template(**changes) -> dict:
    template = {
        "id": "vcsqnm/test",
        "engine": "vcsqnm",
        "problem": "minimization/lj38",
        "code": "vc-SQNM",
        "algorithm": "SQNM",
        "tuning": "default",
        "commit": COMMIT,
        "code_version": CODE_VERSION,
        "sqnm": {**PARAMETERS, "method": "sqnm"},
    }
    template.update(changes)
    return template


def _con(coords: list[float]) -> str:
    rows = [coords[offset : offset + 3] for offset in range(0, len(coords), 3)]
    head = ["a label", "0.0000 TIME", "40.0 40.0 40.0", "90.0 90.0 90.0", "0 0", "0 0 0"]
    atoms = [f"{x!r} {y!r} {z!r} 0 {index}" for index, (x, y, z) in enumerate(rows)]
    return "\n".join([*head, "1", str(len(rows)), "1.008", "H", "Coordinates of Component 1", *atoms, ""])


def _starts(folder: Path, shakes=(0.01, 0.03, 0.05)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for index, shake in enumerate(shakes):
        (folder / f"initial_{index}.con").write_text(_con(_tetrahedron(shake=shake)), encoding="utf-8")
    return folder


def _run(tmp_path: Path, template: dict | None = None, problem: dict | None = None, **options) -> dict:
    options.setdefault("workers", 1)
    options.setdefault("geometries", tmp_path / "cons")
    options.setdefault("output", tmp_path / "out" / "bench.dat")
    options.setdefault("work", tmp_path / "work")
    return vcsqnm_min.run_template(tmp_path, problem or _problem(), template or _template(), **options)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_run_over_three_shaken_tetrahedra_scores_and_writes_every_start(pinned, monkeypatch, tmp_path, capsys):
    _sqnm_standin(monkeypatch)
    _starts(tmp_path / "cons")
    summary = _run(tmp_path)
    output = tmp_path / "out" / "bench.dat"
    rows = _rows(output.with_suffix(".starts.jsonl"))
    counts = [row["force_calls"] for row in rows]
    assert [row["start"] for row in rows] == ["initial_0", "initial_1", "initial_2"]
    assert all(row["success"] and "index" not in row for row in rows)
    assert all(row["energy"] == pytest.approx(-6.0, abs=1e-9) for row in rows)
    assert (summary["starts"], summary["template"], summary["problem"]) == (3, "vcsqnm/test", "minimization/lj38")
    assert (summary["nfailed"], summary["v2_nfailed"]) == (0, 0)
    assert summary["par2_mean"] == pytest.approx(statistics.mean(counts))
    assert (summary["force_calls_min"], summary["force_calls_max"]) == (min(counts), max(counts))
    assert summary["median"] == statistics.median(counts)
    assert summary["energy_min"] == pytest.approx(-6.0, abs=1e-9)
    assert "vcsqnm/test" not in capsys.readouterr().out
    document = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))
    run = document["runs"][0]
    assert (document["schema"], document["problem"]) == ("optbench.result/v2", "minimization/lj38")
    assert (run["entry"], run["code_version"], run["commit"], run["cap"]) == ("bench", CODE_VERSION, COMMIT, CAP)
    assert run["parameters"] == {"gtol": 1.0e-6, "maxiter": CAP, **{**PARAMETERS, "method": "sqnm"}}
    assert run["metrics"]["n"] == 3 and run["metrics"]["successes"] == 3
    assert (document["status"], document["scoring"], document["par_factor"]) == ("archive", "par2-bootstrap", 2)
    assert (document["bootstrap_resamples"], document["bootstrap_seed"]) == (2000, BOOTSTRAP_SEED)
    assert (run["present_source"], run["algorithm"], run["tuning"]) == (True, "SQNM", "default")
    assert len(run["starts"]) == 3 and run["legacy"]["nfailed"] == 0
    assert run["provenance"]["machine"]["node"] == "workstation-a"
    fields = dict(line.split(" ", 1) for line in output.read_text(encoding="utf-8").splitlines() if not line.startswith("#"))
    assert (fields["template"], fields["commit"], fields["code_version"], fields["n"]) == ("vcsqnm/test", COMMIT, CODE_VERSION, "3")
    assert (fields["tuning"], fields["code"], fields["algorithm"], fields["counter"]) == ("default", "vc-SQNM", "SQNM", "harness")


def test_a_template_without_an_algorithm_is_named_by_its_method(pinned, monkeypatch, tmp_path):
    _sqnm_standin(monkeypatch)
    _starts(tmp_path / "cons", shakes=(0.01,))
    template = _template()
    del template["algorithm"]
    _run(tmp_path, template)
    document = json.loads((tmp_path / "out" / "bench.v2.json").read_text(encoding="utf-8"))
    assert document["runs"][0]["algorithm"] == "sqnm"


def test_starts_that_diverge_count_as_failures_in_the_legacy_columns_too(pinned, monkeypatch, tmp_path):
    _sqnm_standin(monkeypatch, jump=True)
    _starts(tmp_path / "cons")
    summary = _run(tmp_path)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert [row["failure"] for row in rows] == ["divergence"] * 3
    assert all(row["force_calls"] == 1 for row in rows)
    assert (summary["nfailed"], summary["v2_nfailed"], summary["par2_mean"]) == (3, 3, 2 * CAP)
    assert summary["median"] is None


def test_a_limit_runs_only_the_first_starts_of_a_vcsqnm_run(pinned, monkeypatch, tmp_path):
    _sqnm_standin(monkeypatch)
    _starts(tmp_path / "cons")
    summary = _run(tmp_path, limit=1)
    assert summary["starts"] == 1
    assert [row["start"] for row in _rows(tmp_path / "out" / "bench.starts.jsonl")] == ["initial_0"]


def test_starts_that_fail_cost_twice_the_cap_each_and_leave_no_median(pinned, monkeypatch, tmp_path):
    _sqnm_standin(monkeypatch, alpha=1.0e-6)
    _starts(tmp_path / "cons")
    summary = _run(tmp_path, problem=_problem(max_force_calls=5))
    assert (summary["nfailed"], summary["v2_nfailed"], summary["par2_mean"]) == (3, 3, 10)
    assert (summary["force_calls"], summary["force_calls_min"], summary["force_calls_max"]) == (None, None, None)
    assert summary["median"] is None
    assert summary["energy_min"] < summary["energy_max"]
    names = {line.split(" ", 1)[0] for line in (tmp_path / "out" / "bench.dat").read_text(encoding="utf-8").splitlines()}
    assert "median" not in names


def test_a_budget_that_only_the_slowest_start_misses_is_scored_with_a_failure_at_twice_the_cap(
    pinned, monkeypatch, tmp_path
):
    _sqnm_standin(monkeypatch)
    _starts(tmp_path / "cons")
    free = _run(tmp_path, output=tmp_path / "free" / "bench.dat")
    counts = [row["force_calls"] for row in _rows(tmp_path / "free" / "bench.starts.jsonl")]
    assert len(set(counts)) == 3 and free["nfailed"] == 0
    cap = max(counts)
    summary = _run(tmp_path, problem=_problem(max_force_calls=cap), output=tmp_path / "tight" / "bench.dat")
    rows = _rows(tmp_path / "tight" / "bench.starts.jsonl")
    kept = [count for count in counts if count < cap]
    assert [row["failure"] for row in rows] == [None if count < cap else "cap" for count in counts]
    assert summary["nfailed"] == summary["v2_nfailed"] == 1
    assert summary["force_calls"] == round(statistics.mean(kept))
    assert (summary["force_calls_min"], summary["force_calls_max"]) == (min(kept), max(kept))
    assert summary["median"] == statistics.median(kept)
    assert summary["par2_mean"] == pytest.approx((sum(kept) + 2 * cap) / 3)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda p, t: t.update(engine="eon"), ValueError, "is not a vc-SQNM template"),
        (lambda p, t: t.update(commit="0" * 40), RuntimeError, "commit is not"),
        (lambda p, t: t.update(code_version="9.9"), RuntimeError, "code_version is not tag"),
        (lambda p, t: t.update(problem="minimization/morse-bulk"), ValueError, "is for minimization/morse-bulk, not minimization/lj38"),
        (
            lambda p, t: t.update(ini={"Optimizer": {"converged_force": 0.5}}),
            ValueError,
            "converged_force is not the problem force norm",
        ),
        (
            lambda p, t: t.update(ini={"Optimizer": {"max_iterations": CAP - 1}}),
            ValueError,
            "max_iterations is below the problem force-call cap",
        ),
        (lambda p, t: t["sqnm"].pop("lattice_weight"), ValueError, "sqnm section needs lattice_weight"),
        (lambda p, t: t["sqnm"].update(method="vc"), ValueError, "moves the cell, and minimization/lj38 has no cell coordinate"),
    ],
)
def test_a_template_that_does_not_fit_the_problem_is_refused_before_any_start_runs(
    pinned, tmp_path, change, error, message
):
    problem, template = _problem(), _template()
    change(problem, template)
    with pytest.raises(error, match=message):
        _run(tmp_path, template, problem)
    assert not (tmp_path / "out").exists()


def test_a_matching_stop_in_the_template_is_accepted(pinned, monkeypatch, tmp_path):
    _sqnm_standin(monkeypatch)
    _starts(tmp_path / "cons", shakes=(0.01,))
    template = _template(ini={"Optimizer": {"converged_force": 1.0e-6, "max_iterations": CAP}})
    assert _run(tmp_path, template)["starts"] == 1


def test_a_package_the_workers_cannot_import_stops_the_run_with_the_import_error(pinned, tmp_path):
    if importlib.util.find_spec("sqnm") is not None:
        pytest.skip("the package is installed, so a worker can import it")
    _starts(tmp_path / "cons", shakes=(0.01, 0.03))
    with pytest.raises(ImportError, match="sqnm"):
        _run(tmp_path, workers=2)


def test_the_run_pins_one_thread_for_the_numerical_libraries(pinned, monkeypatch, tmp_path):
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS")
    for name in names:
        monkeypatch.setenv(name, "8")
    _sqnm_standin(monkeypatch)
    _starts(tmp_path / "cons", shakes=(0.01,))
    _run(tmp_path)
    assert [vcsqnm_min.os.environ[name] for name in names] == ["1"] * len(names)


def test_float_fields_print_whole_numbers_without_a_point_and_others_to_twelve_digits():
    assert vcsqnm_min._format_field(12.0) == "12"
    assert vcsqnm_min._format_field(-6.0) == "-6"
    assert vcsqnm_min._format_field(2.0e12) == "2000000000000"
    assert vcsqnm_min._format_field(0.123456789012345) == "0.123456789012"
    assert vcsqnm_min._format_field(7) == "7"
    assert vcsqnm_min._format_field("a b") == "a b"


def test_a_failed_pin_check_stops_a_run_before_it_reads_the_template(monkeypatch, tmp_path):
    def refuse():
        raise RuntimeError("installed sqnm is not the pinned source")

    monkeypatch.setattr(vcsqnm_min, "assert_pin", refuse)
    with pytest.raises(RuntimeError, match="not the pinned source"):
        _run(tmp_path, _template(engine="eon"))
    with pytest.raises(RuntimeError, match="not the pinned source"):
        vcsqnm_min._minimize_job(_job())


def _pinned_package():
    pytest.importorskip("sqnm.sqnm")
    if importlib.metadata.version("sqnm") != DIST_VERSION:
        pytest.skip("the installed sqnm is not the distribution the registry pins")


def test_the_installed_package_passes_the_pin_check(monkeypatch):
    _pinned_package()
    monkeypatch.setattr(vcsqnm_min, "_PIN_OK", False)
    vcsqnm_min.assert_pin()
    assert vcsqnm_min._PIN_OK is True


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_the_installed_package_settles_a_shaken_tetrahedron_on_the_closed_form_minimum(method):
    _pinned_package()
    record = vcsqnm_min._minimize_job(_job(method=method, coords=_tetrahedron(shake=0.03)))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(-6.0, abs=1e-8)
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx(_spectrum(R_MIN), rel=1e-3)


def test_a_run_with_the_installed_package_gives_one_worker_and_two_the_same_rows(tmp_path):
    _pinned_package()
    _starts(tmp_path / "cons")
    _run(tmp_path, output=tmp_path / "one" / "bench.dat", workers=1)
    _run(tmp_path, output=tmp_path / "two" / "bench.dat", workers=2)
    keep = ("start", "force_calls", "success", "energy", "hessian_index")
    one = _rows(tmp_path / "one" / "bench.starts.jsonl")
    two = _rows(tmp_path / "two" / "bench.starts.jsonl")
    assert [{key: row[key] for key in keep} for row in one] == [{key: row[key] for key in keep} for row in two]
    assert all(row["success"] for row in one)
