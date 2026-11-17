"""The Lennard-Jones noise minimiser on clusters with closed-form answers.

Four atoms at the corners of a regular tetrahedron sit at the pair minimum
``r = 2**(1/6)`` of ``u(r) = 4 (r**-12 - r**-6)`` when the energy is -6 and
the six vibrational eigenvalues are ``k, k, 2k, 2k, 2k, 4k`` with
``k = u''(r)``. At any other edge ``a`` the tension ``t = u'(a)/a`` moves them
to ``k + 3t`` (twice), ``2k + 2t`` (three times) and ``4k`` with
``k = u''(a)``. The planar square at its stationary side is a second-order
saddle.

Solver families whose library is not installed are tested through the seam
the runner calls them by, or as the crash record a missing library leaves.
The real libraries are tried only when they can be imported.
"""

from __future__ import annotations

import enum
import importlib.metadata
import json
import math
import os
import random
import statistics
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.noise import KAPPA, stop_threshold
from optbench.potentials import lennard_jones
from optbench.potserver import RgpotCounter
from optbench.runners import noise_min
from optbench.v2 import HOST_LABEL_ENV

R_MIN = 2.0 ** (1.0 / 6.0)
CAP = 400
SEED = 20261016
PROBLEM_ID = "minimization/lj38-noise"


def _u(r: float) -> float:
    return 4.0 * (r**-12 - r**-6)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


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
    """The side at which the breathing force of the planar square vanishes."""
    low, high = 1.0, 1.2
    for _ in range(200):
        middle = (low + high) / 2.0
        slope = 4.0 * _slope(middle) + 2.0 * math.sqrt(2.0) * _slope(math.sqrt(2.0) * middle)
        if slope < 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _job(**overrides) -> dict:
    job = {
        "index": 3,
        "start": "initial_3",
        "coords": _tetrahedron(shake=0.03),
        "family": "ase",
        "method": "",
        "optimizer": "LBFGS",
        "options": {},
        "sigma": 0.0,
        "seed": 20,
        "clean_tol": 1.0e-6,
        "energy_tol": 1.0e-8,
        "cap": 2000,
    }
    job.update(overrides)
    return job


def _absent(monkeypatch, library: str) -> None:
    """Fail every import of the library, and of a module of it an earlier test loaded, as if it were not installed."""
    loaded = [name for name in sys.modules if name == library or name.startswith(f"{library}.")]
    for name in {*loaded, library}:
        monkeypatch.setitem(sys.modules, name, None)


def _observed(sigma: float, seed: int = 7, cap: int = 50):
    counter = HarnessCounter(cache=True, cap=cap)
    potential = RgpotCounter(lambda positions, _numbers=None, _box=None: lennard_jones(positions), counter)
    return noise_min._Observed(potential, sigma, random.Random(seed)), counter


def test_threads_are_pinned_to_one(monkeypatch):
    for name in noise_min._THREAD_VARS:
        monkeypatch.setenv(name, "8")
    noise_min._quiet_threads()
    assert {os.environ[name] for name in noise_min._THREAD_VARS} == {"1"}


def test_noise_is_added_after_the_clean_call_and_a_repeated_point_repeats_the_sample():
    observed, counter = _observed(0.01)
    point, other = _tetrahedron(shake=0.02), _tetrahedron(shake=0.03)
    clean_energy, clean = lennard_jones(point)
    stream = random.Random(7)
    energy, noisy = observed(point)
    assert energy == clean_energy
    assert noisy == pytest.approx([value + stream.gauss(0.0, 0.01) for value in clean], abs=1e-15)
    assert observed(point) == (energy, noisy)
    assert (counter.calls, observed.energies) == (1, [clean_energy])
    _, second_clean = lennard_jones(other)
    _, second = observed(other)
    assert second == pytest.approx([value + stream.gauss(0.0, 0.01) for value in second_clean], abs=1e-15)
    assert counter.calls == 2


def test_zero_sigma_returns_the_clean_force_and_draws_nothing():
    observed, _ = _observed(0.0)
    point = _tetrahedron(shake=0.02)
    _, clean = lennard_jones(point)
    _, noisy = observed(point)
    assert noisy == tuple(float(value) for value in clean)
    assert observed.rng.getstate() == random.Random(7).getstate()


def test_verification_reads_the_clean_force_and_records_nothing():
    observed, counter = _observed(0.05)
    searched, verified = _tetrahedron(shake=0.02), _tetrahedron(shake=0.03)
    energy, noisy = observed(searched)
    norm = observed.last_norm
    _, clean = lennard_jones(verified)
    with counter.verification():
        _, gradient = observed(verified)
    assert gradient == tuple(float(value) for value in clean)
    assert observed.energies == [energy]
    assert observed.last_norm == norm == pytest.approx(math.sqrt(sum(value * value for value in noisy)))
    assert (counter.calls, counter.verify_calls) == (1, 1)


def test_the_stop_waits_for_a_window_of_energies_inside_the_tolerance():
    observed, _ = _observed(1.0e-9)
    assert observed.met(1.0e-2, 1.0e-6, 12) is False
    for step in range(1, 5):
        observed(_tetrahedron(shake=1.0e-6 * step))
        assert observed.met(1.0e-2, 1.0e-6, 12) is False
    observed(_tetrahedron(shake=5.0e-6))
    assert observed.met(1.0e-2, 1.0e-6, 12) is True
    observed(_tetrahedron(shake=0.05))
    assert observed.met(1.0e-2, 1.0e-6, 12) is False
    for step in range(1, 5):
        observed(_tetrahedron(shake=1.0e-6 * step))
        assert observed.met(1.0e-2, 1.0e-6, 12) is False
    observed(_tetrahedron(shake=5.0e-6))
    assert observed.met(1.0e-2, 1.0e-6, 12) is True


def test_a_clean_run_stops_on_the_force_alone():
    observed, _ = _observed(0.0)
    observed(_tetrahedron(shake=0.03))
    assert observed.met(1.0e-6, 1.0e-8, 12) is False
    observed(_tetrahedron())
    assert observed.met(1.0e-6, 1.0e-8, 12) is True


@pytest.mark.parametrize("edge", [1.2, R_MIN, 1.3, 1.6])
def test_the_index_of_a_tetrahedron_follows_its_closed_form_spectrum(edge):
    index, values = noise_min._index(_tetrahedron(edge), 1.0e-6)
    expected = _spectrum(edge)
    assert values == pytest.approx(expected, rel=1e-9)
    assert index == sum(1 for value in expected if value < -1.0e-6)


def test_a_force_stop_larger_than_a_curvature_hides_it_from_the_index():
    point = _tetrahedron(1.3)
    assert noise_min._index(point, 1.0e-6)[0] == 4
    assert noise_min._index(point, 10.0)[0] == 1


def test_the_index_lists_the_eight_lowest_curvatures_of_a_larger_cluster():
    far = [value + (100.0 if index % 3 == 0 else 0.0) for index, value in enumerate(_tetrahedron())]
    index, values = noise_min._index(_tetrahedron() + far, 1.0e-6)
    stiffness = _curvature(R_MIN)
    assert index == 0
    assert values == pytest.approx([0.0] * 6 + [stiffness, stiffness], abs=1e-8)


def test_a_field_is_written_as_a_short_number_or_text():
    assert noise_min._format_field(2.0) == "2"
    assert noise_min._format_field(0.1 + 0.2) == "0.3"
    assert noise_min._format_field(1.0e-4) == "0.0001"
    assert noise_min._format_field(2.0e12) == "2000000000000"
    assert noise_min._format_field(12) == "12"
    assert noise_min._format_field("LBFGS") == "LBFGS"


def test_the_dat_file_lists_the_set_fields_in_a_fixed_order(monkeypatch):
    summary = {
        "force_calls": 12.5,
        "force_calls_min": 7,
        "force_calls_max": 20,
        "nfailed": 0,
        "median": 10.0,
        "cap": 10000,
        "energy_min": -6.0,
        "energy_max": -5.9999999999,
    }
    template = {
        "id": "noise/ase-lbfgs-s0",
        "code": "ASE",
        "code_version": "3.29.0",
        "commit": "abc",
        "algorithm": "LBFGS",
        "problem": PROBLEM_ID,
        "tuning": "",
        "note": "line one\nline two",
    }
    metrics = {
        "par2_mean": 12.5,
        "rate_low": 0.9,
        "rate_high": 1.0,
        "median": 10.0,
        "median_low": 8.0,
        "median_high": 12.0,
        "nfailed": 0,
        "successes": 4,
        "n": 4,
    }
    text = noise_min._dat(summary, template, metrics, 0.001, None, {"tric": 1, "cart": 2})
    lines = text.splitlines()
    comments = [line for line in lines if line.startswith("#")]
    fields = [line for line in lines if not line.startswith("#")]
    assert text.endswith("\n")
    assert "# final coordinate system: cart 2, tric 1" in comments
    assert "# line one line two" in comments
    assert comments[-1] == "# line one line two"
    assert fields == [
        "force_calls 12.5",
        "force_calls_min 7",
        "force_calls_max 20",
        "nfailed 0",
        "median 10",
        "code ASE",
        "code_version 3.29.0",
        "commit abc",
        "algorithm LBFGS",
        "template noise/ase-lbfgs-s0",
        "counter harness",
        f"problem {PROBLEM_ID}",
        "sigma 0.001",
        "par2_mean 12.5",
        "rate_low 0.9",
        "rate_high 1",
        "par2_median 10",
        "par2_median_low 8",
        "par2_median_high 12",
        "v2_nfailed 0",
        "v2_successes 4",
        "n 4",
        "cap 10000",
        "energy_min -6",
        "energy_max -5.9999999999",
    ]


def test_the_dat_file_opens_with_the_build_that_made_it():
    found = {"wales_sha256": "a" * 64, "wales_svnrev": "r1"}
    text = noise_min._dat({}, {"id": "noise/gmin"}, {}, 0.0, found, None)
    assert text.splitlines()[:2] == [f"# wales_sha256 {'a' * 64}", "# wales_svnrev r1"]
    assert "final coordinate system" not in text


def test_a_fresh_run_drops_the_partial_file_and_a_resumed_run_keeps_its_rows(tmp_path):
    partial = tmp_path / "bench.partial.jsonl"
    assert noise_min._kept_partial(partial, False) == {}
    assert noise_min._kept_partial(partial, True) == {}
    rows = [
        json.dumps({"index": 4, "force_calls": 9}),
        "",
        "  ",
        "{not json",
        json.dumps({"index": 2, "force_calls": 5}),
        json.dumps({"index": 4, "force_calls": 11}),
    ]
    partial.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert noise_min._kept_partial(partial, True) == {4: {"index": 4, "force_calls": 11}, 2: {"index": 2, "force_calls": 5}}
    assert partial.is_file()
    assert noise_min._kept_partial(partial, False) == {}
    assert not partial.exists()


@pytest.mark.parametrize("optimizer", ["LBFGS", "BFGS", "FIRE"])
def test_an_ase_optimizer_settles_a_shaken_tetrahedron_on_the_closed_form_minimum(optimizer):
    record = noise_min._minimize_job(_job(optimizer=optimizer))
    assert record["success"] is True
    assert record["failure"] is None
    assert record["converged"] is True
    assert record["reasons"] == []
    assert record["energy"] == pytest.approx(-6.0, abs=1e-9)
    assert record["force_norm"] <= 1.0e-6
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx(_spectrum(R_MIN), rel=1e-4)
    assert 0 < record["force_calls"] < 2000
    assert record["verify_calls"] == 0
    assert "error" not in record


class _Stepper:
    """What the loop asks of an optimizer: one step at a time, and never more than a handful of them."""

    def __init__(self, move):
        self.move = move
        self.runs = 0

    def run(self, fmax=0.0, steps=1):
        self.runs += 1
        if self.runs > 5:
            raise AssertionError("the loop went on")
        self.move()


def _loop(stepper, observed, counter, cap):
    potential = types.SimpleNamespace(counter=counter)
    return noise_min._step_optimizer(stepper, observed, potential, cap, 1.0e-6, 1.0e-8, 12)


def test_a_step_that_spends_no_call_ends_the_loop_instead_of_spinning():
    observed, counter = _observed(0.0, cap=50)
    stepper = _Stepper(lambda: None)
    assert _loop(stepper, observed, counter, 50) is False
    assert (stepper.runs, counter.calls) == (1, 0)


def test_a_step_that_runs_into_the_budget_ends_the_loop_with_the_calls_it_made():
    observed, counter = _observed(0.0, cap=2)

    def walk():
        for shake in (0.01, 0.02, 0.03):
            observed(_tetrahedron(shake=shake))

    stepper = _Stepper(walk)
    assert _loop(stepper, observed, counter, 2) is False
    assert (stepper.runs, counter.calls) == (1, 2)


def test_a_runtime_error_that_names_the_force_call_cap_ends_the_loop():
    def spent():
        raise RuntimeError("the force-call cap of 50 is spent")

    observed, counter = _observed(0.0, cap=50)
    stepper = _Stepper(spent)
    assert _loop(stepper, observed, counter, 50) is False
    assert stepper.runs == 1


def test_any_other_runtime_error_from_a_step_is_not_hidden():
    def lost():
        raise RuntimeError("the optimizer lost its direction")

    observed, counter = _observed(0.0, cap=50)
    with pytest.raises(RuntimeError, match="lost its direction"):
        _loop(_Stepper(lost), observed, counter, 50)


def test_a_record_names_its_start_seed_and_the_schema_it_belongs_to():
    record = noise_min._minimize_job(_job(index=5, seed=100, sigma=0.0))
    assert record["schema"] == "optbench.start/v2"
    assert (record["start"], record["index"], record["seed"]) == ("initial_3", 5, 105)
    assert (record["counter"], record["comparable"], record["expected_index"]) == ("harness", True, 0)
    assert (record["sigma"], record["hessian_calls"], record["rgmin_stop"]) == (0.0, 0, None)
    assert record["seconds"] >= 0.0


@pytest.mark.parametrize("sigma", [1.0e-4, 1.0e-3])
def test_a_noisy_run_stops_inside_the_noise_floor_at_the_clean_minimum(sigma):
    job = _job(sigma=sigma, energy_tol=max(1.0e-8, KAPPA * sigma))
    record = noise_min._minimize_job(job)
    assert record["success"] is True
    assert record["sigma"] == sigma
    assert record["force_norm"] <= stop_threshold(1.0e-6, sigma, 12, KAPPA)
    assert record["energy"] == pytest.approx(-6.0, abs=1e-5)
    assert record["hessian_index"] == 0


def test_the_noise_stream_of_a_start_is_its_seed_plus_its_index():
    same = [
        noise_min._minimize_job(_job(index=index, seed=seed, sigma=1.0e-3, energy_tol=3.0e-3))
        for index, seed in ((0, 5), (1, 4))
    ]
    other = noise_min._minimize_job(_job(index=1, seed=5, sigma=1.0e-3, energy_tol=3.0e-3))
    assert (same[0]["seed"], same[1]["seed"], other["seed"]) == (5, 5, 6)
    assert (same[0]["force_calls"], same[0]["energy"]) == (same[1]["force_calls"], same[1]["energy"])
    assert (other["force_calls"], other["energy"]) != (same[0]["force_calls"], same[0]["energy"])


def test_a_search_that_spends_its_budget_is_a_cap_failure():
    record = noise_min._minimize_job(_job(cap=5))
    assert record["failure"] == "cap"
    assert record["reasons"] == ["cap"]
    assert record["success"] is False
    assert record["converged"] is False
    assert record["force_calls"] == 5


def test_a_start_on_the_stationary_square_is_a_wrong_stationary_point():
    side = _square_side()
    record = noise_min._minimize_job(_job(coords=_square(side), optimizer="LBFGS"))
    assert record["failure"] == "wrong_stationary"
    assert record["reasons"] == ["hessian_index"]
    assert record["converged"] is True
    assert record["hessian_index"] == 2
    assert record["energy"] == pytest.approx(4.0 * _u(side) + 2.0 * _u(math.sqrt(2.0) * side), rel=1e-9)
    assert record["min_eigenvalues"][0] < 0.0 and record["min_eigenvalues"][1] < 0.0
    assert record["force_calls"] <= 3


def test_a_point_that_is_not_a_number_is_a_divergence(monkeypatch):
    def lost(spec, evaluate, stop):
        return {"point": [float("nan")] * 12, "stopped": True, "steps": 1}

    monkeypatch.setattr("optbench.runners.geometric_min.optimize", lost)
    record = noise_min._minimize_job(_job(family="geometric", method="cart"))
    assert record["failure"] == "divergence"
    assert record["reasons"] == ["divergence"]
    assert "error" not in record


def test_a_point_the_potential_cannot_read_is_a_crash_that_says_so(monkeypatch):
    def short(spec, evaluate, stop):
        return {"point": [0.0, 0.0], "stopped": True, "steps": 1}

    monkeypatch.setattr("optbench.runners.geometric_min.optimize", short)
    record = noise_min._minimize_job(_job(family="geometric", method="cart"))
    assert record["failure"] == "crash"
    assert record["error"].startswith("ValueError: ")
    assert record["hessian_index"] is None


def test_an_unknown_family_is_a_crash_that_names_it():
    record = noise_min._minimize_job(_job(family="steepest"))
    assert record["failure"] == "crash"
    assert record["error"] == "ValueError: unknown solver family steepest"
    assert record["force_calls"] == 0
    assert record["energy"] is None


def test_coordinates_that_are_not_numbers_are_a_crash_before_any_force_call():
    record = noise_min._minimize_job(_job(coords=["x", "y", "z"]))
    assert record["failure"] == "crash"
    assert record["error"].startswith("ValueError: ")
    assert "'x'" in record["error"]
    assert (record["force_calls"], record["verify_calls"], record["energy"]) == (0, 0, None)


def test_the_first_error_of_a_start_is_the_one_its_record_keeps(monkeypatch):
    def fails_at_a_point_nobody_can_read(job, x0, observed, potential, *rest):
        potential.last_point = [0.0, 0.0]
        raise RuntimeError("the driver failed")

    monkeypatch.setattr(noise_min, "_drive_ase", fails_at_a_point_nobody_can_read)
    record = noise_min._minimize_job(_job())
    assert record["failure"] == "crash"
    assert record["error"] == "RuntimeError: the driver failed"
    assert (record["energy"], record["verify_calls"]) == (None, 0)


def test_a_spectrum_that_cannot_be_computed_is_a_crash_that_names_why(monkeypatch):
    def broken(point, force_stop):
        raise np.linalg.LinAlgError("no spectrum")

    monkeypatch.setattr(noise_min, "_index", broken)
    monkeypatch.setattr(noise_min, "_drive_ase", lambda *args: (_tetrahedron(), False))
    record = noise_min._minimize_job(_job())
    assert record["failure"] == "crash"
    assert record["error"] == "LinAlgError: no spectrum"
    assert record["energy"] == pytest.approx(-6.0, abs=1e-9)
    assert (record["hessian_index"], record["verify_calls"]) == (None, 1)


def test_a_spectrum_that_cannot_be_computed_does_not_hide_that_the_budget_ran_out(monkeypatch):
    def spent(job, x0, observed, potential, *rest):
        observed(x0)
        potential.last_point = list(x0)
        raise ForceCallCap("force-call cap 2000")

    def broken(point, force_stop):
        raise np.linalg.LinAlgError("no spectrum")

    monkeypatch.setattr(noise_min, "_drive_ase", spent)
    monkeypatch.setattr(noise_min, "_index", broken)
    record = noise_min._minimize_job(_job())
    assert record["failure"] == "cap"
    assert record["error"] == "ForceCallCap: force-call cap 2000"
    assert record["force_calls"] == 1
    assert record["hessian_index"] is None


@pytest.mark.parametrize(
    ("family", "method", "module"),
    [
        ("rgmin", "lbfgs", "rgmin"),
        ("sella", "", "sella"),
        ("geometric", "cart", "geometric"),
        ("pele", "lbfgs", "pele"),
        ("vcsqnm", "sqnm", "sqnm"),
    ],
)
def test_a_family_without_its_library_is_a_crash_that_names_the_library(monkeypatch, family, method, module):
    _absent(monkeypatch, module)
    record = noise_min._minimize_job(_job(family=family, method=method))
    assert record["failure"] == "crash"
    assert module in record["error"]
    assert record["force_calls"] == 0


def test_gmin_without_its_compiled_library_is_a_crash_that_names_the_file(monkeypatch):
    monkeypatch.setattr("optbench.gmin_lbfgs.library_path", lambda: None)
    monkeypatch.setattr("optbench.gmin_lbfgs._LIBRARY", None)
    record = noise_min._minimize_job(_job(family="gmin", method=""))
    assert record["failure"] == "crash"
    assert "libgminlbfgs.so" in record["error"]
    assert record["force_calls"] == 0


def _geometric_double(report: dict, seen: dict | None = None):
    def optimize(spec, evaluate, stop):
        if seen is not None:
            seen["spec"] = spec
        evaluate(spec["coords"])
        if seen is not None:
            seen["stop"] = stop(None)
        return dict(report)

    return optimize


def test_the_geometric_report_reaches_the_record_and_the_spec_carries_the_options(monkeypatch):
    seen: dict = {}
    point = _tetrahedron()
    report = {"point": point, "stopped": True, "steps": 4, "rebuilds": 1, "coordinate_system": "tric"}
    monkeypatch.setattr("optbench.runners.geometric_min.optimize", _geometric_double(report, seen))
    record = noise_min._minimize_job(
        _job(family="geometric", method="tric", coords=point, options={"neighbors": 3, "element": "Ar"}, cap=77)
    )
    assert record["success"] is True
    assert (record["coordinate_system"], record["geometric_steps"], record["rebuilds"]) == ("tric", 4, 1)
    assert seen["spec"]["kind"] == "lj"
    assert seen["spec"]["coordsys"] == "tric"
    assert (seen["spec"]["neighbors"], seen["spec"]["element"], seen["spec"]["cap"]) == (3, "Ar", 77)
    assert seen["spec"]["box"] is None and seen["spec"]["start"] == "initial_3"
    assert record["force_calls"] == 1


def test_the_stop_given_to_geometric_is_the_noise_aware_test_on_the_last_force_it_evaluated(monkeypatch):
    seen: dict = {}
    report = {"point": _tetrahedron(), "stopped": True}
    monkeypatch.setattr("optbench.runners.geometric_min.optimize", _geometric_double(report, seen))
    noise_min._minimize_job(_job(family="geometric", method="cart", coords=_tetrahedron()))
    assert seen["stop"] is True
    noise_min._minimize_job(_job(family="geometric", method="cart", coords=_tetrahedron(shake=0.03)))
    assert seen["stop"] is False


def test_the_geometric_coordinate_system_is_the_option_before_the_method(monkeypatch):
    seen: dict = {}
    report = {"point": _tetrahedron(), "stopped": True}
    monkeypatch.setattr("optbench.runners.geometric_min.optimize", _geometric_double(report, seen))
    noise_min._minimize_job(_job(family="geometric", method="tric", options={"coordsys": "dlc"}))
    assert seen["spec"]["coordsys"] == "dlc"
    assert seen["spec"]["neighbors"] == 2 and seen["spec"]["element"] == "H"
    noise_min._minimize_job(_job(family="geometric", method="", options={}))
    assert seen["spec"]["coordsys"] == "cart"


def test_a_geometric_cap_message_is_a_cap_and_any_other_message_is_a_crash(monkeypatch):
    point = _tetrahedron(shake=0.02)
    cases = (
        ("ForceCallCap: force-call cap 9", "cap", "ForceCallCap: force-call cap 9"),
        ("force-call cap", "cap", "ForceCallCap: force-call cap"),
        ("RuntimeError: Not bracketed", "crash", "RuntimeError: Not bracketed"),
    )
    for text, failure, recorded in cases:
        report = {"point": point, "stopped": False, "error": text}
        monkeypatch.setattr("optbench.runners.geometric_min.optimize", _geometric_double(report))
        record = noise_min._minimize_job(_job(family="geometric", method="cart", coords=point))
        assert record["failure"] == failure
        assert record["error"] == recorded
        assert record["force_calls"] == 1
        assert record["coordinate_system"] == ""


def _gmin_double(seen: dict, **result):
    def minimize(coords, evaluate, *, gtol, cap, m, maxbfgs, dguess, maxerise, maxefall, ready):
        seen.update(gtol=gtol, cap=cap, m=m, maxbfgs=maxbfgs, dguess=dguess, maxerise=maxerise, maxefall=maxefall)
        evaluate(coords)
        seen["ready"] = ready()
        outcome = {"coords": list(coords), "capped": False, "mflag": True, "info": 0, **result}
        return types.SimpleNamespace(**outcome)

    return minimize


def test_gmin_is_given_the_noise_aware_stop_and_the_options_of_its_template(monkeypatch):
    from optbench.gmin_lbfgs import DEFAULTS

    seen: dict = {}
    monkeypatch.setattr("optbench.gmin_lbfgs.minimize", _gmin_double(seen))
    sigma = 1.0e-3
    point = _tetrahedron()
    record = noise_min._minimize_job(
        _job(family="gmin", coords=point, sigma=sigma, energy_tol=3.0e-3, cap=80, options={"m": 6, "dguess": 0.2})
    )
    assert seen["gtol"] == pytest.approx(stop_threshold(1.0e-6, sigma, 12, KAPPA))
    assert (seen["cap"], seen["m"], seen["dguess"]) == (80, 6, 0.2)
    assert (seen["maxbfgs"], seen["maxerise"], seen["maxefall"]) == (
        DEFAULTS["maxbfgs"],
        DEFAULTS["maxerise"],
        DEFAULTS["maxefall"],
    )
    assert seen["ready"] is False
    assert record["failure"] == "divergence"
    assert record["reasons"] == ["force_norm"]


def test_gmin_is_stopped_only_when_it_says_so_and_the_harness_agrees(monkeypatch):
    point = _tetrahedron()
    for result, success in (({}, True), ({"mflag": False}, False), ({"info": 1}, False)):
        monkeypatch.setattr("optbench.gmin_lbfgs.minimize", _gmin_double({}, **result))
        record = noise_min._minimize_job(_job(family="gmin", coords=point))
        assert record["success"] is success
        assert record["converged"] is success


def test_a_gmin_search_the_budget_refused_is_a_cap_failure_at_the_point_it_reached(monkeypatch):
    point = _tetrahedron(shake=0.02)
    monkeypatch.setattr("optbench.gmin_lbfgs.minimize", _gmin_double({}, capped=True))
    record = noise_min._minimize_job(_job(family="gmin", coords=point))
    assert record["failure"] == "cap"
    assert record["reasons"] == ["cap"]
    assert record["error"] == "ForceCallCap: force-call cap"
    assert record["energy"] == pytest.approx(lennard_jones(point)[0])
    assert isinstance(record["hessian_index"], int)


def _vcsqnm_double(result: dict, seen: dict | None = None):
    def drive(coords, method, options, energy_and_gradient, ready, calls, cap, cell):
        if seen is not None:
            seen.update(method=method, options=options, cap=cap, cell=cell)
        energy_and_gradient(coords)
        seen_calls = calls()
        if seen is not None:
            seen["calls"] = seen_calls
            seen["ready"] = ready()
        return dict(result)

    return drive


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_a_fixed_cell_sqnm_result_reaches_the_record(monkeypatch, method):
    seen: dict = {}
    point = _tetrahedron()
    monkeypatch.setattr(
        "optbench.runners.vcsqnm_min.drive", _vcsqnm_double({"point": point, "stopped": True, "error": ""}, seen)
    )
    record = noise_min._minimize_job(
        _job(family="vcsqnm", method=method, coords=point, options={"nhist_max": 10}, cap=55)
    )
    assert record["success"] is True
    assert seen["method"] == method
    assert seen["options"] == {"nhist_max": 10}
    assert (seen["cap"], seen["cell"], seen["calls"], seen["ready"]) == (55, None, 1, True)


def test_a_variable_cell_method_is_refused_for_a_cluster_and_a_driver_error_is_kept(monkeypatch):
    refused = noise_min._minimize_job(_job(family="vcsqnm", method="vc"))
    assert refused["failure"] == "crash"
    assert refused["error"] == "ValueError: vc is not a fixed-cell SQNM method"
    monkeypatch.setattr(
        "optbench.runners.vcsqnm_min.drive",
        _vcsqnm_double({"point": _tetrahedron(shake=0.02), "stopped": False, "error": "SQNM lost its place"}),
    )
    record = noise_min._minimize_job(_job(family="vcsqnm", method="sqnm", coords=_tetrahedron(shake=0.02)))
    assert (record["failure"], record["error"]) == ("crash", "SQNM lost its place")


def test_a_pele_result_reaches_the_record_and_its_potential_reads_the_noisy_force(monkeypatch):
    seen: dict = {}
    point = _tetrahedron()

    def drive(method, coords, potential, *, stop, cap, options):
        seen.update(method=method, cap=cap, options=options)
        energy, gradient = potential.getEnergyGradient(coords)
        seen["energy"] = energy
        seen["only"] = potential.getEnergy(coords)
        seen["last"] = potential.last
        seen["stop"] = stop(energy=energy, gradient=gradient, tol=1.0e-6)
        return list(point), True, ""

    monkeypatch.setattr("optbench.runners.pele_min.drive_minimizer", drive)
    record = noise_min._minimize_job(_job(family="pele", method="lbfgs", coords=point, cap=60))
    assert record["success"] is True
    assert seen["method"] == "lbfgs"
    assert seen["cap"] == 60
    assert seen["energy"] == seen["only"] == pytest.approx(-6.0, abs=1e-12)
    assert seen["last"] == pytest.approx(point)
    assert seen["stop"] is True
    assert record["force_calls"] == 1


def test_a_pele_error_is_a_crash_unless_it_is_the_cap(monkeypatch):
    point = _tetrahedron(shake=0.02)
    for text, failure in (("pele gave up", "crash"), ("ForceCallCap: force-call cap 9", "cap")):
        monkeypatch.setattr(
            "optbench.runners.pele_min.drive_minimizer",
            lambda method, coords, potential, *, stop, cap, options, text=text: (list(point), False, text),
        )
        record = noise_min._minimize_job(_job(family="pele", method="lbfgs", coords=point))
        assert (record["failure"], record["error"]) == (failure, text)


@pytest.mark.parametrize("method", ["newton", "dogleg", "rfo"])
def test_an_exact_hessian_method_is_refused_under_noise_and_allowed_without_it(monkeypatch, method):
    _absent(monkeypatch, "rgmin")
    with pytest.raises(ValueError, match="exact Hessian"):
        noise_min._minimize_job(_job(family="rgmin", method=method, sigma=1.0e-4))
    record = noise_min._minimize_job(_job(family="rgmin", method=method, sigma=0.0))
    assert record["failure"] == "crash"
    assert "rgmin" in record["error"]


def _rgmin_double(alpha: float = 0.004, stall: bool = False):
    """A stand-in for the solver library. It steps down the function it is handed, or takes Newton steps."""

    class Stop(enum.Enum):
        Budget = 1
        Tolerance = 2
        Stalled = 3

    class Report:
        def __init__(self, stop):
            self.stop = stop

    class SolverConfig:
        def __init__(self, method, gtol=0.0, maxiter=0, **options):
            self.method, self.gtol, self.maxiter, self.options = method, gtol, maxiter, options

    configs: list = []

    class Solver:
        def __init__(self, size, config):
            self.size = size
            self.config = config
            configs.append(config)

        def step(self, fg, x):
            if stall:
                return Report(Stop.Stalled)
            _, gradient = fg(x)
            if np.linalg.norm(gradient) <= self.config.gtol:
                return Report(Stop.Tolerance)
            x -= alpha * gradient
            return Report(Stop.Budget)

        def step_hess(self, fg, hess, x):
            _, gradient = fg(x)
            if np.linalg.norm(gradient) <= self.config.gtol:
                return Report(Stop.Tolerance)
            x -= np.linalg.pinv(hess(x), rcond=1e-9) @ gradient
            return Report(Stop.Budget)

    return types.SimpleNamespace(
        Stop=Stop, SolverConfig=SolverConfig, Solver=Solver, configs=configs, __version__="0.0.0-test"
    )


def test_a_gradient_solver_that_reports_its_tolerance_is_a_success_at_sigma_zero(monkeypatch):
    double = _rgmin_double()
    monkeypatch.setitem(sys.modules, "rgmin", double)
    record = noise_min._minimize_job(_job(family="rgmin", method="lbfgs", cap=CAP, options={"memory": 7}))
    assert record["success"] is True
    assert "Tolerance" in record["rgmin_stop"]
    assert record["hessian_calls"] == 0
    assert record["energy"] == pytest.approx(-6.0, abs=1e-9)
    assert record["hessian_index"] == 0
    config = double.configs[0]
    assert (config.method, config.gtol, config.maxiter, config.options) == ("lbfgs", 1.0e-6, CAP, {"memory": 7})


def test_a_solver_that_reports_the_budget_without_spending_a_call_is_dropped_after_the_cap_plus_one_steps(monkeypatch):
    double = _rgmin_double()
    steps: list[int] = []

    class Idle:
        def __init__(self, size, config):
            pass

        def step(self, fg, x):
            steps.append(len(steps))
            return types.SimpleNamespace(stop=double.Stop.Budget)

    double.Solver = Idle
    monkeypatch.setitem(sys.modules, "rgmin", double)
    record = noise_min._minimize_job(_job(family="rgmin", method="lbfgs", cap=20))
    assert len(steps) == 21
    assert record["force_calls"] == 0
    assert record["converged"] is False
    assert record["reasons"] == ["force_norm"]
    assert record["failure"] == "divergence"


def test_an_exact_hessian_method_counts_one_hessian_for_each_step_it_takes(monkeypatch):
    monkeypatch.setitem(sys.modules, "rgmin", _rgmin_double())
    record = noise_min._minimize_job(_job(family="rgmin", method="newton", cap=CAP))
    assert record["success"] is True
    assert record["hessian_calls"] == record["force_calls"] - 1 > 0
    assert record["energy"] == pytest.approx(-6.0, abs=1e-9)


def test_under_noise_the_solver_has_no_tolerance_and_the_noise_aware_stop_ends_the_search(monkeypatch):
    double = _rgmin_double()
    monkeypatch.setitem(sys.modules, "rgmin", double)
    sigma = 1.0e-4
    record = noise_min._minimize_job(
        _job(family="rgmin", method="lbfgs", cap=CAP, sigma=sigma, energy_tol=max(1.0e-8, KAPPA * sigma))
    )
    assert record["success"] is True
    assert double.configs[0].gtol == 0.0
    assert "Budget" in record["rgmin_stop"]
    assert record["force_norm"] <= stop_threshold(1.0e-6, sigma, 12, KAPPA)


def test_a_solver_that_never_gets_going_is_rebuilt_twice_and_then_given_up_on(monkeypatch):
    double = _rgmin_double(stall=True)
    monkeypatch.setitem(sys.modules, "rgmin", double)
    record = noise_min._minimize_job(_job(family="rgmin", method="lbfgs", cap=CAP, sigma=1.0e-3, energy_tol=3.0e-3))
    assert len(double.configs) == 3
    assert record["failure"] == "divergence"
    assert record["reasons"] == ["force_norm"]
    assert record["force_calls"] == 0
    assert "Stalled" in record["rgmin_stop"]


def test_a_solver_too_slow_for_the_budget_is_a_cap_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "rgmin", _rgmin_double(alpha=1.0e-6))
    record = noise_min._minimize_job(_job(family="rgmin", method="lbfgs", cap=10))
    assert record["failure"] == "cap"
    assert record["force_calls"] == 10


def test_a_solver_that_stops_without_its_tolerance_at_sigma_zero_is_not_converged(monkeypatch):
    monkeypatch.setitem(sys.modules, "rgmin", _rgmin_double(stall=True))
    record = noise_min._minimize_job(_job(family="rgmin", method="lbfgs", cap=CAP))
    assert record["converged"] is False
    assert record["failure"] == "divergence"
    assert "Stalled" in record["rgmin_stop"]


def _problem(**convergence) -> dict:
    return {
        "id": PROBLEM_ID,
        "convergence": {"force_norm": 1.0e-6, "energy_tol": 1.0e-8, "max_force_calls": CAP, **convergence},
        "noise": {"seed": SEED},
    }


def _template(family: str = "ase", sigma: float = 0.0, **solver) -> dict:
    body = {"family": family, **solver}
    if family == "ase":
        body.setdefault("name", "LBFGS")
        body.setdefault("maxstep", 0.1)
    return {
        "id": f"noise/{family}-test",
        "engine": "noise",
        "problem": PROBLEM_ID,
        "code": "ASE",
        "code_version": "3.29.0",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "algorithm": "LBFGS",
        "tuning": "default",
        "note": "A test template.",
        "solver": body,
        "noise": {"sigma": sigma, "seed": SEED},
    }


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
    return noise_min.run_template(tmp_path, problem or _problem(), template or _template(), **options)


@pytest.fixture(autouse=True)
def _label(monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_run_over_three_shaken_tetrahedra_scores_and_writes_every_start(tmp_path):
    _starts(tmp_path / "cons")
    summary = _run(tmp_path)
    output = tmp_path / "out" / "bench.dat"
    rows = _rows(output.with_suffix(".starts.jsonl"))
    counts = [row["force_calls"] for row in rows]
    assert [row["start"] for row in rows] == ["initial_0", "initial_1", "initial_2"]
    assert all(row["success"] and "index" not in row for row in rows)
    assert [row["seed"] for row in rows] == [SEED, SEED + 1, SEED + 2]
    assert all(row["energy"] == pytest.approx(-6.0, abs=1e-9) for row in rows)
    assert (summary["starts"], summary["template"], summary["problem"]) == (3, "noise/ase-test", PROBLEM_ID)
    assert (summary["nfailed"], summary["v2_nfailed"], summary["cap"]) == (0, 0, CAP)
    assert summary["par2_mean"] == pytest.approx(statistics.mean(counts))
    assert (summary["force_calls_min"], summary["force_calls_max"]) == (min(counts), max(counts))
    assert summary["median"] == statistics.median(counts)
    assert summary["energy_min"] == pytest.approx(-6.0, abs=1e-9)
    assert summary["energy_max"] == pytest.approx(-6.0, abs=1e-9)
    assert summary["seconds"] >= 0.0
    assert not output.with_suffix(".partial.jsonl").exists()


def test_a_run_writes_the_dat_summary_and_the_result_document(tmp_path):
    _starts(tmp_path / "cons")
    _run(tmp_path, _template(sigma=0.0))
    output = tmp_path / "out" / "bench.dat"
    fields = dict(line.split(" ", 1) for line in output.read_text(encoding="utf-8").splitlines() if not line.startswith("#"))
    assert fields["template"] == "noise/ase-test"
    assert fields["sigma"] == "0"
    assert fields["cap"] == str(CAP)
    assert fields["nfailed"] == "0"
    assert fields["n"] == "3"
    document = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))
    run = document["runs"][0]
    assert document["schema"] == "optbench.result/v2"
    assert document["problem"] == PROBLEM_ID
    assert (run["entry"], run["code"], run["algorithm"], run["cap"]) == ("bench", "ASE", "LBFGS", CAP)
    assert run["metrics"]["n"] == 3 and run["metrics"]["successes"] == 3
    assert len(run["starts"]) == 3
    assert run["parameters"]["maxstep"] == 0.1
    assert (run["parameters"]["sigma"], run["parameters"]["family"], run["parameters"]["method"]) == (0.0, "ase", "LBFGS")
    assert run["provenance"]["code"] == "ASE"
    assert run["provenance"]["machine"]["node"] == "workstation-a"


def test_two_workers_write_the_same_rows_as_one(tmp_path, capsys):
    _starts(tmp_path / "cons")
    _run(tmp_path, output=tmp_path / "one" / "bench.dat", workers=1)
    assert "noise/ase-test" not in capsys.readouterr().out
    _run(tmp_path, output=tmp_path / "two" / "bench.dat", workers=2)
    assert "noise/ase-test 3/3" in capsys.readouterr().out
    one = _rows(tmp_path / "one" / "bench.starts.jsonl")
    two = _rows(tmp_path / "two" / "bench.starts.jsonl")
    keep = ("start", "force_calls", "success", "energy", "hessian_index", "seed")
    assert [{key: row[key] for key in keep} for row in one] == [{key: row[key] for key in keep} for row in two]


@pytest.mark.parametrize(("kappa", "window"), [(None, 3.0e-3), (10.0, 1.0e-2)])
def test_the_energy_window_follows_the_kappa_of_the_problem(tmp_path, kappa, window):
    _starts(tmp_path / "cons", shakes=(0.01,))
    problem = _problem()
    if kappa is not None:
        problem["noise"]["kappa"] = kappa
    _run(tmp_path, _template(sigma=1.0e-3), problem)
    document = json.loads((tmp_path / "out" / "bench.v2.json").read_text(encoding="utf-8"))
    assert document["runs"][0]["parameters"]["energy_tol"] == pytest.approx(window)


def test_a_limit_runs_only_the_first_starts(tmp_path):
    _starts(tmp_path / "cons")
    summary = _run(tmp_path, limit=2)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert summary["starts"] == 2
    assert [row["start"] for row in rows] == ["initial_0", "initial_1"]


def test_a_resumed_run_keeps_the_rows_it_has_and_a_fresh_run_discards_them(tmp_path):
    _starts(tmp_path / "cons")
    output = tmp_path / "out" / "bench.dat"
    _run(tmp_path)
    first = _rows(output.with_suffix(".starts.jsonl"))
    kept = dict(first[1], index=1, force_calls=7777)
    partial = output.with_suffix(".partial.jsonl")
    partial.write_text(json.dumps(kept) + "\n", encoding="utf-8")
    _run(tmp_path, resume=True)
    resumed = _rows(output.with_suffix(".starts.jsonl"))
    assert [row["force_calls"] for row in resumed] == [first[0]["force_calls"], 7777, first[2]["force_calls"]]
    assert not partial.exists()
    partial.write_text(json.dumps(kept) + "\n", encoding="utf-8")
    _run(tmp_path, resume=False)
    fresh = _rows(output.with_suffix(".starts.jsonl"))
    assert [row["force_calls"] for row in fresh] == [row["force_calls"] for row in first]
    assert not partial.exists()


def test_a_resumed_run_that_has_every_row_runs_no_start_again(tmp_path, monkeypatch):
    _starts(tmp_path / "cons")
    output = tmp_path / "out" / "bench.dat"
    _run(tmp_path)
    first = _rows(output.with_suffix(".starts.jsonl"))
    partial = output.with_suffix(".partial.jsonl")
    partial.write_text("".join(json.dumps(dict(row, index=index)) + "\n" for index, row in enumerate(first)), encoding="utf-8")

    def ran_again(job):
        raise AssertionError(f"{job['start']} ran again")

    monkeypatch.setattr(noise_min, "_minimize_job", ran_again)
    summary = _run(tmp_path, resume=True)
    assert summary["starts"] == 3
    assert _rows(output.with_suffix(".starts.jsonl")) == first
    assert not partial.exists()


def test_a_run_of_starts_that_all_crash_still_writes_its_files_and_pays_twice_the_cap_for_each(tmp_path, monkeypatch):
    _starts(tmp_path / "cons")
    _absent(monkeypatch, "sella")
    template = _template("sella")
    template["solver"] = {"family": "sella", "order": 0}
    summary = _run(tmp_path, template)
    output = tmp_path / "out" / "bench.dat"
    rows = _rows(output.with_suffix(".starts.jsonl"))
    document = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))
    metrics = document["runs"][0]["metrics"]
    assert [row["failure"] for row in rows] == ["crash"] * 3
    assert all("sella" in row["error"] for row in rows)
    assert (summary["nfailed"], summary["v2_nfailed"], summary["par2_mean"]) == (3, 3, 2 * CAP)
    assert (metrics["n"], metrics["successes"], metrics["n_crash"]) == (3, 0, 3)
    assert summary["median"] is None
    assert "energy_min" not in summary
    names = {line.split(" ", 1)[0] for line in output.read_text(encoding="utf-8").splitlines() if not line.startswith("#")}
    assert "median" not in names and "energy_min" not in names
    assert {"nfailed", "par2_mean", "n"} <= names


def test_a_geometric_run_counts_the_coordinate_systems_its_starts_ended_in(tmp_path, monkeypatch):
    _starts(tmp_path / "cons")
    monkeypatch.setitem(sys.modules, "geometric", types.SimpleNamespace(__version__="1.1", __git_revision__="abc"))
    systems = iter(["tric", "cart", "tric"])

    def optimize(spec, evaluate, stop):
        evaluate(spec["coords"])
        return {"point": _tetrahedron(), "stopped": True, "steps": 2, "rebuilds": 0, "coordinate_system": next(systems)}

    monkeypatch.setattr("optbench.runners.geometric_min.optimize", optimize)
    template = _template("geometric", method="tric")
    template["code_version"], template["commit"] = "1.1", "abc"
    _run(tmp_path, template)
    text = (tmp_path / "out" / "bench.dat").read_text(encoding="utf-8")
    assert "# final coordinate system: cart 1, tric 2" in text.splitlines()


def test_a_gmin_run_takes_its_version_and_commit_from_the_build_it_found(tmp_path, monkeypatch):
    _starts(tmp_path / "cons")
    found = {"wales_sha256": "b" * 64, "wales_svnrev": "r123456789abcdef"}
    monkeypatch.setattr("optbench.gmin_lbfgs.build", lambda: dict(found))
    monkeypatch.setattr("optbench.gmin_lbfgs.library_path", lambda: None)
    monkeypatch.setattr("optbench.gmin_lbfgs._LIBRARY", None)
    template = _template("gmin")
    template["solver"] = {"family": "gmin", "m": 4}
    _run(tmp_path, template)
    output = tmp_path / "out" / "bench.dat"
    rows = _rows(output.with_suffix(".starts.jsonl"))
    document = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [f"# wales_sha256 {'b' * 64}", "# wales_svnrev r123456789abcdef"]
    assert all(row["wales_sha256"] == "b" * 64 and row["wales_svnrev"] == found["wales_svnrev"] for row in rows)
    run = document["runs"][0]
    assert (run["code_version"], run["commit"]) == ("r123456789ab", "r123456789abcdef")
    assert run["parameters"]["wales_svnrev"] == "r123456789abcdef"
    assert run["parameters"]["m"] == 4


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda p, t: t.update(engine="eon"), ValueError, "is not a noise template"),
        (lambda p, t: t.update(problem="minimization/lj38"), ValueError, "is for minimization/lj38, not minimization/lj38-noise"),
        (lambda p, t: (t.pop("problem"), p.update(id="minimization/lj38")), ValueError, "does not run on minimization/lj38"),
        (
            lambda p, t: (t["solver"].update(family="rgmin", method="newton"), t["noise"].update(sigma=1.0e-3)),
            ValueError,
            "asks for newton at sigma 0.001",
        ),
        (lambda p, t: p["convergence"].pop("energy_tol"), ValueError, "has no energy_tol"),
    ],
)
def test_a_template_that_does_not_belong_to_the_problem_is_refused_before_any_start_runs(
    tmp_path, change, error, message
):
    problem, template = _problem(), _template()
    change(problem, template)
    with pytest.raises(error, match=message):
        _run(tmp_path, template, problem)
    assert not (tmp_path / "out").exists()


def test_a_template_for_a_different_rgmin_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rgmin", types.SimpleNamespace(__version__="2.0"))
    template = _template("rgmin", method="lbfgs")
    template["code_version"] = "1.0"
    with pytest.raises(RuntimeError, match="rgmin 2.0 does not match template 1.0"):
        _run(tmp_path, template)


def test_a_template_for_a_different_geometric_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "geometric", types.SimpleNamespace(__version__="1.1", __git_revision__="abc"))
    template = _template("geometric")
    template["code_version"] = "1.0"
    with pytest.raises(RuntimeError, match="geometric 1.1 does not match template 1.0"):
        _run(tmp_path, template)
    template["code_version"] = "1.1"
    template["commit"] = "def"
    with pytest.raises(RuntimeError, match="geometric revision abc does not match template def"):
        _run(tmp_path, template)


def test_a_template_for_a_different_pele_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    template = _template("pele")
    template["code_version"] = "1.0.0"
    with pytest.raises(RuntimeError, match="pele 9.9.9 does not match template 1.0.0"):
        _run(tmp_path, template)


def test_a_vcsqnm_template_must_name_the_pinned_commit_and_a_fixed_cell_method(tmp_path, monkeypatch):
    from optbench.vcsqnm_catalog import CODE_VERSION, COMMIT

    monkeypatch.setattr("optbench.runners.vcsqnm_min.assert_pin", lambda: None)
    template = _template("vcsqnm", method="sqnm")
    with pytest.raises(RuntimeError, match="commit is not"):
        _run(tmp_path, template)
    template["commit"] = COMMIT
    with pytest.raises(RuntimeError, match="code_version is not tag"):
        _run(tmp_path, template)
    template["code_version"] = CODE_VERSION
    template["solver"]["method"] = "vc"
    with pytest.raises(ValueError, match="moves a cell this potential does not have"):
        _run(tmp_path, template)


def test_a_template_for_the_installed_rgmin_is_accepted_and_the_run_goes_on(tmp_path, monkeypatch):
    double = _rgmin_double()
    monkeypatch.setitem(sys.modules, "rgmin", double)
    _starts(tmp_path / "cons")
    template = _template("rgmin", method="lbfgs")
    template["code_version"] = double.__version__
    summary = _run(tmp_path, template)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert summary["starts"] == 3
    assert all(row["rgmin_stop"] for row in rows)


def test_a_template_for_the_installed_pele_is_accepted_and_the_run_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    monkeypatch.setattr(
        "optbench.runners.pele_min.drive_minimizer",
        lambda method, coords, potential, *, stop, cap, options: (_tetrahedron(), True, ""),
    )
    _starts(tmp_path / "cons")
    template = _template("pele", method="lbfgs")
    template["code_version"] = "9.9.9"
    summary = _run(tmp_path, template)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert summary["starts"] == 3
    assert [row["success"] for row in rows] == [True, True, True]


def test_a_vcsqnm_template_that_names_the_pin_and_a_fixed_cell_method_is_accepted_and_the_run_goes_on(
    tmp_path, monkeypatch
):
    from optbench.vcsqnm_catalog import CODE_VERSION, COMMIT

    monkeypatch.setattr("optbench.runners.vcsqnm_min.assert_pin", lambda: None)
    monkeypatch.setattr(
        "optbench.runners.vcsqnm_min.drive",
        _vcsqnm_double({"point": _tetrahedron(), "stopped": True, "error": ""}),
    )
    _starts(tmp_path / "cons")
    template = _template("vcsqnm", method="sqnm")
    template["commit"], template["code_version"] = COMMIT, CODE_VERSION
    summary = _run(tmp_path, template)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert summary["starts"] == 3
    assert [row["success"] for row in rows] == [True, True, True]


def test_a_failed_pin_check_stops_a_vcsqnm_run_before_any_start(tmp_path, monkeypatch):
    def refuse():
        raise RuntimeError("installed sqnm is not the pinned source")

    monkeypatch.setattr("optbench.runners.vcsqnm_min.assert_pin", refuse)
    with pytest.raises(RuntimeError, match="not the pinned source"):
        _run(tmp_path, _template("vcsqnm", method="sqnm"))
    assert not (tmp_path / "out").exists()


def test_a_template_seed_does_not_need_a_seed_in_the_problem(tmp_path):
    _starts(tmp_path / "cons")
    problem = _problem()
    problem["noise"] = {"sigma": [0.0]}
    summary = _run(tmp_path, _template(), problem)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert summary["starts"] == 3
    assert [row["seed"] for row in rows] == [SEED, SEED + 1, SEED + 2]


def test_a_run_with_no_seed_anywhere_is_refused(tmp_path):
    _starts(tmp_path / "cons")
    problem, template = _problem(), _template()
    problem["noise"] = {"sigma": [0.0]}
    template["noise"] = {"sigma": 0.0}
    with pytest.raises(KeyError, match="seed"):
        _run(tmp_path, template, problem)
    assert not (tmp_path / "out").exists()


def test_the_problem_seed_is_used_when_the_template_has_none(tmp_path):
    _starts(tmp_path / "cons")
    template = _template()
    template["noise"] = {"sigma": 0.0}
    _run(tmp_path, template)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert [row["seed"] for row in rows] == [SEED, SEED + 1, SEED + 2]


def test_a_template_seed_wins_over_the_problem_seed(tmp_path):
    _starts(tmp_path / "cons")
    template = _template()
    template["noise"] = {"sigma": 0.0, "seed": 5}
    _run(tmp_path, template)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert [row["seed"] for row in rows] == [5, 6, 7]


def test_a_minimum_with_sella_matches_the_closed_form():
    pytest.importorskip("sella")
    record = noise_min._minimize_job(_job(family="sella", method="", options={"order": 0}))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(-6.0, abs=1e-9)
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx(_spectrum(R_MIN), rel=1e-4)


def test_a_minimum_with_geometric_matches_the_closed_form():
    pytest.importorskip("geometric")
    record = noise_min._minimize_job(_job(family="geometric", method="cart", cap=500))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(-6.0, abs=1e-8)
    assert record["hessian_index"] == 0
    assert record["coordinate_system"]


@pytest.mark.parametrize("method", ["sqnm", "ase"])
def test_a_minimum_with_vcsqnm_matches_the_closed_form(method):
    pytest.importorskip("sqnm.sqnm")
    options = {
        "initial_step_size": -1.0e-7,
        "nhist_max": 10,
        "alpha_min": 1.0e-3,
        "eps_subsp": 1.0e-5,
        "lattice_weight": 2.0,
    }
    record = noise_min._minimize_job(_job(family="vcsqnm", method=method, options=options, cap=500))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(-6.0, abs=1e-8)
    assert record["hessian_index"] == 0
