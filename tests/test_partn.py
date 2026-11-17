"""The pARTn runner on a free atom between two fixed atoms.

``libartn.so`` and its ``pypARTn`` interface are not in the lockfile and are
not built here. The search loop runs against a stand-in that has the names and
call signatures the loop touches and returns the displacement each test
scripts, so the loop, the harness count, the record and the stop rules are
tested without the library. The potential is the harness Lennard-Jones
``u(r) = 4 (r**-12 - r**-6)``, and the numbers to check come from closed
forms. A free atom midway between two fixed atoms ``A`` away has no force by
symmetry. Its curvature along the line is ``2 u''(A)`` and across it
``2 u'(A) / A``, so at ``A = 1.5`` it is a stationary point of index 1.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import tempfile
import types
import warnings
from ctypes import byref, c_char_p
from pathlib import Path

import numpy as np
import pytest

from optbench.counter import ForceCallCap
from optbench.engines import partn
from optbench.pes import Pair
from optbench.v2 import HOST_LABEL_ENV

ROOT = Path(__file__).resolve().parents[1]
A = 1.5
COMMIT = "9b753517c6c0eccae51425b83eb4e04ef554550b"
LANC = 4
FREE = [False, True, False]


def _u(r: float) -> float:
    return 4.0 * (r**-12 - r**-6)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


def _line(shift: float = 0.0) -> list[list[float]]:
    return [[-A, 0.0, 0.0], [shift, 0.0, 0.0], [A, 0.0, 0.0]]


def _saddle_energy() -> float:
    return 2.0 * _u(A) + _u(2.0 * A)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    # Setting then deleting makes the teardown restore the caller's value after
    # the code under test has set its own default.
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("OMP_NUM_THREADS")
    monkeypatch.delenv("ARTN_LIB", raising=False)
    monkeypatch.delenv("ARTN_INTERFACE", raising=False)
    monkeypatch.delitem(sys.modules, "pypARTn", raising=False)


@pytest.fixture(autouse=True)
def _watchdog():
    def expired(_signum, _frame):
        raise TimeoutError("a pARTn test ran for 120 s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _pair(shift: float = 0.1, product=None, name: str = "line", free=None) -> Pair:
    return Pair(
        name,
        np.asarray(_line(shift)),
        np.asarray(_line(0.0) if product is None else product),
        np.asarray(FREE if free is None else free),
        None,
    )


def _settings(**changes) -> dict:
    settings = {"potential": "lj", "cap": 50, "force_norm": 1.0e-6}
    settings.update(changes)
    return settings


def _template(**section) -> dict:
    template = {"id": "partn/test", "commit": COMMIT}
    if section:
        template["partn"] = section
    return template


def _job(shift: float = 0.1, **changes) -> dict:
    job = partn._job(_pair(shift), _settings(), _template(), "0")
    job.update(changes)
    return job


def _install(
    monkeypatch,
    plan,
    *,
    version=(1, 1, 0),
    gitinfo=f"pARTn 1.1.0 {COMMIT[:7]} DEVEL",
    refuse=(),
    flags=None,
    error=(0, ""),
    eigenvec=None,
    codes=None,
    destroy_fails=False,
    create_fails=None,
) -> types.SimpleNamespace:
    """A stand-in ``pypARTn`` whose ``next_displ`` is ``plan(index, record)``."""
    seen = types.SimpleNamespace(settings={}, fire={}, calls=[], created=[], destroyed=0, directories=[])
    tallies = {"has_error": False, "has_sad": True, "nevalf": 5}
    tallies.update(flags or {})

    def artn_gitinfo(holder):
        if gitinfo:
            holder._obj.value = gitinfo.encode()

    class Artn:
        def __init__(self, engine, shlib):
            seen.created.append((engine, shlib))
            seen.directories.append(os.getcwd())
            if create_fails is not None:
                raise create_fails
            self.lib = types.SimpleNamespace(artn_gitinfo=artn_gitinfo)

        def version(self):
            return version

        def set(self, name, value):
            if name in refuse:
                raise ValueError(f"{name} is not a parameter")
            seen.settings[name] = value

        def fire_set(self, name, value):
            seen.fire[name] = value

        def get_runparam(self, name):
            if name == "LANC":
                return LANC
            if name == "eigenvec":
                if eigenvec is None:
                    raise KeyError(name)
                return eigenvec(len(seen.calls) - 1)
            raise KeyError(name)

        def next_displ(self, nat, energy, force, typ, pos, cell, if_pos):
            record = {
                "nat": nat,
                "energy": energy,
                "force": np.array(force, copy=True),
                "typ": np.array(typ, copy=True),
                "pos": np.array(pos, copy=True),
                "cell": np.array(cell, copy=True),
                "if_pos": np.array(if_pos, copy=True),
            }
            seen.calls.append(record)
            return plan(len(seen.calls) - 1, record)

        def disp_code(self):
            return LANC if codes is None else codes(len(seen.calls) - 1)

        def extract(self, name):
            if name not in tallies:
                raise KeyError(name)
            value = tallies[name]
            if isinstance(value, Exception):
                raise value
            return value

        def get_error(self):
            return error

        def destroy(self):
            seen.destroyed += 1
            if destroy_fails:
                raise OSError("library already closed")

    module = types.ModuleType("pypARTn")
    module.artn = Artn
    monkeypatch.setitem(sys.modules, "pypARTn", module)
    return seen


def _to_the_saddle(index, record):
    """Walk the free atom to the middle, then say it has converged."""
    if index == 0:
        junk = np.full((3, 3), 0.5)
        junk[1] = [-record["pos"][1][0], 0.0, 0.0]
        return junk, False
    return np.zeros((3, 3)), True


def test_a_search_that_reaches_the_midpoint_is_a_success_with_the_closed_form_energy_and_index(monkeypatch):
    seen = _install(monkeypatch, _to_the_saddle)
    record = partn._execute(_job())
    assert (record["success"], record["failure"], record["reasons"]) == (True, None, [])
    assert record["energy"] == pytest.approx(_saddle_energy(), abs=1e-12)
    assert record["force_norm"] == pytest.approx(0.0, abs=1e-12)
    assert record["hessian_index"] == 1 and record["expected_index"] == 1
    assert (record["force_calls"], record["converged"]) == (2, True)
    assert (record["start"], record["schema"], record["counter"]) == ("line", "optbench.start/v2", "harness")
    assert record["lconv"] is True and record["artn_has_sad"] is True and record["artn_nevalf"] == 5
    assert record["seed"] == "0" and record["zseed"] == _job()["zseed"]
    assert "error" not in record and "positions" not in record
    assert len(seen.calls) == 2 and seen.destroyed == 1


def test_the_library_is_given_the_force_the_positions_and_the_cell_of_the_search(monkeypatch):
    seen = _install(monkeypatch, _to_the_saddle)
    partn._execute(_job(shift=0.1))
    first = seen.calls[0]
    assert first["nat"] == 3
    assert first["energy"] == pytest.approx(_u(A - 0.1) + _u(A + 0.1) + _u(2.0 * A), abs=1e-12)
    assert first["force"][1] == pytest.approx([_slope(A - 0.1) - _slope(A + 0.1), 0.0, 0.0], abs=1e-12)
    assert first["force"].sum(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert first["pos"] == pytest.approx(np.asarray(_line(0.1)))
    assert first["typ"].tolist() == [1, 1, 1] and first["typ"].dtype == np.int32
    assert first["if_pos"].tolist() == [[0, 0, 0], [1, 1, 1], [0, 0, 0]] and first["if_pos"].dtype == np.int32
    assert np.diag(first["cell"]).min() >= 80.0
    assert np.count_nonzero(first["cell"]) == 3
    assert seen.calls[1]["pos"] == pytest.approx(np.asarray(_line(0.0)), abs=1e-15)


def test_fixed_atoms_stay_where_they_were_whatever_the_library_asks_of_them(monkeypatch):
    seen = _install(monkeypatch, _to_the_saddle)
    record = partn._execute(_job(keep_positions=True))
    assert np.asarray(record["positions"]).reshape(3, 3) == pytest.approx(np.asarray(_line(0.0)), abs=1e-15)
    assert seen.calls[1]["pos"][0].tolist() == [-A, 0.0, 0.0] and seen.calls[1]["pos"][2].tolist() == [A, 0.0, 0.0]


def test_the_settings_of_the_job_are_handed_to_the_library_before_the_first_step(monkeypatch):
    seen = _install(monkeypatch, _to_the_saddle)
    job = partn._job(_pair(), _settings(cap=40, force_norm=1.0e-5), _template(), "7")
    partn._execute(job)
    assert seen.created == [("other", job["shlib"])]
    assert seen.settings["engine_units"] == "lammps/lj"
    assert seen.settings["forc_thr"] == 1.0e-5 and seen.settings["nevalf_max"] == 40
    assert seen.settings["zseed"] == job["zseed"] and isinstance(seen.settings["zseed"], int)
    assert (seen.settings["ninit"], seen.settings["lanczos_max_size"]) == (3, 16)
    assert (seen.settings["push_step_size"], seen.settings["eigen_step_size"]) == (0.1, 0.15)
    assert (seen.settings["converge_property"], seen.settings["lpush_final"]) == ("norm", False)
    assert (seen.settings["verbose"], seen.settings["struc_format_out"]) == (0, "none")
    assert "lanczos_disp" not in seen.settings and "etot_diff_limit" not in seen.settings
    assert seen.fire == {"dt_init": 0.05}
    push = seen.settings["push_init"]
    assert push.shape == (3, 3) and push.dtype == np.float64
    assert push.tolist() == [[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_optional_settings_reach_the_library_only_when_the_template_states_them(monkeypatch):
    seen = _install(monkeypatch, _to_the_saddle)
    template = _template(lanczos_disp=0.001, etot_diff_limit=1.0e6, fire_dt_init=0.2, lpush_final=True)
    partn._execute(partn._job(_pair(), _settings(), template, "0"))
    assert seen.settings["lanczos_disp"] == 0.001 and seen.settings["etot_diff_limit"] == 1.0e6
    assert seen.settings["lpush_final"] is True
    assert seen.fire == {"dt_init": 0.2}
    seen = _install(monkeypatch, _to_the_saddle)
    record = partn._execute(partn._job(_pair(), _settings(potential="morse_pt"), _template(), "0"))
    assert seen.fire == {} and seen.settings["engine_units"] == "lammps/metal" and "error" not in record


def test_the_first_push_goes_from_the_reactant_toward_the_product_on_the_atom_that_moves_most(monkeypatch):
    seen = _install(monkeypatch, _to_the_saddle)
    product = [[-A, 0.0, 0.0], [0.1, 0.5, 0.0], [A, 0.0, 0.0]]
    partn._execute(partn._job(_pair(0.1, product=product), _settings(), _template(), "0"))
    assert seen.settings["push_init"].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]]
    seen = _install(monkeypatch, _to_the_saddle)
    job = _job()
    job["product"] = None
    partn._execute(job)
    assert seen.settings["push_init"].tolist() == [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_a_job_with_a_box_gives_the_library_that_cell(monkeypatch):
    seen = _install(monkeypatch, lambda index, record: (np.zeros((3, 3)), True))
    partn._execute(_job(box=[30.0, 40.0, 50.0]))
    assert np.diag(seen.calls[0]["cell"]).tolist() == [30.0, 40.0, 50.0]


def test_a_repeated_geometry_is_not_charged_again_and_a_moved_one_is(monkeypatch):
    def plan(index, record):
        if index < 2:
            return np.zeros((3, 3)), False
        if index == 2:
            return np.array([[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]), False
        return np.zeros((3, 3)), True

    seen = _install(monkeypatch, plan)
    record = partn._execute(_job())
    assert len(seen.calls) == 4
    assert record["force_calls"] == 2
    assert record["success"] is True


def test_three_zero_displacements_in_a_row_are_a_crash(monkeypatch):
    seen = _install(monkeypatch, lambda index, record: (np.zeros((3, 3)), False))
    record = partn._execute(_job())
    assert (record["failure"], record["error"]) == ("crash", "RuntimeError: pARTn displacement stayed at zero")
    assert len(seen.calls) == 3 and record["force_calls"] == 1 and seen.destroyed == 1


def test_a_displacement_that_is_not_finite_is_a_crash(monkeypatch):
    bad = np.zeros((3, 3))
    bad[1, 1] = math.nan
    _install(monkeypatch, lambda index, record: (bad, False))
    record = partn._execute(_job())
    assert (record["failure"], record["error"]) == ("crash", "RuntimeError: pARTn returned a non-finite displacement")


def test_a_search_that_never_converges_stops_at_the_budget_and_is_a_cap_failure(monkeypatch):
    def plan(index, record):
        return np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]), False

    seen = _install(monkeypatch, plan)
    record = partn._execute(_job(cap=4))
    assert len(seen.calls) == 4 and record["force_calls"] == 4
    assert (record["failure"], record["reasons"], record["success"]) == ("cap", ["cap"], False)
    assert record["hessian_index"] is None and "error" not in record


def test_the_position_a_last_step_reached_is_evaluated_and_charged_once_when_the_budget_allows(monkeypatch):
    moves = {2: -0.1, 5: 0.05}

    def plan(index, record):
        step = np.zeros((3, 3))
        step[1, 0] = moves.get(index, 0.0)
        return step, False

    seen = _install(monkeypatch, plan)
    record = partn._execute(_job(cap=4))
    assert len(seen.calls) == 6
    assert record["force_calls"] == 3
    assert record["energy"] == pytest.approx(_u(A - 0.05) + _u(A + 0.05) + _u(2.0 * A), abs=1e-12)
    assert (record["failure"], record["reasons"]) == ("divergence", ["force_norm"])


def test_reaching_the_saddle_on_the_last_permitted_call_is_a_cap_failure(monkeypatch):
    _install(monkeypatch, lambda index, record: (np.zeros((3, 3)), True))
    record = partn._execute(_job(shift=0.0, cap=1))
    assert record["force_calls"] == 1
    assert (record["failure"], record["reasons"], record["converged"]) == ("cap", ["cap"], False)
    assert "error" not in record
    assert partn._execute(_job(shift=0.0, cap=2))["success"] is True


def test_an_error_the_library_reports_is_a_crash_with_its_code_and_message(monkeypatch):
    _install(monkeypatch, _to_the_saddle, flags={"has_error": True}, error=(7, "bad input"))
    record = partn._execute(_job())
    assert (record["failure"], record["error"]) == ("crash", "RuntimeError: pARTn error 7: bad input")
    assert record["hessian_index"] is None


def test_tallies_the_library_cannot_give_are_left_out_of_the_record_from_the_first_missing_one(monkeypatch):
    _install(monkeypatch, _to_the_saddle, flags={"nevalf": KeyError("nevalf")})
    record = partn._execute(_job())
    assert record["success"] is True
    assert record["artn_has_sad"] is True and record["artn_nevalf"] is None
    _install(monkeypatch, _to_the_saddle, flags={"has_sad": KeyError("has_sad")})
    record = partn._execute(_job())
    assert record["success"] is True
    assert record["artn_has_sad"] is None and record["artn_nevalf"] is None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"version": (1, 0, 9)}, "RuntimeError: libartn version is (1, 0, 9), the pin is 1.1.0"),
        ({"gitinfo": "pARTn 1.1.0 0000000 DEVEL"}, "does not contain 9b75351"),
        ({"gitinfo": ""}, "libartn gitinfo '' does not contain 9b75351"),
        ({"refuse": ("forc_thr",)}, "RuntimeError: pARTn set forc_thr failed: forc_thr is not a parameter"),
    ],
)
def test_a_library_that_is_not_the_pinned_build_or_refuses_a_setting_is_a_crash_that_runs_no_step(
    monkeypatch, options, message
):
    seen = _install(monkeypatch, _to_the_saddle, **options)
    record = partn._execute(_job())
    assert record["failure"] == "crash" and message in record["error"]
    assert record["force_calls"] == 0 and seen.calls == []
    assert record["energy"] is None and record["hessian_index"] is None


def test_the_directory_and_the_library_are_restored_after_a_success_and_after_a_crash(monkeypatch):
    before = os.getcwd()
    seen = _install(monkeypatch, _to_the_saddle)
    partn._execute(_job())
    assert os.getcwd() == before and seen.destroyed == 1
    assert Path(seen.directories[0]) != Path(before)
    seen = _install(monkeypatch, _to_the_saddle, version=(0, 0, 1), destroy_fails=True)
    record = partn._execute(_job())
    assert record["failure"] == "crash"
    assert os.getcwd() == before and seen.destroyed == 1


def test_a_potential_the_library_has_no_units_for_is_a_crash_naming_it(monkeypatch):
    _install(monkeypatch, _to_the_saddle)
    record = partn._execute(_job(kind="exotic"))
    assert (record["failure"], record["error"]) == ("crash", "ValueError: exotic has no pARTn engine_units")


def test_the_threads_of_the_numerical_libraries_default_to_one(monkeypatch):
    _install(monkeypatch, _to_the_saddle)
    partn._execute(_job())
    assert os.environ["OMP_NUM_THREADS"] == "1"
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    partn._execute(_job())
    assert os.environ["OMP_NUM_THREADS"] == "6"


def test_the_interface_directory_is_put_first_on_the_import_path_once(monkeypatch, tmp_path):
    _install(monkeypatch, _to_the_saddle)
    interface = str(tmp_path / "interface")
    partn._execute(_job(interface=interface))
    partn._execute(_job(interface=interface))
    assert sys.path[0] == interface and sys.path.count(interface) == 1


def test_a_start_the_library_stops_away_from_a_stationary_point_is_a_force_failure(monkeypatch):
    _install(monkeypatch, lambda index, record: (np.zeros((3, 3)), True))
    record = partn._execute(_job(shift=0.1))
    assert (record["failure"], record["reasons"], record["converged"]) == ("divergence", ["force_norm"], False)
    assert record["force_norm"] == pytest.approx(abs(_slope(A - 0.1) - _slope(A + 0.1)), abs=1e-12)
    assert record["hessian_index"] is None


def test_a_midpoint_between_two_atoms_too_close_to_be_a_saddle_is_the_wrong_stationary_point(monkeypatch):
    close = 1.0
    assert _curvature(close) > 0.0 and _slope(close) < 0.0  # stiff along the line, unstable across it
    _install(monkeypatch, lambda index, record: (np.zeros((3, 3)), True))
    positions = [[-close, 0.0, 0.0], [0.0, 0.0, 0.0], [close, 0.0, 0.0]]
    pair = Pair("tight", np.asarray(positions), np.asarray(positions), np.asarray(FREE), None)
    record = partn._execute(partn._job(pair, _settings(), _template(), "0"))
    assert (record["failure"], record["reasons"]) == ("wrong_stationary", ["hessian_index"])
    assert record["hessian_index"] == 2


def test_the_library_and_the_interface_come_from_the_environment_or_a_default_name(monkeypatch, tmp_path):
    assert partn.library_path().name == "libartn.so"
    assert partn.interface_dir().name == "interface"
    monkeypatch.setenv("ARTN_LIB", str(tmp_path / "build" / "libartn.so"))
    monkeypatch.setenv("ARTN_INTERFACE", str(tmp_path / "pointer"))
    assert partn.library_path() == tmp_path / "build" / "libartn.so"
    assert partn.interface_dir() == tmp_path / "pointer"
    monkeypatch.setenv("ARTN_LIB", "")
    assert partn.library_path().name == "libartn.so" and partn.library_path() != Path("")


def test_the_runner_says_what_is_missing_and_where_to_set_it(monkeypatch, tmp_path):
    library = tmp_path / "build" / "libartn.so"
    interface = tmp_path / "pointer"
    monkeypatch.setenv("ARTN_LIB", str(library))
    monkeypatch.setenv("ARTN_INTERFACE", str(interface))
    assert partn.available() == f"libartn.so is not at {library}. Set ARTN_LIB to the built library."
    library.parent.mkdir()
    library.write_bytes(b"")
    assert partn.available() == (
        f"pypARTn.py is not at {interface / 'pypARTn.py'}. Set ARTN_INTERFACE to the interface directory."
    )
    interface.mkdir()
    (interface / "pypARTn.py").write_text("", encoding="utf-8")
    assert partn.available() is None


def test_units_follow_the_potential_and_any_other_potential_has_none():
    assert partn._units("morse_pt") == "lammps/metal"
    assert partn._units("lj") == "lammps/lj"
    with pytest.raises(ValueError, match="mystery has no pARTn engine_units"):
        partn._units("mystery")


def test_the_section_fills_what_the_template_leaves_out_and_keeps_what_it_states():
    assert partn._section({}) == {
        "push_step_size": 0.1,
        "eigen_step_size": 0.15,
        "ninit": 3,
        "lanczos_max_size": 16,
        "lanczos_disp": None,
        "etot_diff_limit": None,
        "lpush_final": False,
        "converge_property": "norm",
    }
    template = {"partn": {"ninit": 0, "converge_property": "force", "extra": 1}}
    section = partn._section(template)
    assert (section["ninit"], section["converge_property"], section["extra"]) == (0, "force", 1)
    assert section["lanczos_max_size"] == 16
    assert template == {"partn": {"ninit": 0, "converge_property": "force", "extra": 1}}


def test_the_fire_step_is_the_one_the_template_states_else_a_small_one_on_lennard_jones_only():
    assert partn._fire_dt({}, "lj") == 0.05
    assert partn._fire_dt({}, "morse_pt") is None
    assert partn._fire_dt({"fire_dt_init": 0.2}, "morse_pt") == 0.2
    assert partn._fire_dt({"fire_dt_init": 3}, "lj") == 3.0 and isinstance(partn._fire_dt({"fire_dt_init": 3}, "lj"), float)
    assert partn._fire_dt({"fire_dt_init": None}, "lj") == 0.05
    assert partn._fire_dt({"fire_dt_init": 0.0}, "lj") == 0.0


def test_a_pair_becomes_a_job_of_plain_lists_with_the_seed_of_its_name(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTN_LIB", str(tmp_path / "libartn.so"))
    monkeypatch.setenv("ARTN_INTERFACE", str(tmp_path / "pointer"))
    box = np.diag([30.0, 40.0, 50.0])
    pair = Pair("line", np.asarray(_line(0.1)), np.asarray(_line(0.0)), np.asarray(FREE), box)
    job = partn._job(pair, _settings(cap=77, force_norm=0.003), _template(ninit=1, lanczos_disp=0.002, converge_property="maxval"), "5")
    assert job["name"] == "line" and job["kind"] == "lj" and (job["cap"], job["force_norm"]) == (77, 0.003)
    assert job["reactant"] == _line(0.1) and job["product"] == _line(0.0) and job["free"] == FREE
    assert job["box"] == [30.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 50.0]
    assert (job["ninit"], job["lanczos_disp"], job["etot_diff_limit"]) == (1, 0.002, None)
    assert job["converge_property"] == "maxval"
    assert (job["task"], job["reference"], job["overlap"]) == ("saddle", None, 0.99)
    assert (job["seed"], job["commit"]) == ("5", COMMIT)
    assert job["shlib"] == str(tmp_path / "libartn.so") and job["interface"] == str(tmp_path / "pointer")
    assert json.loads(json.dumps(job)) == job
    zseeds = {
        partn._job(Pair(name, pair.reactant, pair.product, pair.free, None), _settings(), _template(), seed)["zseed"]
        for name in ("a", "b")
        for seed in ("0", "1")
    }
    assert len(zseeds) == 4 and all(1 <= value <= 2147483646 for value in zseeds)
    again = partn._job(Pair("a", pair.reactant, pair.product, pair.free, None), _settings(), _template(), "0")
    assert again["zseed"] == partn._job(Pair("a", pair.reactant, pair.product, pair.free, None), _settings(), _template(), "0")["zseed"]
    assert partn._job(Pair("a", pair.reactant, pair.product, pair.free, None), _settings(), _template(), "0")["box"] is None


def test_a_sample_geometry_becomes_a_job_whose_seed_is_the_one_a_pair_of_that_name_gets():
    start = {"name": "line", "positions": _line(0.1), "fixed": [True, False, True], "box": None, "reference": None}
    job = partn._point_job(start, {**_settings(), "task": "saddle", "overlap": 0.5}, _template())
    assert (job["task"], job["product"], job["free"], job["overlap"], job["seed"]) == ("saddle", None, FREE, 0.5, "0")
    assert job["reactant"] == _line(0.1) and job["reference"] is None
    assert job["zseed"] == partn._job(_pair(), _settings(), _template(), "0")["zseed"]
    mode = partn._point_job({**start, "reference": [1.0, 0.0]}, {**_settings(), "task": "mode", "overlap": 0.9}, _template())
    assert (mode["task"], mode["reference"], mode["overlap"]) == ("mode", [1.0, 0.0], 0.9)
    assert partn._point_job(start, _settings() | {"overlap": 0.99}, _template())["task"] == "saddle"


def test_the_free_block_is_put_back_into_a_copy_of_the_saddle():
    saddle = np.arange(9.0).reshape(3, 3)
    free = np.asarray(FREE)
    placed = partn._placed(saddle, free, np.array([7.0, 8.0, 9.0]))
    assert placed.tolist() == [[0.0, 1.0, 2.0], [7.0, 8.0, 9.0], [6.0, 7.0, 8.0]]
    assert saddle[1].tolist() == [3.0, 4.0, 5.0]


def test_the_git_information_is_read_through_a_pointer_to_a_string():
    from ctypes import POINTER

    def artn_gitinfo(holder):
        holder._obj.value = b"pARTn 1.1.0 abc1234"

    art = types.SimpleNamespace(lib=types.SimpleNamespace(artn_gitinfo=artn_gitinfo))
    assert partn._gitinfo(art) == "pARTn 1.1.0 abc1234"
    assert artn_gitinfo.restype is None and artn_gitinfo.argtypes == [POINTER(c_char_p)]
    assert partn._gitinfo(types.SimpleNamespace(lib=types.SimpleNamespace(artn_gitinfo=lambda holder: None))) == ""
    assert byref(c_char_p())._obj.value is None


def test_a_setting_that_fails_names_itself_and_keeps_the_cause():
    class Art:
        def set(self, name, value):
            raise ValueError("no such parameter")

    with pytest.raises(RuntimeError, match="pARTn set ninit failed: no such parameter") as caught:
        partn._set(Art(), "ninit", 3)
    assert isinstance(caught.value.__cause__, ValueError)


def _eigenvec_art(vector=None, error=None):
    def get_runparam(name):
        assert name == "eigenvec"
        if error is not None:
            raise error
        return vector

    return types.SimpleNamespace(get_runparam=get_runparam)


def _mode_with(angle: float) -> tuple[list[float], np.ndarray]:
    """A reference along x on the first free atom, and a mode ``angle`` away from it.

    The mode has ``cos(angle)`` along x on that atom and ``sin(angle)`` along
    x on the second free atom, which the reference does not use. Read the
    other way round, by component, it has no weight on the free atoms.
    """
    full = np.array([[0.3, 0.0, 0.0], [math.cos(angle), 0.0, 0.0], [math.sin(angle), 0.0, 0.0]])
    return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], full


FREE_PAIR = np.asarray([False, True, True])


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.2, math.pi / 2.0 - 1e-3])
def test_the_overlap_of_a_mode_with_the_reference_is_the_cosine_of_the_angle_between_them(angle):
    reference, full = _mode_with(angle)
    assert partn._mode_overlap(_eigenvec_art(full.reshape(-1)), reference, FREE_PAIR) == pytest.approx(math.cos(angle), abs=1e-12)
    assert partn._mode_overlap(_eigenvec_art(-full.reshape(-1)), reference, FREE_PAIR) == pytest.approx(math.cos(angle), abs=1e-12)


def test_a_mode_stored_by_component_is_read_as_well_as_one_stored_by_atom():
    reference, full = _mode_with(0.7)
    by_component = full.T.reshape(-1)
    assert partn._mode_overlap(_eigenvec_art(by_component), reference, FREE_PAIR) == pytest.approx(math.cos(0.7), abs=1e-12)
    assert partn._mode_overlap(_eigenvec_art(list(full.reshape(-1))), reference, FREE_PAIR) == pytest.approx(math.cos(0.7), abs=1e-12)


def test_a_vector_that_can_be_read_either_way_gives_the_larger_of_the_two_overlaps():
    reference = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    by_atom = 1.0 / math.sqrt(2.0)
    for a, b, expected in ((3.0, 4.0, by_atom), (4.0, 3.0, 0.8)):
        vector = np.array([0.0, a, b, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        assert partn._mode_overlap(_eigenvec_art(vector), reference, FREE_PAIR) == pytest.approx(expected, abs=1e-12)


def test_no_overlap_is_reported_without_a_reference_or_a_usable_mode():
    reference, full = _mode_with(0.0)
    flat = full.reshape(-1)
    assert partn._mode_overlap(_eigenvec_art(flat), None, FREE_PAIR) is None
    assert partn._mode_overlap(_eigenvec_art(error=RuntimeError("not computed yet")), reference, FREE_PAIR) is None
    assert partn._mode_overlap(_eigenvec_art([]), reference, FREE_PAIR) is None
    assert partn._mode_overlap(_eigenvec_art(np.where(np.arange(9) == 4, math.nan, flat)), reference, FREE_PAIR) is None
    assert partn._mode_overlap(_eigenvec_art(np.where(np.arange(9) == 4, math.inf, flat)), reference, FREE_PAIR) is None
    assert partn._mode_overlap(_eigenvec_art(flat[:-1]), reference, FREE_PAIR) is None


def test_a_mode_with_no_weight_on_the_free_atoms_or_a_reference_of_the_wrong_size_has_overlap_zero():
    reference, full = _mode_with(0.0)
    on_the_fixed_atom = np.zeros(9)
    on_the_fixed_atom[0] = 1.0
    assert partn._mode_overlap(_eigenvec_art(on_the_fixed_atom), reference, FREE_PAIR) == 0.0
    assert partn._mode_overlap(_eigenvec_art(full.reshape(-1)), reference[:3], FREE_PAIR) == 0.0


def _mode_start() -> dict:
    return {
        "name": "mode-line",
        "positions": [[-A, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 1.4, 0.0]],
        "fixed": [True, False, False],
        "box": None,
        "reference": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _mode_job(**changes) -> dict:
    job = partn._point_job(_mode_start(), {**_settings(cap=20), "task": "mode", "overlap": 0.99}, _template())
    job.update(changes)
    return job


def _mode_install(monkeypatch, angles, **options):
    def eigenvec(index):
        return _mode_with(angles[min(index, len(angles) - 1)])[1].reshape(-1)

    return _install(monkeypatch, lambda index, record: (np.full((3, 3), 0.02), False), eigenvec=eigenvec, **options)


def test_a_lanczos_search_stops_as_soon_as_the_mode_overlaps_the_reference_by_the_threshold(monkeypatch):
    seen = _mode_install(monkeypatch, [1.2, 0.05])
    record = partn._execute(_mode_job())
    assert (record["success"], record["failure"], record["reasons"]) == (True, None, [])
    assert record["overlap"] == pytest.approx(math.cos(0.05), abs=1e-12)
    assert (record["overlap_target"], record["force_calls"], record["verify_calls"]) == (0.99, 2, 0)
    assert record["lconv"] is True and record["hessian_index"] is None and record["expected_index"] is None
    assert len(seen.calls) == 2 and record["error"] == ""
    assert (record["seed"], record["zseed"]) == ("0", _mode_job()["zseed"])


def test_the_overlap_kept_is_the_best_one_reached_even_when_the_search_leaves_lanczos_later(monkeypatch):
    _mode_install(monkeypatch, [0.5, 1.4], codes=lambda index: LANC if index < 1 else LANC + 1)
    record = partn._execute(_mode_job())
    assert (record["failure"], record["reasons"], record["lconv"]) == ("divergence", ["mode_overlap"], False)
    assert record["overlap"] == pytest.approx(math.cos(0.5), abs=1e-12)
    assert record["force_calls"] == 2


def test_a_search_that_leaves_lanczos_before_the_overlap_is_a_failure_after_one_call(monkeypatch):
    seen = _mode_install(monkeypatch, [1.2], codes=lambda index: LANC + 1)
    record = partn._execute(_mode_job())
    assert (record["failure"], record["force_calls"], len(seen.calls)) == ("divergence", 1, 1)
    assert record["overlap"] == pytest.approx(math.cos(1.2), abs=1e-12)


def test_the_overlap_at_the_threshold_counts_and_one_just_below_does_not(monkeypatch):
    angle = math.acos(0.99)
    _mode_install(monkeypatch, [angle])
    assert partn._execute(_mode_job(overlap=math.cos(angle)))["success"] is True
    _mode_install(monkeypatch, [angle], codes=lambda index: LANC + 1)
    assert partn._execute(_mode_job(overlap=math.cos(angle) + 1.0e-9))["success"] is False


def test_a_lanczos_search_that_never_reaches_the_overlap_stops_at_the_budget(monkeypatch):
    seen = _mode_install(monkeypatch, [1.2])
    record = partn._execute(_mode_job(cap=3))
    assert (record["failure"], record["reasons"], record["force_calls"]) == ("cap", ["cap"], 3)
    assert len(seen.calls) == 3


def test_a_lanczos_search_with_no_reference_has_no_overlap_and_runs_to_the_budget(monkeypatch):
    _mode_install(monkeypatch, [0.0])
    record = partn._execute(_mode_job(reference=None, cap=3))
    assert (record["failure"], record["overlap"], record["force_calls"]) == ("cap", 0.0, 3)


def test_a_lanczos_search_the_library_cannot_start_is_a_crash_with_its_error(monkeypatch):
    _mode_install(monkeypatch, [0.0], version=(2, 0, 0))
    record = partn._execute(_mode_job())
    assert (record["failure"], record["reasons"]) == ("crash", ["crash"])
    assert record["error"] == "RuntimeError: libartn version is (2, 0, 0), the pin is 1.1.0"
    assert record["force_calls"] == 0 and record["lconv"] is False


@pytest.fixture
def library_files(monkeypatch, tmp_path):
    library = tmp_path / "build" / "libartn.so"
    interface = tmp_path / "pointer"
    library.parent.mkdir()
    interface.mkdir()
    library.write_bytes(b"")
    (interface / "pypARTn.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("ARTN_LIB", str(library))
    monkeypatch.setenv("ARTN_INTERFACE", str(interface))
    return library, interface


def test_a_run_with_no_library_is_refused_with_the_reason_before_anything_is_planned(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTN_LIB", str(tmp_path / "missing" / "libartn.so"))
    with pytest.raises(RuntimeError, match="libartn.so is not at .*Set ARTN_LIB"):
        partn.plan(ROOT, "tsopt/LJ38", "partn/ts-lj38", tmp_path / "work")
    with pytest.raises(RuntimeError, match="libartn.so is not at"):
        partn.run(ROOT, "tsopt/LJ38", "partn/ts-lj38", output=tmp_path / "out" / "bench.dat")
    with pytest.raises(RuntimeError, match="libartn.so is not at"):
        partn.search_named(ROOT, "neb/Pt-heptamer", "partn/artn", "min01")
    assert not (tmp_path / "out").exists()


def test_the_transition_state_plan_has_one_saddle_search_per_guess_and_no_blind_push(library_files, tmp_path):
    planned = partn.plan(ROOT, "tsopt/LJ38", "partn/ts-lj38", tmp_path / "work")
    jobs = planned["jobs"]
    assert planned["execute"] is partn._execute
    assert planned["problem"]["id"] == "tsopt/LJ38" and planned["template"]["id"] == "partn/ts-lj38"
    assert planned["settings"]["task"] == "saddle" and planned["settings"]["potential"] == "lj"
    assert len(jobs) == 200 and jobs[0]["name"] == "0000" and jobs[-1]["name"] == "0199"
    first = jobs[0]
    assert (first["task"], first["kind"], first["ninit"], first["product"], first["box"]) == ("saddle", "lj", 0, None, None)
    assert len(first["reactant"]) == 38 and all(first["free"])
    section = planned["template"]["partn"]
    assert (first["lanczos_disp"], first["lanczos_max_size"], first["etot_diff_limit"]) == (
        section["lanczos_disp"],
        section["lanczos_max_size"],
        section["etot_diff_limit"],
    )
    assert (first["fire_dt"], first["eigen_step_size"], first["cap"]) == (
        partn._fire_dt(section, "lj"),
        section["eigen_step_size"],
        planned["settings"]["cap"],
    )
    assert first["force_norm"] == planned["settings"]["force_norm"] and first["reference"] is None
    assert len({job["zseed"] for job in jobs}) == len(jobs)
    assert json.loads(json.dumps(jobs)) == jobs


def test_the_island_transition_state_plan_fixes_the_slab_and_keeps_the_cell(library_files, tmp_path):
    planned = partn.plan(ROOT, "tsopt/pt-island", "partn/ts-pt", tmp_path / "work")
    first = planned["jobs"][0]
    assert len(planned["jobs"]) == 49 and first["name"] == "initial_0"
    assert first["kind"] == "morse_pt" and first["fire_dt"] is None and first["ninit"] == 0
    assert 0 < sum(first["free"]) < len(first["free"]) == len(first["reactant"])
    assert len(first["box"]) == 3 and min(first["box"]) > 0.0


def test_a_template_with_no_initial_pushes_stated_gets_none_on_a_transition_state_guess(library_files, tmp_path, monkeypatch):
    from optbench.campaign import load_template, resolve_problem

    template = load_template(ROOT, "partn/ts-lj38")
    del template["partn"]["ninit"]
    monkeypatch.setattr(partn, "_resolved", lambda root, problem_name, template_name: (resolve_problem(ROOT, "tsopt/LJ38"), template))
    assert partn.plan(ROOT, "tsopt/LJ38", "partn/ts-lj38", tmp_path / "work")["jobs"][0]["ninit"] == 0
    pairs = load_template(ROOT, "partn/artn")
    del pairs["partn"]["ninit"]
    monkeypatch.setattr(partn, "_resolved", lambda root, problem_name, template_name: (resolve_problem(ROOT, "neb/Pt-heptamer"), pairs))
    assert partn.plan(ROOT, "neb/Pt-heptamer", "partn/artn", tmp_path / "work")["jobs"][0]["ninit"] == 3


def test_the_lowest_mode_plan_carries_the_reference_mode_and_the_overlap_to_reach(library_files, tmp_path):
    planned = partn.plan(ROOT, "lowest-eigenvector/lj38", "partn/mode-lj38", tmp_path / "work")
    first = planned["jobs"][0]
    assert planned["settings"]["task"] == "mode" and first["task"] == "mode"
    assert len(planned["jobs"]) == 200 and first["overlap"] == planned["settings"]["overlap"] == 0.99
    assert len(first["reference"]) == 38 * 3
    assert np.linalg.norm(first["reference"]) == pytest.approx(1.0, abs=1e-12)


def test_the_pair_plan_has_one_search_per_pair_pushed_toward_its_product(library_files, tmp_path):
    planned = partn.plan(ROOT, "neb/Pt-heptamer", "partn/artn", tmp_path / "work")
    jobs = planned["jobs"]
    assert len(jobs) == 13 and "task" not in planned["settings"]
    first = jobs[0]
    assert first["name"] == "min01" and first["task"] == "saddle" and first["ninit"] == 3
    assert first["product"] is not None and first["product"] != first["reactant"]
    assert first["seed"] == "0" and first["kind"] == "morse_pt"
    assert len({job["zseed"] for job in jobs}) == len(jobs)


def test_a_plan_refuses_what_pairs_with_neither_the_template_nor_the_potential(library_files, tmp_path, monkeypatch):
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="partn/ts-lj38 belongs to tsopt/LJ38"):
        partn.plan(ROOT, "tsopt/pt-island", "partn/ts-lj38", work)
    with pytest.raises(ValueError, match="tsase/ci-neb code is not artn"):
        partn.plan(ROOT, "neb/Pt-heptamer", "tsase/ci-neb", work)
    with pytest.raises(RuntimeError, match="one catalog walk"):
        partn.plan(ROOT, "saddle-search/pt-island", "partn/saddle-pt", work)
    with pytest.raises(ValueError, match="minimization/lj38 has no band potential"):
        partn.plan(ROOT, "minimization/lj38", "partn/artn", work)
    from optbench.campaign import load_template, resolve_problem

    problem = resolve_problem(ROOT, "tsopt/LJ38")
    del problem["convergence"]["max_force_calls"]
    template = load_template(ROOT, "partn/ts-lj38")
    monkeypatch.setattr(partn, "_resolved", lambda root, problem_name, template_name: (problem, template))
    with pytest.raises(ValueError, match="tsopt/LJ38 has no force-call cap"):
        partn.plan(ROOT, "tsopt/LJ38", "partn/ts-lj38", work)


def _crashing_library(monkeypatch):
    return _install(monkeypatch, _to_the_saddle, version=(0, 0, 1))


def _template_of(template_name: str) -> dict:
    from optbench.campaign import load_template

    return load_template(ROOT, template_name)


def _pinned(monkeypatch, template_name: str, plan, **options):
    pin = _template_of(template_name)["commit"]
    return _install(monkeypatch, plan, gitinfo=f"pARTn 1.1.0 {pin[:7]} DEVEL", **options)


def test_a_named_search_carries_the_names_of_the_code_and_the_template_on_its_record(monkeypatch, library_files):
    _crashing_library(monkeypatch)
    record = partn.search_named(ROOT, "neb/Pt-heptamer", "partn/artn", "min01", seed="3")
    assert (record["problem"], record["template"], record["start"]) == ("neb/Pt-heptamer", "partn/artn", "min01")
    assert (record["code"], record["code_version"], record["algorithm"], record["tuning"]) == ("artn", "1.1.0", "ARTn", "default")
    assert record["commit"] == _template_of("partn/artn")["commit"]
    assert record["seed"] == "3" and record["failure"] == "crash"
    other = partn.search_named(ROOT, "neb/Pt-heptamer", "partn/artn", "min01")
    assert other["seed"] == "0" and other["zseed"] != record["zseed"]
    with pytest.raises(KeyError, match="neb/Pt-heptamer has no start nope"):
        partn.search_named(ROOT, "neb/Pt-heptamer", "partn/artn", "nope")


def test_a_named_search_refuses_a_template_of_another_code(library_files):
    with pytest.raises(ValueError, match="tsase/ci-neb code is not artn"):
        partn.search_named(ROOT, "neb/Pt-heptamer", "tsase/ci-neb", "min01")


def test_a_run_over_pairs_writes_the_starts_the_score_and_the_data_file(monkeypatch, library_files, tmp_path):
    seen = _crashing_library(monkeypatch)
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run(ROOT, "neb/Pt-heptamer", "partn/artn", output=output, workers=1, limit=1)
    assert summary == {
        "problem": "neb/Pt-heptamer",
        "template": "partn/artn",
        "force_calls": None,
        "nfailed": 1,
        "starts": 1,
        "successes": 0,
        "output": str(output),
    }
    assert seen.created and seen.calls == []
    rows = [json.loads(line) for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["start"] for row in rows] == ["min01"]
    assert (rows[0]["problem"], rows[0]["template"], rows[0]["failure"]) == ("neb/Pt-heptamer", "partn/artn", "crash")
    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[:4] == [
        "# counter harness. One force call is one potential evaluation in the next_displ loop.",
        "# nevalf is the pARTn tally and is not this count.",
        "# The product sets the initial push. This search does not build a band to the product.",
        "# A success is under the cap, at the force stop, and Hessian index 1.",
    ]
    assert {"nfailed 1", "successes 0", "n 1", "template partn/artn", "problem neb/Pt-heptamer", "counter harness"} <= set(text.splitlines())
    assert not [line for line in text.splitlines() if line.startswith("images")]
    document = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))
    parameters = document["provenance"]["parameters"]
    assert (document["schema"], document["problem"], document["counter"]) == ("optbench.score/v2", "neb/Pt-heptamer", "harness")
    assert parameters["zseed_base"] == 20261015 and parameters["seed"] == "0" and parameters["ninit"] == 3
    assert parameters["push"] == "one free atom from the reactant toward the product"
    assert parameters["potential"] == "morse_pt" and "task" not in parameters
    assert parameters["counter"] == "harness" and parameters["engine_units"] == "lammps/metal for morse_pt, lammps/lj for lj"
    assert parameters["push_step_size"] == 0.1 and parameters["converge_property"] == "norm"
    assert (output.parent / "work").is_dir() is False
    assert document["provenance"]["machine"]["node"] == "workstation-a"


def test_a_run_over_transition_state_guesses_runs_each_from_the_guess_and_says_so(monkeypatch, library_files, tmp_path):
    seen = _pinned(monkeypatch, "partn/ts-lj38", lambda index, record: (np.zeros(record["pos"].shape), True))
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run(ROOT, "tsopt/LJ38", "partn/ts-lj38", output=output, workers=1, limit=1)
    section = partn._section(_template_of("partn/ts-lj38"))
    assert (summary["starts"], summary["problem"], summary["template"]) == (1, "tsopt/LJ38", "partn/ts-lj38")
    assert seen.settings["ninit"] == 0 and seen.fire == {"dt_init": partn._fire_dt(section, "lj")}
    assert seen.calls[0]["nat"] == 38
    rows = [json.loads(line) for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["start"] == "0000" and rows[0]["failure"] == "divergence" and rows[0]["reasons"] == ["force_norm"]
    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[:4] == [
        "# counter harness. The start is the sample geometry, not a reactant/product pair.",
        "# ninit initial pushes use push_init along x of the first free atom.",
        "# A success is under the cap, at the force stop, and Hessian index 1.",
        "# nevalf is the pARTn tally and is not this count.",
    ]
    assert not [line for line in text.splitlines() if line.startswith("images")]
    assert (output.parent / "work" / "partn-ts-lj38" / "geometries").is_dir()
    parameters = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))["provenance"]["parameters"]
    assert parameters["task"] == "saddle" and parameters["fire_dt_init"] == partn._fire_dt(section, "lj")
    assert (parameters["counter"], parameters["seed"], parameters["zseed_base"]) == ("harness", "0", 20261015)
    assert parameters["converge_property"] == section["converge_property"]
    assert parameters["lanczos_disp"] == section["lanczos_disp"]
    assert parameters["engine_units"] == "lammps/metal for morse_pt, lammps/lj for lj"


def test_a_run_over_lowest_modes_succeeds_where_the_mode_the_library_finds_is_the_reference(monkeypatch, library_files, tmp_path):
    reference = np.asarray(partn.plan(ROOT, "lowest-eigenvector/lj38", "partn/mode-lj38", tmp_path / "work")["jobs"][0]["reference"])
    seen = _pinned(
        monkeypatch,
        "partn/mode-lj38",
        lambda index, record: (np.full(record["pos"].shape, 0.01), False),
        eigenvec=lambda index: reference,
    )
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run(ROOT, "lowest-eigenvector/lj38", "partn/mode-lj38", output=output, workers=1, limit=1)
    assert (summary["starts"], summary["successes"], summary["nfailed"], summary["force_calls"]) == (1, 1, 0, 1)
    assert len(seen.calls) == 1
    rows = [json.loads(line) for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["success"] is True and rows[0]["overlap"] == pytest.approx(1.0, abs=1e-12)
    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[:3] == [
        "# counter harness. The search stops when the pARTn eigenvector overlaps the reference mode by the problem threshold.",
        "# The displacement that would leave that geometry is not applied after the overlap is met.",
        "# nevalf is the pARTn tally and is not this count.",
    ]
    assert not [line for line in text.splitlines() if line.startswith("images")]
    parameters = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))["provenance"]["parameters"]
    assert parameters["task"] == "mode" and parameters["overlap"] == 0.99
    assert (parameters["counter"], parameters["seed"], parameters["zseed_base"]) == ("harness", "0", 20261015)


def test_the_island_catalog_is_one_walk_that_run_hands_on_with_its_options(monkeypatch, library_files, tmp_path):
    taken = {}

    def walk(root, problem, template, *, output, workers, limit=0):
        taken.update(problem=problem["id"], template=template["id"], output=output, workers=workers, limit=limit)
        return {"starts": 0}

    monkeypatch.setattr(partn, "run_catalog", walk)
    output = tmp_path / "out" / "bench.dat"
    assert partn.run(ROOT, "saddle-search/pt-island", "partn/saddle-pt", output=output, workers=3, limit=4) == {"starts": 0}
    assert taken == {
        "problem": "saddle-search/pt-island",
        "template": "partn/saddle-pt",
        "output": output,
        "workers": 3,
        "limit": 4,
    }


def test_free_atoms_relax_to_the_pair_distance_while_the_fixed_atom_stays():
    positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    free = np.array([False, True])
    out, ok, calls, energy = partn._minimize_free("lj", positions, free, None, 1.0e-9, 500)
    assert ok is True and 2 < calls < 500
    assert out[0].tolist() == [0.0, 0.0, 0.0]
    assert out[1] == pytest.approx([2.0 ** (1.0 / 6.0), 0.0, 0.0], abs=1e-8)
    assert energy == pytest.approx(-1.0, abs=1e-12)
    assert positions[1].tolist() == [1.4, 0.0, 0.0]


def test_a_relaxation_that_is_given_one_call_stops_where_it_started_and_says_so():
    positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    out, ok, calls, energy = partn._minimize_free("lj", positions, np.array([False, True]), None, 1.0e-9, 1)
    assert ok is False and calls == 1
    assert out[1].tolist() == [1.4, 0.0, 0.0]
    assert energy == pytest.approx(_u(1.4), abs=1e-12)


def _island(**search):
    from optbench.campaign import load_template, resolve_problem

    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    template = load_template(ROOT, "partn/saddle-pt")
    template["search"] = {**template["search"], **search}
    return problem, template


class _Walk:
    """Stands in for the two costly steps of the island walk.

    The scripted searches climb to a saddle a known shift away from the
    reactant minimum, and the scripted relaxations end on that minimum
    or on a structure one angstrom from it.
    """

    ENERGY = -100.0

    def __init__(self, monkeypatch, script, endpoints, *, minimum_converged=True):
        self.script = list(script)
        self.endpoints = list(endpoints)
        self.minimum_converged = minimum_converged
        self.jobs = []
        self.relaxations = []
        self.starts = []
        self.boxes = []
        self.minimum = None
        monkeypatch.setattr(partn, "_execute", self.execute)
        monkeypatch.setattr(partn, "_minimize_free", self.minimize)

    def minimize(self, kind, positions, free, box, force_tol, budget):
        self.relaxations.append((kind, force_tol, budget))
        self.starts.append(np.array(positions, dtype=np.float64, copy=True))
        self.boxes.append(None if box is None else np.asarray(box, dtype=np.float64).tolist())
        if budget == 5000:
            self.minimum = np.array(positions, dtype=np.float64, copy=True)
            return np.array(positions, dtype=np.float64, copy=True), self.minimum_converged, 11, self.ENERGY
        end = self.endpoints.pop(0)
        out = np.array(self.minimum, copy=True)
        if end == "other":
            out[np.flatnonzero(free)[0], 0] += 1.0
        if end == "wrapped":
            out[np.flatnonzero(free)[0], 0] += box[0] - 0.05
        return out, True, 7, self.ENERGY if end in ("minimum", "wrapped") else self.ENERGY + 1.0

    def execute(self, job):
        step = self.script[len(self.jobs)]
        self.jobs.append(job)
        free = np.asarray(job["free"], dtype=bool)
        if step["barrier"] is None:
            return {
                "start": job["name"],
                "success": False,
                "failure": "cap",
                "reasons": ["cap"],
                "force_calls": step["climb"],
                "energy": None,
                "positions": None,
                "lowest_mode": None,
            }
        saddle = np.array(self.minimum, copy=True)
        saddle[np.flatnonzero(free)[0], 0] += step["shift"]
        mode = [0.0] * (3 * int(free.sum()))
        mode[0] = 2.0
        wrong = step.get("wrong", False)
        return {
            "start": job["name"],
            "success": not wrong,
            "failure": "wrong_stationary" if wrong else None,
            "reasons": ["hessian_index"] if wrong else [],
            "force_calls": step["climb"],
            "energy": self.ENERGY + step["barrier"],
            "positions": saddle.reshape(-1).tolist(),
            "lowest_mode": mode,
        }


def _rows(path):
    return [json.loads(line) for line in path.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]


def _comment(line):
    words = line.removeprefix("# ").split()
    return dict(zip(words[0::2], words[1::2]))


SIX_SEARCHES = [
    {"climb": 10, "barrier": 0.5, "shift": 0.3},
    {"climb": 8, "barrier": 0.505, "shift": 0.3},
    {"climb": 77, "barrier": None},
    {"climb": 9, "barrier": 2.0, "shift": 0.6},
    {"climb": 6, "barrier": 0.7, "shift": 0.9},
    {"climb": 5, "barrier": 0.9, "shift": 1.2},
]
SIX_ENDS = ["minimum", "other"] * 3 + ["other", "other"] + ["minimum", "other"]


def test_the_island_walk_counts_a_new_connected_saddle_under_the_barrier_and_reports_why_the_rest_do_not(
    monkeypatch, library_files, tmp_path, capsys
):
    problem, template = _island(max_searches=10, search_budget=77, displace_atoms="0,1", repetitions=1)
    problem["goal"] = {"connected_saddles": 2, "barrier_below_eV": 1.5}
    walk = _Walk(monkeypatch, SIX_SEARCHES, SIX_ENDS)
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run_catalog(ROOT, problem, template, output=output, workers=7)
    assert summary == {
        "force_calls": 196.0,
        "nfailed": 0,
        "starts": 6,
        "template": "partn/saddle-pt",
        "problem": "saddle-search/pt-island",
    }
    rows = _rows(output)
    assert [(row["start"], row["success"], row["failure"], row["reasons"], row["force_calls"]) for row in rows] == [
        ("rep1-search1", True, None, [], 24),
        ("rep1-search2", False, "divergence", ["duplicate"], 22),
        ("rep1-search3", False, "cap", ["cap"], 77),
        ("rep1-search4", False, "divergence", ["barrier"], 23),
        ("rep1-search5", False, "divergence", ["not_connected"], 20),
        ("rep1-search6", True, None, [], 19),
    ]
    barriers = [row.get("barrier") for row in rows]
    assert barriers[2] is None
    assert [barriers[i] for i in (0, 1, 3, 4, 5)] == pytest.approx([0.5, 0.505, 2.0, 0.7, 0.9], abs=1e-9)
    assert rows[0]["endpoint_calls"] == 14 and rows[0]["per_atom_plus"] == 0.0
    assert rows[0]["per_atom_minus"] == pytest.approx(1.0, abs=1e-12)
    assert rows[4]["per_atom_plus"] == pytest.approx(1.0, abs=1e-12)
    assert all("positions" not in row and "lowest_mode" not in row for row in rows)
    lines = output.read_text(encoding="utf-8").splitlines()
    found = _comment(lines[0])
    assert (found["rep"], found["force_calls"], found["unique_saddles"], found["total_searches"]) == ("1", "196", "3", "6")
    assert (found["connected"], found["minimize_calls"], found["minimum_converged"], found["seed"]) == ("2", "11", "1", "20261016")
    assert float(found["fraction_connected"]) == pytest.approx(5.0 / 6.0, abs=1e-12)
    assert lines[1:] == [
        "force_calls 196.0",
        "force_calls_min 196",
        "force_calls_max 196",
        "code artn",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template['algorithm']}",
        f"tuning {template['tuning']}",
        "template partn/saddle-pt",
        "problem saddle-search/pt-island",
        "counter harness",
        "repetitions 1",
        "nfailed 0",
        "goal 2",
    ]
    assert walk.relaxations[0] == ("morse_pt", partn.point_settings(problem)["force_norm"], 5000)
    assert [budget for _kind, _tol, budget in walk.relaxations] == [5000] + [2000] * 10
    assert capsys.readouterr().out.splitlines() == ["partn/saddle-pt rep 1 searches 1 connected 0"]


def test_each_search_starts_from_the_minimum_kicked_on_one_listed_free_atom_by_the_stated_distance(
    monkeypatch, library_files, tmp_path
):
    problem, template = _island(max_searches=6, search_budget=77, displace_atoms="0,1", displace_magnitude=0.1)
    problem["goal"] = {"connected_saddles": 2, "barrier_below_eV": 1.5}
    walk = _Walk(monkeypatch, SIX_SEARCHES, SIX_ENDS)
    partn.run_catalog(ROOT, problem, template, output=tmp_path / "out" / "bench.dat", workers=1)
    assert [job["name"] for job in walk.jobs] == [f"rep1-search{n}" for n in range(1, 7)]
    assert len({job["zseed"] for job in walk.jobs}) == 6
    assert walk.boxes[0] is not None and len(walk.boxes[0]) == 3 and all(job["box"] == walk.boxes[0] for job in walk.jobs)
    assert all(job["free"] == walk.jobs[0]["free"] for job in walk.jobs) and sum(walk.jobs[0]["free"]) == 175
    for job in walk.jobs:
        assert (job["task"], job["cap"], job["ninit"], job["keep_mode"], job["keep_positions"]) == ("saddle", 77, 3, True, True)
        assert job["reference"] is None and job["product"] is None and job["overlap"] == 0.99
        moved = np.flatnonzero(np.abs(np.asarray(job["reactant"]) - walk.minimum).sum(axis=1))
        assert moved.tolist() in ([0], [1])
        step = np.asarray(job["reactant"])[moved[0]] - walk.minimum[moved[0]]
        assert np.linalg.norm(step) == pytest.approx(0.1, abs=1e-12)


def test_every_listed_atom_is_kicked_in_a_long_walk_and_no_other(monkeypatch, library_files, tmp_path):
    problem, template = _island(max_searches=60, search_budget=50, displace_atoms="1, 4,5,", displace_magnitude=0.25)
    problem["goal"] = {"connected_saddles": 9, "barrier_below_eV": 1.5}
    walk = _Walk(monkeypatch, [{"climb": 5, "barrier": None}] * 60, [])
    partn.run_catalog(ROOT, problem, template, output=tmp_path / "out" / "bench.dat", workers=1)
    kicked = set()
    for job in walk.jobs:
        moved = np.flatnonzero(np.abs(np.asarray(job["reactant"]) - walk.minimum).sum(axis=1))
        assert len(moved) == 1
        assert np.linalg.norm(np.asarray(job["reactant"])[moved[0]] - walk.minimum[moved[0]]) == pytest.approx(0.25, abs=1e-12)
        kicked.add(int(moved[0]))
    assert kicked == {1, 4, 5}


def test_the_kicks_are_the_same_on_every_walk_and_differ_between_repetitions(monkeypatch, library_files, tmp_path):
    problem, template = _island(max_searches=2, search_budget=50, displace_atoms="0,1,2,3,4,5,6", repetitions=2)
    problem["goal"] = {"connected_saddles": 1, "barrier_below_eV": 1.5}
    never = [{"climb": 5, "barrier": None}] * 4

    def kicks(name):
        walk = _Walk(monkeypatch, never, [])
        partn.run_catalog(ROOT, problem, template, output=tmp_path / name / "bench.dat", workers=1)
        return [np.asarray(job["reactant"]).tolist() for job in walk.jobs]

    first, second = kicks("a"), kicks("b")
    assert first == second
    assert len(first) == 4 and first[0] != first[2]


def test_repetitions_are_totalled_separately_and_a_repetition_that_misses_the_goal_is_counted_failed(
    monkeypatch, library_files, tmp_path
):
    problem, template = _island(max_searches=2, search_budget=50, displace_atoms="0", repetitions=2)
    problem["goal"] = {"connected_saddles": 1, "barrier_below_eV": 1.5}
    script = [
        {"climb": 10, "barrier": 0.5, "shift": 0.3},
        {"climb": 50, "barrier": None},
        {"climb": 40, "barrier": None},
    ]
    walk = _Walk(monkeypatch, script, ["minimum", "other"])
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    assert (summary["force_calls"], summary["nfailed"], summary["starts"]) == (68.0, 1, 3)
    lines = output.read_text(encoding="utf-8").splitlines()
    first, second = _comment(lines[0]), _comment(lines[1])
    assert (first["force_calls"], first["connected"], first["total_searches"], first["seed"]) == ("35", "1", "1", "20261016")
    assert (second["force_calls"], second["connected"], second["total_searches"], second["seed"]) == ("101", "0", "2", "20261017")
    assert (second["unique_saddles"], float(second["fraction_connected"])) == ("0", 1.0)
    assert lines[2:5] == ["force_calls 68.0", "force_calls_min 35", "force_calls_max 101"]
    assert {"repetitions 2", "nfailed 1", "goal 1"} <= set(lines)
    assert [job["name"] for job in walk.jobs] == ["rep1-search1", "rep2-search1", "rep2-search2"]
    assert [budget for _kind, _tol, budget in walk.relaxations].count(5000) == 2


@pytest.mark.parametrize(("limit", "searched"), [(1, 1), (2, 2), (50, 3)])
def test_a_limit_caps_the_searches_of_the_walk_and_never_raises_them(monkeypatch, library_files, tmp_path, limit, searched):
    problem, template = _island(max_searches=3, search_budget=50, displace_atoms="0")
    problem["goal"] = {"connected_saddles": 9, "barrier_below_eV": 1.5}
    walk = _Walk(monkeypatch, [{"climb": 5, "barrier": None}] * 3, [])
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run_catalog(ROOT, problem, template, output=output, workers=1, limit=limit)
    assert (summary["starts"], summary["nfailed"]) == (searched, 1)
    assert len(walk.jobs) == searched and _comment(output.read_text(encoding="utf-8").splitlines()[0])["total_searches"] == str(searched)


def test_a_walk_that_is_allowed_no_search_reports_only_the_cost_of_the_first_minimization(
    monkeypatch, library_files, tmp_path
):
    problem, template = _island(max_searches=0)
    walk = _Walk(monkeypatch, [], [], minimum_converged=False)
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    assert summary == {
        "force_calls": 11.0,
        "nfailed": 1,
        "starts": 0,
        "template": "partn/saddle-pt",
        "problem": "saddle-search/pt-island",
    }
    found = _comment(output.read_text(encoding="utf-8").splitlines()[0])
    assert (found["total_searches"], found["connected"], found["minimum_converged"]) == ("0", "0", "0")
    assert float(found["fraction_connected"]) == 0.0
    assert walk.jobs == [] and _rows(output) == []


def test_a_walk_refuses_to_kick_an_atom_that_is_held_fixed(monkeypatch, library_files, tmp_path):
    problem, template = _island(displace_atoms="7")
    _Walk(monkeypatch, [], [])
    with pytest.raises(RuntimeError, match="displace atom 7 is not free"):
        partn.run_catalog(ROOT, problem, template, output=tmp_path / "out" / "bench.dat", workers=1)


def test_a_library_that_raises_the_harness_cap_is_recorded_as_a_cap_and_not_as_a_crash(monkeypatch):
    def refuse(index, record):
        raise ForceCallCap("force-call cap 50")

    seen = _install(monkeypatch, refuse)
    record = partn._execute(_job())
    assert (record["success"], record["failure"], record["reasons"]) == (False, "cap", ["cap"])
    assert record["error"] == "force-call cap: force-call cap 50"
    assert seen.destroyed == 1


def test_a_library_that_cannot_be_created_is_a_crash_and_nothing_is_destroyed(monkeypatch):
    seen = _install(monkeypatch, _to_the_saddle, create_fails=OSError("cannot open the shared library"))
    before = Path.cwd()
    record = partn._execute(_job())
    assert (record["success"], record["failure"], record["reasons"]) == (False, "crash", ["crash"])
    assert record["error"] == "OSError: cannot open the shared library"
    assert len(seen.created) == 1 and seen.destroyed == 0 and seen.calls == []
    assert Path.cwd() == before


def test_a_template_of_another_engine_is_refused_by_a_named_search_and_by_the_plan(monkeypatch, library_files, tmp_path):
    monkeypatch.setattr(partn, "ENGINE", "another-engine")
    with pytest.raises(ValueError, match="partn/artn engine is"):
        partn.search_named(ROOT, "neb/Pt-heptamer", "partn/artn", "min01")
    with pytest.raises(ValueError, match="partn/artn engine is"):
        partn.plan(ROOT, "neb/Pt-heptamer", "partn/artn", tmp_path / "work")


def test_a_walk_of_no_repetitions_has_no_totals_to_report(monkeypatch, library_files, tmp_path):
    problem, template = _island(repetitions=0)
    walk = _Walk(monkeypatch, [], [])
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    assert (summary["force_calls"], summary["nfailed"], summary["starts"]) == (None, 0, 0)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert not [line for line in lines if line.startswith(("# rep", "force_calls"))]
    assert {"repetitions 0", "nfailed 0"} <= set(lines)
    assert walk.relaxations == [] and walk.jobs == []


def test_the_two_endpoints_are_relaxed_from_the_saddle_pushed_a_tenth_of_an_angstrom_each_way_along_the_mode(
    monkeypatch, library_files, tmp_path
):
    problem, template = _island(max_searches=1, displace_atoms="0")
    problem["goal"] = {"connected_saddles": 1, "barrier_below_eV": 1.5}
    walk = _Walk(monkeypatch, [{"climb": 4, "barrier": 0.5, "shift": 0.3}], ["minimum", "other"])
    partn.run_catalog(ROOT, problem, template, output=tmp_path / "out" / "bench.dat", workers=1)
    free = np.asarray(walk.jobs[0]["free"], dtype=bool)
    first = int(np.flatnonzero(free)[0])
    saddle_x = walk.minimum[first, 0] + 0.3
    plus, minus = walk.starts[1], walk.starts[2]
    assert plus[first, 0] - saddle_x == pytest.approx(0.1, abs=1e-12)
    assert minus[first, 0] - saddle_x == pytest.approx(-0.1, abs=1e-12)
    for start in (plus, minus):
        assert np.array_equal(start[~free], walk.minimum[~free])
        moved = np.abs(start[free] - walk.minimum[free]).sum(axis=1)
        assert np.count_nonzero(moved) == 1


def test_an_endpoint_that_is_the_minimum_across_the_periodic_boundary_is_the_minimum(monkeypatch, library_files, tmp_path):
    problem, template = _island(max_searches=1, displace_atoms="0")
    problem["goal"] = {"connected_saddles": 1, "barrier_below_eV": 1.5}
    _Walk(monkeypatch, [{"climb": 4, "barrier": 0.5, "shift": 0.3}], ["wrapped", "other"])
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    assert summary["nfailed"] == 0
    rows = _rows(output)
    assert rows[0]["success"] is True and rows[0]["per_atom_plus"] == pytest.approx(0.05, abs=1e-9)
    assert rows[0]["per_atom_minus"] == pytest.approx(1.0, abs=1e-12)


def test_a_barrier_at_the_limit_or_at_zero_is_refused_and_one_just_inside_is_kept(monkeypatch, library_files, tmp_path):
    problem, template = _island(max_searches=4, displace_atoms="0")
    problem["goal"] = {"connected_saddles": 9, "barrier_below_eV": 1.5}
    script = [
        {"climb": 3, "barrier": 1.5, "shift": 0.3},
        {"climb": 3, "barrier": 0.0, "shift": 0.6},
        {"climb": 3, "barrier": 1.5 - 2.0**-20, "shift": 0.9},
        {"climb": 3, "barrier": 2.0**-20, "shift": 1.2},
    ]
    _Walk(monkeypatch, script, ["minimum", "other"] * 4)
    output = tmp_path / "out" / "bench.dat"
    partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    assert [(row["success"], row["reasons"]) for row in _rows(output)] == [
        (False, ["barrier"]),
        (False, ["barrier"]),
        (True, []),
        (True, []),
    ]
    assert _comment(output.read_text(encoding="utf-8").splitlines()[0])["connected"] == "2"


def test_a_saddle_with_the_same_structure_and_a_different_barrier_is_a_new_saddle(monkeypatch, library_files, tmp_path):
    problem, template = _island(max_searches=3, displace_atoms="0")
    problem["goal"] = {"connected_saddles": 9, "barrier_below_eV": 1.5}
    script = [
        {"climb": 3, "barrier": 0.5, "shift": 0.3},
        {"climb": 3, "barrier": 0.52, "shift": 0.3},
        {"climb": 3, "barrier": 0.5, "shift": 0.45},
    ]
    _Walk(monkeypatch, script, ["minimum", "other"] * 3)
    output = tmp_path / "out" / "bench.dat"
    partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    assert [row["success"] for row in _rows(output)] == [True, True, True]
    assert _comment(output.read_text(encoding="utf-8").splitlines()[0])["unique_saddles"] == "3"


def test_the_first_kick_is_the_draw_of_the_seed_the_output_states(monkeypatch, library_files, tmp_path):
    problem, template = _island(max_searches=1, displace_atoms="3, 0,5", displace_magnitude=0.2)
    walk = _Walk(monkeypatch, [{"climb": 1, "barrier": None}], [])
    output = tmp_path / "out" / "bench.dat"
    partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    stated = _comment(output.read_text(encoding="utf-8").splitlines()[0])["seed"]
    assert stated == "20261016"
    rng = np.random.default_rng(int(stated))
    atom = [3, 0, 5][int(rng.integers(0, 3))]
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    expected = np.array(walk.minimum, copy=True)
    expected[atom] += 0.2 * direction
    assert np.asarray(walk.jobs[0]["reactant"]) == pytest.approx(expected, abs=1e-12)


def test_a_walk_with_nothing_stated_uses_the_island_defaults(monkeypatch, library_files, tmp_path):
    problem, template = _island()
    problem.pop("goal")
    template["search"] = {}
    script = [{"climb": 2, "barrier": 1.4, "shift": 0.3}, {"climb": 2, "barrier": 1.6, "shift": 0.6}]
    script += [{"climb": 2, "barrier": None}] * 118
    walk = _Walk(monkeypatch, script, ["minimum", "other"] * 2)
    output = tmp_path / "out" / "bench.dat"
    summary = partn.run_catalog(ROOT, problem, template, output=output, workers=1, limit=120)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert {"goal 27", "repetitions 1", "nfailed 1"} <= set(lines)
    found = _comment(lines[0])
    assert (found["total_searches"], found["connected"]) == ("120", "1")
    assert [row["reasons"] for row in _rows(output)[:2]] == [[], ["barrier"]]
    assert {job["cap"] for job in walk.jobs} == {10000}
    kicked = set()
    for job in walk.jobs:
        moved = np.flatnonzero(np.abs(np.asarray(job["reactant"]) - walk.minimum).sum(axis=1))
        assert len(moved) == 1
        assert np.linalg.norm(np.asarray(job["reactant"])[moved[0]] - walk.minimum[moved[0]]) == pytest.approx(0.1, abs=1e-12)
        kicked.add(int(moved[0]))
    assert kicked == set(range(7))


def test_the_walk_says_where_it_is_on_the_first_search_and_every_twenty_fifth(monkeypatch, library_files, tmp_path, capsys):
    problem, template = _island(max_searches=60, displace_atoms="0")
    problem["goal"] = {"connected_saddles": 9, "barrier_below_eV": 1.5}
    _Walk(monkeypatch, [{"climb": 1, "barrier": None}] * 60, [])
    partn.run_catalog(ROOT, problem, template, output=tmp_path / "out" / "bench.dat", workers=1)
    assert capsys.readouterr().out.splitlines() == [
        "partn/saddle-pt rep 1 searches 1 connected 0",
        "partn/saddle-pt rep 1 searches 25 connected 0",
        "partn/saddle-pt rep 1 searches 50 connected 0",
    ]


def test_a_template_that_changes_the_defaults_changes_what_the_library_is_given(monkeypatch):
    seen = _install(monkeypatch, lambda index, record: (np.zeros((3, 3)), True))
    template = _template(push_step_size=0.25, eigen_step_size=0.35, ninit=5, lanczos_max_size=8, converge_property="maxval")
    job = partn._job(_pair(0.1, free=[True, True, False]), _settings(), template, "0")
    job["product"] = None
    partn._execute(job)
    assert (seen.settings["push_step_size"], seen.settings["eigen_step_size"]) == (0.25, 0.35)
    assert (seen.settings["ninit"], seen.settings["lanczos_max_size"], seen.settings["converge_property"]) == (5, 8, "maxval")
    assert seen.settings["push_init"].tolist() == [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert seen.calls[0]["if_pos"].tolist() == [[1, 1, 1], [1, 1, 1], [0, 0, 0]]
    seen = _install(monkeypatch, lambda index, record: (np.zeros((3, 3)), True))
    product = [[-A, 0.0, 0.0], [0.1, 0.5, 0.0], [A, 0.0, 0.0]]
    partn._execute(partn._job(_pair(0.1, product=product), _settings(), template, "0"))
    assert seen.settings["push_init"].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.0]]


def test_a_job_with_no_task_is_a_saddle_search(monkeypatch):
    for task in ("missing", None):
        _install(monkeypatch, _to_the_saddle)
        job = _job()
        if task == "missing":
            del job["task"]
        else:
            job["task"] = task
        record = partn._execute(job)
        assert record["success"] is True and record["hessian_index"] == 1


def test_an_error_text_that_names_the_cap_class_is_a_cap_failure_and_not_a_crash(monkeypatch):
    def refuse(index, record):
        raise RuntimeError("ForceCallCap raised in the library")

    _install(monkeypatch, refuse)
    record = partn._execute(_job())
    assert (record["failure"], record["reasons"]) == ("cap", ["cap"])
    assert record["error"] == "RuntimeError: ForceCallCap raised in the library"


def test_displacements_of_a_millionth_of_an_angstrom_are_steps_and_not_a_stall(monkeypatch):
    def plan(index, record):
        step = np.zeros((3, 3))
        step[1, 0] = 1.0e-6
        return step, index >= 4

    seen = _install(monkeypatch, plan)
    record = partn._execute(_job(cap=20))
    assert len(seen.calls) == 5 and record["force_calls"] == 5 and "error" not in record


def test_a_search_asked_to_keep_the_mode_reports_it_along_the_line_of_the_atoms(monkeypatch):
    _install(monkeypatch, _to_the_saddle)
    record = partn._execute(_job(keep_mode=True))
    assert [abs(value) for value in record["lowest_mode"]] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    _install(monkeypatch, _to_the_saddle)
    assert partn._execute(_job())["lowest_mode"] is None


def test_every_evaluation_of_the_search_is_made_in_the_cell_of_the_job(monkeypatch):
    import optbench.pes as pes

    cells = []
    real = pes.evaluate

    def spy(kind, positions, box):
        cells.append((kind, None if box is None else np.asarray(box).tolist()))
        return real(kind, positions, box)

    monkeypatch.setattr(pes, "evaluate", spy)
    moves = {2: -0.1, 5: 0.05}

    def plan(index, record):
        step = np.zeros((3, 3))
        step[1, 0] = moves.get(index, 0.0)
        return step, False

    _install(monkeypatch, plan)
    partn._execute(_job(cap=4, box=[30.0, 40.0, 50.0]))
    assert len(cells) >= 3 and set(map(repr, cells)) == {repr(("lj", [30.0, 40.0, 50.0]))}


def test_the_verdict_is_judged_on_the_potential_the_free_atoms_and_the_cell_of_the_job(monkeypatch):
    taken = {}
    real = partn.judge_saddle

    def spy(*args, **kwargs):
        taken["args"] = args
        taken["kwargs"] = kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(partn, "judge_saddle", spy)
    _install(monkeypatch, _to_the_saddle)
    partn._execute(_job(box=[30.0, 40.0, 50.0], keep_mode=True, cap=40, force_norm=1.0e-5))
    name, position, energy, gradient = taken["args"]
    kwargs = taken["kwargs"]
    assert name == "line" and np.asarray(position) == pytest.approx(np.asarray(_line(0.0)), abs=1e-15)
    assert energy == pytest.approx(_saddle_energy(), abs=1e-12)
    assert np.asarray(gradient)[1] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert (kwargs["kind"], kwargs["cap"], kwargs["force_norm"], kwargs["keep_mode"]) == ("lj", 40, 1.0e-5, True)
    assert kwargs["free"].tolist() == FREE and kwargs["box"].tolist() == [30.0, 40.0, 50.0] and kwargs["prefix"] == []
    assert kwargs["counter"].calls == 2
    _install(monkeypatch, _to_the_saddle, version=(0, 0, 1))
    partn._execute(_job())
    assert taken["kwargs"]["prefix"] == ["crash"] and taken["kwargs"]["keep_mode"] is False


def test_a_mode_record_carries_the_tallies_the_library_gave(monkeypatch):
    _mode_install(monkeypatch, [0.0], flags={"has_sad": False, "nevalf": 9})
    record = partn._execute(_mode_job())
    assert record["artn_has_sad"] is False and record["artn_nevalf"] == 9


def test_the_overlap_does_not_depend_on_the_length_of_the_reference():
    reference, full = _mode_with(0.7)
    scaled = [5.0 * value for value in reference]
    assert partn._mode_overlap(_eigenvec_art(full.reshape(-1)), scaled, FREE_PAIR) == pytest.approx(math.cos(0.7), abs=1e-12)


def test_a_mode_of_zero_length_has_overlap_zero_without_a_numerical_warning():
    reference, _ = _mode_with(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert partn._mode_overlap(_eigenvec_art(np.zeros(9)), reference, FREE_PAIR) == 0.0


def test_a_sample_geometry_job_carries_the_template_and_the_places_of_the_library(library_files):
    library, interface = library_files
    start = {"name": "line", "positions": _line(0.1), "fixed": [True, False, True], "box": [30.0, 40.0, 50.0], "reference": None}
    template = _template(push_step_size=0.25, lpush_final=True, converge_property="maxval")
    job = partn._point_job(start, {**_settings(), "task": "saddle", "overlap": 0.5}, template)
    assert (job["push_step_size"], job["lpush_final"], job["converge_property"]) == (0.25, True, "maxval")
    assert job["shlib"] == str(library) and job["interface"] == str(interface)
    assert job["box"] == [30.0, 40.0, 50.0] and job["commit"] == COMMIT


def test_a_template_that_states_initial_pushes_keeps_them_on_a_transition_state_guess(library_files, tmp_path, monkeypatch):
    from optbench.campaign import load_template, resolve_problem

    template = load_template(ROOT, "partn/ts-lj38")
    template["partn"]["ninit"] = 2
    monkeypatch.setattr(partn, "_resolved", lambda root, problem_name, template_name: (resolve_problem(ROOT, "tsopt/LJ38"), template))
    planned = partn.plan(ROOT, "tsopt/LJ38", "partn/ts-lj38", tmp_path / "work")
    assert planned["jobs"][0]["ninit"] == 2 and planned["template"]["partn"]["ninit"] == 2
    assert (tmp_path / "work" / "partn-ts-lj38" / "geometries").is_dir()


def test_a_search_that_failed_is_not_relaxed_or_counted_even_when_it_carries_a_saddle_and_a_mode(
    monkeypatch, library_files, tmp_path
):
    problem, template = _island(max_searches=2, displace_atoms="0")
    problem["goal"] = {"connected_saddles": 1, "barrier_below_eV": 1.5}
    script = [
        {"climb": 6, "barrier": 0.5, "shift": 0.3, "wrong": True},
        {"climb": 6, "barrier": 0.5, "shift": 0.6},
    ]
    walk = _Walk(monkeypatch, script, ["minimum", "other"])
    output = tmp_path / "out" / "bench.dat"
    partn.run_catalog(ROOT, problem, template, output=output, workers=1)
    rows = _rows(output)
    assert [(row["success"], row["failure"], row["reasons"], row["force_calls"]) for row in rows] == [
        (False, "wrong_stationary", ["hessian_index"], 6),
        (True, None, [], 20),
    ]
    assert "barrier" not in rows[0] and "endpoint_calls" not in rows[0]
    assert [budget for _kind, _tol, budget in walk.relaxations] == [5000, 2000, 2000]
    found = _comment(output.read_text(encoding="utf-8").splitlines()[0])
    assert (found["unique_saddles"], found["connected"], found["total_searches"]) == ("1", "1", "2")
