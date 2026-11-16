"""Surfaces and the rgsaddle saddle driver.

The library test loads librgsaddle.so from RGSADDLE_LIB. CI does not
build that library, so the test skips when the variable is unset.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem
from optbench.surfaces import lennard_jones, morse_smooth_linear, read_con

ROOT = Path(__file__).resolve().parents[1]


def test_a_limit_runs_the_first_rgsaddle_geometries(monkeypatch, tmp_path: Path):
    from optbench.campaign import run
    import optbench.rgsaddle_minmode as driver

    monkeypatch.setattr(driver, "_geometries", lambda *args, **kwargs: [{"name": f"g{i}"} for i in range(200)])
    seen = []

    def fake_dispatch(spec):
        seen.append(spec["start"])
        return {"force_calls": 4, "success": True, "failure": None, "start": spec["start"]}

    monkeypatch.setattr(driver, "_spec_from_geometry", lambda row, *args, **kwargs: {"start": row["name"]})
    monkeypatch.setattr(driver, "_dispatch", fake_dispatch)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "tsopt/LJ38",
        "rgsaddle/lanczos",
        output=output,
        work=tmp_path / "work",
        limit=1,
    )
    assert seen == ["g0"]
    assert summary["starts"] == 1


def test_island_saddle_success_is_the_eon_comparison():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    success = problem["scoring"]["v2"]["success"]
    assert "Matter::compare" in success
    assert "RMSD" not in success
    assert problem["connection"] == {"distance_difference": 0.1, "energy_difference": 0.01}
    assert "max_force_calls" not in (problem.get("convergence") or {})


ISLAND_TEMPLATES = sorted(path.stem for path in (ROOT / "templates" / "rgsaddle").glob("saddle-*.toml"))


def test_island_search_bounds_are_the_archived_eon_search():
    archived = configparser.ConfigParser(strict=False)
    archived.read(ROOT / "archive" / "entries" / "saddle-search" / "pt-island" / "eon-dimer" / "config.ini")
    assert len(ISLAND_TEMPLATES) == 4
    for name in ISLAND_TEMPLATES:
        template = load_template(ROOT, f"rgsaddle/{name}")
        ini = template["ini"]
        assert int(ini["SaddleSearch"]["max_iterations"]) == archived.getint("Saddle Search", "max_iterations")
        assert int(ini["Optimizer"]["max_iterations"]) == archived.getint("Optimizer", "max_iterations")
        if template["tuning"] == "default":
            assert float(ini["Optimizer"]["max_move"]) == archived.getfloat("Optimizer", "max_move")


def test_island_search_cap_is_the_step_ceiling_and_two_minimizations():
    from optbench.rgsaddle_minmode import _step_ceiling

    caps = {}
    for name in ISLAND_TEMPLATES:
        ini = load_template(ROOT, f"rgsaddle/{name}")["ini"]
        steps = int(ini["SaddleSearch"]["max_iterations"])
        caps[name] = steps * _step_ceiling(ini["SaddleSearch"]) + 2 * int(ini["Optimizer"]["max_iterations"])
    assert caps == {
        "saddle-dimer-fire": 48000,
        "saddle-dimer-lbfgs": 48000,
        "saddle-dimer-rotskip": 48000,
        "saddle-lanczos": 30000,
    }
    assert _step_ceiling({"min_mode_method": "dimer", "max_rotations": 20, "difference": "central"}) == 88


@pytest.mark.skipif(not os.environ.get("RGSADDLE_LIB"), reason="RGSADDLE_LIB is not set")
@pytest.mark.parametrize("name", ISLAND_TEMPLATES)
def test_an_island_climb_step_stays_under_its_ceiling(name):
    from optbench.connection import periodic_lengths
    from optbench.displacement import displace, neighbours
    from optbench.rgsaddle_minmode import _dispatch, _section, _step_ceiling, _stored
    from optbench.surfaces import load_con

    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    template = load_template(ROOT, f"rgsaddle/{name}")
    positions, fixed, cell = load_con(_stored(ROOT, "website/saddle-search/pt-island/pt-island.con"))
    search = _section(template, "Search")
    listed = [int(part) for part in str(search["displace_atoms"]).split(",")]
    near = neighbours(positions, periodic_lengths(cell, problem["potential"]["pbc"]), float(search["displace_radius"]), listed)
    _epicenter, shift, seed = displace(np.random.default_rng(1), ~fixed, listed, near, float(search["displace_magnitude"]))
    saddle = _section(template, "SaddleSearch")
    ceiling = _step_ceiling(saddle)
    steps = 40
    outcome = _dispatch(
        {
            "name": name,
            "positions": positions + shift,
            "fixed": fixed,
            "cell": cell,
            "potential": problem["potential"],
            "session": "minmode",
            "kind": str(saddle["min_mode_method"]),
            "method": str(_section(template, "Optimizer")["opt_method"]),
            "mode_seed": "file",
            "task": "saddle",
            "saddle": saddle,
            "optimizer": _section(template, "Optimizer"),
            "sella": {},
            "cap": steps * ceiling,
            "max_steps": steps,
            "force_tol": float(problem["convergence"]["force_norm"]),
            "reactant": None,
            "file_mode": seed[~fixed].reshape(-1),
            "reference_mode": None,
            "overlap": 0.99,
            "defer_hessian": True,
        }
    )
    assert 0 < outcome["step_calls_max"] <= ceiling
    assert outcome["calls"] <= outcome["steps"] * ceiling


@pytest.mark.skipif(not os.environ.get("RGSADDLE_LIB"), reason="RGSADDLE_LIB is not set")
def test_one_catalog_repetition_reports_its_own_totals(tmp_path):
    import copy

    from optbench.rgsaddle_minmode import CATALOG_SEED, run_template

    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    template = copy.deepcopy(load_template(ROOT, "rgsaddle/saddle-dimer-rotskip"))
    template["ini"]["Search"]["max_searches"] = 2
    result = run_template(ROOT, problem, template, output=tmp_path / "catalog.dat", repetition=1)
    (summary,) = result["repetitions"]
    assert summary["seed"] == CATALOG_SEED + 1
    assert summary["total_searches"] == len(result["records"]) == 2
    assert summary["force_calls"] == summary["minimize_calls"] + sum(row["force_calls"] for row in result["records"])
    assert summary["reached"] is (summary["connected"] >= summary["goal"])
    with pytest.raises(RuntimeError, match="repetitions 1 to 1"):
        run_template(ROOT, problem, template, output=tmp_path / "other.dat", repetition=2)


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


def test_morse_pair_loop_equals_the_numpy_sum():
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
    shifted = positions.copy()
    for move in ([0, 0.0, 0.0, 0.0], [1, 0.05, -0.02, 0.01], [0, 1.4, -0.8, 0.6]):
        shifted[int(move[0])] += np.array(move[1:])
        energy, gradient = morse_smooth_linear(shifted, **kwargs)
        reference_energy, reference_gradient = _morse_numpy(shifted, **kwargs)
        assert np.float64(energy).tobytes() == np.float64(reference_energy).tobytes()
        assert gradient.tobytes() == reference_gradient.tobytes()


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


def test_a_second_component_stays_in_the_geometry():
    text = "\n".join(
        [
            "0",
            "20 20 20",
            "90 90 90",
            "0 0",
            "0 0",
            "2",
            "2 1",
            "1.0 1.0",
            "Al",
            "Coordinates of Component 1",
            "0 0 0 1 0",
            "1 0 0 1 1",
            "Al",
            "Coordinates of Component 2",
            "2 3 4 0 0",
        ]
    )
    positions, fixed, _cell = read_con(text)
    assert positions.shape == (3, 3)
    assert positions[2].tolist() == [2.0, 3.0, 4.0]
    assert fixed.tolist() == [True, True, False]


def test_a_short_coordinate_block_is_not_a_geometry():
    text = "\n".join(
        [
            "0",
            "20 20 20",
            "90 90 90",
            "0 0",
            "0 0",
            "1",
            "2",
            "1.0",
            "Pt",
            "Coordinates",
            "0 0 0 0 0",
        ]
    )
    with pytest.raises(ValueError, match="header says 2"):
        read_con(text)
    nan = text + "\nnan 0 0 0 0\n"
    with pytest.raises(ValueError, match="not finite"):
        read_con(nan)


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
        fire = {
            "rgsaddle/dimer-fire",
            "rgsaddle/pt-dimer-fire",
            "rgsaddle/saddle-dimer-fire",
            "rgsaddle/saddle-dimer-lbfgs",
            "rgsaddle/saddle-dimer-rotskip",
            "rgsaddle/saddle-lanczos",
        }
        if template["id"] in fire:
            assert template["commit"] == "8fa0eb1559c92b6ae9d1de190e253787a183729d"
            assert template["code_version"] == "(main 8fa0eb15)"
        else:
            assert template["commit"] == "2db34b42e6e029ebcf5b9868b94fcdf969103c9f"
            assert template["code_version"] == "0.2.2"
        assert template["engine"] == "rgsaddle"
    assert "rgsaddle/dimer-lbfgs" in found
    assert "rgsaddle/prfo" in found
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


def test_a_fresh_rgsaddle_search_does_not_keep_its_checkpoint():
    from optbench.rgsaddle_minmode import _continues

    header = {"template": "rgsaddle/saddle-lanczos", "rep": 1}
    assert _continues(header, "rgsaddle/saddle-lanczos", 1, False) is False
    assert _continues(header, "rgsaddle/saddle-lanczos", 1, True) is True
    assert _continues(header, "rgsaddle/saddle-lanczos", 2, True) is False
    assert _continues(None, "rgsaddle/saddle-lanczos", 1, True) is False


def test_run_asks_rgsaddle_to_resume(monkeypatch, tmp_path):
    from optbench.campaign import run

    captured = {}

    def fake(_root, _problem, _template, **kwargs):
        captured["resume"] = kwargs["resume"]
        return {"force_calls": 1, "nfailed": 0, "starts": 1}

    monkeypatch.setattr("optbench.rgsaddle_minmode.run_template", fake)
    run(
        ROOT,
        "saddle-search/pt-island",
        "rgsaddle/saddle-lanczos",
        output=tmp_path / "bench.dat",
        resume=True,
    )
    assert captured["resume"] is True
