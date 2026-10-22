"""vc-SQNM rows use the harness potential and the v1.3 pin."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import list_templates, resolve_problem, run
from optbench.potentials import morse_bulk, morse_bulk_hessian
from optbench.vcsqnm_catalog import CODE_VERSION, COMMIT, specs, write_templates
from optbench.vcsqnm_pes import (
    as_cell,
    cell_stress,
    morse_cell,
    morse_cell_hessian,
    sqnm_lattice_derivative,
)

ROOT = Path(__file__).resolve().parents[1]


def test_column_layout_roundtrips():
    from optbench.runners.vcsqnm_min import _columns, _flatten

    flat = np.arange(12, dtype=np.float64)
    assert np.array_equal(_flatten(_columns(flat)), flat)


def test_morse_cell_matches_the_orthorhombic_kernel():
    from optbench.runners.rgmin_min import _load_geometries

    morse = resolve_problem(ROOT, "minimization/morse-bulk")
    _name, coords, box = _load_geometries(ROOT, morse, None, Path("/tmp/optbench-vcsqnm-geo"), Path("/tmp/u.dat"))[0]
    energy, grad, deriv = morse_cell(coords, box)
    bulk_energy, bulk_grad = morse_bulk(coords, box)
    assert energy == pytest.approx(bulk_energy, abs=1e-9)
    assert np.max(np.abs(grad - bulk_grad)) < 1e-9
    hessian = morse_cell_hessian(coords, box)
    bulk = morse_bulk_hessian(coords, box)
    assert np.max(np.abs(hessian - bulk)) < 1e-8
    cell = as_cell(box)
    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    frac = pos @ np.linalg.inv(cell).T
    eps = 1e-6
    for index in ((0, 0), (0, 1), (1, 2)):
        up = cell.copy()
        down = cell.copy()
        up[index] += eps
        down[index] -= eps
        high, _grad, _deriv = morse_cell(frac @ up.T, up)
        low, _grad, _deriv = morse_cell(frac @ down.T, down)
        assert (high - low) / (2 * eps) == pytest.approx(deriv[index], abs=1e-4)
    voigt = pytest.importorskip("ase.stress")

    stress = voigt.voigt_6_to_full_3x3_stress(voigt.full_3x3_to_voigt_6_stress(cell_stress(deriv, cell)))
    deralat = -float(np.linalg.det(cell)) * stress @ np.linalg.inv(cell).T
    assert np.max(np.abs(deralat - sqnm_lattice_derivative(deriv))) < 1e-8


def test_templates_match_the_catalog_and_the_registry():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "vcsqnm")
    assert pinned["commit"] == COMMIT
    assert pinned["ref"] == "v1.3"
    catalog = specs()
    assert len(catalog) == 12
    assert {row["id"] for row in catalog if row["problem"] == "minimization/lj38"} == {
        "vcsqnm/lj38-sqnm",
        "vcsqnm/lj38-sqnm-tuned",
        "vcsqnm/lj38-ase",
        "vcsqnm/lj38-ase-tuned",
    }
    on_disk = [row for row in list_templates(ROOT) if row.get("engine") == "vcsqnm"]
    assert {row["id"] for row in on_disk} == {row["id"] for row in catalog}
    by_id = {row["id"]: row for row in on_disk}
    for spec in catalog:
        stored = by_id[spec["id"]]
        problem = resolve_problem(ROOT, spec["problem"])
        assert stored["commit"] == COMMIT
        assert stored["code_version"] == CODE_VERSION
        assert stored["code"] == "vc-SQNM"
        assert stored["sqnm"]["initial_step_size"] == -1e-7
        assert stored["sqnm"]["alpha_min"] == 1e-3
        assert stored["sqnm"]["eps_subsp"] == 1e-5
        assert stored["ini"]["Optimizer"]["converged_force"] == problem["convergence"]["force_norm"]
        assert stored["ini"]["Optimizer"]["max_iterations"] == 10000
        if spec["problem"] != "minimization/morse-bulk":
            assert spec["sqnm"]["method"] not in {"vc", "ase-vc"}
    rendered = ROOT / "templates" / "vcsqnm"
    assert {path.name for path in rendered.glob("*.toml")} == {f"{spec['id'].split('/', 1)[1]}.toml" for spec in catalog}


def test_write_templates_matches_the_catalog(tmp_path: Path):
    paths = write_templates(tmp_path)
    assert len(paths) == 12
    text = (tmp_path / "templates" / "vcsqnm" / "morse-ase-vc-tuned.toml").read_text(encoding="utf-8")
    assert 'method = "ase-vc"' in text
    assert "nhist_max = 20" in text
    assert "eps_subsp = 1e-05" in text


def test_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["limit"] = kwargs["limit"]
        return {"force_calls": 1, "nfailed": 0, "starts": 1, "template": template["id"], "problem": problem["id"]}

    monkeypatch.setattr("optbench.runners.vcsqnm_min.run_template", fake)
    summary = run(ROOT, "minimization/lj38", "vcsqnm/lj38-sqnm", output=tmp_path / "benchmark.dat", workers=2, limit=3)
    assert captured == {"template": "vcsqnm/lj38-sqnm", "problem": "minimization/lj38", "limit": 3}
    assert summary["template"] == "vcsqnm/lj38-sqnm"


def _sqnm():
    return pytest.importorskip("sqnm.sqnm")


def _first(problem_id: str):
    from optbench.runners.rgmin_min import _load_geometries

    problem = resolve_problem(ROOT, problem_id)
    return _load_geometries(ROOT, problem, None, Path("/tmp/optbench-vcsqnm-geo"), Path("/tmp/u.dat"))[0]


def _job(name, coords, box, kind, method, gtol, **overrides):
    from optbench.vcsqnm_catalog import DEFAULT

    parameters = dict(DEFAULT)
    parameters.update(overrides)
    return {
        "index": 0,
        "start": name,
        "coords": np.asarray(coords, dtype=np.float64).reshape(-1).tolist(),
        "box": None if box is None else np.asarray(box, dtype=np.float64).tolist(),
        "kind": kind,
        "method": method,
        "parameters": parameters,
        "gtol": gtol,
        "cap": 10000,
    }


def test_cluster_zero_matches_the_repaired_coefficient():
    _sqnm()
    from optbench.runners.vcsqnm_min import _columns, _flatten, _minimize_job
    from optbench.vcsqnm_catalog import TUNED
    from sqnm.free_or_fixed_cell_sqnm import free_sqnm
    from optbench.potentials import lennard_jones

    name, coords, _box = _first("minimization/lj38")
    native = _minimize_job(_job(name, coords, None, "lj", "sqnm", 0.01))
    wrapped = _minimize_job(_job(name, coords, None, "lj", "ase", 0.01))
    tuned = _minimize_job(_job(name, coords, None, "lj", "sqnm", 0.01, **TUNED))
    assert native["success"] is True
    assert native["force_calls"] == 383
    assert native["energy"] == pytest.approx(-163.773633746714, abs=1e-9)
    assert native["hessian_index"] == 0
    assert native["force_norm"] <= 0.01
    assert wrapped["force_calls"] == native["force_calls"]
    assert wrapped["energy"] == pytest.approx(native["energy"], abs=1e-9)
    assert wrapped["success"] is True
    assert tuned["success"] is True
    assert tuned["force_calls"] == 348
    assert tuned["energy"] == pytest.approx(-161.569235822923, abs=1e-9)
    flat = np.asarray(coords, dtype=np.float64).reshape(-1)
    energy, grad = lennard_jones(flat)
    opt = free_sqnm(flat.size // 3, -0.01, 10, 1e-3, 1e-3)
    moved = _flatten(opt.optimizer_step(_columns(flat), float(energy), _columns(-grad)))
    assert float(np.max(np.abs(moved - flat))) > 100.0
    energy_out, _grad_out = lennard_jones(moved)
    assert energy_out > -1.0


def test_morse_start_matches_between_the_wrapper_and_the_native_step():
    _sqnm()
    from optbench.runners.vcsqnm_min import _minimize_job

    name, coords, box = _first("minimization/morse-bulk")
    native = _minimize_job(_job(name, coords, box, "morse", "sqnm", 1e-3))
    wrapped = _minimize_job(_job(name, coords, box, "morse", "ase", 1e-3))
    cell = _minimize_job(_job(name, coords, box, "morse", "vc", 1e-3))
    cell_wrapped = _minimize_job(_job(name, coords, box, "morse", "ase-vc", 1e-3))
    assert native["success"] is True
    assert native["force_calls"] == 23
    assert native["energy"] == pytest.approx(-1396.173237628638, abs=1e-8)
    assert native["hessian_index"] == 0
    assert wrapped["force_calls"] == 23
    assert wrapped["energy"] == pytest.approx(native["energy"], abs=1e-8)
    assert cell["success"] is True
    assert cell["force_calls"] == 31
    assert cell["energy"] == pytest.approx(-1464.284640899119, abs=1e-8)
    assert cell["hessian_index"] == 0
    assert cell["lattice_deriv_max"] <= 1e-3
    assert cell_wrapped["force_calls"] == 31
    assert cell_wrapped["energy"] == pytest.approx(cell["energy"], abs=1e-8)
    assert cell["energy"] < native["energy"] - 50.0


def test_sigma_zero_matches_the_clean_count():
    _sqnm()
    from optbench.noise import KAPPA, window_energy_tolerance
    from optbench.runners.noise_min import _minimize_job
    from optbench.vcsqnm_catalog import DEFAULT

    name, coords, _box = _first("minimization/lj38")
    record = _minimize_job(
        {
            "index": 0,
            "start": name,
            "coords": np.asarray(coords, dtype=np.float64).reshape(-1).tolist(),
            "family": "vcsqnm",
            "method": "sqnm",
            "optimizer": "",
            "options": dict(DEFAULT),
            "sigma": 0.0,
            "seed": 20261016,
            "clean_tol": 0.01,
            "energy_tol": window_energy_tolerance(0.01, 0.0, KAPPA),
            "cap": 10000,
        }
    )
    assert record["success"] is True
    assert record["force_calls"] == 383
    assert record["energy"] == pytest.approx(-163.773633746714, abs=1e-9)
    assert math.isclose(record["force_norm"], 0.00965627042457, rel_tol=0.0, abs_tol=1e-9)
