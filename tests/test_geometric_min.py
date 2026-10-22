"""geomeTRIC templates, the primitive batches, and one Cartesian minimization."""

from __future__ import annotations

from pathlib import Path

import pytest

from optbench.campaign import list_templates, resolve_problem, run

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f657e69bae9c81bf7a23e7375db98502ca5c8c30"


def test_geometric_templates_match_the_pin():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "geometric")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == "geomeTRIC"
    assert pinned["ref"] == "1.1.1"
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("geometric/")]
    assert {row["id"] for row in found} == {
        "geometric/lj38-cart",
        "geometric/lj38-tric",
        "geometric/morse-cart",
        "geometric/morse-tric",
    }
    for template in found:
        assert template["engine"] == "geometric"
        assert template["code"] == "geomeTRIC"
        assert template["code_version"] == "1.1.1"
        assert template["commit"] == COMMIT
        assert template["tuning"] == "default"
        assert template["geometric"]["coordsys"] in {"cart", "tric"}
        problem = resolve_problem(ROOT, template["problem"])
        assert float(problem["convergence"]["force_norm"]) > 0.0
    noise = [row for row in list_templates(ROOT) if str(row["id"]).startswith("noise/geometric-")]
    assert len(noise) == 8
    for template in noise:
        assert template["engine"] == "noise"
        assert template["solver"]["family"] == "geometric"
        assert template["commit"] == COMMIT


def test_geometric_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        return {"force_calls": 1, "nfailed": 0, "starts": 1, "problem": problem["id"], "template": template["id"]}

    monkeypatch.setattr("optbench.runners.geometric_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/lj38",
        "geometric/lj38-cart",
        client=None,
        output=tmp_path / "benchmark.dat",
    )
    assert captured == {"template": "geometric/lj38-cart", "problem": "minimization/lj38"}
    assert summary["starts"] == 1


def test_batched_primitives_match_the_library_formulas():
    geometric = pytest.importorskip("geometric")
    assert geometric.__version__ == "1.1.1"
    import numpy as np

    from optbench.runners.geometric_min import install_batch_primitives, install_fast_add, knearest
    from optbench.runners.rgmin_min import _load_geometries

    install_fast_add()
    work = Path("/tmp/optbench-geo-test")
    loaded = _load_geometries(ROOT, resolve_problem(ROOT, "minimization/lj38"), None, work, work / "lj.dat")
    _name, coords, _box = loaded[0]
    bonds = knearest(coords, 2, None)
    from geometric.molecule import Molecule
    from geometric.nifty import ang2bohr
    from geometric.optimize import DelocalizedInternalCoordinates

    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    molecule = Molecule()
    molecule.elem = ["H"] * len(pos)
    molecule.xyzs = [pos.copy()]
    molecule.Data["bonds"] = list(bonds)
    molecule.top_settings["read_bonds"] = True
    molecule.build_topology(force_bonds=False)
    ic = DelocalizedInternalCoordinates(molecule, build=True, connect=False, addcart=False)
    xyz = molecule.xyzs[0].flatten() * ang2bohr
    shifted = xyz + 1e-3
    reference_values = np.array([item.value(xyz) for item in ic.Prims.Internals])
    reference_deriv = np.array([item.derivative(xyz) for item in ic.Prims.Internals])
    reference_diff = np.array([item.calcDiff(shifted, xyz) for item in ic.Prims.Internals])
    install_batch_primitives()
    values = ic.Prims.calculate(xyz)
    deriv = ic.Prims.derivatives(xyz)
    diff = ic.Prims.calcDiff(shifted, xyz)
    assert np.max(np.abs(values - reference_values)) < 1e-8
    assert np.max(np.abs(deriv - reference_deriv)) < 1e-8
    assert np.max(np.abs(diff - reference_diff)) < 1e-8


def test_coordinate_rebuild_keeps_the_stored_bond_list():
    pytest.importorskip("geometric")
    import numpy as np

    from geometric.molecule import Molecule

    from optbench.runners.geometric_min import install_keep_read_bonds

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )

    def fresh():
        molecule = Molecule()
        molecule.elem = ["H", "H", "H", "H"]
        molecule.xyzs = [pos.copy()]
        molecule.Data["bonds"] = [(0, 1)]
        molecule.top_settings["read_bonds"] = True
        return molecule

    discovered = fresh()
    discovered.build_bonds()
    assert len(discovered.Data["bonds"]) > 1

    install_keep_read_bonds()
    kept = fresh()
    kept.build_topology(force_bonds=True)
    assert kept.Data["bonds"] == [(0, 1)]

    open_list = fresh()
    open_list.top_settings["read_bonds"] = False
    open_list.build_topology(force_bonds=True)
    assert open_list.Data["bonds"] == discovered.Data["bonds"]


def test_cartesian_lj38_start_reaches_the_force_stop():
    pytest.importorskip("geometric")
    from optbench.runners.geometric_min import _minimize_job
    from optbench.runners.rgmin_min import _load_geometries

    work = Path("/tmp/optbench-geo-test")
    loaded = _load_geometries(ROOT, resolve_problem(ROOT, "minimization/lj38"), None, work, work / "lj.dat")
    name, coords, _box = loaded[0]
    record = _minimize_job(
        {
            "index": 0,
            "start": name,
            "coords": coords.tolist(),
            "box": None,
            "kind": "lj",
            "coordsys": "cart",
            "neighbors": 0,
            "element": "H",
            "gtol": 0.01,
            "cap": 10000,
        }
    )
    assert record["success"] is True
    assert record["hessian_index"] == 0
    assert record["force_norm"] <= 0.01
    assert record["force_calls"] < 10000
    assert record["coordinate_system"] == "CartesianCoordinates"
