"""PyAMFF templates stay on the two machine-learning problems."""

import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

from optbench.board import stated_full_n
from optbench.campaign import list_templates, resolve_problem
from optbench.runners.pyamff_ml import OPTIMIZERS, _section, _stop, classify_stop, max_abs_converged

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "240f51d01ca22969369f9bc21760c61cbb3b6665"
PROBLEMS = ("mlopt/emt-pt38", "mlopt/C2H4")


def test_stop_classes():
    assert classify_stop(calls=12, cap=100, converged=True, finite=True, error="") == (True, None)
    assert classify_stop(calls=100, cap=100, converged=True, finite=True, error="") == (False, "cap")
    assert classify_stop(calls=100, cap=100, converged=False, finite=True, error="") == (False, "cap")
    assert classify_stop(calls=4, cap=100, converged=False, finite=False, error="") == (False, "divergence")
    assert classify_stop(calls=4, cap=100, converged=False, finite=True, error="RuntimeError") == (False, "crash")
    assert classify_stop(calls=4, cap=100, converged=False, finite=True, error="") == (False, "wrong_stationary")


def test_max_abs_stop_is_strict():
    assert max_abs_converged(0.099999, 0.1) is True
    assert max_abs_converged(0.1, 0.1) is False
    assert max_abs_converged(0.02, 0.02) is False


def test_pyamff_templates_cover_the_archive_cells():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pyamff")
    assert pinned["commit"] == COMMIT
    assert not any(row["id"] == "pyamff-8037eb98" for row in registry["code"])
    found = [row for row in list_templates(ROOT) if row.get("engine") == "pyamff"]
    assert len(found) == 12
    seen = set()
    for template in found:
        assert template["commit"] == COMMIT
        assert template["code_version"] in {"0.1.0-beta", "240f51d0", "240f51d"}
        assert template["code"] == "PyAMFF"
        assert template["tuning"] in {"default", "stock", "archive"}
        assert template.get("ranked") is not False
        assert "8037eb98" not in template["id"]
        assert template["problem"] in PROBLEMS
        problem = resolve_problem(ROOT, template["problem"])
        assert stated_full_n(ROOT, problem) == 50
        cap, tol, coeff = _stop(problem, template)
        section = template.get("pyamff") or {}
        optimizer = _section(template)["optimizer"]
        assert optimizer in OPTIMIZERS
        assert "seed_base" not in section
        if "init_model_parameters" in section:
            assert section["init_model_parameters"] is True
        if template["id"].endswith("-force") or template["id"].endswith("fcoeff0_1"):
            assert coeff == 0.1
            assert cap == 30000
            assert tol == 0.1
        else:
            assert coeff == 0.0
            assert cap == 10000
            assert tol == 0.02
        seen.add((template["problem"], optimizer, coeff))
    for problem in PROBLEMS:
        for name in OPTIMIZERS:
            assert (problem, name, 0.0) in seen
            assert (problem, name, 0.1) in seen


# The Optax derivative note uses this periodic four-atom cell.
_PT4 = [
    [8.6964, 7.8873, 1.5741],
    [8.8873, 4.1428, 8.8971],
    [1.0862, 9.5476, 1.2094],
    [7.3070, 7.0155, 7.4174],
]


def test_fortran_derivative_matches_central_difference(tmp_path, monkeypatch):
    pytest.importorskip("pyamff.fmodules")
    from ase import Atoms
    from pyamff.fingerprints.fingerprints import Fingerprints

    paras = ROOT / "archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0/fpParas.dat"
    shutil.copy(paras, tmp_path / "fpParas.dat")
    monkeypatch.chdir(tmp_path)
    atoms = Atoms("Pt4", positions=_PT4, cell=[10.0, 10.0, 10.0], pbc=True)
    calc = Fingerprints(uniq_elements=["Pt"], filename="fpParas.dat", nfps={"Pt": 8})

    def grab(geom):
        values, primes = calc.calcFPs(geom, ["Pt"] * 4)
        copied = {
            (int(key[0]), key[1], int(key[2]), key[3], int(key[4])): np.array(val, dtype=float, copy=True)[:8]
            for key, val in primes.items()
        }
        return np.array(values, dtype=float, copy=True)[:, :8], copied

    _g0, primes = grab(atoms)
    step = 1e-5
    worst = 0.0
    for atom in range(4):
        for coord in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            moved = plus.get_positions()
            moved[atom, coord] += step
            plus.set_positions(moved)
            moved = minus.get_positions()
            moved[atom, coord] -= step
            minus.set_positions(moved)
            high, _ = grab(plus)
            low, _ = grab(minus)
            finite = (high - low) / (2.0 * step)
            for center in range(4):
                stored = primes.get((atom, "Pt", center, "Pt", coord), np.zeros(8))
                worst = max(worst, float(np.max(np.abs(finite[center] - stored))))
    assert worst < 5e-9
