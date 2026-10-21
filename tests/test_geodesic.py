"""Geodesic initialiser ablation: pin, templates, and the repaired path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import list_templates, load_template, resolve_problem, _check_stop
from optbench.geodesic_path import COMMIT, VERSION, symbols_for
from optbench.rgsaddle_band import _registry

ROOT = Path(__file__).resolve().parents[1]

PROBLEMS = {
    "double-ended-saddle-search/lj38",
    "double-ended-saddle-search/lj38_2ts_separation",
    "double-ended-saddle-search/pt-island",
    "neb/Pt-heptamer",
}

ASE_IDS = {
    "geodesic/ase-de-lj38-default",
    "geodesic/ase-de-lj38-tuned",
    "geodesic/ase-de-2ts-default",
    "geodesic/ase-de-2ts-tuned",
    "geodesic/ase-de-pt-default",
    "geodesic/ase-de-pt-tuned",
    "geodesic/ase-heptamer-default",
    "geodesic/ase-heptamer-tuned",
}

RGSADDLE_IDS = {"geodesic/ci-neb", "geodesic/ci-neb-lbfgs"}


def test_registry_pin_matches_the_path_module():
    pin = _registry(ROOT, "geodesic")
    assert pin["name"] == "geodesic"
    assert pin["commit"] == COMMIT
    assert pin["committed"] == "2026-06-01"
    assert "virtualzx-nad/geodesic-interpolate" in pin["repository"]
    assert VERSION == "1.0.0"


def test_templates_are_an_initialiser_ablation():
    found = [item for item in list_templates(ROOT) if item["id"].startswith("geodesic/")]
    assert {item["id"] for item in found} == ASE_IDS | RGSADDLE_IDS
    band = [item for item in list_templates(ROOT) if item.get("engine") == "rgsaddle-band"]
    assert len(band) == 12
    pin = _registry(ROOT, "geodesic")
    for item in found:
        loaded = load_template(ROOT, item["id"])
        assert loaded["engine"] == "geodesic"
        assert loaded["code"] == "geodesic"
        assert loaded["code_version"] == "1.0.0"
        assert loaded["commit"] == pin["commit"]
        assert loaded["tuning"] == "ablation"
        assert "Initialiser ablation" in loaded["note"]
        assert loaded["backend"] in {"ase", "rgsaddle"}
    for ident in RGSADDLE_IDS:
        loaded = load_template(ROOT, ident)
        assert loaded["backend"] == "rgsaddle"
        assert loaded["band"]["path"] == "geodesic"
        _check_stop(resolve_problem(ROOT, "minimization/lj38"), loaded)
    caps = {
        "geodesic/ase-de-lj38-default": 2500,
        "geodesic/ase-de-lj38-tuned": 2500,
        "geodesic/ase-de-2ts-default": 250000,
        "geodesic/ase-de-2ts-tuned": 250000,
        "geodesic/ase-de-pt-default": 3000,
        "geodesic/ase-de-pt-tuned": 3000,
        "geodesic/ase-heptamer-default": 10000,
        "geodesic/ase-heptamer-tuned": 10000,
    }
    for ident, cap in caps.items():
        loaded = load_template(ROOT, ident)
        assert loaded["problem"] in PROBLEMS
        assert loaded["neb"]["interpolate"] == "geodesic"
        assert int(loaded["stop"]["max_force_calls"]) == cap
        problem = resolve_problem(ROOT, loaded["problem"])
        assert _check_stop(problem, loaded) == cap


def test_symbols_follow_the_potential():
    assert symbols_for("lj", 3) == ["A", "A", "A"]
    assert symbols_for("lennard-jones", 2) == ["A", "A"]
    assert symbols_for("morse_pt", 2) == ["Pt", "Pt"]
    assert symbols_for("morse", 1) == ["Pt"]
    with pytest.raises(ValueError):
        symbols_for("unknown", 1)


def test_path_keeps_endpoints_and_is_not_linear(monkeypatch, tmp_path: Path):
    pytest.importorskip("geodesic_interpolate")
    monkeypatch.setenv("OPTBENCH_GEODESIC_CACHE", str(tmp_path))
    from optbench.geodesic_path import geodesic_positions
    from optbench.pes import interpolate_free

    reactant = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [2.1, 0.8, 0.2],
            [0.4, 1.5, 0.3],
            [1.6, 1.7, 0.1],
            [2.4, 0.2, 1.1],
        ],
        dtype=np.float64,
    )
    product = reactant + np.array(
        [
            [0.3, -0.2, 0.4],
            [-0.4, 0.5, 0.1],
            [0.2, -0.6, 0.3],
            [0.5, 0.2, -0.4],
            [-0.3, 0.4, 0.2],
            [0.1, -0.2, 0.6],
        ],
        dtype=np.float64,
    )
    symbols = symbols_for("lj", len(reactant))
    info: dict = {}
    frames = geodesic_positions(reactant, product, symbols, 2, None, seed=0, info=info)
    again = geodesic_positions(reactant, product, symbols, 2, None, seed=0, info={})
    assert frames.shape == (4, 6, 3)
    assert np.allclose(frames[0], reactant)
    assert np.allclose(frames[-1], product)
    assert np.allclose(frames, again)
    assert info["cached"] is False
    cached: dict = {}
    geodesic_positions(reactant, product, symbols, 2, None, seed=0, info=cached)
    assert cached["cached"] is True
    assert cached["seconds"] == 0.0
    linear = interpolate_free(reactant, product, np.ones(len(reactant), dtype=bool), 2, None)
    linear = linear.reshape(4, 6, 3)
    assert np.max(np.abs(frames[1:-1] - linear[1:-1])) > 1e-3


def test_fixed_atoms_stay_on_the_reactant(monkeypatch, tmp_path: Path):
    pytest.importorskip("geodesic_interpolate")
    monkeypatch.setenv("OPTBENCH_GEODESIC_CACHE", "0")
    from optbench.geodesic_path import geodesic_positions

    reactant = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [1.2, 1.2, 2.4],
            [2.0, 0.4, 2.2],
            [0.3, 1.8, 2.6],
        ],
        dtype=np.float64,
    )
    product = reactant.copy()
    product[3:] += np.array([[0.4, -0.3, 0.2], [-0.2, 0.5, -0.1], [0.3, 0.1, 0.4]])
    free = np.array([False, False, False, True, True, True])
    box = np.array([8.0, 8.0, 12.0])
    frames = geodesic_positions(reactant, product, symbols_for("morse", 6), 2, box, free=free, seed=0)
    assert np.allclose(frames[:, ~free], reactant[~free])
    assert np.allclose(frames[0], reactant)
    assert np.allclose(frames[-1, free], product[free])
