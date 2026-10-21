"""LAMMPS climbing-image NEB templates, call counts, and the spec potential."""

import os
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import geometry_archive, list_templates, load_template, resolve_problem
from optbench.lammps_neb import (
    COMMIT,
    calls_from_blocks,
    iteration_budget,
    lammps_executable,
    minimization_blocks,
    probe_pair,
    render_neb_input,
    search_named,
)
from optbench.pes import evaluate, load_pairs

ROOT = Path(__file__).resolve().parents[1]
BINARY = lammps_executable()


def _settings(min_style: str, kind: str, images: int, cap: int, force_norm: float, max_move: float) -> dict:
    return {
        "potential": kind,
        "n_interior": images,
        "n_images": images + 2,
        "max_move": max_move,
        "force_norm": force_norm,
        "cap": cap,
        "min_style": min_style,
        "spring_k": 1.0,
        "parallel": "neigh",
        "timestep": 0.005 if kind == "lj" else 0.001,
        "units": "lj" if kind == "lj" else "metal",
    }


def test_registry_commit_matches_both_templates():
    registry = __import__("tomllib").loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "lammpsneb")
    assert row["commit"] == COMMIT
    assert row["name"] == "LAMMPS"
    assert row["ref"] == "stable_30Sep2026"
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "lammps-neb":
            continue
        found.append(template["id"])
        loaded = load_template(ROOT, template["id"])
        assert loaded["commit"] == row["commit"]
        assert loaded["code"] == "LAMMPS"
        assert loaded["code_version"] == "30 Sep 2026"
        assert loaded["tuning"] == "default"
        assert loaded["band"]["climbing"] is True
        assert loaded["band"]["spring_k"] == 1.0
        assert loaded["band"]["parallel"] == "neigh"
    assert load_template(ROOT, "lammpsneb/ci-fire")["band"]["call_reserve"] == 0.2
    assert load_template(ROOT, "lammpsneb/ci-quickmin")["band"]["call_reserve"] == 0.0
    assert found == ["lammpsneb/ci-fire", "lammpsneb/ci-quickmin"]


def test_iteration_budget_stays_under_each_problem_cap():
    expected = {
        "double-ended-saddle-search/lj38": (7, 2500),
        "double-ended-saddle-search/lj38_2ts_separation": (7, 250000),
        "double-ended-saddle-search/pt-island": (5, 3000),
        "neb/Pt-heptamer": (10, 10000),
    }
    for name, (images, cap) in expected.items():
        problem = resolve_problem(ROOT, name)
        assert int(problem["band"]["interior_images"]) + 2 == images
        assert int(problem["convergence"]["max_force_calls"]) == cap
        n1, n2 = iteration_budget(images, cap, 0.0)
        assert n1 >= 1 and n2 >= 1
        assert images * (n1 + n2 + 4) <= cap - 1
        fire_n1, fire_n2 = iteration_budget(images, cap, 0.2)
        assert fire_n1 + fire_n2 < n1 + n2
        assert images * (fire_n1 + fire_n2 + 4) <= int((cap - 1) * 0.8)


def test_minimization_stats_count_setup_and_the_early_extra():
    text = "\n".join(
        [
            "Minimization stats:",
            "  Stopping criterion = max iterations",
            "  Iterations, force evaluations = 20 21",
            "Minimization stats:",
            "  Stopping criterion = force tolerance",
            "  Iterations, force evaluations = 3 3",
        ]
    )
    blocks = minimization_blocks(text)
    assert blocks == [("max iterations", 21), ("force tolerance", 3)]
    assert calls_from_blocks(blocks) == (1 + 21) + (1 + 3 + 1)


def test_input_uses_the_spec_pair_style_and_holds_the_endpoints():
    quick = render_neb_input(_settings("quickmin", "lj", 5, 2500, 0.001, 0.05), n1=176, n2=177, frozen=False)
    assert quick.index("neb ") < quick.index("write_dump")
    assert "pair_style lj/cut 40.0" in quick
    assert "pair_modify shift no" in quick
    assert "neighbor 0.3 bin" in quick
    assert "boundary f f f" in quick
    assert "min_style quickmin" in quick
    assert "min_modify dmax 0.05 norm two" in quick
    assert "fix neb1 neb neb 1 parallel neigh" in quick
    assert "partition yes 1 fix hold neb setforce 0.0 0.0 0.0" in quick
    assert "partition yes 7 fix hold neb setforce 0.0 0.0 0.0" in quick
    assert "neb 0.0 0.001 176 177 1 final final.neb" in quick
    assert "timestep" not in quick
    fire = render_neb_input(_settings("fire", "morse_pt", 8, 10000, 0.01, 0.2), n1=497, n2=498, frozen=True)
    assert "pair_style morse 9.5" in fire
    assert "pair_coeff * * 0.7102 1.6047 2.897" in fire
    assert "pair_modify shift yes" in fire
    assert "neighbor 0.0 multi" in fire
    assert "comm_modify mode multi" in fire
    assert "boundary p p p" in fire
    assert "min_style fire" in fire
    assert "fix freeze frozen setforce 0.0 0.0 0.0" in fire
    assert "partition yes 10 fix hold neb setforce" in fire


def test_board_index_names_the_lammps_classes():
    from optbench.board import _benchmark_index

    index = _benchmark_index()
    quick = index["bench_lammpsneb.HeptamerLammpsnebCiQuickmin"]
    fire = index["bench_lammpsneb.Lj38DeLammpsnebCiFire"]
    assert quick == {"template": "lammpsneb/ci-quickmin", "problem": "neb/Pt-heptamer", "cap": 10000}
    assert fire["template"] == "lammpsneb/ci-fire"
    assert fire["problem"] == "double-ended-saddle-search/lj38"
    assert fire["cap"] == 2500


def test_default_asv_grid_is_one_start():
    from benchmarks.bench_lammpsneb import HeptamerLammpsnebCiFire, Lj38DeLammpsnebCiQuickmin

    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    assert len(HeptamerLammpsnebCiFire.params[1]) == (13 if full else 1)
    assert len(Lj38DeLammpsnebCiQuickmin.params[1]) == (50 if full else 1)
    assert HeptamerLammpsnebCiFire.param_names == ["seed", "start"]
    assert HeptamerLammpsnebCiFire.track_force_calls.unit == "force calls"


@pytest.mark.skipif(BINARY is None, reason="LAMMPS binary is not available")
def test_lammps_pair_matches_the_harness_on_lj_and_morse():
    lj = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    pair = load_pairs(geometry_archive(ROOT, lj), lj["geometry"]["kind"], 50)[0]
    energy, grad = probe_pair("lj", pair.reactant, None, binary=BINARY)
    harness_energy, harness_grad = evaluate("lj", pair.reactant, None)
    assert energy == pytest.approx(harness_energy, abs=1e-8)
    assert np.allclose(grad, harness_grad, atol=1e-8)

    hept = resolve_problem(ROOT, "neb/Pt-heptamer")
    slab = load_pairs(geometry_archive(ROOT, hept), hept["geometry"]["kind"], 13)[0]
    moved = np.array(slab.reactant, copy=True)
    moved[slab.free][0, 0] += 0.2
    morse_energy, morse_grad = probe_pair("morse_pt", moved, slab.box, binary=BINARY)
    harness_energy, harness_grad = evaluate("morse_pt", moved, slab.box)
    assert morse_energy == pytest.approx(harness_energy, abs=1e-8)
    assert np.allclose(morse_grad, harness_grad, atol=1e-8)


@pytest.mark.skipif(BINARY is None, reason="LAMMPS binary is not available")
def test_short_quickmin_band_counts_pair_computes_and_holds_the_reactant(tmp_path: Path):
    record = search_named(
        ROOT,
        "double-ended-saddle-search/lj38",
        "lammpsneb/ci-quickmin",
        "pair_00",
        n1=2,
        n2=2,
        work=tmp_path,
    )
    assert record["failure"] == "divergence"
    assert record["success"] is False
    assert record["n_images"] == 7
    logs = [tmp_path.joinpath(f"log.lammps.{index}").read_text(encoding="utf-8") for index in range(7)]
    assert record["force_calls"] == sum(calls_from_blocks(minimization_blocks(text)) for text in logs)
    assert record["force_calls"] == 7 * (2 + 2 + 2)
    from optbench.lammps_neb import _coords_from_dump

    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    pair = load_pairs(geometry_archive(ROOT, problem), problem["geometry"]["kind"], 50)[0]
    held = _coords_from_dump(tmp_path / "final.0.dump", pair.reactant.shape[0])
    assert np.allclose(held, pair.reactant, atol=1e-8)
