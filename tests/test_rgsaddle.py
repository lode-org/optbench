"""Geometry, band layout, and asv shape for the rgsaddle entries."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_rgsaddle import (
    PROBLEMS,
    TEMPLATES,
    TRACKS,
    HeptamerRgsaddleCiNeb,
    verify_asv_tracks,
)
from optbench.asv_tracks import TrackedSearch
from optbench.campaign import geometry_archive, list_templates, load_template, resolve_problem
from optbench.pes import (
    energy_weighted_springs,
    idpp_objective,
    lennard_jones,
    load_pairs,
    morse_pt,
    read_xyz,
)
from optbench.rgsaddle_band import (
    BandConfig,
    MinModeConfig,
    Report,
    _Request,
    _dat,
    _registry,
    library_path,
    publish_sidecars,
    vibrational_index,
)
from optbench.stationary import jacobi_eigenvalues

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "38c3137f62ce1436d662bb37188a8d7cd2d789b8"


def _problem_pairs(name: str):
    problem = resolve_problem(ROOT, name)
    return load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )


def test_struct_sizes_match_the_c_layout():
    assert __import__("ctypes").sizeof(BandConfig) == 96
    assert __import__("ctypes").sizeof(Report) == 64
    assert __import__("ctypes").sizeof(_Request) == 64
    assert __import__("ctypes").sizeof(MinModeConfig) == 88


def test_jacobi_matches_numpy_on_a_symmetric_matrix():
    rng = np.random.default_rng(0)
    factor = rng.normal(size=(6, 6))
    matrix = factor.T @ factor
    values, _vectors = jacobi_eigenvalues(matrix.tolist())
    assert np.allclose(values, np.linalg.eigvalsh(matrix), atol=1e-8)


def test_rigid_projection_drops_a_translational_negative_mode():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    translation = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    translation /= np.linalg.norm(translation)
    hessian = -np.outer(translation, translation)
    assert vibrational_index(hessian, coords, project_rigid=False) == 1
    assert vibrational_index(hessian, coords, project_rigid=True) == 0
    stretch = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    stretch /= np.linalg.norm(stretch)
    bond = -np.outer(stretch, stretch)
    assert vibrational_index(bond, coords, project_rigid=True) == 1


def test_morse_and_lj_gradients_match_a_central_difference():
    rng = np.random.default_rng(1)
    morse_coords = rng.normal(size=(5, 3))
    _energy, morse_grad = morse_pt(morse_coords, None)
    lj_coords = rng.normal(size=(4, 3))
    _lj_energy, lj_grad = lennard_jones(lj_coords, None)
    step = 1e-6
    for coords, grad, potential in (
        (morse_coords, morse_grad, morse_pt),
        (lj_coords, lj_grad, lennard_jones),
    ):
        numeric = np.zeros_like(coords)
        for index in np.ndindex(coords.shape):
            plus = coords.copy()
            minus = coords.copy()
            plus[index] += step
            minus[index] -= step
            numeric[index] = (potential(plus, None)[0] - potential(minus, None)[0]) / (2 * step)
        assert np.allclose(grad, numeric, atol=1e-6)


def test_energy_weighted_springs_follow_the_eon_span():
    energies = np.array([0.0, 1.0, 5.0, 2.0, 0.0])
    constants = energy_weighted_springs(energies, 0.97, 9.7)
    assert constants[0] == pytest.approx((1 - 0.8) * 9.7 + 0.8 * 0.97)
    assert constants[1] == pytest.approx(9.7)
    assert constants[2] == pytest.approx(9.7)
    assert constants[3] == pytest.approx(0.4 * 9.7 + 0.6 * 0.97)
    assert np.allclose(energy_weighted_springs(np.array([1.0, 1.0, 1.0])), np.full(2, 0.97))


def test_idpp_gradient_matches_a_central_difference():
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(4, 3)) + np.array([0.0, 0.0, 0.0])
    reactant = coords - 0.2
    product = coords + 0.4
    target = 0.5 * (
        np.linalg.norm(reactant[:, None, :] - reactant[None, :, :], axis=-1)
        + np.linalg.norm(product[:, None, :] - product[None, :, :], axis=-1)
    )
    _energy, grad = idpp_objective(coords, target, None)
    step = 1e-6
    numeric = np.zeros_like(coords)
    for index in np.ndindex(coords.shape):
        plus = coords.copy()
        minus = coords.copy()
        plus[index] += step
        minus[index] -= step
        numeric[index] = (idpp_objective(plus, target)[0] - idpp_objective(minus, target)[0]) / (2 * step)
    assert np.allclose(grad, numeric, atol=1e-5)


def test_xyz_keeps_a_blank_comment_line():
    text = "2\n\nPt 0 0 0\nPt 1 0 0\n"
    coords = read_xyz(text)
    assert coords.shape == (2, 3)
    assert coords[1, 0] == 1.0


def test_archived_pair_counts_and_a_relaxed_heptamer_endpoint():
    expected = {
        "neb/Pt-heptamer": 13,
        "double-ended-saddle-search/pt-island": 59,
        "double-ended-saddle-search/lj38": 50,
        "double-ended-saddle-search/lj38_2ts_separation": 51,
    }
    for name, count in expected.items():
        pairs = _problem_pairs(name)
        assert len(pairs) == count
    heptamer = _problem_pairs("neb/Pt-heptamer")[0]
    _energy, grad = morse_pt(heptamer.reactant, heptamer.box)
    assert np.linalg.norm(grad[heptamer.free]) < 1e-8
    assert int(np.count_nonzero(heptamer.free)) == 175
    lj = _problem_pairs("double-ended-saddle-search/lj38")[0]
    assert lj.reactant.shape == (38, 3)
    assert bool(np.all(lj.free))


def test_templates_name_the_registry_commit():
    registry = _registry(ROOT)
    assert registry["commit"] == COMMIT
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "rgsaddle-band":
            continue
        found.append(template["id"])
        loaded = load_template(ROOT, template["id"])
        assert loaded["engine"] == "rgsaddle-band"
        assert loaded["commit"] == registry["commit"]
        assert loaded["code_version"] == "0.2.2"
        assert "ini" in loaded
    assert len(found) == 12
    assert {name for _prefix, name in TEMPLATES} == set(found)


def test_dat_mean_is_half_even_over_successes_only():
    template = {
        "code": "rgsaddle",
        "code_version": "0.2.2",
        "commit": COMMIT,
        "algorithm": "CI-NEB",
        "id": "rgsaddle/ci-neb",
        "tuning": "default",
    }
    settings = {"n_interior": 8}
    records = [
        {"success": True, "force_calls": 10, "hessian_index": 1},
        {"success": False, "force_calls": 99, "hessian_index": 0},
        {"success": True, "force_calls": 11, "hessian_index": 1},
    ]
    text = _dat(records, template, settings)
    fields = {
        line.split()[0]: line.split()[1]
        for line in text.splitlines()
        if line and not line.startswith("#")
    }
    assert fields["force_calls"] == "10"
    assert fields["successes"] == "2"
    assert fields["nfailed"] == "1"
    assert fields["n"] == "3"
    assert fields["transition_states"] == "2"
    assert fields["counter"] == "harness"
    assert fields["commit"] == COMMIT
    failed = _dat(
        [{"success": False, "force_calls": 4, "hessian_index": None}],
        template,
        settings,
    )
    assert not any(line.startswith("force_calls ") for line in failed.splitlines())


def test_asv_classes_parameterize_seed_and_the_selected_start(monkeypatch):
    assert issubclass(HeptamerRgsaddleCiNeb, TrackedSearch)
    assert HeptamerRgsaddleCiNeb.param_names == ["seed", "start"]
    assert HeptamerRgsaddleCiNeb.params[0] == ["0"]
    assert HeptamerRgsaddleCiNeb.repeat == 1
    assert HeptamerRgsaddleCiNeb.rounds == 1
    assert HeptamerRgsaddleCiNeb.number == 1
    assert HeptamerRgsaddleCiNeb.min_run_count == 1
    assert HeptamerRgsaddleCiNeb.warmup_time == 0
    assert HeptamerRgsaddleCiNeb.sample_time == 0
    assert HeptamerRgsaddleCiNeb.timeout == 1800
    assert HeptamerRgsaddleCiNeb.cap == 10000
    assert HeptamerRgsaddleCiNeb.track_force_calls.unit == "force calls"
    assert HeptamerRgsaddleCiNeb.track_success.unit == "fraction"
    assert HeptamerRgsaddleCiNeb.track_failure_divergence.unit == "count"
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    counts = {name: count for name, count in (
        ("neb/Pt-heptamer", 13),
        ("double-ended-saddle-search/pt-island", 59),
        ("double-ended-saddle-search/lj38", 50),
        ("double-ended-saddle-search/lj38_2ts_separation", 51),
    )}
    caps = {
        "neb/Pt-heptamer": 10000,
        "double-ended-saddle-search/pt-island": 3000,
        "double-ended-saddle-search/lj38": 2500,
        "double-ended-saddle-search/lj38_2ts_separation": 10000,
    }
    import benchmarks.bench_rgsaddle as bench

    assert len(bench.CLASS_TEMPLATE) == len(PROBLEMS) * len(TEMPLATES)
    for prefix, problem_name in PROBLEMS:
        names = [pair.name for pair in _problem_pairs(problem_name)]
        assert len(names) == counts[problem_name]
        assert bench.selected_starts(names) == names[:1]
        monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
        assert bench.selected_starts(names) == names
        monkeypatch.delenv("OPTBENCH_ASV_FULL")
        for template_prefix, template_name in TEMPLATES:
            cls = getattr(bench, f"{prefix}Rgsaddle{template_prefix}")
            assert cls.params[1] == names[:1]
            assert cls.cap == caps[problem_name]
            assert cls.template_name == template_name
            for track in TRACKS:
                assert hasattr(cls, track)
            assert hasattr(cls, "time_search")


def test_publish_refuses_a_partial_start_set(tmp_path: Path):
    names = [pair.name for pair in _problem_pairs("neb/Pt-heptamer")]
    directory = tmp_path / "sidecars"
    for index, name in enumerate(names[:-1]):
        _write_fake(directory, name, index)
    with pytest.raises(ValueError):
        publish_sidecars(ROOT, directory, tmp_path / "out")


def test_publish_writes_the_full_set(tmp_path: Path):
    names = [pair.name for pair in _problem_pairs("neb/Pt-heptamer")]
    directory = tmp_path / "sidecars"
    for index, name in enumerate(names):
        _write_fake(directory, name, index)
    summaries = publish_sidecars(ROOT, directory, tmp_path / "out")
    assert summaries[0]["successes"] == 12
    assert summaries[0]["nfailed"] == 1
    assert summaries[0]["starts"] == 13
    text = Path(summaries[0]["output"]).read_text(encoding="utf-8")
    assert "template rgsaddle/ci-neb" in text
    starts = [
        json.loads(line)
        for line in Path(summaries[0]["output"]).with_suffix(".starts.jsonl").read_text().splitlines()
    ]
    assert [row["start"] for row in starts] == names
    assert {row["counter"] for row in starts} == {"harness"}


def test_verify_asv_tracks_reads_the_sidecar(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPTBENCH_RESULT_ROOT", str(tmp_path))
    record = _fake_record(HeptamerRgsaddleCiNeb.params[1][0], 0)
    path = (
        tmp_path
        / "results"
        / "rgsaddle-asv"
        / "neb-Pt-heptamer--rgsaddle-ci-neb"
        / "0"
        / f"{record['start']}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    benchmark = "bench_rgsaddle.HeptamerRgsaddleCiNeb.track_force_calls"
    document = {
        "result_columns": ["result", "params"],
        "results": {
            benchmark: [
                [float(record["force_calls"])],
                [["0"], [record["start"]]],
            ]
        },
    }
    verify_asv_tracks(document)
    document["results"][benchmark][0] = [float(record["force_calls"]) + 1]
    with pytest.raises(ValueError):
        verify_asv_tracks(document)


def _fake_record(name: str, index: int) -> dict:
    success = index % 13 != 0
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "seed": "0",
        "force_calls": 10 + index,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": None if success else "divergence",
        "reasons": [] if success else ["force_norm"],
        "energy": 0.0,
        "force_norm": 0.0 if success else 1.0,
        "hessian_index": 1 if success else 0,
        "converged": True,
        "candidate": "climbing",
        "ci_index": 1,
        "refined": False,
        "surrogate_steps": 0,
        "problem": "neb/Pt-heptamer",
        "template": "rgsaddle/ci-neb",
        "code": "rgsaddle",
        "code_version": "0.2.2",
        "commit": COMMIT,
        "algorithm": "CI-NEB",
        "tuning": "default",
    }


def _write_fake(directory: Path, name: str, index: int) -> None:
    path = directory / "rgsaddle-ci-neb" / "0" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_fake_record(name, index)) + "\n", encoding="utf-8")


@pytest.mark.skipif(library_path() is None, reason="librgsaddle.so is not built")
def test_double_well_ci_neb_is_an_index_one_saddle(monkeypatch):
    from optbench.potserver import RgpotCounter
    from optbench.rgsaddle_band import run_band_search

    charged = {"n": 0}
    real_calculate = RgpotCounter.calculate

    def _calculate(self, positions, atomic_numbers=None, box=None):
        charged["n"] += 1
        return real_calculate(self, positions, atomic_numbers, box)

    monkeypatch.setattr(RgpotCounter, "calculate", _calculate)

    def evaluate_full(coords):
        # Minima at (+-1, 0, 0). The valley bows out to y = 0.5, so the
        # straight initial path has a perpendicular force and climbing
        # can arm. A straight double well has zero NEB force on the
        # chord and the climbing trigger never leaves its baseline.
        xyz = np.asarray(coords, dtype=np.float64).reshape(1, 3)
        x, y, z = xyz[0]
        bend = y - 0.5 + 0.5 * x * x
        energy = (x * x - 1.0) ** 2 + bend * bend + z * z
        grad = np.array(
            [[4.0 * x * (x * x - 1.0) + 2.0 * x * bend, 2.0 * bend, 2.0 * z]]
        )
        return float(energy), grad

    reactant = np.array([[-1.0, 0.0, 0.0]])
    product = np.array([[1.0, 0.0, 0.0]])
    record = run_band_search(
        name="double-well",
        reactant=reactant,
        product=product,
        free=np.array([True]),
        box=None,
        evaluate_full=evaluate_full,
        n_interior=4,
        force_norm=1e-2,
        cap=2000,
        max_move=0.2,
        climbing=True,
        spring="uniform",
        path="linear",
        refine="",
        method="fire",
        memory=20,
        spring_k=5.0,
        k_min=0.97,
        k_max=9.7,
        ci_trigger_factor=0.5,
        ci_trigger_force=0.0,
        idpp_spring_k=1.0,
        idpp_force_tol=0.01,
        idpp_max_steps=80,
        sidpp_stage_steps=50,
        project_rigid=False,
        keep_translation=True,
    )
    assert record["success"], record
    assert record["failure"] is None
    assert record["hessian_index"] == 1
    assert record["verify_calls"] == 6
    assert record["force_calls"] < 2000
    assert record["counter"] == "harness"
    assert charged["n"] >= record["force_calls"] + record["verify_calls"]
    assert record["energy"] == pytest.approx(1.0, abs=1e-3)
