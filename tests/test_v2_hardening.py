"""v2 hardening: one counter, seeds, failures, noise, and problem keys."""

import json
import math
import random
import shutil
from pathlib import Path

import pytest

from optbench.extpot import format_extpot, read_extpot, respond
from optbench.failures import FAILURES, classify_failure, failure_of
from optbench.lj import lj_cluster
from optbench.noise import (
    KAPPA,
    SIGMA_LEVELS,
    WINDOW,
    converged_under_noise,
    gaussian_forces,
    noise_floor,
    stop_threshold,
)
from optbench.potserver import PotServer, RgpotCounter, remote_calculate, remote_count, remote_reset
from optbench.problems import load_all, load_problem
from optbench.profile import performance_profile, profile_cost
from optbench.score import score_v2
from optbench.seeds import descend, dimer_search, displace, random_orientation, run_seeded
from optbench.quadratic import ENERGY_TOL, FORCE_NORM, REFERENCE_MINIMUM, REFERENCE_SADDLE, RMSD_TOL, bowl, saddle
from optbench.statistics import build_statistics
from optbench.v2 import V2_KEYS, record_start

ROOT = Path(__file__).resolve().parents[1]


def _lj(positions, atomic_numbers, box):
    return lj_cluster(positions, atomic_numbers, box)


def test_socket_and_rgpot_share_one_counter(tmp_path):
    server = PotServer(_lj).start()
    try:
        box = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        numbers = [1, 1]
        positions = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0]
        first = remote_calculate(server.host, server.port, positions, numbers, box)
        second = remote_calculate(server.host, server.port, positions, numbers, box)
        direct = lj_cluster(positions, numbers, box)
        assert first[2] == 1
        assert second[2] == 2
        assert first[0] == direct[0]
        assert first[1] == direct[1]
        calls, verify = remote_count(server.host, server.port)
        assert (calls, verify) == (2, 0)
        same = RgpotCounter(_lj, server.counter)
        same.calculate(positions, numbers, box)
        assert server.counter.calls == 3
        remote_reset(server.host, server.port)
        assert server.counter.calls == 0
    finally:
        server.stop()

    request = "\n".join(
        [
            "0 0 0",
            "0 0 0",
            "0 0 0",
            "1 0 0 0",
            "1 1.2 0 0",
        ]
    )
    box, numbers, positions = read_extpot(request)
    assert len(numbers) == 2
    assert len(positions) == 6
    assert len(box) == 9
    (tmp_path / "from_eon_to_extpot").write_text(request, encoding="utf-8")
    server = PotServer(_lj).start()
    try:
        calls = respond(tmp_path, server.endpoint())
        assert calls == 1
        body = (tmp_path / "from_extpot_to_eon").read_text(encoding="utf-8")
        assert body == format_extpot(*lj_cluster(positions, numbers, box))
    finally:
        server.stop()


def test_socket_refuses_the_call_past_the_cap():
    evaluated = []

    def evaluate(positions, atomic_numbers, box):
        evaluated.append(tuple(positions))
        return lj_cluster(positions, atomic_numbers, box)

    server = PotServer(evaluate, cap=2).start()
    try:
        box = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        numbers = [1, 1]
        first = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0]
        second = [0.0, 0.0, 0.0, 1.3, 0.0, 0.0]
        third = [0.0, 0.0, 0.0, 1.4, 0.0, 0.0]
        remote_calculate(server.host, server.port, first, numbers, box)
        remote_calculate(server.host, server.port, second, numbers, box)
        with pytest.raises(RuntimeError, match="cap"):
            remote_calculate(server.host, server.port, third, numbers, box)
        calls, verify = remote_count(server.host, server.port)
        assert (calls, verify) == (2, 0)
        assert evaluated == [tuple(first), tuple(second)]
    finally:
        server.stop()


def test_profile_keeps_an_unsolved_problem_in_the_denominator():
    costs = {"fast": [10.0, 10.0], "capped": [20.0, math.inf]}
    profile = performance_profile(costs)
    assert profile["reference"] == [10.0, 10.0]
    assert profile["solved"]["capped"] == 0.5
    assert profile["rho"]["capped"][0] == 0.0
    assert profile["rho"]["capped"][1] == 0.5
    assert profile["rho"]["fast"] == [1.0, 1.0, 1.0, 1.0, 1.0]
    failed = {"success": False, "force_calls": 30}
    solved = {"success": True, "force_calls": 30}
    assert profile_cost(failed) == math.inf
    assert profile_cost(solved) == 30


def test_seeded_sample_scores_every_start():
    rng = random.Random(1)
    state = rng.getstate()
    assert displace((1.0, 2.0), rng, 0.0) == [1.0, 2.0]
    assert rng.getstate() == state
    orientation = random_orientation(random.Random(3), 2)
    assert math.isclose(math.hypot(*orientation), 1.0)

    starts, metrics = run_seeded(
        bowl,
        x0=(0.0, 0.0),
        kind="minimum",
        reference_energy=0.0,
        reference_coords=REFERENCE_MINIMUM,
        cap=5000,
        force_norm=FORCE_NORM,
        energy_tol=ENERGY_TOL,
        rmsd_tol=RMSD_TOL,
        samples=2,
        seed=20261016,
        displacement=0.02,
        step=0.05,
        mode="displacement",
    )
    assert len(starts) == 2
    assert metrics["n"] == 2
    assert metrics["includes_failures"] is True
    assert sum(metrics[f"n_{name}"] for name in FAILURES) == metrics["nfailed"]
    assert all(start["counter"] == "harness" for start in starts)
    assert all(start["success"] for start in starts)
    assert "median_low" in metrics and "rate_low" in metrics

    outcome = dimer_search(type("P", (), {"energy_forces": staticmethod(saddle)})(), (0.4, -0.3), (1.0, 0.0))
    assert outcome["converged"] is True
    assert outcome["force_norm"] <= FORCE_NORM


def test_descent_from_the_origin_matches_the_smoke_count():
    # The clean bowl from (0, 0) with step 0.05 is the quadratic smoke: 139 calls.
    calls = {"n": 0}

    class Potential:
        def energy_forces(self, coords):
            calls["n"] += 1
            return bowl(coords)

    outcome = descend(Potential(), (0.0, 0.0), 0.05, 5000, FORCE_NORM)
    assert outcome["converged"] is True
    assert calls["n"] == 139


def test_failure_classes_stay_in_the_par2_mean():
    assert classify_failure(["cap", "force_norm"]) == "cap"
    assert classify_failure(["hessian_index", "rmsd"]) == "wrong_stationary"
    assert classify_failure(["force_norm"]) == "wrong_stationary"
    assert classify_failure(["divergence", "force_norm"]) == "divergence"
    assert classify_failure(["duplicate", "barrier"]) == "duplicate"
    assert classify_failure(["barrier"]) == "over_barrier"
    assert classify_failure(["not_connected", "barrier"]) == "over_barrier"
    assert classify_failure(["not_connected"]) == "disconnected"
    assert classify_failure(["crash"]) == "crash"
    assert failure_of({"force_calls": 40, "success": False}, 100) == "divergence"
    assert failure_of({"force_calls": 100, "success": False}, 100) == "cap"

    def crash(potential):
        potential.energy_forces((0.0, 0.0))
        raise RuntimeError("optimizer aborted")

    crashed = record_start(
        "crash",
        bowl,
        crash,
        cap=100,
        kind="minimum",
        reference_energy=0.0,
        reference_coords=REFERENCE_MINIMUM,
        force_norm=FORCE_NORM,
        energy_tol=ENERGY_TOL,
        rmsd_tol=RMSD_TOL,
    )
    assert crashed["failure"] == "crash"
    assert crashed["force_calls"] == 1
    assert crashed["error"] == "RuntimeError"
    assert crashed["hessian_index"] is None

    records = [
        {"force_calls": 139, "success": True, "failure": None},
        {"force_calls": 200, "success": False, "failure": "cap"},
    ]
    summary = score_v2(records, 200, resamples=20, seed=1)
    assert summary["par2_mean"] == (139 + 400) / 2
    assert summary["success_only_mean"] == 139
    assert summary["nfailed"] == 1
    assert summary["n_cap"] == 1
    assert summary["n_wrong_stationary"] == summary["n_divergence"] == summary["n_crash"] == 0
    with pytest.raises(ValueError):
        failure_of({"force_calls": 1, "success": True, "failure": "cap"}, 10)


def test_noise_stop_needs_a_window_and_the_clean_path_is_unchanged():
    assert noise_floor(0.0, 2) == 0.0
    assert stop_threshold(FORCE_NORM, 0.0, 2) == FORCE_NORM
    assert math.isclose(stop_threshold(FORCE_NORM, 1e-2, 2), KAPPA * 1e-2 * math.sqrt(2))
    assert (
        converged_under_noise(
            force_norm=0.0,
            energies=[0.0],
            clean_tol=FORCE_NORM,
            sigma=1e-3,
            components=2,
            energy_tol=1e-3,
            window=WINDOW,
        )
        is False
    )
    assert (
        converged_under_noise(
            force_norm=0.0,
            energies=[0.0] * WINDOW,
            clean_tol=FORCE_NORM,
            sigma=1e-3,
            components=2,
            energy_tol=1e-3,
            window=WINDOW,
        )
        is True
    )
    rng = random.Random(5)
    state = rng.getstate()
    energy, gradient = gaussian_forces(bowl, 0.0, rng)((0.0, 0.0))
    assert rng.getstate() == state
    assert energy == bowl((0.0, 0.0))[0]
    assert gradient == bowl((0.0, 0.0))[1]

    from optbench.noise import noisy_descent

    optimizer = noisy_descent(
        bowl,
        (0.0, 0.0),
        0.05,
        5000,
        sigma=0.0,
        rng=random.Random(1),
        clean_tol=FORCE_NORM,
        energy_tol=ENERGY_TOL,
    )
    start = record_start(
        "clean",
        optimizer.evaluate,
        optimizer,
        cap=5000,
        kind="minimum",
        reference_energy=0.0,
        reference_coords=REFERENCE_MINIMUM,
        force_norm=FORCE_NORM,
        energy_tol=ENERGY_TOL,
        rmsd_tol=RMSD_TOL,
    )
    assert start["force_calls"] == 139
    assert start["success"] is True
    assert start["hessian_index"] == 0


def test_every_problem_carries_legacy_and_the_v2_keys():
    problems = load_all(ROOT / "problems")
    assert "minimization/lj38-noise" in problems
    for ident, problem in problems.items():
        half_even = (problem.get("scoring") or {}).get("half_even") or {}
        version2 = (problem.get("scoring") or {}).get("v2") or {}
        assert half_even.get("label") == "half-even", ident
        assert tuple(version2.get("required") or ()) == V2_KEYS, ident
    noise = load_problem(ROOT / "problems" / "minimization" / "lj38-noise.toml")
    assert noise["noise"]["distribution"] == "gaussian"
    assert noise["noise"]["component"] == "force"
    assert noise["noise"]["sigma"] == list(SIGMA_LEVELS)
    assert noise["noise"]["kappa"] == KAPPA
    assert noise["noise"]["window"] == WINDOW
    assert noise["noise"]["seed"] == 20261016
    text = (ROOT / "docs" / "v2.md").read_text(encoding="utf-8")
    for phrase in ("HarnessCounter", "Dolan", "PAR-2", "rate interval", "wrong_stationary", "half-even"):
        assert phrase in text


def test_committed_statistics_match_a_fresh_run():
    from optbench.asv_score import statistics_from_asv

    path = ROOT / "results" / "validation" / "statistics.json"
    assert path.is_file(), "results/validation/statistics.json is produced by optbench statistics"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["source"] == "asv"
    provenance = stored["provenance"]
    fresh = build_statistics(commit=provenance["commit"], machine=provenance["machine"])
    reread = statistics_from_asv(ROOT, ROOT / "results" / "asv")
    assert reread["provenance"]["commit"] == provenance["commit"]

    def calls(document, section):
        return [start["force_calls"] for start in document[section]["starts"]]

    assert fresh["displacement"]["metrics"]["n"] == stored["displacement"]["metrics"]["n"]
    assert calls(fresh, "displacement") == calls(stored, "displacement") == calls(reread, "displacement")
    assert [start["success"] for start in fresh["dimer"]["starts"]] == [
        start["success"] for start in stored["dimer"]["starts"]
    ]
    assert calls(fresh, "dimer") == calls(stored, "dimer") == calls(reread, "dimer")
    assert fresh["profile"]["calls"] == stored["profile"]["calls"] == reread["profile"]["calls"]
    assert fresh["profile"]["par2"]["gd-step-0.001-cap-30"]["par2_mean"] == stored["profile"]["par2"]["gd-step-0.001-cap-30"]["par2_mean"]
    assert fresh["failures"]["metrics"]["par2_mean"] == stored["failures"]["metrics"]["par2_mean"]
    assert fresh["failures"]["metrics"]["success_only_mean"] == stored["failures"]["metrics"]["success_only_mean"]
    for left, right in zip(fresh["noise"]["runs"], stored["noise"]["runs"], strict=True):
        assert left["sigma"] == right["sigma"]
        assert left["start"]["force_calls"] == right["start"]["force_calls"]
        assert left["start"]["success"] == right["start"]["success"]
        assert left["start"]["failure"] == right["start"]["failure"]
        assert left["start"]["hessian_index"] == 0
    assert stored["status"] == "quadratic"
    assert stored["problem"] == "quadratic/bowl"


def test_committed_counter_matches_eonclient_results():
    dat = ROOT / "results" / "validation" / "lj38-counter.dat"
    starts = ROOT / "results" / "validation" / "lj38-counter.starts.jsonl"
    assert dat.is_file()
    fields = {}
    for line in dat.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        fields[key] = value
    assert fields["published_mean"] == "false"
    assert fields["counter"] == "harness"
    assert fields["problem"] == "minimization/lj38"
    assert fields["template"] == "eon-client/lbfgs-m100"
    assert fields["code_version"] == "3.5.0"
    assert fields["commit"] == "7ec1840"
    assert fields["lbfgs_memory"] == "100"
    rows = [json.loads(line) for line in starts.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == int(fields["n"])
    assert sum(1 for row in rows if row["matched_client"]) == int(fields["matched_client"])
    assert int(fields["matched_client"]) == int(fields["n"])
    client_dir = ROOT / "results" / "validation" / "lj38-counter-client"
    for row in rows:
        assert row["counter"] == "harness"
        assert row["comparable"] is True
        assert row["force_calls"] == row["client_force_calls"]
        text = (client_dir / f"{row['start']}-extpot-results.dat").read_text(encoding="utf-8")
        calls = next(int(float(line.split()[0])) for line in text.splitlines() if line.endswith("total_force_calls"))
        assert calls == row["client_force_calls"]
        builtin = (client_dir / f"{row['start']}-builtin-results.dat").read_text(encoding="utf-8")
        builtin_calls = next(
            int(float(line.split()[0])) for line in builtin.splitlines() if line.endswith("total_force_calls")
        )
        assert builtin_calls == row["builtin_force_calls"]


@pytest.mark.skipif(shutil.which("eonclient") is None, reason="eonclient is not on PATH")
def test_eonclient_subset_matches_the_harness_counter(tmp_path):
    from optbench.validate_counter import validate_subset

    client = Path(shutil.which("eonclient"))
    summary = validate_subset(
        ROOT,
        client,
        tmp_path / "counter.dat",
        limit=1,
        work=tmp_path / "work",
    )
    assert summary["matched_client"] == 1
    assert summary["rows"][0]["force_calls"] == summary["rows"][0]["client_force_calls"]
    assert summary["version"]
