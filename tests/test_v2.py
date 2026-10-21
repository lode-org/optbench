import json
from pathlib import Path

from optbench.score import distribution_summary, penalised_cost, quantile, score_v2, wilson_interval
from optbench.smoke import build_smoke
from optbench.v2 import parse_repetitions

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_locks_the_published_counts():
    full_low, full_high = wilson_interval(1000, 1000)
    quickmin_low, quickmin_high = wilson_interval(975, 1000)
    steep_low, steep_high = wilson_interval(904, 1000)
    assert f"{full_low:.6f}" == "0.996173"
    assert f"{full_high:.6f}" == "1.000000"
    assert f"{quickmin_low:.6f}" == "0.963355"
    assert f"{quickmin_high:.6f}" == "0.983010"
    assert f"{steep_low:.6f}" == "0.884165"
    assert f"{steep_high:.6f}" == "0.920743"
    zero_low, _zero_high = wilson_interval(0, 10)
    _all_low, all_high = wilson_interval(10, 10)
    assert zero_low == 0.0
    assert all_high == 1.0


def test_rounded_legacy_means_are_not_a_v2_score():
    # Illustration only: every success is charged the rounded 2014 mean.
    assert (975 * 3514 + 25 * 2 * 10000) / 1000 == 3926.15
    assert (904 * 4901 + 96 * 2 * 10000) / 1000 == 6350.504
    assert (999 * 1849 + 1 * 2 * 10000) / 1000 == 1867.151


def test_par2_charges_the_failure_and_bootstrap_is_stable():
    assert penalised_cost(10, True, 10000) == 10
    assert penalised_cost(10000, False, 10000) == 20000
    records = [
        {"force_calls": 10, "success": True},
        {"force_calls": 30, "success": True},
        {"force_calls": 40, "success": False},
    ]
    first = score_v2(records, cap=100, resamples=50, seed=7)
    second = score_v2(records, cap=100, resamples=50, seed=7)
    assert first["par2_mean"] == (10 + 30 + 200) / 3
    assert first["successes"] == 2
    assert first["median"] == 30
    assert first["median_low"] == second["median_low"]
    assert first["median_low"] <= first["median"] <= first["median_high"]
    assert first["iqr"] == quantile([10, 30, 200], 0.75) - quantile([10, 30, 200], 0.25)


def test_pt_island_repetitions_are_per_search():
    text = (ROOT / "results" / "campaign" / "eon-3.5.0-saddle-pt-dimer.dat").read_text(encoding="utf-8")
    rows = parse_repetitions(text)
    assert len(rows) == 10
    assert rows[0]["force_calls"] == 2611052
    assert rows[0]["searches"] == 9936
    assert rows[0]["per_search"] == 2611052 / 9936
    summary = distribution_summary([row["per_search"] for row in rows], resamples=20, seed=1)
    assert f"{summary['median']:.6f}" == "263.284133"
    assert f"{summary['mean']:.6f}" == "263.590450"
    classic = parse_repetitions(
        (ROOT / "results" / "campaign" / "eon-3.5.0-saddle-pt-classic.dat").read_text(encoding="utf-8")
    )
    classic_summary = distribution_summary([row["per_search"] for row in classic], resamples=20, seed=1)
    assert f"{classic_summary['median']:.6f}" == "469.759056"
    no_cap = score_v2(
        [{"force_calls": row["per_search"], "success": True} for row in rows],
        cap=None,
        resamples=20,
        seed=1,
    )
    assert no_cap["par_applied"] is False
    assert no_cap["par2_mean"] is None
    assert no_cap["successes"] == 10


def test_quadratic_smoke_counts_and_rejects_a_false_saddle():
    document = build_smoke(commit="abc", machine="testbox")
    runs = {run["entry"]: run for run in document["runs"]}
    default = runs["gd-step-0.05"]
    assert [start["force_calls"] for start in default["starts"]] == [139, 139]
    assert all(start["success"] for start in default["starts"])
    assert all(start["hessian_index"] == 0 for start in default["starts"])
    assert all(start["verify_calls"] == 4 for start in default["starts"])
    assert default["metrics"]["par2_mean"] == 139
    assert default["tuning"] == "default"
    assert runs["gd-step-0.01"]["starts"][0]["force_calls"] == 720
    assert runs["gd-step-0.01"]["tuning"] == "tuned"
    failed = runs["gd-step-0.001-cap-200"]
    assert failed["starts"][0]["force_calls"] == 200
    assert failed["metrics"]["par2_mean"] == 400
    assert failed["starts"][0]["success"] is False
    saddle = runs["newton-saddle"]["starts"][0]
    assert saddle["force_calls"] == 2
    assert saddle["hessian_index"] == 1
    assert saddle["success"] is True
    assert saddle["rmsd"] == 0.0
    false = runs["bowl-reported-as-saddle"]
    assert false["starts"][0]["force_calls"] == 139
    assert false["metrics"]["par2_mean"] == 10000
    assert "hessian_index" in false["starts"][0]["reasons"]
    assert "rmsd" in false["starts"][0]["reasons"]
    provenance = default["provenance"]
    assert provenance["code"] == "optbench"
    assert provenance["commit"] == "abc"
    assert provenance["parameters"]["step"] == 0.05
    assert provenance["machine"]["node"] == "testbox"
    encoded = json.dumps(document)
    assert "optbench.result/v2" in encoded
    assert "optbench.start/v2" in encoded
