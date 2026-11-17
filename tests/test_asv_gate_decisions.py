"""The asv gate on result files written the way asv writes them.

Timing rows are a regression only when the sample sets differ and the head is
slower. Track rows are exact. A row that cannot be compared is unmeasured, and
an unmeasured row fails the job, so a result that went missing cannot pass.
The result files are real ``asv`` result objects saved to disk and read back.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from asv.results import Results
from asv_runner.statistics import compute_stats

from optbench.asv_gate import (
    _as_list,
    _finite_samples,
    classify_timing,
    classify_track,
    compare_results,
    find_result,
    format_report,
    iter_points,
    main,
)

BASE = "a" * 40
HEAD = "b" * 40
TRACK = "bench.Search.track_force_calls"
TIME = "bench.Search.time_search"
SAMPLES = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01, 0.99]
SLOWER = [1.5, 1.52, 1.48, 1.51, 1.49, 1.5, 1.52, 1.48]
GRID = [["0"], ["0"]]


def _track(value: float = 139.0, *, params=None, version: str = "v1") -> tuple:
    return (value, [value], {"repeat": 1}, GRID if params is None else params, version)


def _timing(samples: list[float], *, version: str = "v1") -> tuple:
    value, stats = compute_stats(samples, 1)
    return (value, samples, stats, GRID, version)


def _save(directory: Path, commit: str, entries: dict[str, tuple], *, machine: str = "optbench-ci", env: str = "uv-pip-py3.12") -> Path:
    header = {"machine": machine, "os": "Linux", "arch": "x86_64", "cpu": "test", "num_cpu": "1", "ram": "1"}
    result = Results(header, [], commit, 0, "3.12", env, {})
    for key, (value, samples, stats, params, version) in entries.items():
        many = isinstance(value, list)
        result._results[key] = value if many else [value]
        result._benchmark_params[key] = params
        result._samples[key] = samples if many else [samples]
        result._stats[key] = stats if many else [stats]
        result._benchmark_version[key] = version
        result._started_at[key] = 0
        result._duration[key] = 0.1
    result.save(str(directory))
    return directory / result._filename


def _pair(tmp_path: Path, base: dict[str, tuple], head: dict[str, tuple]) -> tuple[Results, Results]:
    return (
        Results.load(str(_save(tmp_path / "base", BASE, base))),
        Results.load(str(_save(tmp_path / "head", HEAD, head))),
    )


def _argv(tmp_path: Path, **override) -> list[str]:
    values = {
        "--results": str(tmp_path / "results"),
        "--machine": "optbench-ci",
        "--base": BASE,
        "--head": HEAD,
        "--report": str(tmp_path / "out" / "report.md"),
        "--decision": str(tmp_path / "out" / "decision.json"),
    }
    values.update(override)
    return [item for pair in values.items() for item in pair]


def _results_dir(tmp_path: Path, base: dict[str, tuple], head: dict[str, tuple]) -> None:
    _save(tmp_path / "results", BASE, base)
    _save(tmp_path / "results", HEAD, head)


def test_a_commit_has_one_result_file_on_a_machine_and_the_machine_record_is_not_one(tmp_path):
    saved = _save(tmp_path, BASE, {TRACK: _track()})
    (saved.parent / "machine.json").write_text("{}", encoding="utf-8")
    assert find_result(tmp_path, "optbench-ci", BASE) == saved
    with pytest.raises(SystemExit, match=r"expected one result for cccccccc on machine optbench-ci in .*, found none"):
        find_result(tmp_path, "optbench-ci", "c" * 40)
    other = _save(tmp_path, BASE, {TRACK: _track()}, env="other-env")
    with pytest.raises(SystemExit) as caught:
        find_result(tmp_path, "optbench-ci", BASE)
    assert f"found {other.name}, {saved.name}" in str(caught.value)


def test_the_gate_passes_when_the_tracks_match_and_the_timing_stays_in_the_noise(tmp_path, capsys):
    _results_dir(
        tmp_path,
        {TRACK: _track(), TIME: _timing(SAMPLES)},
        {TRACK: _track(), TIME: _timing(list(reversed(SAMPLES)))},
    )
    assert main(_argv(tmp_path)) == 0
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out == report + "\n"
    assert report.splitlines()[-1] == "No regression: timing shifts sit inside the sample noise, and tracks match."
    decision = json.loads((tmp_path / "out" / "decision.json").read_text(encoding="utf-8"))
    assert (decision["machine"], decision["base"], decision["head"]) == ("optbench-ci", BASE, HEAD)
    assert decision["regression"] is False
    assert decision["unmeasured"] == []
    assert [(row["benchmark"], row["kind"]) for row in decision["rows"]] == [(TIME, "ok"), (TRACK, "ok")]


def test_the_gate_fails_when_a_track_moves_in_either_direction_and_says_which(tmp_path, capsys):
    _results_dir(tmp_path, {TRACK: _track(139.0)}, {TRACK: _track(140.0)})
    assert main(_argv(tmp_path)) == 1
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert f"| + | 139 | 140 | {140 / 139:.6g} | {TRACK} |" in report
    assert report.splitlines()[-1].startswith("Regression: a timing change cleared the sample test")
    decision = json.loads((tmp_path / "out" / "decision.json").read_text(encoding="utf-8"))
    assert decision["regression"] is True
    assert decision["rows"][0]["ratio"] == pytest.approx(140 / 139)
    capsys.readouterr()


def test_a_result_file_that_holds_another_commit_than_its_name_is_refused(tmp_path):
    other = "c" * 40
    saved = _save(tmp_path / "results", other, {TRACK: _track()})
    saved.rename(saved.with_name("aaaaaaaa-uv-pip-py3.12.json"))
    _save(tmp_path / "results", HEAD, {TRACK: _track()})
    with pytest.raises(SystemExit, match=f"base result {other} does not match {BASE}"):
        main(_argv(tmp_path))
    assert not (tmp_path / "out").exists()


def test_results_from_two_machines_are_not_compared(tmp_path):
    base = Results.load(str(_save(tmp_path / "base", BASE, {TRACK: _track()}, machine="optbench-ci")))
    head = Results.load(str(_save(tmp_path / "head", HEAD, {TRACK: _track()}, machine="other-machine")))
    with pytest.raises(SystemExit, match="result machines differ: 'optbench-ci' and 'other-machine'"):
        compare_results(base, head)


def test_a_benchmark_only_one_side_ran_is_unmeasured_and_fails_the_job(tmp_path):
    base, head = _pair(tmp_path, {TRACK: _track(), TIME: _timing(SAMPLES)}, {TRACK: _track()})
    decision = compare_results(base, head)
    assert decision["regression"] is True
    assert decision["unmeasured"] == [TIME]
    assert [(row["benchmark"], row["mark"], row["kind"], row["before"], row["after"], row["ratio"]) for row in decision["rows"]] == [
        (TIME, "x", "unmeasured", None, None, None),
        (TRACK, " ", "ok", 139.0, 139.0, 1.0),
    ]


def test_a_point_asv_has_no_value_for_is_unmeasured_and_the_others_are_still_compared(tmp_path):
    grid = [["1", "2"], ["0"]]
    base, head = _pair(tmp_path, {TRACK: _track(params=grid)}, {TRACK: _track(params=grid)})
    decision = compare_results(base, head)
    assert decision["regression"] is True
    assert decision["unmeasured"] == [f"{TRACK}['2', '0']"]
    assert [(row["benchmark"], row["kind"]) for row in decision["rows"]] == [
        (f"{TRACK}['1', '0']", "ok"),
        (f"{TRACK}['2', '0']", "unmeasured"),
    ]


def test_a_parameter_grid_that_repeats_a_tuple_leaves_the_whole_benchmark_unmeasured(tmp_path):
    grid = [["1", "1"], ["0"]]
    base, head = _pair(tmp_path, {TRACK: _track(params=grid)}, {TRACK: _track(params=grid)})
    with pytest.raises(ValueError, match="parameter grid does not match its values"):
        list(iter_points(base, TRACK))
    decision = compare_results(base, head)
    assert decision["regression"] is True
    assert decision["unmeasured"] == [TRACK]
    assert [row["kind"] for row in decision["rows"]] == ["unmeasured"]


def test_the_points_of_a_grid_come_in_product_order_with_the_benchmark_version(tmp_path):
    grid = [["1", "2"], ["x", "y"]]
    entry = ([10.0, 20.0, 30.0, 40.0], [[1.0], [2.0], [3.0], [4.0]], [{"repeat": 1}] * 4, grid, "v7")
    saved = Results.load(str(_save(tmp_path, BASE, {TRACK: entry})))
    points = list(iter_points(saved, TRACK))
    assert [(combo, version, value) for combo, version, value, _stats, _samples in points] == [
        (("1", "x"), "v7", 10.0),
        (("1", "y"), "v7", 20.0),
        (("2", "x"), "v7", 30.0),
        (("2", "y"), "v7", 40.0),
    ]


def test_a_value_the_stats_and_the_samples_are_spread_over_a_grid_the_way_the_grid_needs():
    assert _as_list([1, 2, 3], 3) == [1, 2, 3]
    assert _as_list([7], 3) == [7, 7, 7]
    assert _as_list(7, 2) == [7, 7]
    assert _as_list([1, 2], 3) == [[1, 2], [1, 2], [1, 2]]


def test_samples_keep_their_finite_numbers_and_nothing_else():
    assert _finite_samples([1.0, None, float("nan"), 2]) == [1.0, 2.0]
    assert _finite_samples([None, float("nan")]) is None
    assert _finite_samples([]) is None
    assert _finite_samples(None) is None
    assert _finite_samples([[1.0], [2.0]]) is None


def test_a_head_that_is_faster_beyond_the_noise_is_an_improvement():
    _, base_stats = compute_stats(SLOWER, 1)
    _, head_stats = compute_stats(SAMPLES, 1)
    kind, ratio = classify_timing(SLOWER, SAMPLES, base_stats, head_stats)
    assert kind == "improved"
    assert ratio == pytest.approx(1 / 1.5, rel=0.05)


def test_unusable_samples_or_stats_leave_a_timing_row_unmeasured():
    _, stats = compute_stats(SAMPLES, 1)
    assert classify_timing(SAMPLES, SAMPLES, None, stats) == ("unmeasured", None)
    assert classify_timing(SAMPLES, SAMPLES, stats, "none") == ("unmeasured", None)
    assert classify_timing([0.0, 0.0, 0.0], SAMPLES, stats, stats) == ("unmeasured", None)
    assert classify_timing([None, float("nan"), 1.0], SAMPLES, stats, stats) == ("unmeasured", None)
    assert classify_timing(SAMPLES, [[1.0], [2.0]], stats, stats) == ("unmeasured", None)


def test_missing_samples_are_dropped_before_the_medians_are_compared():
    noisy = [1.0, None, float("nan"), 1.01, 0.99, 1.02, 0.98]
    _, stats = compute_stats([1.0, 1.01, 0.99, 1.02, 0.98], 1)
    kind, ratio = classify_timing(noisy, list(reversed(noisy)), stats, stats)
    assert kind == "ok"
    assert ratio == pytest.approx(1.0)


def test_a_track_is_unmeasured_when_either_value_is_missing_and_exact_otherwise():
    assert classify_track(None, 3.0) == ("unmeasured", None)
    assert classify_track(3.0, None) == ("unmeasured", None)
    assert classify_track(0.0, 0.0) == ("ok", None)
    assert classify_track(0.0, 3.0) == ("regression", None)
    assert classify_track(139.0, 139.0 + 1e-10) == ("ok", 1.0)
    assert classify_track(139.0, 139.0 + 1e-6)[0] == "regression"
    kind, ratio = classify_track(139.0, 278.0)
    assert (kind, ratio) == ("regression", 2.0)
    assert math.isclose(classify_track(200.0, 100.0)[1], 0.5)


def test_the_report_is_a_table_of_marks_values_and_ratios_with_a_verdict_line():
    decision = {
        "regression": True,
        "rows": [
            {"benchmark": "bench.A.track_x", "mark": "+", "kind": "regression", "before": 139.0, "after": 140.5, "ratio": 140.5 / 139.0},
            {"benchmark": "bench.B.time_y", "mark": " ", "kind": "ok", "before": 0.0123456789, "after": 0.0124, "ratio": 1.0045},
            {"benchmark": "bench.C.track_z", "mark": "x", "kind": "unmeasured", "before": None, "after": None, "ratio": None},
        ],
    }
    assert format_report(decision) == (
        "| Change | Before | After | Ratio | Benchmark |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| + | 139 | 140.5 | 1.01079 | bench.A.track_x |\n"
        "|   | 0.0123457 | 0.0124 | 1.0045 | bench.B.time_y |\n"
        "| x | n/a | n/a | n/a | bench.C.track_z |\n"
        "\n"
        "Regression: a timing change cleared the sample test, or a track changed, or a row was unmeasured.\n"
    )
