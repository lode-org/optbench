import json
from pathlib import Path

import pytest
from asv.results import Results
from asv_runner.statistics import compute_stats

from optbench.asv_gate import (
    classify_timing,
    classify_track,
    compare_results,
    ensure_machine,
    format_report,
)


def _result(
    tmp: Path,
    commit: str,
    key: str,
    value,
    samples,
    stats,
    params=None,
    version: str = "v",
) -> Results:
    params = [["0"], ["0"]] if params is None else params
    header = {
        "machine": "optbench-ci",
        "os": "Linux",
        "arch": "x86_64",
        "cpu": "test",
        "num_cpu": "1",
        "ram": "1",
    }
    result = Results(header, [], commit, 0, "3.12", "uv-pip-py3.12", {})
    multi = isinstance(value, list)
    result._results[key] = value if multi else [value]
    result._benchmark_params[key] = params
    result._samples[key] = samples if multi else [samples]
    result._stats[key] = stats if multi else [stats]
    result._benchmark_version[key] = version
    result._started_at[key] = 0
    result._duration[key] = 0.1
    out = tmp / result._filename
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(tmp))
    return Results.load(str(out))


def test_one_timing_sample_is_not_a_decision():
    kind, ratio = classify_timing([1.0], [2.0], {"repeat": 1}, {"repeat": 1})
    assert kind == "unmeasured"
    assert ratio is None


def test_track_change_is_exact():
    assert classify_track(139.0, 139.0)[0] == "ok"
    assert classify_track(139.0, 140.0)[0] == "regression"
    assert classify_track(140.0, 139.0)[0] == "regression"


def test_a_shift_outside_the_sample_noise_is_a_regression(tmp_path: Path):
    base_samples = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01, 0.99]
    head_samples = [1.5, 1.52, 1.48, 1.51, 1.49, 1.5, 1.52, 1.48]
    _, base_stats = compute_stats(base_samples, 1)
    base_value, _ = compute_stats(base_samples, 1)
    head_value, head_stats = compute_stats(head_samples, 1)
    kind, ratio = classify_timing(base_samples, head_samples, base_stats, head_stats)
    assert kind == "regression"
    assert ratio == pytest.approx(1.5, rel=0.05)
    base = _result(
        tmp_path / "base",
        "a" * 40,
        "quadratic.QuadraticKept.time_search",
        base_value,
        base_samples,
        base_stats,
    )
    head = _result(
        tmp_path / "head",
        "b" * 40,
        "quadratic.QuadraticKept.time_search",
        head_value,
        head_samples,
        head_stats,
    )
    decision = compare_results(base, head)
    assert decision["regression"] is True
    assert decision["machine"] == "optbench-ci"
    text = format_report(decision)
    assert "| + |" in text
    assert "quadratic.QuadraticKept.time_search" in text


def test_matching_samples_are_not_a_regression():
    samples = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01]
    _, stats = compute_stats(samples, 1)
    kind, ratio = classify_timing(samples, list(reversed(samples)), stats, stats)
    assert kind == "ok"
    assert ratio == pytest.approx(1.0, rel=0.02)


def _track(tmp: Path, commit: str, value, params=None, version: str = "v") -> Results:
    values = value if isinstance(value, list) else [value]
    samples = [[item] for item in values]
    stats = [{"repeat": 1} for _ in values]
    stored_value = values if len(values) > 1 else values[0]
    stored_samples = samples if len(values) > 1 else samples[0]
    stored_stats = stats if len(values) > 1 else stats[0]
    return _result(
        tmp,
        commit,
        "quadratic.QuadraticKept.track_force_calls",
        stored_value,
        stored_samples,
        stored_stats,
        params=params,
        version=version,
    )


def test_an_increased_track_fails_the_gate(tmp_path: Path):
    base = _track(tmp_path / "base", "a" * 40, 139.0)
    head = _track(tmp_path / "head", "b" * 40, 140.0)
    decision = compare_results(base, head)
    assert decision["regression"] is True
    assert decision["rows"][0]["kind"] == "regression"
    assert decision["rows"][0]["mark"] == "+"
    assert decision["rows"][0]["before"] == 139.0
    assert decision["rows"][0]["after"] == 140.0


def test_a_decreased_track_fails_the_gate(tmp_path: Path):
    base = _track(tmp_path / "base", "c" * 40, 140.0)
    head = _track(tmp_path / "head", "d" * 40, 139.0)
    decision = compare_results(base, head)
    assert decision["regression"] is True
    assert decision["rows"][0]["kind"] == "regression"
    assert decision["rows"][0]["mark"] == "+"
    assert decision["rows"][0]["before"] == 140.0
    assert decision["rows"][0]["after"] == 139.0


def test_parameter_tuples_align_when_the_grid_is_reordered(tmp_path: Path):
    base = _track(
        tmp_path / "base",
        "a" * 40,
        [139.0, 200.0],
        params=[["0"], ["a", "b"]],
        version="same",
    )
    head = _track(
        tmp_path / "head",
        "b" * 40,
        [200.0, 139.0],
        params=[["0"], ["b", "a"]],
        version="same",
    )
    decision = compare_results(base, head)
    assert decision["regression"] is False
    assert decision["unmeasured"] == []
    assert [row["kind"] for row in decision["rows"]] == ["ok", "ok"]


def test_a_changed_parameter_tuple_fails(tmp_path: Path):
    base = _track(
        tmp_path / "base",
        "a" * 40,
        [139.0, 139.0],
        params=[["0"], ["a", "b"]],
        version="same",
    )
    head = _track(
        tmp_path / "head",
        "b" * 40,
        [139.0, 139.0],
        params=[["0"], ["b", "c"]],
        version="same",
    )
    decision = compare_results(base, head)
    assert decision["regression"] is True
    kinds = [row["kind"] for row in decision["rows"]]
    assert kinds.count("ok") == 1
    assert kinds.count("unmeasured") == 2


def test_a_benchmark_version_mismatch_fails(tmp_path: Path):
    base = _track(tmp_path / "base", "a" * 40, 139.0, version="left")
    head = _track(tmp_path / "head", "b" * 40, 139.0, version="right")
    decision = compare_results(base, head)
    assert decision["regression"] is True
    assert [row["kind"] for row in decision["rows"]] == ["unmeasured"]
    assert decision["rows"][0]["before"] is None
    assert decision["rows"][0]["after"] is None


def test_machine_record_drops_the_node_name(tmp_path: Path):
    path = tmp_path / "machines.json"
    ensure_machine("optbench-ci", path)
    text = path.read_text(encoding="utf-8")
    saved = json.loads(text)
    assert saved["optbench-ci"]["machine"] == "optbench-ci"
    import platform

    node = platform.node()
    if node:
        assert node not in text


def test_unrelated_machine_record_keeps_its_own_name(tmp_path: Path):
    import platform

    node = platform.node()
    assert node
    path = tmp_path / "machines.json"
    path.write_text(
        json.dumps(
            {
                "other-host": {
                    "machine": "other-host",
                    "os": node,
                    "arch": "x86_64",
                    "cpu": node,
                    "num_cpu": "1",
                    "ram": "1",
                }
            }
        ),
        encoding="utf-8",
    )
    ensure_machine("optbench-ci", path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["other-host"]["os"] == node
    assert saved["other-host"]["cpu"] == node
    record = saved["optbench-ci"]
    assert record["machine"] == "optbench-ci"
    assert node not in json.dumps(record)
