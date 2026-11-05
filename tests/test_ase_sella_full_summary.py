"""The full-grid summary scores asv tracks with score_v2."""

import importlib.util
import json
from pathlib import Path

from optbench.score import score_v2

ROOT = Path(__file__).resolve().parents[1]


def _summary():
    path = ROOT / "results" / "ase-sella-full" / "summarize.py"
    spec = importlib.util.spec_from_file_location("ase_sella_full_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_result(directory: Path, rows: list[dict]) -> None:
    machine = directory / "xeon-8c"
    machine.mkdir()
    (machine / "machine.json").write_text(
        json.dumps(
            {
                "arch": "x86_64",
                "cpu": "test",
                "machine": "xeon-8c",
                "num_cpu": "1",
                "os": "test",
                "ram": "1",
                "version": 1,
            }
        ),
        encoding="utf-8",
    )
    columns = {
        "track_force_calls": [],
        "track_success": [],
        "track_failure_cap": [],
        "track_failure_crash": [],
        "track_failure_divergence": [],
        "track_failure_wrong_stationary": [],
    }
    starts = []
    for row in rows:
        starts.append(repr(row["start"]))
        calls = row.get("force_calls", float("nan"))
        success = row.get("success")
        failure = row.get("failure")
        columns["track_force_calls"].append(calls)
        if success is None:
            for name in columns:
                if name != "track_force_calls":
                    columns[name].append(float("nan"))
            continue
        columns["track_success"].append(1.0 if success else 0.0)
        for name in (
            "cap",
            "crash",
            "divergence",
            "wrong_stationary",
        ):
            columns[f"track_failure_{name}"].append(1.0 if failure == name else 0.0)
    params = [["'0'"], starts]
    results = {}
    result_columns = ["result", "params", "version", "started_at", "duration"]
    for method, values in columns.items():
        results[f"bench_ase_sella.Example.{method}"] = [
            values,
            params,
            "abc",
            0,
            1.0,
        ]
    payload = {
        "commit_hash": "abc123def4567890",
        "env_name": "uv-pip-py3.12",
        "date": 1,
        "params": {
            "arch": "x86_64",
            "cpu": "test",
            "machine": "xeon-8c",
            "num_cpu": "1",
            "os": "test",
            "ram": "1",
            "python": "3.12",
        },
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": result_columns,
        "results": results,
        "version": 2,
    }
    (machine / "abc123de-uv-pip-py3.12.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summary_uses_score_v2_and_keeps_blank_tracks_out(tmp_path: Path):
    module = _summary()
    _write_result(
        tmp_path,
        [
            {"start": "a", "force_calls": 10.0, "success": True, "failure": None},
            {"start": "b", "force_calls": 100.0, "success": False, "failure": "cap"},
            {"start": "c", "force_calls": float("nan"), "success": None, "failure": None},
        ],
    )
    from asv.results import iter_results

    result = next(iter_results(str(tmp_path)))
    row = module.summarize_class(result, "Example", cap=100, expected=3)
    assert row["n_starts"] == 3
    assert row["n_tracked"] == 2
    assert row["n_missing"] == 1
    assert row["partial"] is True
    assert row["successes"] == 1
    assert row["n_cap"] == 1
    assert row["n_crash"] == 0
    assert row["n_divergence"] == 0
    assert row["n_wrong_stationary"] == 0
    assert row["force_calls_mean"] == 10
    assert row["force_calls_min"] == 10
    assert row["force_calls_max"] == 10
    scored = score_v2(
        [
            {"force_calls": 10.0, "success": True, "failure": None},
            {"force_calls": 100.0, "success": False, "failure": "cap"},
        ],
        100,
    )
    assert row["par2"]["par2_mean"] == scored["par2_mean"]
    assert row["par2"]["n_cap"] == 1
    assert row["missing_starts"] == ["0,c"]
