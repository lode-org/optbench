import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_a_bad_client_count_is_missing():
    from optbench.campaign import client_count, parse_client_dat

    assert client_count("181") == 181
    assert client_count("181.0") == 181
    assert client_count("0") == 0
    assert client_count("nan") is None
    assert client_count("inf") is None
    assert client_count("-3") is None
    assert client_count("1e") is None
    assert parse_client_dat("nan total_force_calls\n0 termination_reason\n") == (None, "0")
    assert parse_client_dat("12 total_force_calls\n") == (12, None)


def test_a_nan_job_result_is_a_missing_count(tmp_path):
    from optbench.eon_jobs import _execute

    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('results.dat').write_text('nan total_force_calls\\nnan number_of_extrema\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    row = _execute((str(client), "", str(tmp_path / "run"), [], [], "total_force_calls"))
    assert row["raw"] is None
    assert row["force_calls"] is None
    assert row["extrema"] is None


def test_client_record_is_the_campaign_count():
    from optbench.campaign import client_record

    assert client_record({"force_calls": 40, "reason": "0"}, 100) == {
        "force_calls": 40.0,
        "success": True,
        "failure": None,
        "client_calls": 40,
        "reason": "0",
    }
    assert client_record({"force_calls": None, "reason": None}, 100)["failure"] == "crash"
    bad = client_record({"force_calls": 40, "reason": "1"}, 100)
    assert (bad["force_calls"], bad["success"], bad["failure"]) == (100.0, False, "cap")
    over = client_record({"force_calls": 120, "reason": "0"}, 100)
    assert (over["force_calls"], over["success"], over["failure"]) == (120.0, False, "cap")


def test_job_start_is_scored_as_the_campaign_file(monkeypatch, tmp_path):
    from optbench import eon_jobs

    start_path = str(tmp_path / "w" / "runs" / "pair_00")
    run_root = tmp_path / "stub"
    planned = {
        "template": {"id": "eon/x", "failure_cap": 100},
        "ini": "",
        "specs": [(start_path, [], [], "total_force_calls")],
        "images": 1,
    }
    for raw, reason, expected in (
        (40, "0", (40.0, True, None)),
        (40, "1", (40.0, False, None)),
        (100, "0", (100.0, False, "cap")),
        (None, None, (100.0, False, "crash")),
    ):
        monkeypatch.setattr(eon_jobs, "_execute", lambda spec, raw=raw, reason=reason: {"raw": raw, "reason": reason})
        record = eon_jobs.run_start(run_root, planned, "pair_00")
        assert (record["force_calls"], record["success"], record["failure"]) == expected
    with pytest.raises(KeyError):
        eon_jobs.run_start(run_root, planned, "pair_01")


def test_plan_is_the_spec_list_run_named_runs(tmp_path):
    from optbench import eon_jobs

    for name, (template, fn) in eon_jobs.JOBS.items():
        if fn not in eon_jobs.SPECS:
            with pytest.raises(ValueError):
                eon_jobs.plan(ROOT, name, tmp_path / name)
            continue
        planned = eon_jobs.plan(ROOT, name, tmp_path / name)
        assert planned["template"]["id"] == template
        names = [eon_jobs.start_name(spec) for spec in planned["specs"]]
        assert names and len(names) == len(set(names))
