"""The archived GMIN log line is the function-call counter."""

import importlib
import json
import math

from optbench.runners.gmin_paper import ROWS, campaign_name, parse_log, run_one, write_dat
from optbench.wales import header, snapshot


def test_target_hit_line_matches_run_gmin():
    text = (
        "Qu       1768 E=    -173.928416337074     steps=  112\n"
        "gsaveit> Target hit after            1768 quenches          254024 function calls.\n"
        "finalio> time elapsed               2.45 seconds\n"
    )
    parsed = parse_log(text)
    assert parsed["quenches"] == 1768
    assert parsed["ncalls"] == 254024
    assert parsed["success"] == 1
    assert parsed["seconds"] == 2.45


def test_a_miss_is_a_failure():
    parsed = parse_log("mc> run finished without the target\n")
    assert parsed["success"] == 0
    assert parsed["ncalls"] is None


def test_rows_record_the_wales_build(tmp_path):
    found = snapshot("a" * 64, "5271791321b85779783debf84617812fd1b267a0")
    other = snapshot("b" * 64, found["wales_svnrev"])
    output = tmp_path / "row.dat"
    write_dat(output, {"n": "1"}, ROWS[0], found)
    assert output.read_text(encoding="utf-8").splitlines()[:2] == header(found)
    binary = tmp_path / "GMIN"
    binary.write_text(
        "#!/bin/sh\necho 'gsaveit> Target hit after 7 quenches 70 function calls.'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    work = tmp_path / "start"
    job = {"index": 0, "work": str(work), "data": "", "coords": "", "binary": str(binary), "resume": True}
    work.mkdir()
    (work / "record.json").write_text(json.dumps({"index": 0, "ncalls": 1, **other}), encoding="utf-8")
    record = run_one(dict(job, wales=found))
    assert record["ncalls"] == 70
    assert {key: record[key] for key in found} == found
    assert run_one(dict(job, wales=found)) == record


def test_a_stop_on_the_data_file_is_an_error(tmp_path):
    binary = tmp_path / "GMIN"
    binary.write_text(
        "#!/bin/sh\necho 'Starting serial execution'\necho 'STOP input> Error while trying to read integer' >&2\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    job = {"index": 0, "work": str(tmp_path / "start"), "data": "", "coords": "", "binary": str(binary)}
    record = run_one(job)
    assert record["returncode"] == 0
    assert record["success"] == 0
    assert record["error"].startswith("STOP input>")


def test_asv_classes_are_the_campaign_rows(monkeypatch):
    import benchmarks.bench_gmin_paper as bench

    names = {spec["campaign"]: spec["class_name"] for spec in bench._specs()}
    assert sorted(names) == sorted(campaign_name(row) for row in ROWS)
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    try:
        full = importlib.reload(bench)
        for row in ROWS:
            benchmark = getattr(full, names[campaign_name(row)])
            assert benchmark.problem_id == f"globalopt/{row['problem']}"
            assert benchmark.params == [["0"], [str(index) for index in range(row["start"], row["stop"] + 1)]]
    finally:
        monkeypatch.delenv("OPTBENCH_ASV_FULL")
        default = importlib.reload(bench)
    for row in ROWS:
        assert getattr(default, names[campaign_name(row)]).params == [["0"], [str(row["start"])]]


def test_asv_every_track_has_a_value_in_a_fresh_process(monkeypatch, tmp_path):
    import benchmarks.bench_gmin_paper as bench
    from optbench.runners import gmin_go, gmin_paper

    calls = []
    found = {
        "1": {"ncalls": 254024, "quenches": 1768, "success": 1, "returncode": 0},
        "2": {"ncalls": None, "quenches": None, "success": 0, "returncode": 0},
        "3": {"ncalls": None, "quenches": None, "success": 0, "returncode": 1},
        "4": {"ncalls": None, "quenches": None, "success": 0, "returncode": 0, "error": "STOP input> Error"},
    }

    def prepare_jobs(root, row, work):
        assert row["start"] == row["stop"]
        return [{"index": row["start"], "work": str(work / str(row["start"])), "resume": True}]

    def run_one(job):
        assert job["resume"] is False
        calls.append(job["index"])
        return dict(found[str(job["index"])], index=job["index"])

    monkeypatch.setenv("OPTBENCH_ASV_CACHE", str(tmp_path))
    monkeypatch.setattr(bench, "_binary_ready", lambda: True)
    monkeypatch.setattr(gmin_paper, "prepare_jobs", prepare_jobs)
    monkeypatch.setattr(gmin_paper, "run_one", run_one)
    monkeypatch.setattr(gmin_paper, "serial_binary", lambda: tmp_path / "GMIN")
    build = snapshot("a" * 64, "5271791321b85779783debf84617812fd1b267a0")
    monkeypatch.setattr(gmin_go, "binary_build", lambda binary: build)
    benchmark = bench.GminPaperLj75
    expected = {
        "1": (254024.0, 1.0, 0.0, 0.0),
        "2": (math.nan, 0.0, 1.0, 0.0),
        "3": (math.nan, 0.0, 0.0, 1.0),
        "4": (math.nan, 0.0, 0.0, 1.0),
    }
    for start, (force, success, cap, crash) in expected.items():
        values = {}
        for name in (
            "track_force_calls",
            "track_success",
            "track_failure_cap",
            "track_failure_wrong_stationary",
            "track_failure_divergence",
            "track_failure_crash",
        ):
            fresh = benchmark()
            fresh.setup("0", start)
            values[name] = getattr(fresh, name)("0", start)
            assert isinstance(values[name], float)
        assert values["track_force_calls"] == force or (math.isnan(force) and math.isnan(values["track_force_calls"]))
        assert values["track_success"] == success
        assert values["track_failure_cap"] == cap
        assert values["track_failure_crash"] == crash
        assert values["track_failure_wrong_stationary"] == 0.0
        assert values["track_failure_divergence"] == 0.0
    assert calls == [1, 2, 3, 4]
    timed = benchmark()
    timed.setup("0", "1")
    timed.time_search("0", "1")
    assert calls == [1, 2, 3, 4, 1]
    build = snapshot("b" * 64, build["wales_svnrev"])
    fresh = benchmark()
    fresh.setup("0", "1")
    assert fresh.track_force_calls("0", "1") == 254024.0
    assert calls == [1, 2, 3, 4, 1, 1]


def test_one_start_is_the_job_of_the_whole_row(monkeypatch, tmp_path):
    from pathlib import Path

    from optbench.runners import gmin_paper

    monkeypatch.setattr(gmin_paper, "serial_binary", lambda: tmp_path / "GMIN")
    root = Path(__file__).resolve().parents[1]
    for row in (ROWS[0], ROWS[3]):
        whole = {job["index"]: job for job in gmin_paper.prepare_jobs(root, row, tmp_path)}
        for index in (row["start"], row["stop"]):
            (job,) = gmin_paper.prepare_jobs(root, dict(row, start=index, stop=index), tmp_path)
            assert job == whole[index]


def test_pending_drops_the_globalopt_cells_these_classes_measure():
    from benchmarks import pending

    measured = {f"globalopt/{row['problem']}" for row in ROWS}
    assert measured <= pending.stated_problems("gmin_paper")
    assert not measured & set(pending.pending_cells()["gmin"])
