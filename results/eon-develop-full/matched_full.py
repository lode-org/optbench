"""Run the four r2549-ini develop templates on every start.

The client is develop binary 7095b23. Each start goes through
``run_case``, which sets ``Main.parallel`` false and counts force calls
on the harness. ``--write-asv COMMIT`` reads ``matched-full/records.jsonl``
and writes one asv result file for the matched classes. It does not read
the harness cache: those paths still hold the energy-accept runs recorded
under the old template ids.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CASES = (
    {
        "cls": "EonDevelopTsLj38DimerMatched",
        "template": "eon/develop-ts-lj38-dimer",
        "kind": "ts-lj38",
        "starts": [f"{i:04d}" for i in range(200)],
    },
    {
        "cls": "EonDevelopTsPtDimerMatched",
        "template": "eon/develop-ts-pt-dimer",
        "kind": "ts-pt",
        "starts": [f"initial_{i}" for i in range(49)],
    },
    {
        "cls": "EonDevelopDeLj38CapMatched",
        "template": "eon/develop-de-lj38-cap",
        "kind": "de-lj38",
        "starts": [f"pair_{i:02d}" for i in range(50)],
    },
    {
        "cls": "EonDevelopDe2tsMatched",
        "template": "eon/develop-de-2ts",
        "kind": "de-2ts",
        "starts": [f"pair_{i:02d}" for i in range(51)],
    },
)


def _one(job: tuple[str, str, str]) -> dict:
    template, kind, start = job
    from benchmarks.bench_eon_develop import run_case

    started = time.monotonic()
    record = run_case(template, kind, 1, start)
    record["template"] = template
    record["kind"] = kind
    record["start"] = start
    record["seed"] = 1
    record["wall_seconds"] = time.monotonic() - started
    return record


def _jobs() -> list[tuple[str, str, str]]:
    jobs = []
    for case in CASES:
        for start in case["starts"]:
            jobs.append((case["template"], case["kind"], start))
    return jobs


def run_all(workers: int) -> list[dict]:
    jobs = _jobs()
    done: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, job): job for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            done.append(record)
            print(
                f"{len(done)}/{len(jobs)} {record['template']} {record['start']} "
                f"harness={record['force_calls']} client={record.get('client_calls')} "
                f"failure={record.get('failure')} seconds={record.get('seconds')}",
                flush=True,
            )
    return done


def _jsonl_records() -> dict[tuple[str, str], dict]:
    path = Path(__file__).resolve().parent / "matched-full" / "records.jsonl"
    found: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        found[(record["template"], record["start"])] = record
    return found


def _track(record: dict, method: str):
    if method == "time_search":
        return float(record["seconds"])
    if method == "track_force_calls":
        return float(record["force_calls"])
    if method == "track_success":
        return 1.0 if record["success"] else 0.0
    if method.startswith("track_failure_"):
        name = method.removeprefix("track_failure_")
        return 1.0 if record.get("failure") == name else 0.0
    if method == "track_extrema":
        value = record.get("extrema")
        if value is None:
            return float("nan")
        return float(value)
    raise KeyError(method)


def write_asv(commit: str, dest: Path) -> None:
    from benchmarks.bench_eon_develop import EON_COMMIT

    sample = json.loads(
        (ROOT / "results/asv/xeon-8c/06efc6dd-existing-py_workspace_.venv_bin_python.json").read_text(
            encoding="utf-8"
        )
    )
    methods = [
        "time_search",
        "track_force_calls",
        "track_success",
        "track_failure_cap",
        "track_failure_wrong_stationary",
        "track_failure_divergence",
        "track_failure_crash",
    ]
    saved = _jsonl_records()
    results = {}
    now_ms = int(time.time() * 1000)
    for case in CASES:
        records = []
        for start in case["starts"]:
            record = saved.get((case["template"], start))
            if record is None:
                raise FileNotFoundError(f"missing {case['template']} {start}")
            records.append(record)
        case_methods = list(methods)
        if case["kind"] == "de-2ts":
            case_methods.append("track_extrema")
        params = [[repr("1")], [repr(start) for start in case["starts"]]]
        for method in case_methods:
            values = [_track(record, method) for record in records]
            results[f"bench_eon_develop.{case['cls']}.{method}"] = [
                values,
                params,
                EON_COMMIT,
                now_ms,
                None,
            ]
    payload = {
        "commit_hash": commit,
        "env_name": sample["env_name"],
        "date": now_ms,
        "params": sample["params"],
        "python": sample["python"],
        "requirements": sample["requirements"],
        "env_vars": sample.get("env_vars") or {},
        "result_columns": sample["result_columns"],
        "results": results,
        "durations": {"<build>": 0.0},
        "version": sample["version"],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {dest}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--write-asv":
        commit = args[1]
        prefix = commit[:8]
        env = "existing-py_workspace_.venv_bin_python"
        dest = ROOT / "results" / "asv" / "xeon-8c" / f"{prefix}-{env}.json"
        write_asv(commit, dest)
        return
    workers = int(os.environ.get("OPTBENCH_MATCHED_WORKERS", "8"))
    records = run_all(workers)
    out = Path(__file__).resolve().parent / "matched-full" / "records.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in sorted(records, key=lambda row: (row["template"], row["start"]))]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} starts)")


if __name__ == "__main__":
    main()
