"""Write asv tracks from finished geomeTRIC campaigns.

The force counts are the campaign records. ``time_search`` stores the
measured per-start seconds. The result commit is a commit that contains
the templates.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from benchmarks.bench_geometric import _class_name as geometric_class
from benchmarks.bench_noise import _class_name as noise_class

TRACKS = (
    "time_search",
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)


def _version(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _quote(value: str) -> str:
    return repr(str(value))


def _records(path: Path, expected: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) != expected:
        raise ValueError(f"{path} has {len(rows)} starts, expected {expected}")
    return rows


def _template(dat: Path) -> str:
    for line in dat.read_text(encoding="utf-8").splitlines():
        if line.startswith("template "):
            return line.split(None, 1)[1].strip()
    raise ValueError(f"{dat} has no template")


def write_results(root: Path, commit: str, *, started_at: int | None = None) -> Path:
    """One result file for the geomeTRIC clean and noise campaigns."""
    campaign = root / "results" / "campaign"
    groups: list[tuple[str, str, list[dict]]] = []
    for path in sorted(campaign.glob("geometric-*.starts.jsonl")):
        dat = campaign / (path.name[: -len(".starts.jsonl")] + ".dat")
        template = _template(dat)
        expected = 100 if "morse" in template else 1000
        module = "bench_geometric"
        groups.append((module, geometric_class(template), _records(path, expected)))
    for path in sorted(campaign.glob("noise-geometric-*.starts.jsonl")):
        dat = campaign / (path.name[: -len(".starts.jsonl")] + ".dat")
        template = _template(dat)
        groups.append(("bench_noise", noise_class(template), _records(path, 1000)))
    if len(groups) != 12:
        raise ValueError(f"expected 12 geomeTRIC campaigns, found {len(groups)}")
    machine = json.loads((root / "results" / "asv" / "cursor" / "machine.json").read_text(encoding="utf-8"))
    params = {
        "arch": machine["arch"],
        "cpu": machine["cpu"],
        "machine": machine["machine"],
        "num_cpu": machine["num_cpu"],
        "os": machine["os"],
        "ram": machine["ram"],
        "python": "3.12",
    }
    now = int(time.time() * 1000) if started_at is None else int(started_at)
    columns = ["result", "params", "version", "started_at", "duration"]
    results = {}
    for module, class_name, rows in groups:
        calls = [float(row["force_calls"]) for row in rows]
        success = [1.0 if row["success"] else 0.0 for row in rows]
        failure = {name: [] for name in ("cap", "wrong_stationary", "divergence", "crash")}
        seconds = []
        for row in rows:
            label = row.get("failure")
            for name in failure:
                failure[name].append(1.0 if label == name else 0.0)
            seconds.append(float(row.get("seconds") or 0.0))
        payload = {
            "time_search": seconds,
            "track_force_calls": calls,
            "track_success": success,
            "track_failure_cap": failure["cap"],
            "track_failure_wrong_stationary": failure["wrong_stationary"],
            "track_failure_divergence": failure["divergence"],
            "track_failure_crash": failure["crash"],
        }
        seeds = [_quote("0")]
        starts = [_quote(str(index)) for index in range(len(rows))]
        duration = sum(seconds)
        for method in TRACKS:
            key = f"{module}.{class_name}.{method}"
            results[key] = [payload[method], [seeds, starts], _version(key), now, duration]
    document = {
        "commit_hash": commit,
        "env_name": "uv-pip-py3.12",
        "date": now,
        "params": params,
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }
    dest = root / "results" / "asv" / "cursor" / f"{commit[:8]}-uv-pip-py3.12.json"
    dest.write_text(json.dumps(document, separators=(", ", ": ")), encoding="utf-8")
    return dest
