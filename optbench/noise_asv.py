"""Write asv tracks from a finished lj38-noise campaign.

The force counts are the campaign records. ``time_search`` stores the
measured per-start seconds. The result commit is the commit that
contains ``templates/noise``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from benchmarks.bench_noise import _class_name
from optbench.asv_machine import machine_dir, machine_record

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


def _records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) != 1000:
        raise ValueError(f"{path} has {len(rows)} starts, expected 1000")
    return rows


def write_results(root: Path, commit: str, *, started_at: int | None = None) -> Path:
    """One result file for every ``results/campaign/noise-*.starts.jsonl``."""
    campaign = root / "results" / "campaign"
    groups: dict[str, list[dict]] = {}
    for path in sorted(campaign.glob("noise-*.starts.jsonl")):
        template = None
        dat = campaign / (path.name[: -len(".starts.jsonl")] + ".dat")
        text = dat.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("template "):
                template = line.split(None, 1)[1].strip()
        if not template:
            raise ValueError(f"{dat} has no template")
        groups[template] = _records(path)
    from optbench.noise_catalog import specs

    expected = len(specs())
    if len(groups) != expected:
        raise ValueError(f"expected {expected} noise campaigns, found {len(groups)}")
    params = {**machine_record(), "python": "3.12"}
    now = int(time.time() * 1000) if started_at is None else int(started_at)
    seeds = [_quote("0")]
    starts = [_quote(str(index)) for index in range(1000)]
    columns = ["result", "params", "version", "started_at", "duration"]
    results = {}
    for template, rows in groups.items():
        class_name = _class_name(template)
        calls = [float(row["force_calls"]) for row in rows]
        success = [1.0 if row["success"] else 0.0 for row in rows]
        failure = {name: [] for name in ("cap", "wrong_stationary", "divergence", "crash")}
        seconds = []
        for row in rows:
            label = row.get("failure")
            for name in failure:
                failure[name].append(1.0 if label == name else 0.0)
            seconds.append(float(row["seconds"]))
        payload = {
            "time_search": seconds,
            "track_force_calls": calls,
            "track_success": success,
            "track_failure_cap": failure["cap"],
            "track_failure_wrong_stationary": failure["wrong_stationary"],
            "track_failure_divergence": failure["divergence"],
            "track_failure_crash": failure["crash"],
        }
        duration = sum(seconds)
        for method in TRACKS:
            key = f"bench_noise.{class_name}.{method}"
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
    # The other result files use asv's short hash in the name and the full hash inside.
    dest = machine_dir(root / "results" / "asv") / f"{commit[:8]}-uv-pip-py3.12.json"
    dest.write_text(json.dumps(document, separators=(", ", ": ")), encoding="utf-8")
    return dest
