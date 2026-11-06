"""Write asv tracks from a finished pele minimization campaign.

The force counts and the per-start seconds are the campaign records.
The result commit is the optbench commit that contains ``templates/pele``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from benchmarks.bench_pele import _class_name
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
    return rows


def write_results(root: Path, commit: str, *, started_at: int | None = None) -> Path:
    """One result file for every ``results/campaign/pele-b7081b93-*.starts.jsonl``."""
    from optbench.campaign import list_templates

    campaign = root / "results" / "campaign"
    expected = {
        str(template["id"])
        for template in list_templates(root)
        if template.get("engine") == "pele"
    }
    groups: dict[str, list[dict]] = {}
    for path in sorted(campaign.glob("pele-b7081b93-*.starts.jsonl")):
        template = None
        dat = campaign / (path.name[: -len(".starts.jsonl")] + ".dat")
        for line in dat.read_text(encoding="utf-8").splitlines():
            if line.startswith("template "):
                template = line.split(None, 1)[1].strip()
        if not template:
            raise ValueError(f"{dat} has no template")
        rows = _records(path)
        if any(row.get("partial") for row in rows):
            raise ValueError(f"{path} is a partial campaign")
        groups[template] = rows
    missing = expected - set(groups)
    if missing:
        raise ValueError(f"missing pele campaigns: {sorted(missing)}")
    params = {**machine_record(), "python": "3.12"}
    now = int(time.time() * 1000) if started_at is None else int(started_at)
    columns = ["result", "params", "version", "started_at", "duration"]
    results = {}
    for template, rows in groups.items():
        class_name = _class_name(template)
        seeds = [_quote("0")]
        starts = [_quote(str(index)) for index in range(len(rows))]
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
        duration = sum(seconds)
        for method in TRACKS:
            key = f"bench_pele.{class_name}.{method}"
            results[key] = [payload[method], [seeds, starts], _version(key), now, duration]
    document = {
        "commit_hash": commit,
        "env_name": "pele-uv-pip-py3.12",
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
    dest = machine_dir(root / "results" / "asv") / f"{commit[:8]}-pele-uv-pip-py3.12.json"
    dest.write_text(json.dumps(document, separators=(", ", ": ")), encoding="utf-8")
    return dest
