"""Merge strided asv workers and fill gaps from the harness cache.

Workers were started with ``OPTBENCH_ASV_FULL=1`` and
``OPTBENCH_ASV_SLICE=k/n``. Parameters are put back in campaign order.
A parameter no worker recorded stays ``None`` unless the harness cache
has that seed and start. Cache values are the fields the tracks read.
An asv value is never replaced, and a disagreement with the cache raises.
"""

from __future__ import annotations

import ast
import itertools
import json
import os
import sys
from pathlib import Path


def _unwrap(value: str):
    return ast.literal_eval(value)


def _load_specs():
    os.environ["OPTBENCH_ASV_FULL"] = "1"
    os.environ.pop("OPTBENCH_ASV_SLICE", None)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from benchmarks.bench_eon_develop import EON_COMMIT, _specs

    return EON_COMMIT, _specs()


def _result_files(directory: Path) -> list[Path]:
    machine_dir = directory / "results" / "xeon-8c"
    found = list(machine_dir.glob("*.json")) if machine_dir.is_dir() else list(directory.rglob("*.json"))
    return [path for path in found if path.name not in {"machine.json", "benchmarks.json"}]


def _track_value(record: dict, method: str):
    if method == "time_search":
        return float(record["seconds"])
    if method == "track_force_calls":
        return float(record["force_calls"])
    if method == "track_success":
        return 1.0 if record["success"] else 0.0
    if method.startswith("track_failure_"):
        name = method.removeprefix("track_failure_")
        return 1.0 if record.get("failure") == name else 0.0
    key = method.removeprefix("track_")
    value = record.get(key)
    if value is None:
        return None
    return float(value)


def _methods(spec: dict) -> list[str]:
    names = [
        "time_search",
        "track_force_calls",
        "track_success",
        "track_failure_cap",
        "track_failure_wrong_stationary",
        "track_failure_divergence",
        "track_failure_crash",
    ]
    if spec["kind"] == "akmc":
        names.extend(
            [
                "track_unique_saddles",
                "track_fraction_connected",
                "track_total_searches",
            ]
        )
    if spec["kind"] == "de-2ts":
        names.append("track_extrema")
    return names


def merge(worker_dirs: list[Path], cache: Path | None, cache_commit: str | None) -> dict:
    eon_commit, specs = _load_specs()
    by_class = {spec["cls"]: spec for spec in specs}
    payloads = []
    for directory in worker_dirs:
        result_files = _result_files(directory)
        if len(result_files) != 1:
            raise FileNotFoundError(f"{directory} has {len(result_files)} result files")
        payloads.append(json.loads(result_files[0].read_text(encoding="utf-8")))
    if not payloads:
        raise ValueError("no worker results")
    base = json.loads(json.dumps(payloads[0]))
    names = sorted({name for payload in payloads for name in payload["results"]})
    # Classes that only exist in the cache still need a row.
    if cache is not None:
        for spec in specs:
            for method in _methods(spec):
                names.append(f"bench_eon_develop.{spec['cls']}.{method}")
        names = sorted(set(names))
    merged: dict = {}
    coverage: dict[str, dict] = {}
    grid_commits: dict[str, list[str]] = {}
    filled_from_cache: dict[str, list] = {}
    for name in names:
        cls = name.split(".")[1]
        spec = by_class[cls]
        order = [
            (str(seed), str(start))
            for seed, start in itertools.product(spec["campaign_seeds"], spec["campaign_starts"])
        ]
        index = {pair: position for position, pair in enumerate(order)}
        values = [None] * len(order)
        donors = []
        commits: list[str] = []
        for payload in payloads:
            if name not in payload["results"]:
                continue
            row = payload["results"][name]
            seeds = [_unwrap(item) for item in row[1][0]]
            starts = [_unwrap(item) for item in row[1][1]]
            pairs = list(itertools.product(seeds, starts))
            raw = row[0]
            if raw is None or len(raw) != len(pairs):
                raise ValueError(f"{name} value list does not match its parameters")
            touched = False
            for pair, value in zip(pairs, raw, strict=True):
                key = (str(pair[0]), str(pair[1]))
                if key not in index:
                    raise ValueError(f"{name} has unexpected parameter {key}")
                slot = index[key]
                if values[slot] is not None and values[slot] != value:
                    raise ValueError(f"{name} {key} disagrees: {values[slot]} vs {value}")
                if value is not None:
                    touched = True
                values[slot] = value
            if touched:
                commit = payload.get("commit_hash")
                if commit and commit not in commits:
                    commits.append(commit)
            donors.append(row)
        cache_pairs = []
        method = name.split(".", 2)[2]
        if cache is not None:
            passes = int(spec.get("passes", 0) or 0)
            safe = spec["template"].replace("/", "-")
            for pair in order:
                slot = index[pair]
                path = cache / "harness-1" / eon_commit / safe / f"passes-{passes}" / pair[0] / f"{pair[1]}.json"
                if not path.is_file():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                value = _track_value(record, method)
                # asv times the whole measurement. The cache stores the
                # harness timer, which is a few milliseconds shorter.
                if method == "time_search" and values[slot] is not None:
                    continue
                if values[slot] is not None and values[slot] != value:
                    raise ValueError(f"{name} {pair} cache disagrees: {values[slot]} vs {value}")
                if values[slot] is None and value is not None:
                    values[slot] = value
                    if method == "track_force_calls":
                        cache_pairs.append(list(pair))
        if cache_pairs and cache_commit and cache_commit not in commits:
            commits.append(cache_commit)
        if not donors and not any(value is not None for value in values):
            continue
        if donors:
            template = json.loads(json.dumps(donors[0]))
        else:
            template = [None, None, eon_commit, None, None]
        template[0] = values
        template[1] = [
            [repr(str(seed)) for seed in spec["campaign_seeds"]],
            [repr(str(start)) for start in spec["campaign_starts"]],
        ]
        for column in range(5, len(template)):
            if isinstance(template[column], list):
                template[column] = None
        merged[name] = template
        if commits:
            grid_commits[cls] = commits
        if method == "track_force_calls" and cache_pairs:
            filled_from_cache[cls] = cache_pairs
        missing = [order[i] for i, value in enumerate(values) if value is None]
        coverage[cls] = {
            "benchmarks": coverage.get(cls, {}).get("benchmarks", 0) + 1,
            "n": len(order),
            "filled": sum(value is not None for value in values),
            "missing": [list(pair) for pair in missing],
        }
    base["results"] = merged
    base["grid_commits"] = grid_commits
    base["filled_from_cache"] = filled_from_cache
    return base, coverage


def main() -> None:
    args = sys.argv[1:]
    cache: Path | None = None
    cache_commit: str | None = None
    positional: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--cache":
            cache = Path(args[index + 1])
            index += 2
            continue
        if item == "--cache-commit":
            cache_commit = args[index + 1]
            index += 2
            continue
        positional.append(item)
        index += 1
    if len(positional) < 2:
        raise SystemExit(
            "usage: assemble.py OUTPUT [--cache DIR] [--cache-commit SHA] WORKER_RESULT_DIR..."
        )
    output = Path(positional[0])
    workers = [Path(arg) for arg in positional[1:]]
    payload, coverage = merge(workers, cache, cache_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    for cls, row in sorted(coverage.items()):
        print(f"{cls} filled {row['filled']}/{row['n']} missing {len(row['missing'])}")


if __name__ == "__main__":
    main()
