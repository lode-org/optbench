"""Noise-aware decision for one asv comparison.

Both result files must come from the same machine record. Timing rows
are a regression only when ``asv._stats.is_different`` says the sample
sets differ and the head median is slower. That test is Mann-Whitney
when there are enough samples, and a 99% interval overlap check
otherwise. A single sample has an infinite interval, so it cannot
decide a timing regression. Track rows are exact: any force-call
change is a regression, whether the head value rises or falls.
A missing track value, including a pair of missing values, is
unmeasured and fails the job.
Points inside one benchmark are paired by parameter tuple, and the
benchmark version on the two results must be the same string. A
missing tuple or a version mismatch is unmeasured and fails the job.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import platform

from asv._stats import is_different
from asv.machine import Machine, MachineCollection
from asv.results import Results


def _as_list(value, count: int):
    if isinstance(value, list):
        if len(value) == count:
            return value
        if len(value) == 1 and count > 1:
            return value * count
    return [value] * count


def iter_points(result: Results, key: str):
    """Yield ``(param_tuple, version, value, stats, samples)`` in product order.

    ``param_tuple`` is one combination from the benchmark's parameter
    grid. ``version`` is that benchmark's version string, or ``None``
    when the result file has no version for the key. The caller pairs
    rows by ``(version, param_tuple)`` rather than by index.
    """
    stored = result.get_result_params(key)
    if stored is None:
        raise ValueError(f"{key} has no parameter grid")
    combos = list(itertools.product(*stored)) if stored else [()]
    value = result.get_result_value(key, stored)
    stats = result.get_result_stats(key, stored)
    samples = result.get_result_samples(key, stored)
    values = value if isinstance(value, list) else [value]
    if len(combos) != len(values) or len(combos) != len(set(combos)):
        raise ValueError(f"{key} parameter grid does not match its values")
    version = result.benchmark_version.get(key)
    stats_list = _as_list(stats, len(values))
    samples_list = _as_list(samples, len(values))
    for combo, item, stat, sample in zip(combos, values, stats_list, samples_list):
        yield combo, version, item, stat, sample


def _finite_samples(samples) -> list[float] | None:
    if not isinstance(samples, list) or not samples:
        return None
    if any(isinstance(item, list) for item in samples):
        return None
    values = []
    for item in samples:
        if item is None or (isinstance(item, float) and math.isnan(item)):
            continue
        values.append(float(item))
    return values or None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def classify_timing(base_samples, head_samples, base_stats, head_stats) -> tuple[str, float | None]:
    """Return ``ok``, ``regression``, ``improved``, or ``unmeasured``."""
    left = _finite_samples(base_samples)
    right = _finite_samples(head_samples)
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return "unmeasured", None
    if not isinstance(base_stats, dict) or not isinstance(head_stats, dict):
        return "unmeasured", None
    base_median = _median(left)
    head_median = _median(right)
    if base_median == 0:
        return "unmeasured", None
    ratio = head_median / base_median
    different = is_different(left, right, base_stats, head_stats)
    if not different:
        return "ok", ratio
    if head_median > base_median:
        return "regression", ratio
    if head_median < base_median:
        return "improved", ratio
    return "ok", ratio


def classify_track(base_value, head_value) -> tuple[str, float | None]:
    if base_value is None or head_value is None:
        return "unmeasured", None
    base = float(base_value)
    head = float(head_value)
    if math.isnan(base) or math.isnan(head):
        return "unmeasured", None
    if math.isclose(base, head, rel_tol=0.0, abs_tol=1e-9):
        ratio = 1.0 if base != 0 else None
        return "ok", ratio
    ratio = None if base == 0 else head / base
    return "regression", ratio


def ensure_machine(name: str = "optbench-ci", path: Path | None = None) -> None:
    """Record a fixed machine name for asv.

    The default asv name is the node name. This record uses ``name`` and
    drops any field that still contains the node name. The saved file is
    a collection: only ``saved[name]`` is inspected, so an unrelated
    record may keep its own name.
    """
    node = platform.node()
    info = Machine.get_defaults()
    info["machine"] = name
    for key, value in list(info.items()):
        if isinstance(value, str) and node and node in value:
            info[key] = value.replace(node, name)
    target = None if path is None else str(path)
    MachineCollection.save(name, info, _path=target)
    file_path = Path(target or MachineCollection.get_machine_file_path())
    saved = json.loads(file_path.read_text(encoding="utf-8"))
    record = saved.get(name)
    if not isinstance(record, dict):
        raise SystemExit(f"machine record {name} was not saved")
    if node and _contains(record, node):
        raise SystemExit("machine record still contains the node name")


def _contains(value, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(_contains(key, needle) for key in value) or any(
            _contains(item, needle) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains(item, needle) for item in value)
    return False


def find_result(results_dir: Path, machine: str, commit: str) -> Path:
    folder = results_dir / machine
    prefix = commit[:8]
    matches = sorted(
        path
        for path in folder.glob(f"{prefix}-*.json")
        if path.name != "machine.json"
    )
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise SystemExit(
            f"expected one result for {commit[:8]} on machine {machine} in {folder}, found {found}"
        )
    return matches[0]


def _load(path: Path) -> Results:
    return Results.load(str(path))


def _machine(result: Results) -> str:
    return str(result._params.get("machine", ""))


def _unmeasured(name: str) -> dict:
    return {
        "benchmark": name,
        "mark": "x",
        "kind": "unmeasured",
        "before": None,
        "after": None,
        "ratio": None,
    }


def _point_name(key: str, combo: tuple, alone: bool) -> str:
    if alone:
        return key
    shown = ", ".join(repr(part) for part in combo)
    return f"{key}[{shown}]"


def compare_results(base: Results, head: Results) -> dict:
    if _machine(base) != _machine(head) or not _machine(base):
        raise SystemExit(
            f"result machines differ: {_machine(base)!r} and {_machine(head)!r}"
        )
    keys = sorted(set(base.get_all_result_keys()) | set(head.get_all_result_keys()))
    rows = []
    unmeasured = []
    for key in keys:
        if key not in base.get_all_result_keys() or key not in head.get_all_result_keys():
            rows.append(
                {
                    "benchmark": key,
                    "mark": "x",
                    "kind": "unmeasured",
                    "before": None,
                    "after": None,
                    "ratio": None,
                }
            )
            unmeasured.append(key)
            continue
        try:
            base_points = list(iter_points(base, key))
            head_points = list(iter_points(head, key))
        except ValueError:
            rows.append(_unmeasured(key))
            unmeasured.append(key)
            continue
        base_version = base.benchmark_version.get(key)
        head_version = head.benchmark_version.get(key)
        versions_match = base_version == head_version and all(
            version == base_version for _, version, _value, _stats, _samples in base_points + head_points
        )
        if not versions_match:
            rows.append(_unmeasured(key))
            unmeasured.append(key)
            continue
        base_map = {
            combo: (value, stats, samples)
            for combo, _version, value, stats, samples in base_points
        }
        head_map = {
            combo: (value, stats, samples)
            for combo, _version, value, stats, samples in head_points
        }
        combos = sorted(set(base_map) | set(head_map), key=repr)
        timing = ".time_" in key
        for combo in combos:
            if combo not in base_map or combo not in head_map:
                name = _point_name(key, combo, alone=False)
                rows.append(_unmeasured(name))
                unmeasured.append(name)
                continue
            base_value, base_stats, base_samples = base_map[combo]
            head_value, head_stats, head_samples = head_map[combo]
            alone = len(combos) == 1
            name = _point_name(key, combo, alone)
            if timing:
                kind, ratio = classify_timing(base_samples, head_samples, base_stats, head_stats)
                before = _median(_finite_samples(base_samples) or []) if _finite_samples(base_samples) else base_value
                after = _median(_finite_samples(head_samples) or []) if _finite_samples(head_samples) else head_value
            else:
                kind, ratio = classify_track(base_value, head_value)
                before, after = base_value, head_value
            mark = {"ok": " ", "regression": "+", "improved": "-", "unmeasured": "x"}[kind]
            if kind == "unmeasured":
                unmeasured.append(name)
            rows.append(
                {
                    "benchmark": name,
                    "mark": mark,
                    "kind": kind,
                    "before": before,
                    "after": after,
                    "ratio": ratio,
                }
            )
    regression = any(row["kind"] == "regression" for row in rows) or bool(unmeasured)
    return {
        "machine": _machine(base),
        "base": base._commit_hash,
        "head": head._commit_hash,
        "regression": regression,
        "unmeasured": unmeasured,
        "rows": rows,
    }


def format_report(decision: dict) -> str:
    lines = [
        "| Change | Before | After | Ratio | Benchmark |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in decision["rows"]:
        ratio = "n/a" if row["ratio"] is None else f"{row['ratio']:.6g}"
        before = "n/a" if row["before"] is None else f"{row['before']:.6g}"
        after = "n/a" if row["after"] is None else f"{row['after']:.6g}"
        lines.append(
            f"| {row['mark']} | {before} | {after} | {ratio} | {row['benchmark']} |"
        )
    lines.append("")
    if decision["regression"]:
        lines.append("Regression: a timing change cleared the sample test, or a track changed, or a row was unmeasured.")
    else:
        lines.append("No regression: timing shifts sit inside the sample noise, and tracks match.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Noise-aware asv comparison on one machine.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args(argv)

    base_path = find_result(args.results, args.machine, args.base)
    head_path = find_result(args.results, args.machine, args.head)
    decision = compare_results(_load(base_path), _load(head_path))
    if not decision["base"].startswith(args.base) and args.base not in ("HEAD",):
        if not decision["base"].startswith(args.base[:8]):
            raise SystemExit(f"base result {decision['base']} does not match {args.base}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(format_report(decision), encoding="utf-8")
    args.decision.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(args.report.read_text(encoding="utf-8"))
    return 1 if decision["regression"] else 0


if __name__ == "__main__":
    sys.exit(main())
