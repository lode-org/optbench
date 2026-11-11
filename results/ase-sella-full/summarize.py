"""Summarize a full ASE/Sella asv grid.

Reads ``results/asv`` and writes one row per benchmark class. Force-call
tracks come from the asv result file. The PAR-2 value is ``score_v2``.
Starts that asv left blank stay out of that score.

Run from the repository root:

    uv run python results/ase-sella-full/summarize.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from optbench.asv_score import TRACKS, _count, _key, _missing, _unwrap, track_rows
from optbench.failures import FAILURES
from optbench.score import score_v2
from optbench.v2 import relabel_machine

ROOT = Path(__file__).resolve().parents[2]


def _combo_label(combo) -> str:
    return ",".join(str(_unwrap(part)) for part in combo)


def tolerant_rows(result, class_name: str) -> tuple[list[dict], list[str]]:
    """Rows with a complete track, and labels of parameter rows asv skipped."""
    import itertools

    force_key = _key(result, class_name, "track_force_calls")
    params = result.get_result_params(force_key)
    combos = list(itertools.product(*params)) if params else [()]
    columns = {}
    for method in TRACKS:
        key = _key(result, class_name, method)
        values = result.get_result_value(key, params)
        if not isinstance(values, list):
            values = [values]
        if len(values) != len(combos):
            raise SystemExit(
                f"{class_name}.{method} has {len(values)} values for {len(combos)} parameters"
            )
        columns[method] = values
    version = result.benchmark_version.get(force_key)
    rows = []
    missing = []
    for index, combo in enumerate(combos):
        raw = {method: columns[method][index] for method in TRACKS}
        label = _combo_label(combo)
        if any(_missing(value) for value in raw.values()):
            missing.append(label)
            continue
        success = float(raw["track_success"])
        hits = [name for name in FAILURES if float(raw[f"track_failure_{name}"]) == 1.0]
        if success == 1.0:
            if hits:
                raise SystemExit(f"{class_name} {combo} is a success with failure tracks {hits}")
            failure = None
            ok = True
        elif success == 0.0:
            if len(hits) != 1:
                raise SystemExit(f"{class_name} {combo} failure tracks are {hits}")
            failure = hits[0]
            ok = False
        else:
            raise SystemExit(f"{class_name} {combo} success track is {success}")
        rows.append(
            {
                "params": tuple(str(_unwrap(part)) for part in combo),
                "force_calls": _count(raw["track_force_calls"]),
                "success": ok,
                "failure": failure,
                "version": version,
            }
        )
    return rows, missing


def rows_for_class(result, class_name: str) -> tuple[list[dict] | None, list[str]]:
    """Return tracked rows, or ``None`` when the class is absent."""
    try:
        return track_rows(result, class_name), []
    except SystemExit as exc:
        text = str(exc)
        if f"{class_name}.track_force_calls matched nothing" in text:
            return None, []
        if "was skipped or failed" not in text and "matched nothing" not in text:
            raise
        try:
            return tolerant_rows(result, class_name)
        except SystemExit as inner:
            inner_text = str(inner)
            if "matched nothing" not in inner_text:
                raise
            # A track method was not saved. No start has a complete record.
            force_key = _key(result, class_name, "track_force_calls")
            params = result.get_result_params(force_key)
            import itertools

            combos = list(itertools.product(*params)) if params else [()]
            return [], [_combo_label(combo) for combo in combos]


def _success_stats(rows: list[dict]) -> dict[str, float | None]:
    calls = [float(row["force_calls"]) for row in rows if row["success"]]
    if not calls:
        return {
            "force_calls_mean": None,
            "force_calls_median": None,
            "force_calls_min": None,
            "force_calls_max": None,
        }
    return {
        "force_calls_mean": statistics.fmean(calls),
        "force_calls_median": statistics.median(calls),
        "force_calls_min": min(calls),
        "force_calls_max": max(calls),
    }


def _failure_counts(rows: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in FAILURES}
    for row in rows:
        if row["success"]:
            continue
        counts[str(row["failure"])] += 1
    return counts


def summarize_class(result, class_name: str, cap: int, expected: int | None) -> dict | None:
    rows, missing = rows_for_class(result, class_name)
    if rows is None:
        return None
    tracked = len(rows)
    failures = _failure_counts(rows)
    stats = _success_stats(rows)
    scored = None
    if rows:
        scored = score_v2(
            [
                {
                    "force_calls": row["force_calls"],
                    "success": row["success"],
                    "failure": row["failure"],
                }
                for row in rows
            ],
            cap,
        )
    n_starts = tracked + len(missing)
    partial = bool(missing) or (expected is not None and n_starts != expected)
    return {
        "class": class_name,
        "cap": cap,
        "n_starts": n_starts,
        "n_tracked": tracked,
        "n_missing": len(missing),
        "expected_starts": expected,
        "partial": partial,
        "successes": sum(1 for row in rows if row["success"]),
        **{f"n_{name}": failures[name] for name in FAILURES},
        **stats,
        "par2": scored,
        "missing_starts": missing,
    }


def _param_total(result) -> int:
    total = 0
    for key in result.get_all_result_keys():
        if not (key.startswith("bench_ase_sella.") and key.endswith(".track_force_calls")):
            continue
        params = result.get_result_params(key)
        count = 1
        for axis in params or [[]]:
            count *= len(axis)
        total += count
    return total


def _ase_results(results_dir: Path, commit: str | None):
    from asv.results import iter_results

    found = []
    for result in iter_results(str(results_dir)):
        if _param_total(result) == 0:
            continue
        if commit is not None and not result.commit_hash.startswith(commit):
            continue
        found.append(result)
    if not found:
        raise SystemExit(f"{results_dir} has no bench_ase_sella tracks")
    # The one-seed file and the full-grid file can sit side by side.
    # The full grid is the file with the most parameter rows.
    found.sort(key=_param_total, reverse=True)
    best = found[0]
    same = [result for result in found if result.commit_hash == best.commit_hash]
    if len(same) != 1:
        raise SystemExit(
            f"commit {best.commit_hash} has ASE/Sella tracks in {len(same)} files; merge them first"
        )
    return best


def expected_grid() -> dict[str, dict]:
    """Full start counts from the benchmark module with OPTBENCH_ASV_FULL=1."""
    code = """
import json, os
os.environ["OPTBENCH_ASV_FULL"] = "1"
import benchmarks.bench_ase_sella as module
out = {}
for name, obj in vars(module).items():
    if isinstance(obj, type) and getattr(obj, "template_id", None):
        seeds, starts = obj.params
        out[name] = {
            "template_id": obj.template_id,
            "cap": int(obj.cap),
            "expected_starts": len(seeds) * len(starts),
        }
print(json.dumps(out, sort_keys=True))
"""
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["OPTBENCH_ASV_FULL"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _versions(python: str | None) -> dict:
    """Package versions from the interpreter that ran the benchmarks.

    asv itself is not installed in that environment. Its version is taken
    from the interpreter that launched ``asv run`` and stored separately.
    """
    versions = interpreter_versions(python)
    if versions.get("asv") is None:
        launcher = interpreter_versions(sys.executable)
        versions["launcher"] = {
            "python_version": launcher.get("python_version"),
            "asv": launcher.get("asv"),
            "asv-runner": launcher.get("asv-runner"),
        }
    return versions


def interpreter_versions(python: str | None) -> dict:
    code = """
import json, platform, sys
from importlib import metadata
names = ["ase", "sella", "numpy", "scipy", "asv", "asv-runner"]
versions = {"python": sys.version, "python_version": platform.python_version()}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps(versions))
"""
    executable = python or sys.executable
    completed = subprocess.run(
        [executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    payload["interpreter"] = executable
    return payload


def build_document(
    results_dir: Path,
    *,
    python: str | None,
    run_env: dict | None,
    commit: str | None = None,
) -> dict:
    result = _ase_results(results_dir, commit)
    grid = expected_grid()
    classes = []
    for name in sorted(grid):
        spec = grid[name]
        row = summarize_class(result, name, int(spec["cap"]), int(spec["expected_starts"]))
        if row is None:
            classes.append(
                {
                    "class": name,
                    "template_id": spec["template_id"],
                    "cap": spec["cap"],
                    "n_starts": 0,
                    "n_tracked": 0,
                    "n_missing": spec["expected_starts"],
                    "expected_starts": spec["expected_starts"],
                    "partial": True,
                    "successes": 0,
                    **{f"n_{name_}": 0 for name_ in FAILURES},
                    "force_calls_mean": None,
                    "force_calls_median": None,
                    "force_calls_min": None,
                    "force_calls_max": None,
                    "par2": None,
                    "missing_starts": [],
                    "note": "not run",
                }
            )
            continue
        row["template_id"] = spec["template_id"]
        if row["partial"] and row["n_tracked"] == 0:
            row["note"] = "asv recorded no track"
        elif row["partial"]:
            row["note"] = "grid incomplete; par2 is the tracked starts only"
        else:
            row["note"] = "full grid"
        classes.append(row)
    present = set()
    for key in result.get_all_result_keys():
        if key.startswith("bench_ase_sella.") and key.endswith(".track_force_calls"):
            present.add(key.split(".", 1)[1].rsplit(".", 1)[0])
    unknown = sorted(present - set(grid))
    if unknown:
        raise SystemExit(f"result has classes that are not in the catalog: {unknown}")
    return {
        "schema": "optbench.ase-sella-full/v1",
        "commit": result.commit_hash,
        "env_name": result._env_name,
        "python_result": result._python,
        "date": result.date,
        "machine": relabel_machine(dict(result.params)),
        "versions": _versions(python),
        "run_env": run_env or {},
        "classes": classes,
    }


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def render_markdown(document: dict) -> str:
    lines = [
        "# ASE/Sella full-grid summary",
        "",
        f"Commit `{document['commit']}`. PAR-2 is `score_v2` on tracked starts.",
        "",
        "| class | n | tracked | success | cap | divergence | wrong_stationary | crash | missing | mean | median | min | max | par2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in document["classes"]:
        par2 = None if row["par2"] is None else row["par2"].get("par2_mean")
        lines.append(
            "| "
            + " | ".join(
                [
                    row["class"],
                    str(row["n_tracked"] + row["n_missing"] if row["note"] != "not run" else 0),
                    str(row["n_tracked"]),
                    str(row["successes"]),
                    str(row["n_cap"]),
                    str(row["n_divergence"]),
                    str(row["n_wrong_stationary"]),
                    str(row["n_crash"]),
                    str(row["n_missing"] if row["note"] != "not run" else row["expected_starts"]),
                    _fmt(row["force_calls_mean"]),
                    _fmt(row["force_calls_median"]),
                    _fmt(row["force_calls_min"]),
                    _fmt(row["force_calls_max"]),
                    _fmt(par2),
                ]
            )
            + " |"
        )
    partial = [row for row in document["classes"] if row["partial"]]
    lines.extend(["", "## Partial or absent grids", ""])
    if not partial:
        lines.append("Every class has a track for every start.")
    for row in partial:
        lines.append(
            f"- `{row['class']}`: tracked {row['n_tracked']} of {row['expected_starts']} ({row['note']})"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "asv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ase-sella-full" / "summary.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results" / "ase-sella-full" / "summary.md")
    parser.add_argument("--interpreter", default=None, help="Python that has the ASE and Sella that ran")
    parser.add_argument("--run-env", type=Path, default=None)
    parser.add_argument("--commit", default=None, help="Commit prefix to score. Default: fullest file.")
    args = parser.parse_args(argv)
    run_env = None
    if args.run_env is not None:
        run_env = json.loads(args.run_env.read_text(encoding="utf-8"))
    document = build_document(
        args.results,
        python=args.interpreter,
        run_env=run_env,
        commit=args.commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(document), encoding="utf-8")


if __name__ == "__main__":
    main()
