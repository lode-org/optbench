"""Turn an ``asv run`` result file into campaign ``benchmark.dat`` files.

The numbers are the track values asv stored. Nothing here reruns eonclient.
``track_force_calls`` is the ``HarnessCounter`` tally from the ext_pot
bridge. A file written from the default one-start grid sets
``published_mean`` false.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
from pathlib import Path

STEMS = {
    "EonDevelopNebPtLbfgs": "eon-asv-neb-pt-lbfgs",
    "EonDevelopNebPtLbfgsM100": "eon-asv-neb-pt-lbfgs-m100",
    "EonDevelopSaddlePtDimer": "eon-develop-saddle-pt-dimer",
    "EonDevelopSaddlePtLanczos": "eon-develop-saddle-pt-lanczos",
}


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _flag(value) -> bool:
    return not _missing(value) and float(value) == 1.0


def _reason(success, cap, wrong, divergence, crash) -> str | None:
    """Client-file reason string from the four failure tracks."""
    if _flag(success):
        return "0"
    if _flag(crash):
        return "crash"
    if _flag(divergence):
        return "divergence"
    if _flag(cap):
        return "cap"
    if _flag(wrong):
        return "wrong_stationary"
    return None


def _track(payload: dict, benchmark: str) -> list:
    row = payload["results"][benchmark]
    values = row[0]
    if values is None:
        raise ValueError(f"{benchmark} has no result list")
    return list(values)


def _pairs(spec: dict) -> list[tuple[int, str]]:
    return [(int(seed), str(start)) for seed, start in itertools.product(spec["seeds"], spec["starts"])]


def _append(path: Path, lines: list[str]) -> None:
    text = path.read_text(encoding="utf-8").rstrip("\n")
    path.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def write_one(
    root: Path,
    spec: dict,
    tracks: dict[str, list],
    output: Path,
    *,
    asv_commit: str,
    workers: int,
) -> dict:
    """Write one benchmark.dat and its ``.starts`` from aligned track lists."""
    from .campaign import load_template
    from .eon_akmc import _write_saddle
    from .eon_jobs import _finish, _pair_count

    template = load_template(root, spec["template"])
    pairs = _pairs(spec)
    count = len(pairs)
    for name, values in tracks.items():
        if len(values) != count:
            raise ValueError(f"{spec['cls']} {name} has {len(values)} values, expected {count}")
    calls = tracks["force_calls"]
    success = tracks["success"]
    seconds = tracks.get("seconds")
    if spec["kind"] == "akmc":
        rows = []
        for index, (seed, _start) in enumerate(pairs):
            raw = None if _missing(calls[index]) else int(round(float(calls[index])))
            reached = (not _missing(success[index])) and float(success[index]) == 1.0
            def optional(name: str):
                value = tracks[name][index]
                return None if _missing(value) else float(value)

            rows.append(
                {
                    "force_calls": raw,
                    "fraction_connected": optional("fraction_connected"),
                    "passes": None,
                    "qualifying": None,
                    "reached": reached,
                    "repetition": index + 1,
                    "seconds": None if seconds is None or _missing(seconds[index]) else float(seconds[index]),
                    "seed": seed,
                    "total_searches": optional("total_searches"),
                    "unique_saddles": optional("unique_saddles"),
                }
            )
        summary = _write_saddle(template, rows, output, workers)
        text = output.read_text(encoding="utf-8")
        text = text.replace(
            "counter client-search-results\ncount_keys sad-fcs+mins-fcs\n",
            "counter harness\ncount_keys HarnessCounter\n",
        )
        output.write_text(text, encoding="utf-8")
        _append(output, ["published_mean false"])
    else:
        rows = []
        for index, (seed, start) in enumerate(pairs):
            raw = None if _missing(calls[index]) else int(round(float(calls[index])))
            row = {
                "count_key": "harness",
                "geometry": start,
                "raw": raw,
                "seed": seed,
            }
            if seconds is not None and not _missing(seconds[index]):
                row["seconds"] = float(seconds[index])
            if "extrema" in tracks and not _missing(tracks["extrema"][index]):
                row["extrema"] = int(round(float(tracks["extrema"][index])))
            row["reason"] = _reason(
                None if _missing(success[index]) else float(success[index]),
                tracks.get("failure_cap", [None] * count)[index],
                tracks.get("failure_wrong_stationary", [None] * count)[index],
                tracks.get("failure_divergence", [None] * count)[index],
                tracks.get("failure_crash", [None] * count)[index],
            )
            rows.append(row)
        extras = _pair_count(rows) if spec["kind"] == "de-2ts" else None
        summary = _finish(template, rows, output, 1, extras)
        _append(
            output,
            [
                "counter harness",
                "counter_file potserver",
                "counter_note HarnessCounter through ext_pot",
                "published_mean false",
            ],
        )
    _append(
        output,
        [
            f"asv_benchmark benchmarks.bench_eon_develop.{spec['cls']}",
            f"asv_commit {asv_commit}",
        ],
    )
    return summary


def _class_result(payload: dict, cls: str, method: str) -> list | None:
    name = f"benchmarks.bench_eon_develop.{cls}.{method}"
    if name not in payload["results"]:
        return None
    values = payload["results"][name][0]
    if values is None:
        return None
    return list(values)


def write_results(root: Path, result_path: Path, campaign: Path, *, workers: int = 8) -> list[str]:
    """Write a dat for every develop class present in one asv result file."""
    from benchmarks.bench_eon_develop import _specs

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    commit = str(payload["commit_hash"])
    written: list[str] = []
    campaign.mkdir(parents=True, exist_ok=True)
    for spec in _specs():
        calls = _class_result(payload, spec["cls"], "track_force_calls")
        success = _class_result(payload, spec["cls"], "track_success")
        if calls is None or success is None:
            continue
        tracks = {"force_calls": calls, "success": success}
        for name in (
            "failure_cap",
            "failure_wrong_stationary",
            "failure_divergence",
            "failure_crash",
        ):
            values = _class_result(payload, spec["cls"], f"track_{name}")
            if values is None:
                raise ValueError(f"{spec['cls']} is missing track_{name}")
            tracks[name] = values
        seconds = _class_result(payload, spec["cls"], "time_search")
        if seconds is not None:
            tracks["seconds"] = seconds
        if spec["kind"] == "akmc":
            for name in ("unique_saddles", "fraction_connected", "total_searches"):
                values = _class_result(payload, spec["cls"], f"track_{name}")
                if values is None:
                    raise ValueError(f"{spec['cls']} is missing track_{name}")
                tracks[name] = values
        if spec["kind"] == "de-2ts":
            extrema = _class_result(payload, spec["cls"], "track_extrema")
            if extrema is None:
                raise ValueError(f"{spec['cls']} is missing track_extrema")
            tracks["extrema"] = extrema
        stem = STEMS[spec["cls"]]
        write_one(root, spec, tracks, campaign / f"{stem}.dat", asv_commit=commit, workers=workers)
        written.append(stem)
    return written


def copy_asv_tree(asv_results: Path, dest: Path) -> list[Path]:
    """Copy machine.json and result JSON into a tracked directory."""
    copied: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(asv_results.rglob("*.json")):
        target = dest / path.relative_to(asv_results)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied
