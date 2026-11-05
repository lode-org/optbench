"""Compare eOn develop rows with the SVN r2549 campaign.

Reads the develop asv tracks, the harness cache, and the SVN campaign
files on ``origin/entries/eon-svn-fortran``. PAR-2 is ``score_v2`` on
one SVN replicate and on the develop tracks. The develop cap is the
template ``failure_cap`` in this tree. A previous cap is scored as well
when the template used to say 100000, so the old ranked number is
recomputed from the same calls.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SVN_REF = "origin/entries/eon-svn-fortran"
ASV = ROOT / "results/asv/xeon-8c/06efc6dd-existing-py_workspace_.venv_bin_python.json"
CACHE = Path("/tmp/optbench-asv/harness-1/7095b23ac17d49f434a5e08ccb072897ceca698e")
MATCHED = Path(__file__).resolve().parent / "matched-svn-ini"

# Defaults in include/eon/ParametersOptions.h on 7095b23.
LBFGS_DEFAULTS = {
    "lbfgs_memory": 20,
    "lbfgs_inverse_curvature": 0.01,
    "lbfgs_max_inverse_curvature": 0.0,
    "lbfgs_auto_scale": True,
    "lbfgs_angle_reset": True,
    "lbfgs_distance_reset": True,
    "lbfgs_curvature": "reset",
    "lbfgs_project_rigid": False,
    "lbfgs_secant": "standard",
    "lbfgs_precon": "none",
    "lbfgs_step": "lbfgs",
    "lbfgs_h0": "sy_yy",
    # ParametersOptions.h initializes accept to "energy". The generated
    # SSOT default, which is what a missing key leaves in place, is "none".
    "lbfgs_accept": "none",
    "lbfgs_extra_updates": 0,
    "lbfgs_cautious_eps": 1.0e-6,
    "lbfgs_cautious_alpha": 0.01,
    "lbfgs_precon_A": 3.0,
    "lbfgs_precon_mu": 1.0,
    "lbfgs_precon_rcut": 0.0,
}

PAIRS = (
    {
        "name": "ts-lj38-dimer",
        "develop_template": "eon/develop-ts-lj38-dimer-accept-energy",
        "develop_class": "EonDevelopTsLj38Dimer",
        "svn_template": "templates/eon-svn/ts-lj38-dimer.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-ts-lj38-dimer.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-ts-lj38-dimer.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"{i:04d}" for i in range(200)],
        "previous_cap": None,
    },
    {
        "name": "ts-lj38-lanczos",
        "develop_template": "eon/develop-ts-lj38-lanczos",
        "develop_class": "EonDevelopTsLj38Lanczos",
        "svn_template": "templates/eon-svn/ts-lj38-lanczos.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-ts-lj38-lanczos.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-ts-lj38-lanczos.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"{i:04d}" for i in range(200)],
        "previous_cap": None,
    },
    {
        "name": "ts-pt-dimer",
        "develop_template": "eon/develop-ts-pt-dimer-accept-energy",
        "develop_class": "EonDevelopTsPtDimer",
        "svn_template": "templates/eon-svn/ts-pt-dimer.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-ts-pt-dimer.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-ts-pt-dimer.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"initial_{i}" for i in range(49)],
        "previous_cap": None,
    },
    {
        "name": "ts-pt-lanczos",
        "develop_template": "eon/develop-ts-pt-lanczos",
        "develop_class": "EonDevelopTsPtLanczos",
        "svn_template": "templates/eon-svn/ts-pt-lanczos.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-ts-pt-lanczos.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-ts-pt-lanczos.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"initial_{i}" for i in range(49)],
        "previous_cap": None,
    },
    {
        "name": "de-lj38-ci",
        "develop_template": "eon/develop-de-lj38-cap-accept-energy",
        "develop_class": "EonDevelopDeLj38Cap",
        "svn_template": "templates/eon-svn/de-lj38-ci.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-de-lj38-ci.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-de-lj38-ci.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"pair_{i:02d}" for i in range(50)],
        "previous_cap": None,
    },
    {
        "name": "de-2ts-ci",
        "develop_template": "eon/develop-de-2ts-accept-energy",
        "develop_class": "EonDevelopDe2ts",
        "svn_template": "templates/eon-svn/de-2ts-ci.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-de-2ts-ci.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-de-2ts-ci.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"pair_{i:02d}" for i in range(51)],
        "previous_cap": 100000,
    },
    {
        "name": "low-lj38-dimer",
        "develop_template": "eon/develop-low-lj38-dimer",
        "develop_class": "EonDevelopLowLj38Dimer",
        "svn_template": "templates/eon-svn/low-lj38-dimer.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-low-lj38-dimer.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-low-lj38-dimer.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"{i:04d}" for i in range(200)],
        "previous_cap": None,
    },
    {
        "name": "low-lj38-lanczos",
        "develop_template": "eon/develop-low-lj38-lanczos",
        "develop_class": "EonDevelopLowLj38Lanczos",
        "svn_template": "templates/eon-svn/low-lj38-lanczos.toml",
        "svn_dat": "results/campaign/eon-svn-r2549-low-lj38-lanczos.dat",
        "svn_jsonl": "results/campaign/eon-svn-r2549-low-lj38-lanczos.starts.jsonl",
        "seeds": ["1"],
        "starts": [f"{i:04d}" for i in range(200)],
        "previous_cap": None,
    },
)


def _git_text(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{SVN_REF}:{path}"],
        text=True,
    )


def _dat_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        fields[key] = value
    return fields


def _flatten(ini: dict) -> dict[str, object]:
    flat = {}
    for section, body in ini.items():
        if not isinstance(body, dict):
            flat[section] = body
            continue
        for key, value in body.items():
            flat[f"{section}.{key}"] = value
    return flat


def _same(left, right) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def _config_diff(develop: dict, svn: dict) -> dict:
    left = _flatten(develop.get("ini") or {})
    right = _flatten(svn.get("ini") or {})
    only_develop = []
    only_svn = []
    changed = []
    for key in sorted(set(left) | set(right)):
        if key not in right:
            only_develop.append({"key": key, "develop": left[key]})
        elif key not in left:
            only_svn.append({"key": key, "svn": right[key]})
        elif not _same(left[key], right[key]):
            changed.append({"key": key, "develop": left[key], "svn": right[key]})
    catalog = []
    for key, default in LBFGS_DEFAULTS.items():
        value = (develop.get("ini") or {}).get("LBFGS", {}).get(key)
        if value is None:
            continue
        catalog.append({"key": key, "develop": value, "header_default": default, "equals_default": _same(value, default)})
    return {
        "develop_failure_cap": develop.get("failure_cap"),
        "svn_failure_cap": svn.get("failure_cap"),
        "only_develop": only_develop,
        "only_svn": only_svn,
        "changed": changed,
        "lbfgs_catalog_versus_header_defaults": catalog,
    }


def _track(payload: dict, cls: str, method: str):
    name = f"bench_eon_develop.{cls}.{method}"
    if name not in payload["results"]:
        return None
    return payload["results"][name][0]


def _develop_records(payload: dict, pair: dict, cap: int) -> list[dict]:
    order = [(seed, start) for seed, start in itertools.product(pair["seeds"], pair["starts"])]
    calls = _track(payload, pair["develop_class"], "track_force_calls")
    success = _track(payload, pair["develop_class"], "track_success")
    flags = {
        name: _track(payload, pair["develop_class"], f"track_failure_{name}")
        for name in ("cap", "wrong_stationary", "divergence", "crash")
    }
    if calls is None or success is None or len(calls) != len(order):
        raise ValueError(f"{pair['develop_class']} track length does not match the start list")
    safe = pair["develop_template"].replace("/", "-")
    records = []
    for index, (seed, start) in enumerate(order):
        counted = float(calls[index])
        ok = float(success[index]) == 1.0
        failure = None
        if not ok:
            for name in ("crash", "divergence", "cap", "wrong_stationary"):
                values = flags[name]
                if values is not None and values[index] == 1.0:
                    failure = name
                    break
        harness_ok = ok
        harness_failure = failure
        if counted >= cap and ok:
            ok = False
            failure = "cap"
        cache_path = CACHE / safe / "passes-0" / str(seed) / f"{start}.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else None
        records.append(
            {
                "seed": seed,
                "start": start,
                "force_calls": counted,
                "success": ok,
                "failure": failure,
                "harness_success": harness_ok,
                "harness_failure": harness_failure,
                "client_calls": None if cache is None else cache.get("client_calls"),
                "cache_seconds": None if cache is None else cache.get("seconds"),
                "cache_overlap": None if cache is None else cache.get("overlap"),
                "cache_force_calls": None if cache is None else cache.get("force_calls"),
            }
        )
    return records


def _svn_replicates(text: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row.get("replicate")), []).append(row)
    return grouped


def _ratio_stats(ratios: list[float]) -> dict | None:
    if not ratios:
        return None
    return {
        "n": len(ratios),
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
    }


def _counting(develop: list[dict], svn: list[dict]) -> dict:
    by_start = {row["start"]: row for row in svn}
    both_success = []
    harness_eq_client = 0
    client_known = 0
    cache_matches_harness = 0
    cache_known = 0
    svn_ok_develop_fail = 0
    develop_ok_svn_fail = 0
    ratios = []
    for row in develop:
        other = by_start.get(row["start"])
        if other is None:
            continue
        if row["client_calls"] is not None:
            client_known += 1
            if float(row["client_calls"]) == float(row["force_calls"]):
                harness_eq_client += 1
        if row["cache_force_calls"] is not None:
            cache_known += 1
            if float(row["cache_force_calls"]) == float(row["force_calls"]):
                cache_matches_harness += 1
        if other["success"] and not row["harness_success"]:
            svn_ok_develop_fail += 1
        if row["harness_success"] and not other["success"]:
            develop_ok_svn_fail += 1
        if row["harness_success"] and other["success"] and float(other["force_calls"]) > 0:
            ratios.append(float(row["force_calls"]) / float(other["force_calls"]))
            both_success.append(row["start"])
    return {
        "starts": len(develop),
        "client_known": client_known,
        "harness_equals_client": harness_eq_client,
        "cache_known": cache_known,
        "cache_equals_asv_harness": cache_matches_harness,
        "svn_success_develop_harness_fail": svn_ok_develop_fail,
        "develop_harness_success_svn_fail": develop_ok_svn_fail,
        "both_harness_success_ratio_develop_over_svn": _ratio_stats(ratios),
    }


def _score_block(records: list[dict], cap: int) -> dict:
    from optbench.score import score_v2

    scored = score_v2(records, cap)
    keep = (
        "par2_mean",
        "successes",
        "nfailed",
        "success_only_mean",
        "median",
        "n_cap",
        "n_divergence",
        "n_crash",
        "n_wrong_stationary",
        "cap",
        "n",
    )
    return {key: scored[key] for key in keep}


def _matched(name: str) -> dict | None:
    path = MATCHED / f"{name}.json"
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    row.pop("ini", None)
    return row


def _ablation(name: str) -> list[dict] | None:
    path = MATCHED / f"{name}-ablation.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "label": row["label"],
            "start": row["start"],
            "force_calls": row["force_calls"],
            "client_calls": row["client_calls"],
            "client_reason": row["client_reason"],
            "seconds": row["seconds"],
        }
        for row in rows
    ]


def _confirm(name: str) -> list[dict]:
    path = MATCHED / "confirm.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "label": row["label"],
            "start": row["start"],
            "force_calls": row["force_calls"],
            "client_calls": row["client_calls"],
            "client_reason": row["client_reason"],
            "seconds": row["seconds"],
        }
        for row in rows
        if row.get("name") == name
    ]


def _verdict(diff: dict, counting: dict, matched: dict | None, ablation: list[dict] | None) -> str:
    changed_keys = {item["key"] for item in diff["changed"]}
    method = {
        "Potential.potential",
        "Nudged Elastic Band.climbing_image_method",
        "Nudged Elastic Band.max_iterations",
        "Nudged Elastic Band.images",
        "Optimizer.max_iterations",
        "Optimizer.max_move",
        "Dimer.converged_angle",
        "Dimer.rotations_max",
        "Saddle Search.min_mode_method",
    }
    if changed_keys & method:
        return "config mismatch"
    ratio = (counting.get("both_harness_success_ratio_develop_over_svn") or {}).get("median")
    identical = not diff["changed"] and not diff["only_develop"] and not diff["only_svn"]
    if identical and ratio is not None and 0.9 <= ratio <= 1.1:
        return "not a regression"
    accept = next((item for item in diff["only_develop"] if item["key"] == "LBFGS.lbfgs_accept"), None)
    accept_moves = False
    if ablation:
        by_label = {row["label"]: row["force_calls"] for row in ablation}
        if "svn-ini" in by_label and "svn-plus-lbfgs_accept" in by_label:
            accept_moves = float(by_label["svn-plus-lbfgs_accept"]) > 2.0 * float(by_label["svn-ini"])
    if accept is not None and accept.get("develop") == "energy" and (accept_moves or matched is not None):
        return "config mismatch"
    if matched is not None:
        return "config mismatch"
    if ratio is not None and 0.9 <= ratio <= 1.1:
        return "not a regression"
    client_gap = counting["client_known"] and counting["harness_equals_client"] < counting["client_known"]
    if client_gap and ratio is not None and ratio < 1.2:
        return "counting mismatch"
    if ratio is not None and ratio > 3:
        return "genuine algorithmic regression"
    return "config mismatch"


def _pair_row(payload: dict, spec: dict) -> dict:
    develop = tomllib.loads((ROOT / "templates" / f"{spec['develop_template']}.toml").read_text(encoding="utf-8"))
    svn = tomllib.loads(_git_text(spec["svn_template"]))
    diff = _config_diff(develop, svn)
    cap = int(develop["failure_cap"])
    develop_records = _develop_records(payload, spec, cap)
    replicates = _svn_replicates(_git_text(spec["svn_jsonl"]))
    names = sorted(replicates)
    base_name = "svn-all" if "svn-all" in replicates else names[0]
    base = replicates[base_name]
    same = {
        name: [row["force_calls"] for row in rows] == [row["force_calls"] for row in base]
        for name, rows in replicates.items()
    }
    svn_records = [
        {
            "start": row["start"],
            "force_calls": float(row["force_calls"]),
            "success": bool(row["success"]),
            "failure": row.get("failure"),
            "client_calls": row.get("client_calls"),
            "seconds": row.get("seconds"),
            "overlap": row.get("overlap"),
        }
        for row in base
    ]
    dat = _dat_fields(_git_text(spec["svn_dat"]))
    matched = _matched(spec["name"])
    counting = _counting(develop_records, svn_records)
    develop_score = _score_block(develop_records, cap)
    svn_score = _score_block(svn_records, int(svn["failure_cap"]))
    previous = None
    if spec["previous_cap"] is not None:
        previous_records = _develop_records(payload, spec, int(spec["previous_cap"]))
        previous = _score_block(previous_records, int(spec["previous_cap"]))
    first = spec["starts"][0]
    develop_first = next(row for row in develop_records if row["start"] == first)
    svn_first = next(row for row in svn_records if row["start"] == first)
    return {
        "template": spec["develop_template"],
        "svn_template": svn["id"],
        "develop_par2": develop_score["par2_mean"],
        "develop_cap": cap,
        "develop_score": develop_score,
        "develop_par2_at_previous_cap": None if previous is None else previous["par2_mean"],
        "previous_cap_score": previous,
        "svn_par2": svn_score["par2_mean"],
        "svn_cap": int(svn["failure_cap"]),
        "svn_score": svn_score,
        "svn_dat_par2": float(dat["par2_mean"]),
        "svn_dat_n": int(dat["n"]),
        "svn_replicate": base_name,
        "svn_replicates_identical_force_calls": same,
        "config": diff,
        "counting": counting,
        "first_start": {
            "start": first,
            "develop_harness": develop_first["force_calls"],
            "develop_client": develop_first["client_calls"],
            "develop_cache_seconds": develop_first["cache_seconds"],
            "svn_harness": svn_first["force_calls"],
            "svn_client": svn_first["client_calls"],
            "svn_seconds": svn_first["seconds"],
        },
        "matched_svn_ini": matched,
        "ablation": _ablation(spec["name"]),
        "confirm": _confirm(spec["name"]),
        "reruns": _reruns(develop_records, svn_records, matched, _ablation(spec["name"]), _confirm(spec["name"])),
        "verdict": _verdict(diff, counting, matched, _ablation(spec["name"])),
    }


def _lowest(payload: dict, spec: dict, row: dict) -> dict:
    """Show the equal PAR-2 came from the develop cache, not a copied file."""
    develop = _develop_records(payload, spec, int(row["develop_cap"]))
    replicates = _svn_replicates(_git_text(spec["svn_jsonl"]))
    base = replicates[row["svn_replicate"]]
    svn_calls = [float(item["force_calls"]) for item in base]
    dev_calls = [float(item["force_calls"]) for item in develop]
    seconds_differ = 0
    overlap_differ = 0
    compared = 0
    client_values = []
    for item, other in zip(develop, base, strict=True):
        compared += 1
        if item["cache_seconds"] is not None and item["cache_seconds"] != other.get("seconds"):
            seconds_differ += 1
        if item["cache_overlap"] is not None and other.get("overlap") is not None:
            if not math.isclose(float(item["cache_overlap"]), float(other["overlap"]), rel_tol=0.0, abs_tol=0.0):
                overlap_differ += 1
        if item["client_calls"] is not None:
            client_values.append(item["client_calls"])
    return {
        "template": spec["develop_template"],
        "harness_sequence_equals_svn_replicate": dev_calls == svn_calls,
        "starts": compared,
        "cache_seconds_differ_from_svn": seconds_differ,
        "cache_overlap_differ_from_svn": overlap_differ,
        "first_develop_seconds": develop[0]["cache_seconds"],
        "first_svn_seconds": base[0].get("seconds"),
        "first_develop_overlap": develop[0]["cache_overlap"],
        "first_svn_overlap": base[0].get("overlap"),
        "client_calls_min": None if not client_values else min(client_values),
        "client_calls_max": None if not client_values else max(client_values),
        "copied_from_svn_files": False if dev_calls == svn_calls and seconds_differ == compared else None,
    }


def _reruns(develop: list[dict], svn: list[dict], matched: dict | None, ablation: list[dict] | None, confirm: list[dict]) -> list[dict]:
    svn_by = {row["start"]: row for row in svn}
    dev_by = {row["start"]: row for row in develop}
    slots: dict[str, dict] = {}

    def slot(start: str) -> dict:
        found = slots.setdefault(start, {"start": start})
        if start in dev_by:
            found["develop_harness"] = dev_by[start]["force_calls"]
            found["develop_client"] = dev_by[start]["client_calls"]
        if start in svn_by:
            found["svn_harness"] = svn_by[start]["force_calls"]
            found["svn_client"] = svn_by[start]["client_calls"]
        return found

    if matched:
        slot(matched["start"])["svn_ini"] = matched["force_calls"]
        slot(matched["start"])["svn_ini_client"] = matched["client_calls"]
    for item in ablation or []:
        if item["label"] == "svn-ini":
            slot(item["start"])["svn_ini"] = item["force_calls"]
        elif item["label"] == "svn-plus-lbfgs_accept":
            slot(item["start"])["accept_energy"] = item["force_calls"]
        elif item["label"] == "develop-template":
            slot(item["start"])["develop_template_rerun"] = item["force_calls"]
    for item in confirm:
        if item["label"] == "svn-ini":
            slot(item["start"])["svn_ini"] = item["force_calls"]
        elif item["label"] == "svn-plus-accept":
            slot(item["start"])["accept_energy"] = item["force_calls"]
    return [slots[key] for key in slots]


def _table(rows: list[dict]) -> str:
    lines = [
        "template | SVN PAR-2 | develop PAR-2 | cap SVN/develop | config differences | verdict",
        "---|---|---|---|---|---",
    ]
    for row in rows:
        diff = row["config"]
        bits = []
        if diff["develop_failure_cap"] != diff["svn_failure_cap"]:
            bits.append(f"failure_cap {diff['svn_failure_cap']} vs {diff['develop_failure_cap']}")
        if diff["changed"]:
            bits.append("changed " + ", ".join(item["key"] for item in diff["changed"]))
        if diff["only_develop"]:
            bits.append("develop-only " + ", ".join(item["key"] for item in diff["only_develop"]))
        if diff["only_svn"]:
            bits.append("svn-only " + ", ".join(item["key"] for item in diff["only_svn"]))
        if not bits:
            bits.append("ini identical")
        develop_par2 = f"{row['develop_par2']}"
        if row["develop_par2_at_previous_cap"] is not None:
            develop_par2 += f" (was {row['develop_par2_at_previous_cap']} at cap {row['previous_cap_score']['cap']})"
        lines.append(
            " | ".join(
                [
                    row["template"],
                    f"{row['svn_par2']}",
                    develop_par2,
                    f"{row['svn_cap']}/{row['develop_cap']}",
                    "; ".join(bits),
                    row["verdict"],
                ]
            )
        )
    lines.append("")
    lines.append("matched reruns on 7095b23")
    lines.append("template | start | svn harness | develop cache harness | svn ini | plus lbfgs_accept=energy")
    for row in rows:
        for item in row.get("reruns") or []:
            lines.append(
                " | ".join(
                    [
                        row["template"],
                        str(item["start"]),
                        str(item.get("svn_harness", "")),
                        str(item.get("develop_harness", "")),
                        str(item.get("svn_ini", "")),
                        str(item.get("accept_energy", "")),
                    ]
                )
            )
    notes = []
    for row in rows:
        same = row["svn_replicates_identical_force_calls"]
        if not all(same.values()):
            notes.append(
                f"{row['svn_template']}: replicates are not identical, dat PAR-2 {row['svn_dat_par2']} "
                f"is the published doubled sample, one-replicate score_v2 on {row['svn_replicate']} "
                f"is {row['svn_par2']}"
            )
    if notes:
        lines.append("")
        lines.extend(notes)
    for row in rows:
        ablation = {item["label"]: item["force_calls"] for item in row.get("ablation") or []}
        if "svn-ini" not in ablation or "svn-plus-lbfgs_accept" not in ablation:
            continue
        others = [
            label
            for label in ablation
            if label.startswith("svn-plus-lbfgs_") and label != "svn-plus-lbfgs_accept"
        ]
        if others and all(ablation[label] == ablation["svn-ini"] for label in others):
            lines.append(
                f"{row['template']} start {row['first_start']['start']}: "
                f"svn ini {ablation['svn-ini']}, each other catalog key {ablation['svn-ini']}, "
                f"lbfgs_accept=energy {ablation['svn-plus-lbfgs_accept']}, "
                f"develop template {ablation.get('develop-template')}. "
                "A missing lbfgs_accept stays the generated SSOT default none. "
                "ParametersOptions.h initializes that field to energy, and the SSOT "
                "value replaces it before the ini is read."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = json.loads(ASV.read_text(encoding="utf-8"))
    rows = [_pair_row(payload, spec) for spec in PAIRS]
    lowest = {
        spec["name"]: _lowest(payload, spec, row)
        for spec, row in zip(PAIRS, rows, strict=True)
        if spec["name"].startswith("low-")
    }
    body = {"svn_ref": SVN_REF, "asv": str(ASV.relative_to(ROOT)), "pairs": rows, "lowest": lowest}
    dest = Path(__file__).resolve().parent / "svn-compare.json"
    dest.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    table = _table(rows)
    table_path = Path(__file__).resolve().parent / "svn-compare.txt"
    table_path.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
