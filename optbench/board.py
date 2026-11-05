"""Leaderboard from committed campaign files, asv tracks, and archived scores.

``uv run optbench board`` reads
``results/campaign``, ``results/asv``, and ``oracle/entries.jsonl``.
A ranked row is one code, one version, and one stock method or preset.
Harness rows use the v2 PAR-2 mean. An archived row uses its published
primary number and sits in the same tables.

The ranked cost of a band method is the harness force-call total across
every image. The per-image mean divides that same total by the problem's
image count, and every code on the problem uses that divisor.

A row whose start count is below the problem's full set is short.
An asv result that covers one start is short.
The full set is the problem file's sample, or the Morse archive when
that file gives no sample count. It is not the largest campaign file.
A result that sets ``ranked false``, or the previous short token, stays
off the ranked table. A row that is not the stock method stays off too.
A campaign file and an asv track for the same run are one row. Two full
versions of one method are two rows.
"""

from __future__ import annotations

import ast
import html
import json
import math
import re
import shutil
import subprocess
import tarfile
import tomllib
from collections import defaultdict
from pathlib import Path

from .campaign import geometry_archive, list_templates
from .corrections import corrected, load_corrections
from .differences import difference_note, load_differences
from .counts import entry_count, load_counts
from .dat import DatError, parse_lenient
from .failures import FAILURES, failure_of
from .families import (
    FAMILIES,
    coordinate_count,
    display_code,
    family_of,
    family_title,
    finite_difference_calls,
    is_adjusted,
    method_name,
    row_label,
    version_text,
)
from .pins import pin_gap, same_commit
from .problems import load_all
from .profile import performance_profile, profile_cost
from .score import PAR_FACTOR, normalize_start, score_v2
from .site import SUITE_ORDER, _float, _held, _primary, campaign_problem
from .units import FORCE_CALL_FIELDS, harness_unit, rank_unit, row_unit, unit_source
from .v2 import parse_repetitions

BOARD_SCHEMA = "optbench.board/v2"


def _slug(problem_id: str) -> str:
    return problem_id.replace("/", "-")


def _float_field(fields: dict[str, str], key: str) -> float | None:
    raw = fields.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _json_lines(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def stated_full_n(root: Path, problem: dict) -> int | None:
    """How many starts the problem file calls a complete score.

    ``sample.starts``, ``pairs``, ``repetitions`` or ``initializations``
    wins. A minimization archive with no sample count uses the number of
    ``.con`` files, which is the Morse bulk set.
    """
    sample = problem.get("sample") or {}
    for key in ("starts", "pairs", "repetitions", "initializations"):
        value = sample.get(key)
        if value not in (None, "", 0, "0"):
            return int(value)
    archive_path = (problem.get("input") or {}).get("archive_path")
    if not archive_path:
        return None
    try:
        archive = geometry_archive(root, problem)
    except (OSError, ValueError, FileNotFoundError):
        return None
    if not tarfile.is_tarfile(archive):
        return None
    with tarfile.open(archive) as packed:
        count = sum(
            1
            for member in packed.getmembers()
            if member.isfile() and member.name.endswith(".con")
        )
    return count or None


def _registry(root: Path) -> list[dict]:
    path = root / "codes" / "registry.toml"
    if not path.is_file():
        return []
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    return [code for code in record.get("code") or [] if isinstance(code, dict)]


def _registry_match(row: dict, registry: list[dict]) -> bool:
    """True when this row's commit is the pin for its code.

    Several registry entries share the name eOn. The commit selects which pin.
    """
    recorded = str(row.get("commit") or "")
    code = str(row.get("code") or "")
    if not recorded or not code:
        return False
    return pin_gap(code, recorded, registry) is None


class MissingAsvCommit(RuntimeError):
    """An asv result names a commit this checkout does not contain."""


def _git_root(root: Path) -> str:
    return str(Path(root).resolve())


def _is_work_tree(root: Path) -> bool:
    cache = getattr(_is_work_tree, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_is_work_tree, "_cache", cache)
    key = _git_root(root)
    if key not in cache:
        try:
            out = subprocess.check_output(
                ["git", "-C", key, "rev-parse", "--is-inside-work-tree"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            out = ""
        cache[key] = out.strip() == "true"
    return cache[key]


def _commit_in_checkout(root: Path, commit: str) -> bool:
    cache = getattr(_commit_in_checkout, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_commit_in_checkout, "_cache", cache)
    key = (_git_root(root), commit)
    if key not in cache:
        try:
            subprocess.check_output(
                ["git", "-C", key[0], "cat-file", "-e", f"{commit}^{{commit}}"],
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            cache[key] = False
        else:
            cache[key] = True
    return cache[key]


def _require_asv_commit(root: Path, commit: str) -> None:
    """Stop when a real checkout is missing the commit that recorded a track.

    A fixture with no git repository still falls back to the working tree.
    A shallow checkout must not: the working-tree pin is a different run.
    """
    if not commit or not _is_work_tree(root) or _commit_in_checkout(root, commit):
        return
    raise MissingAsvCommit(
        f"asv result {commit} is not in this checkout, so its template pin cannot be read"
    )


def _template_at_commit(root: Path, commit: str, template_id: str) -> dict | None:
    """Template as it was when an asv result was recorded.

    The working tree can move the code pin forward. The result's commit still
    names the pin that produced the track.
    """
    cache = getattr(_template_at_commit, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_template_at_commit, "_cache", cache)
    key = (str(root.resolve()), commit, template_id)
    if key in cache:
        return cache[key]
    _require_asv_commit(root, commit)
    loaded = None
    if commit and template_id:
        rel = f"templates/{template_id}.toml"
        try:
            text = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{commit}:{rel}"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            text = ""
        if text:
            try:
                loaded = tomllib.loads(text)
            except tomllib.TOMLDecodeError:
                loaded = None
    cache[key] = loaded
    return loaded


_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _module_pin(root: Path, commit: str, module_name: str) -> str:
    """The client build a benchmark module checked, as the module stood at ``commit``.

    A module that refuses any other build names it in one top-level
    ``*_COMMIT`` string. Its tracks ran on that build, whatever commit the
    template it reads names.
    """
    cache = getattr(_module_pin, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_module_pin, "_cache", cache)
    key = (str(root.resolve()), commit, module_name)
    if key in cache:
        return cache[key]
    text = ""
    if commit and module_name:
        try:
            text = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{commit}:benchmarks/{module_name}.py"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            text = ""
    try:
        body = ast.parse(text).body if text else []
    except SyntaxError:
        body = []
    named = {
        node.value.value
        for node in body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.endswith("_COMMIT")
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and _FULL_COMMIT.fullmatch(node.value.value)
    }
    pin = named.pop() if len(named) == 1 else ""
    cache[key] = pin
    return pin


def _geometry_key(start) -> str:
    """One token per geometry so campaign names and asv indexes line up.

    ``cluster_0000.con``, ``pos_0000``, and asv start ``0`` are the same index.
    """
    if isinstance(start, tuple):
        start = start[-1] if start else ""
    text = str(start if start is not None else "").strip()
    if text.endswith(".con"):
        text = text[: -len(".con")]
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    match = re.fullmatch(r"[A-Za-z][A-Za-z0-9]*[_-](\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _image_count(problem: dict | None, template: dict | None, fields: dict[str, str]) -> int | None:
    """Moving images for this problem. Every code uses the same divisor."""
    band = (problem or {}).get("band") or {}
    raw = band.get("interior_images")
    if raw in (None, ""):
        neb = (template or {}).get("neb") or {}
        raw = neb.get("moving_images")
    if raw in (None, ""):
        raw = fields.get("images")
    if raw in (None, ""):
        return None
    images = int(raw)
    return images if images > 0 else None


_PER_IMAGE_TRACK = 'float(record["force_calls"]) / images'


def _commit_divides_band(root: Path, commit: str) -> bool:
    """True when that commit's ASE benchmark stored a per-image quotient."""
    cache = getattr(_commit_divides_band, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_commit_divides_band, "_cache", cache)
    key = (str(root.resolve()), commit)
    if key in cache:
        return cache[key]
    divides = False
    if commit:
        _require_asv_commit(root, commit)
        try:
            text = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{commit}:benchmarks/bench_ase_sella.py"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            text = ""
        divides = _PER_IMAGE_TRACK in text
    cache[key] = divides
    return divides


def _cap_of(problem: dict | None, template: dict | None, fields: dict[str, str]) -> int | None:
    """The PAR-2 cap of a row from any source.

    The problem's ``max_force_calls`` comes first, so a campaign file and an
    asv track of the same starts get the same score. A template limit is
    the cap only when the problem states none, and the result file's
    ``cap`` field only when neither does.
    """
    if problem is not None:
        stated = (problem.get("convergence") or {}).get("max_force_calls")
        if stated not in (None, ""):
            return int(stated)
    if template is not None:
        if template.get("failure_cap") not in (None, ""):
            return int(template["failure_cap"])
        stop = template.get("stop") or {}
        raw = stop.get("max_force_calls", stop.get("max_steps"))
        if raw not in (None, ""):
            return int(raw)
    raw_field = _float_field(fields, "cap")
    if raw_field is not None:
        return int(raw_field)
    return None


def _force_tol(problem: dict | None, template: dict | None) -> float | None:
    convergence = (problem or {}).get("convergence") or {}
    for key in ("force_norm", "gradient_norm"):
        if key in convergence:
            return float(convergence[key])
    stop = (template or {}).get("stop") or {}
    for key in ("force_norm", "gradient_norm", "fmax"):
        if key in stop:
            return float(stop[key])
    return None


def _record_from_raw(raw: dict, cap: int | None, force_tol: float | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    calls = raw.get("force_calls", raw.get("raw", raw.get("per_search")))
    if calls is None:
        return None
    try:
        calls = float(calls)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(calls):
        return None
    failure = raw.get("failure")
    if "success" in raw and raw.get("success") is not None:
        success = bool(raw["success"])
    elif "failed" in raw:
        success = not bool(raw["failed"])
    elif "reason" in raw:
        success = str(raw.get("reason")) in {"0", "0.0"}
        failure = None
    else:
        return None
    if cap is not None and calls >= cap:
        success = False
    record: dict = {"force_calls": calls, "success": success, "start": raw.get("start") or raw.get("geometry")}
    # Kept so a later preference can see whether this start applied the index rule.
    if raw.get("hessian_index") is not None:
        record["hessian_index"] = raw["hessian_index"]
    if raw.get("hessian_calls") is not None:
        try:
            record["hessian_calls"] = float(raw["hessian_calls"])
        except (TypeError, ValueError):
            pass
    if raw.get("per_image") is not None:
        try:
            record["per_image"] = float(raw["per_image"])
        except (TypeError, ValueError):
            pass
    if raw.get("call_unit"):
        record["call_unit"] = str(raw["call_unit"])
    if success:
        return record
    if failure in FAILURES:
        record["failure"] = failure
        return record
    reasons = raw.get("reasons")
    if isinstance(reasons, list) and reasons:
        record["reasons"] = [str(item) for item in reasons]
        return record
    record["failure_inferred"] = True
    if raw.get("error"):
        record["failure"] = "crash"
    elif cap is not None and calls >= cap:
        record["failure"] = "cap"
    elif force_tol is not None and raw.get("force_norm") not in (None, ""):
        try:
            above = float(raw["force_norm"]) > force_tol
        except (TypeError, ValueError):
            above = True
        record["failure"] = "divergence" if above else "wrong_stationary"
    return record


def _records_from_repetitions(text: str, fields: dict[str, str]) -> list[dict]:
    rows = parse_repetitions(text)
    if not rows:
        return []
    failed = int(float(fields.get("nfailed") or 0))
    if failed:
        return []
    return [
        {"force_calls": float(row["per_search"]), "success": True, "failure": None, "start": str(row["repetition"])}
        for row in rows
    ]


def _load_group_records(
    group: dict[str, Path],
    cap: int | None,
    force_tol: float | None,
    images: int | None = None,
) -> tuple[list[dict], str]:
    if "starts_jsonl" in group:
        collected = []
        for raw in _json_lines(group["starts_jsonl"]):
            record = _record_from_raw(normalize_start(raw, images), cap, force_tol)
            if record is not None:
                collected.append(record)
        if collected:
            return collected, "starts.jsonl"
    if "starts" in group:
        collected = []
        for raw in _json_lines(group["starts"]):
            record = _record_from_raw(normalize_start(raw, images), cap, force_tol)
            if record is not None:
                collected.append(record)
        if collected:
            return collected, "starts"
    if "v2" in group:
        document = json.loads(group["v2"].read_text(encoding="utf-8"))
        runs = document.get("runs") or []
        collected = []
        for run in runs:
            for raw in run.get("starts") or []:
                record = _record_from_raw(normalize_start(raw, images), cap, force_tol)
                if record is not None:
                    collected.append(record)
        if collected:
            return collected, "v2.json"
    if "dat" in group:
        text = group["dat"].read_text(encoding="utf-8")
        fields, _duplicates = parse_lenient(text)
        collected = _records_from_repetitions(text, fields)
        if collected:
            return collected, "repetitions"
    return [], ""


def _campaign_groups(directory: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    if not directory.is_dir():
        return groups
    for path in sorted(directory.iterdir()):
        name = path.name
        if not path.is_file():
            continue
        if name.endswith(".starts.jsonl"):
            if path.stat().st_size == 0:
                continue
            stem, kind = name[: -len(".starts.jsonl")], "starts_jsonl"
        elif name.endswith(".v2.json"):
            stem, kind = name[: -len(".v2.json")], "v2"
        elif name.endswith(".starts"):
            stem, kind = name[: -len(".starts")], "starts"
        elif name.endswith(".dat"):
            stem, kind = name[: -len(".dat")], "dat"
        else:
            continue
        groups.setdefault(stem, {})[kind] = path
    return groups


def _comparable(fields: dict[str, str], raw_rows: list[dict]) -> bool:
    counter = str(fields.get("counter") or "").lower()
    reducer = str(fields.get("reducer") or "").lower()
    if counter == "client":
        return False
    if counter == "harness" or reducer.startswith("harness"):
        return True
    for raw in raw_rows:
        if str(raw.get("counter") or "").lower() == "harness":
            return True
    return False


def _first_raw(group: dict[str, Path]) -> dict:
    for kind in ("starts_jsonl", "starts"):
        path = group.get(kind)
        if path is None:
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _blank_row(**values) -> dict:
    row = {
        "problem": "",
        "template": "",
        "code": "",
        "code_version": "",
        "commit": "",
        "tuning": "",
        "entry": "",
        "source": "campaign",
        "path": "",
        "asv_commit": "",
        "counter": "",
        "comparable": False,
        "short": False,
        "short_reason": "",
        "origin": "harness",
        "method": "",
        "kind": "",
        "label": "",
        "family": "",
        "family_rank": None,
        "hessian_calls": None,
        "fd_calls": None,
        "published": None,
        "rank_cost": None,
        "registry_match": False,
        "unit": "",
        "call_unit": "",
        "mean_per_image": None,
        "median_per_image": None,
        "n": 0,
        "full_n": None,
        "par2_mean": None,
        "success_rate": None,
        "median": None,
        "median_low": None,
        "median_high": None,
        "mean_calls": None,
        "median_calls": None,
        "nfailed": None,
        "min_calls": None,
        "max_calls": None,
        "failures": {},
        "failure_note": "",
        "cap": None,
        "scored": False,
        "note": "",
    }
    row.update(values)
    return row


def _score_records(records: list[dict], cap: int | None) -> dict:
    summary = score_v2(records, cap)
    calls = [float(record["force_calls"]) for record in records]
    ordered = sorted(calls)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median_calls = ordered[mid]
    else:
        median_calls = (ordered[mid - 1] + ordered[mid]) / 2
    summary["mean_calls"] = sum(calls) / len(calls)
    summary["median_calls"] = median_calls
    summary["min_calls"] = ordered[0]
    summary["max_calls"] = ordered[-1]
    summary["call_unit"] = "total"
    per_image = [record["per_image"] for record in records if record.get("per_image") is not None]
    if len(per_image) == len(records):
        per_ordered = sorted(per_image)
        per_mid = len(per_ordered) // 2
        if len(per_ordered) % 2:
            median_per = per_ordered[per_mid]
        else:
            median_per = (per_ordered[per_mid - 1] + per_ordered[per_mid]) / 2
        summary["mean_per_image"] = sum(per_image) / len(per_image)
        summary["median_per_image"] = median_per
    return summary


def _apply_summary(row: dict, summary: dict, records: list[dict]) -> None:
    row["scored"] = True
    row["n"] = int(summary["n"])
    row["par2_mean"] = summary["par2_mean"]
    row["success_rate"] = summary["success_rate"]
    row["median"] = summary["median"]
    row["median_low"] = summary["median_low"]
    row["median_high"] = summary["median_high"]
    row["mean_calls"] = summary["mean_calls"]
    row["median_calls"] = summary["median_calls"]
    row["min_calls"] = summary["min_calls"]
    row["max_calls"] = summary["max_calls"]
    row["nfailed"] = summary["nfailed"]
    row["call_unit"] = summary.get("call_unit") or "total"
    row["mean_per_image"] = summary.get("mean_per_image")
    row["median_per_image"] = summary.get("median_per_image")
    row["failures"] = {name: summary[f"n_{name}"] for name in FAILURES}
    inferred: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("failure_inferred"):
            label = failure_of(record, row.get("cap"))
            if label is not None:
                inferred[label] += 1
    if inferred:
        row["failures_inferred"] = {name: inferred[name] for name in FAILURES if inferred[name]}
    hessians = [
        float(record["hessian_calls"])
        for record in records
        if record.get("hessian_calls") is not None
    ]
    row["hessian_calls"] = (sum(hessians) / len(hessians)) if hessians else 0.0
    row["_profile"] = [(_geometry_key(record.get("start")), profile_cost(record)) for record in records]


def _source_parts(row: dict) -> set[str]:
    return {part.strip() for part in str(row.get("source") or "").split(",") if part.strip()}


def _mark_short(row: dict, full_n: int | None) -> None:
    row["full_n"] = full_n
    n = int(row.get("n") or 0)
    if row.get("origin") == "archive":
        row["short"] = False
        row["short_reason"] = ""
        return
    if row.get("_declared_short"):
        row["short"] = True
        row["short_reason"] = row["_declared_short"]
        return
    if _source_parts(row) == {"asv"} and n <= 1:
        row["short"] = True
        if full_n:
            row["short_reason"] = f"scored {n} of {full_n} starts"
        else:
            row["short_reason"] = "one start"
        return
    if full_n is not None and n < full_n:
        row["short"] = True
        row["short_reason"] = f"scored {n} of {full_n} starts"
        return
    row["short"] = False
    row["short_reason"] = ""


def _dedup_key(row: dict) -> tuple:
    commit = str(row.get("commit") or "")
    identity = commit[:12] if commit else str(row.get("path") or "")
    return (
        row.get("problem"),
        row.get("template"),
        row.get("code_version") or "",
        identity,
        row.get("n"),
        row.get("par2_mean"),
        row.get("mean_calls"),
        row.get("median_calls"),
        row.get("min_calls"),
        row.get("max_calls"),
        row.get("nfailed"),
    )


def _join_sources(rows: list[dict]) -> str:
    found: list[str] = []
    for row in rows:
        for part in str(row.get("source") or "").split(","):
            part = part.strip()
            if part and part not in found:
                found.append(part)
    found.sort(key=lambda name: (name != "campaign", name))
    return ", ".join(found)


def _collapse_run(group: list[dict]) -> dict:
    ordered = sorted(
        group,
        key=lambda row: ("campaign" not in _source_parts(row), str(row.get("path") or "")),
    )
    base = dict(ordered[0])
    paths: list[str] = []
    asv_commits: list[str] = []
    for row in ordered:
        path = str(row.get("path") or "")
        if path and path not in paths:
            paths.append(path)
        asv_commit = str(row.get("asv_commit") or "")
        if asv_commit and asv_commit not in asv_commits:
            asv_commits.append(asv_commit)
    base["source"] = _join_sources(ordered)
    base["path"] = "; ".join(paths)
    if asv_commits:
        base["asv_commit"] = asv_commits[0]
    tracked = [row for row in ordered if "asv" in _source_parts(row) and row.get("scored")]
    if tracked:
        base["failures"] = dict(tracked[0].get("failures") or {})
        if tracked[0].get("failures_inferred"):
            base["failures_inferred"] = dict(tracked[0]["failures_inferred"])
        else:
            base.pop("failures_inferred", None)
    # Same score means the asv track already applied the success rule.
    # The merged row is that track, so the missing-index mark does not stick.
    if any("asv" in _source_parts(row) for row in ordered):
        base.pop("_missing_hessian_index", None)
    return base


def _dedup(rows: list[dict]) -> list[dict]:
    """One row per run. A campaign file and its asv track share that row."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_dedup_key(row)].append(row)
    merged = []
    for group in groups.values():
        merged.append(group[0] if len(group) == 1 else _collapse_run(group))
    return merged


def _campaign_rows(root: Path, problems: dict[str, dict], templates: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    unparsed: list[dict] = []
    for stem, group in _campaign_groups(root / "results" / "campaign").items():
        path = group.get("dat") or group.get("v2") or group.get("starts_jsonl") or group.get("starts")
        fields: dict[str, str] = {}
        parse_note = ""
        if "dat" in group:
            try:
                fields, _duplicates = parse_lenient(group["dat"].read_text(encoding="utf-8"))
            except DatError as exc:
                unparsed.append(
                    _blank_row(
                        entry=stem,
                        path=str(group["dat"].relative_to(root)),
                        note=str(exc),
                        source="campaign",
                    )
                )
                continue
        probe = _first_raw(group)
        template_id = fields.get("template") or str(probe.get("template") or "")
        template = templates.get(template_id)
        problem_id = (
            fields.get("problem")
            or str(probe.get("problem") or "")
            or (template or {}).get("problem")
            or campaign_problem(stem)
            or ""
        )
        problem = problems.get(problem_id)
        cap = _cap_of(problem, template, fields)
        force_tol = _force_tol(problem, template)
        images = _image_count(problem, template, fields)
        try:
            records, record_source = _load_group_records(group, cap, force_tol, images)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            unparsed.append(
                _blank_row(
                    problem=problem_id,
                    template=template_id,
                    entry=stem,
                    path=str(path.relative_to(root)) if path else "",
                    note=str(exc),
                    code=fields.get("code", ""),
                    code_version=fields.get("code_version", ""),
                    commit=fields.get("commit", ""),
                )
            )
            continue
        counter = fields.get("counter") or str(probe.get("counter") or "")
        if not counter and str(fields.get("reducer") or "").lower().startswith("harness"):
            counter = "harness"
        tuning = (
            fields.get("tuning")
            or fields.get("profile")
            or str(probe.get("tuning") or "")
            or str((template or {}).get("tuning") or (template or {}).get("profile") or "")
        )
        row = _blank_row(
            problem=problem_id,
            template=template_id or str((template or {}).get("id") or ""),
            code=fields.get("code") or str(probe.get("code") or (template or {}).get("code") or ""),
            code_version=fields.get("code_version") or str(probe.get("code_version") or (template or {}).get("code_version") or ""),
            commit=fields.get("commit") or str(probe.get("commit") or (template or {}).get("commit") or ""),
            tuning=tuning,
            entry=stem,
            source="campaign",
            path=str(path.relative_to(root)) if path else "",
            counter=counter,
            comparable=_comparable(fields, [probe] if probe else []),
            cap=cap,
            note=parse_note or record_source,
            method=method_name(template, fields),
            kind=str((template or {}).get("kind") or fields.get("kind") or ""),
            origin="harness",
        )
        named = str(fields.get("potential") or "").strip()
        declared = ""
        if str(fields.get("ranked") or "").lower() == "false":
            declared = "result file sets ranked false"
        elif str(fields.get("partial") or "").lower() == "true" or str(fields.get("short") or "").lower() == "true":
            declared = "result file sets short true"
        if declared:
            if named:
                declared = f"{declared}; potential {named}"
            row["_declared_short"] = declared
        if not records:
            row["note"] = "no per-start record"
            unparsed.append(row)
            continue
        if _success_needs_hessian(problem) and any(
            record.get("success") and "hessian_index" not in record for record in records
        ):
            # A success with no index did not apply the rule. A cap or a crash
            # has no stationary point, so a missing index there is not this mark.
            row["_missing_hessian_index"] = True
        try:
            summary = _score_records(records, cap)
        except (ValueError, RuntimeError) as exc:
            row["note"] = str(exc)
            unparsed.append(row)
            continue
        _apply_summary(row, summary, records)
        rows.append(row)
    return rows, unparsed


def _benchmark_index() -> dict[str, dict]:
    import importlib

    index: dict[str, dict] = {}
    modules = (
        "benchmarks.bench_lj38",
        "benchmarks.bench_eon_develop",
        "benchmarks.bench_eon_svn",
        "benchmarks.bench_eon_lane",
        "benchmarks.bench_eon_client",
        "benchmarks.bench_rgmin",
        "benchmarks.bench_ase_sella",
        "benchmarks.bench_rgsaddle",
        "benchmarks.bench_geodesic",
        "benchmarks.bench_airss",
        "benchmarks.bench_softsaddle",
        "benchmarks.bench_lammpsneb",
        "benchmarks.bench_noise",
        "benchmarks.bench_n2p2",
        "benchmarks.bench_gmin",
        "benchmarks.bench_vcsqnm",
        "benchmarks.bench_pele",
        "benchmarks.bench_optking",
        "benchmarks.quadratic",
    )
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        short = module_name.split(".")[-1]
        for name, value in vars(module).items():
            if not isinstance(value, type):
                continue
            spec = getattr(value, "spec", None)
            template = getattr(value, "template_id", None) or getattr(value, "template_name", None)
            problem = getattr(value, "problem_id", None) or getattr(value, "problem_name", None)
            cap = getattr(value, "cap", None)
            if isinstance(spec, dict):
                template = template or spec.get("template_id") or spec.get("template")
                problem = problem or spec.get("problem_id") or spec.get("problem")
                if cap is None:
                    cap = spec.get("cap")
            index[f"{short}.{name}"] = {
                "template": str(template) if template else "",
                "problem": str(problem) if problem else "",
                "cap": int(cap) if isinstance(cap, int) or (isinstance(cap, float) and float(cap).is_integer()) else None,
            }
    return index


def _asv_rows(root: Path, problems: dict[str, dict], templates: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict]]:
    results_dir = root / "results" / "asv"
    if not results_dir.is_dir():
        return [], [], []
    from asv.results import iter_results

    from .asv_score import track_rows

    index = _benchmark_index()
    rows: list[dict] = []
    unparsed: list[dict] = []
    unmapped: list[dict] = []
    for result in iter_results(str(results_dir)):
        keys = [key for key in result.get_all_result_keys() if str(key).endswith("track_force_calls")]
        seen: set[str] = set()
        for key in keys:
            head = str(key)[: -len(".track_force_calls")] if str(key).endswith(".track_force_calls") else str(key)
            parts = head.split(".")
            class_name = parts[-1]
            if class_name in seen:
                continue
            seen.add(class_name)
            module_name = parts[-2] if len(parts) >= 2 else ""
            described = index.get(f"{module_name}.{class_name}") or index.get(class_name) or {}
            template_id = described.get("template") or ""
            template = templates.get(template_id)
            pinned = _template_at_commit(root, str(result.commit_hash), template_id) if template_id else None
            source_template = pinned or template
            problem_id = (
                described.get("problem")
                or (source_template or {}).get("problem")
                or (template or {}).get("problem")
                or ""
            )
            problem = problems.get(problem_id)
            class_cap = described.get("cap")
            cap = _cap_of(problem, source_template or template, {} if class_cap is None else {"cap": str(class_cap)})
            pin_note = "asv track_force_calls"
            if template_id and pinned is None:
                pin_note = "asv track_force_calls; code pin is the working-tree template"
            code_commit = str((source_template or {}).get("commit") or "")
            built = _module_pin(root, str(result.commit_hash), module_name)
            if built and not same_commit(code_commit, built):
                code_commit = built
                pin_note = "asv track_force_calls; code pin is the client build the benchmark module checks"
            try:
                tracked = track_rows(result, class_name)
            except SystemExit as exc:
                unparsed.append(
                    _blank_row(
                        problem=problem_id,
                        template=template_id,
                        entry=class_name,
                        source="asv",
                        path=f"results/asv/{result.commit_hash}",
                        asv_commit=str(result.commit_hash),
                        note=str(exc),
                    )
                )
                continue
            reducer = str((source_template or template or {}).get("reducer") or "")
            images = _image_count(problem, source_template or template, {})
            per_image_track = reducer == "per_image" and _commit_divides_band(root, str(result.commit_hash))
            records = []
            try:
                for item in tracked:
                    raw = {
                        "force_calls": item["force_calls"],
                        "success": item["success"],
                        "failure": item["failure"],
                    }
                    raw = normalize_start(raw, images, per_image_track=per_image_track)
                    record = _record_from_raw(raw, cap, _force_tol(problem, source_template or template))
                    if record is not None:
                        record["start"] = tuple(item.get("params") or ())
                        records.append(record)
            except ValueError as exc:
                unparsed.append(
                    _blank_row(
                        problem=problem_id,
                        template=template_id,
                        entry=f"{module_name}.{class_name}",
                        source="asv",
                        path=f"results/asv/{result.commit_hash}",
                        asv_commit=str(result.commit_hash),
                        note=str(exc),
                    )
                )
                continue
            client_counter = module_name == "bench_lj38"
            row = _blank_row(
                problem=problem_id,
                template=template_id,
                code=str((source_template or {}).get("code") or ""),
                code_version=str((source_template or {}).get("code_version") or ""),
                commit=code_commit,
                tuning=str((source_template or {}).get("tuning") or (source_template or {}).get("profile") or ""),
                entry=f"{module_name}.{class_name}",
                source="asv",
                path=f"results/asv/{result.commit_hash}",
                asv_commit=str(result.commit_hash),
                counter="client" if client_counter else "harness",
                comparable=bool(template_id) and bool(problem_id) and not client_counter,
                cap=cap,
                note=pin_note,
                method=method_name(source_template or template, {}),
                kind=str((source_template or template or {}).get("kind") or ""),
                origin="harness",
            )
            if not problem_id or not template_id:
                row["note"] = f"benchmark {row['entry']} is not a problem template"
                row["n"] = len(records)
                unmapped.append(row)
                continue
            if not records:
                row["note"] = "asv track has no starts"
                unparsed.append(row)
                continue
            try:
                summary = _score_records(records, cap)
            except (ValueError, RuntimeError) as exc:
                row["note"] = str(exc)
                unparsed.append(row)
                continue
            _apply_summary(row, summary, records)
            rows.append(row)
    return rows, unparsed, unmapped


def _public(row: dict) -> dict:
    hidden = {"comparable"}
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_") and key not in hidden
    }


def _rank_key(row: dict) -> tuple:
    score = row.get("rank_cost")
    if score is None:
        score = row.get("par2_mean")
    if score is None:
        score = row.get("published")
    return (
        score is None,
        float(score) if score is not None else 0.0,
        str(row.get("label") or ""),
        str(row.get("template") or ""),
        str(row.get("path") or ""),
    )


def _over_budget(row: dict, problem: dict | None) -> bool:
    """An archived run above the problem cap with no failure for it ran under a larger cap."""
    most = row.get("_archived_max")
    cap = _cap_of(problem, None, {})
    if row.get("origin") != "archive" or most is None or cap is None:
        return False
    return row.get("unit") == harness_unit(problem) and most > cap


def _archive_par2(row: dict, problem: dict | None) -> tuple[float | None, str]:
    """PAR-2 of an archived mean, or why the harness failure rule cannot score it.

    The half-even rule each problem file states leaves failures out of the
    archived mean, so the finished runs sum to that mean times their count.
    The reason is empty when the mean leaves nothing out.
    """
    published = row.get("published")
    nfailed = row.get("nfailed")
    if published is None:
        return None, ""
    if nfailed is None:
        return None, "the entry records no failure count"
    cap = _cap_of(problem, None, {})
    if cap is None:
        return None, "" if nfailed == 0 else "the problem states no cap"
    harness = harness_unit(problem)
    if row.get("unit") != harness:
        return None, "" if nfailed == 0 else f"it counts {row.get('unit')}, and the cap counts {harness}"
    if _over_budget(row, problem):
        return None, f"its archived maximum, {_format_number(row['_archived_max'])}, is above the cap of {cap}"
    if nfailed == 0:
        return float(published), ""
    n = int(row.get("n") or 0)
    if n < nfailed:
        return None, f"the entry records {nfailed} failures and no count of {nfailed} or more runs"
    return ((n - nfailed) * float(published) + nfailed * PAR_FACTOR * cap) / n, ""


def _annotate_row(row: dict, problem: dict | None, registry: list[dict]) -> None:
    code = display_code(str(row.get("code") or ""), registry)
    version = version_text(str(row.get("code_version") or ""), str(row.get("commit") or ""))
    row["label"] = row_label(code, version, str(row.get("method") or ""))
    suite = str((problem or {}).get("suite") or str(row.get("problem") or "").split("/")[0])
    row["family"] = family_of(str(row.get("method") or ""), suite=suite, kind=str(row.get("kind") or ""))
    if row.get("hessian_calls") is None:
        row["hessian_calls"] = 0.0
    row["fd_calls"] = finite_difference_calls(float(row["hessian_calls"]), coordinate_count(problem))
    row["unit"] = row_unit(problem, str(row.get("origin") or "harness"))
    if row.get("origin") == "archive":
        par2, row["failure_note"] = _archive_par2(row, problem)
        row["_over_budget"] = _over_budget(row, problem)
        row["par2_mean"] = par2
        row["rank_cost"] = par2 if par2 is not None else row.get("published")
    else:
        row["published"] = None
        row["rank_cost"] = row.get("par2_mean")


def _method_key(row: dict) -> tuple:
    return (
        row.get("problem"),
        str(row.get("code") or "").casefold(),
        version_text(str(row.get("code_version") or ""), str(row.get("commit") or "")).casefold(),
        str(row.get("method") or "").casefold(),
        row.get("origin") or "harness",
    )


def _success_needs_hessian(problem: dict | None) -> bool:
    """The v2 success rule counts a Hessian index.

    The sentence is the rule. ``Hessian index`` is the explicit form.
    ``The index is vibrational`` and ``index 1 at eigenvalue`` are the
    same count without that pair of words.
    """
    if not problem:
        return False
    text = str(((problem.get("scoring") or {}).get("v2") or {}).get("success") or "").casefold()
    if "hessian index" in text:
        return True
    if "index" not in text:
        return False
    return "vibrational" in text or "eigenvalue" in text


def _prefer_row(group: list[dict]) -> dict:
    """One row for a code, version, and method.

    On a problem whose success rule counts a Hessian index, a campaign
    row whose successes stored no index loses to a row that recorded
    the check. Cost then picks among the rows that remain, so a checked
    row that is already cheaper stays the ranked one.
    """

    def sort_key(row: dict) -> tuple:
        cost = row.get("rank_cost")
        return (
            1 if row.get("_missing_hessian_index") else 0,
            cost is None,
            float(cost) if cost is not None else 0.0,
            -int(row.get("n") or 0),
            0 if row.get("registry_match") else 1,
            str(row.get("path") or ""),
        )

    return sorted(group, key=sort_key)[0]


def _collapse_methods(rows: list[dict]) -> list[dict]:
    """One ranked row per code, version, and method. Short rows stay as they are."""
    other: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("short") or is_adjusted(row) or not row.get("comparable"):
            other.append(row)
            continue
        groups[_method_key(row)].append(row)
    return other + [_prefer_row(group) for group in groups.values()]


def _archive_rows(root: Path, problems: dict[str, dict], registry: list[dict], full_n: dict[str, int | None]) -> list[dict]:
    from .check import load_oracle
    from .dat import normalize

    path = root / "oracle" / "entries.jsonl"
    if not path.is_file():
        return []
    corrections = load_corrections(root)
    differences = load_differences(root)
    counts = load_counts(root)
    rows: list[dict] = []
    for item in load_oracle(path):
        ident = f"{item['suite']}/{item['problem']}"
        problem = problems.get(ident)
        if problem is None:
            continue
        fields, _renamed = normalize(item["fields"])
        fields = corrected(str(item.get("path") or ""), fields, corrections)
        if _held(fields):
            continue
        primary = _primary(problem)
        published = _float(fields.get(primary, "")) if primary else None
        method = str(fields.get("algorithm") or fields.get("preset") or fields.get("method") or item.get("entry") or "")
        nfailed_raw = fields.get("nfailed")
        try:
            nfailed = int(float(nfailed_raw)) if nfailed_raw not in (None, "") else None
        except ValueError:
            nfailed = None
        own = entry_count(item, primary, counts) if primary else None
        stated = own if own is not None else full_n.get(ident)
        if fields.get("force_calls"):
            mean_calls = _float(fields.get("force_calls", ""))
        else:
            mean_calls = published if primary in FORCE_CALL_FIELDS else None
        row = _blank_row(
            problem=ident,
            entry=str(item.get("entry") or ""),
            code=str(fields.get("code") or ""),
            code_version=str(fields.get("code_version") or ""),
            source="archive",
            path=str(item.get("path") or ""),
            counter="archive",
            comparable=True,
            origin="archive",
            method=method,
            scored=True,
            published=published,
            mean_calls=mean_calls,
            median_calls=_float(fields.get("force_calls_median", "")),
            nfailed=nfailed,
            n=stated or 0,
            hessian_calls=0.0,
            note="archived score",
            _archived_max=_float(fields["force_calls_max"]) if fields.get("force_calls_max") else None,
        )
        difference = difference_note(row["path"], fields, differences)
        if difference:
            row["difference"] = difference
        rows.append(row)
    return rows


def _unit_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Rows of each other unit, ranked only among themselves."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("unit") or "")].append(row)
    out = []
    for name in sorted(grouped):
        group = sorted(grouped[name], key=_rank_key)
        _assign_family_ranks(group)
        out.append((name, group))
    return out


def board_rows(problem: dict) -> list[dict]:
    """Ranked rows, then the rows of each other unit."""
    rows = list(problem.get("ranked") or [])
    for group in problem.get("other_units") or []:
        rows.extend(group.get("rows") or [])
    return rows


def _assign_family_ranks(ranked: list[dict]) -> None:
    seen: dict[str, int] = defaultdict(int)
    for row in ranked:
        family = str(row.get("family") or "")
        seen[family] += 1
        row["family_rank"] = seen[family]


def rank_violations(document: dict) -> list[str]:
    """Ranked rows that the checker must still reject."""
    errors: list[str] = []
    for problem in (document.get("problems") or {}).values():
        ident = str(problem.get("id") or "")
        for row in board_rows(problem):
            label = str(row.get("label") or row.get("template") or row.get("entry") or "")
            if row.get("short"):
                errors.append(f"short row {ident} {label}")
            if is_adjusted(row):
                errors.append(f"adjusted row {ident} {label}")
            reason = str(row.get("short_reason") or "")
            if "ranked false" in reason:
                errors.append(f"unranked row {ident} {label}")
    return errors


def build_board(root: Path) -> dict:
    """Score every committed campaign file, asv track, and archived row under ``root``."""
    root = Path(root)
    problems = load_all(root / "problems") if (root / "problems").is_dir() else {}
    templates = {item["id"]: item for item in list_templates(root)} if (root / "templates").is_dir() else {}
    registry = _registry(root)
    campaign, unparsed = _campaign_rows(root, problems, templates)
    asv_rows, asv_unparsed, unmapped = _asv_rows(root, problems, templates)
    unparsed.extend(asv_unparsed)
    scored = [_public(row) for row in campaign + asv_rows]
    rows = _dedup(campaign + asv_rows)
    full_n: dict[str, int | None] = {}
    for problem_id, problem in problems.items():
        full_n[problem_id] = stated_full_n(root, problem)
    rows.extend(_archive_rows(root, problems, registry, full_n))
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem"]].append(row)
    for problem_id, group in by_problem.items():
        stated = full_n.get(problem_id)
        problem = problems.get(problem_id) or {}
        if problem_id and problem_id not in problems and stated is None:
            for row in group:
                row["full_n"] = None
                if row.get("_declared_short"):
                    row["short"] = True
                    row["short_reason"] = row["_declared_short"]
                elif _source_parts(row) == {"asv"} and int(row.get("n") or 0) <= 1:
                    row["short"] = True
                    row["short_reason"] = "one start"
                else:
                    row["short"] = True
                    row["short_reason"] = "no problem file states the full set"
        else:
            for row in group:
                _mark_short(row, stated)
        for row in group:
            row["registry_match"] = _registry_match(row, registry)
            _annotate_row(row, problem, registry)
        by_problem[problem_id] = _collapse_methods(group)

    problems_out: dict[str, dict] = {}
    short: list[dict] = []
    withheld: list[dict] = []
    not_comparable: list[dict] = []
    over_budget: list[dict] = []
    for problem_id in sorted(by_problem):
        group = by_problem[problem_id]
        ranked = []
        held_short = []
        held_adjusted = []
        held_client = []
        held_budget = []
        for row in group:
            if row.get("short"):
                held_short.append(row)
                short.append(row)
            elif not row.get("comparable"):
                held_client.append(row)
                not_comparable.append(row)
            elif is_adjusted(row):
                held_adjusted.append(row)
                withheld.append(row)
            elif row.get("_over_budget"):
                held_budget.append(row)
                over_budget.append(row)
            else:
                ranked.append(row)
        problem = problems.get(problem_id) or {}
        unit = rank_unit(problem, ranked)
        other_units = _unit_groups([row for row in ranked if row.get("unit") != unit])
        ranked = [row for row in ranked if row.get("unit") == unit]
        ranked.sort(key=_rank_key)
        _assign_family_ranks(ranked)
        problems_out[problem_id] = {
            "id": problem_id,
            "title": problem.get("title", problem_id),
            "full_n": full_n.get(problem_id),
            "unit": unit,
            "unit_source": unit_source(problem),
            "ranked": [_public(row) for row in ranked],
            "other_units": [
                {"unit": name, "rows": [_public(row) for row in rows]} for name, rows in other_units
            ],
            "short": [_public(row) for row in held_short],
            "withheld": [_public(row) for row in held_adjusted],
            "not_comparable": [_public(row) for row in held_client],
            "over_budget": [_public(row) for row in sorted(held_budget, key=_rank_key)],
            "_ranked_rows": ranked,
        }
    return {
        "schema": BOARD_SCHEMA,
        "problems": problems_out,
        "short": [_public(row) for row in short],
        "withheld": [_public(row) for row in withheld],
        "unparsed": [_public(row) for row in unparsed],
        "unmapped": [_public(row) for row in unmapped],
        "not_comparable": [_public(row) for row in not_comparable],
        "over_budget": [_public(row) for row in over_budget],
        "ranked_count": sum(len(board_rows(item)) for item in problems_out.values()),
        "_scored_rows": scored,
    }


def _series_label(row: dict, used: set[str]) -> str:
    label = str(row.get("label") or row.get("template") or row.get("entry") or "row")
    if label in used:
        label = f"{label} {row.get('path') or row.get('asv_commit') or len(used)}"
    used.add(label)
    return label


def _keyed_costs(row: dict) -> list[tuple[str, float]]:
    ordered = []
    seen = set()
    for start, cost in row.get("_profile") or []:
        key = str(start)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((key, float(cost)))
    return ordered


def _aligned_costs(rows: list[dict]) -> tuple[dict[str, list[float]] | None, str]:
    """Start-aligned costs for a Dolan-Moré profile, or the reason there is none."""
    usable = [row for row in rows if row.get("_profile")]
    if not usable:
        return None, "The ranked rows have no per-start record."
    used: set[str] = set()
    series = []
    for row in usable:
        keyed = _keyed_costs(row)
        if keyed:
            series.append((_series_label(row, used), keyed))
    if not series:
        return None, "The ranked rows have no per-start record."
    key_sets = [{key for key, _cost in keyed} for _label, keyed in series]
    common = set.intersection(*key_sets)
    shortest = min(len(keys) for keys in key_sets)
    if common and len(common) >= shortest:
        order = [key for key, _cost in series[0][1] if key in common]
        costs: dict[str, list[float]] = {}
        for label, keyed in series:
            found = dict(keyed)
            costs[label] = [found[key] for key in order]
        return costs, ""
    if len({len(keyed) for _label, keyed in series}) == 1:
        return {label: [cost for _key, cost in keyed] for label, keyed in series}, ""
    return None, "The ranked rows do not share one start set."


def _profile_for(rows: list[dict]) -> dict | None:
    costs, _reason = _aligned_costs(rows)
    if costs is None:
        return None
    try:
        return performance_profile(costs)
    except ValueError:
        return None


def profile_gap(rows: list[dict]) -> str:
    """Why a ranked problem has no Dolan-Moré profile. Empty when one exists."""
    costs, reason = _aligned_costs(rows)
    if costs is None:
        return f"No performance profile. {reason}"
    try:
        performance_profile(costs)
    except ValueError as exc:
        return f"No performance profile. {_profile_reason(exc)}"
    return ""


def _profile_reason(exc: ValueError) -> str:
    text = str(exc)
    if text.startswith("problem ") and text.endswith(" has no finite cost"):
        return "One start has no finite cost."
    if not text:
        return "The profile could not be built."
    if text[-1] not in ".!?":
        text += "."
    return text[0].upper() + text[1:]


def _plain_tick(value: float, _position: object) -> str:
    """Tick text without mathtext, so the SVG does not outline every glyph."""
    if value == 0:
        return "0"
    return f"{value:.4g}"


def _raster_thumbnails(figure, stem: Path) -> list[str]:
    """Write a PNG fallback and a compressed WebP at the figure's pixel size."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image = Image.open(buffer)
    image.load()
    if image.mode == "RGBA":
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.getchannel("A"))
        image = flat
    elif image.mode != "RGB":
        image = image.convert("RGB")
    png_path = stem.with_suffix(".png")
    webp_path = stem.with_suffix(".webp")
    image.save(png_path, format="PNG", optimize=True)
    image.save(webp_path, format="WEBP", quality=72, method=6)
    return [str(png_path), str(webp_path)]


def _save_figure(figure, directory: Path, stem: str) -> list[str]:
    svg_path = directory / f"{stem}.svg"
    figure.savefig(svg_path, format="svg")
    written = [str(svg_path)]
    written.extend(_raster_thumbnails(figure, directory / stem))
    return written


def _write_plots(problem_id: str, ranked: list[dict], profile: dict | None, directory: Path) -> list[str]:
    import os

    import matplotlib
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    matplotlib.use("Agg", force=False)
    matplotlib.rcParams["svg.hashsalt"] = "optbench"
    # Glyphs stay as <text>. Path-cloned markers were hundreds of <use> nodes per plot.
    matplotlib.rcParams["svg.fonttype"] = "none"
    # Same bytes on the next render. The date is metadata, not a score.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    slug = _slug(problem_id)
    written = []
    plain = FuncFormatter(_plain_tick)
    if profile is not None:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for label, values in profile["rho"].items():
            axis.step(profile["taus"], values, where="post", label=label)
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_formatter(plain)
        axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axis.set_xlabel("tau")
        axis.set_ylabel("fraction of starts")
        axis.set_ylim(0.0, 1.05)
        axis.set_title(f"Dolan-More profile: {problem_id}")
        if len(profile["rho"]) <= 12:
            axis.legend(fontsize=7, loc="lower right")
        figure.tight_layout()
        written.extend(_save_figure(figure, directory, f"{slug}-profile"))
        plt.close(figure)
    plotted = [row for row in ranked if row.get("par2_mean") is not None and row.get("success_rate") is not None]
    if plotted:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        rates = [float(row["success_rate"]) for row in plotted]
        costs = [float(row["par2_mean"]) for row in plotted]
        axis.scatter(rates, costs, s=16, marker="o", linewidths=0)
        for index, (rate, cost) in enumerate(zip(rates, costs), start=1):
            axis.annotate(str(index), (rate, cost), fontsize=7, textcoords="offset points", xytext=(3, 3))
        axis.xaxis.set_major_formatter(plain)
        axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axis.set_xlabel("success rate")
        axis.set_ylabel("PAR-2 mean" if any(row["par2_mean"] is not None for row in ranked) else "median calls")
        axis.set_title(f"Cost and success: {problem_id}")
        figure.tight_layout()
        written.extend(_save_figure(figure, directory, f"{slug}-cost"))
        plt.close(figure)
    return written


def _org_html(fragment: str) -> str:
    lines = ["#+begin_export rst", ".. raw:: html", ""]
    for line in fragment.splitlines():
        lines.append("   " + line)
    lines.append("#+end_export")
    lines.append("")
    return "\n".join(lines)


def _format_number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _failure_text(failures: dict, inferred: dict | None = None) -> str:
    parts = []
    for name in FAILURES:
        part = f"{name} {failures.get(name, 0)}"
        if (inferred or {}).get(name):
            part += f" ({inferred[name]} inferred)"
        parts.append(part)
    return ", ".join(parts)


def failure_cell(row: dict) -> str:
    """Failure classes of a harness row. An archived entry records only a count."""
    if row.get("origin") != "archive":
        return _failure_text(row.get("failures") or {}, row.get("failures_inferred"))
    nfailed = row.get("nfailed")
    if nfailed is None:
        return "not recorded"
    if nfailed == 0:
        return "0 failed"
    return f"{nfailed} failed, class not recorded"


def _cell(value) -> str:
    return html.escape(str(value))


def _show_per_image(problem: dict) -> bool:
    rows = list(problem.get("ranked") or [])
    return any(row.get("mean_per_image") is not None for row in rows)


def _headers(problem: dict) -> list[str]:
    headers = [
        "Row",
        "Family",
        "Family rank",
        "Unit",
        "PAR-2",
        "Published",
        "Median calls",
        "Mean calls",
    ]
    if _show_per_image(problem):
        headers.append("Per-image mean")
    headers.extend(["Hessians", "FD calls", "Failures", "Runs"])
    return headers


def ranked_cells(row: dict, per_image: bool) -> list[tuple[str, str]]:
    """Visible overall-table cells, in column order.

    The site check compares these strings with the rendered table.
    """
    cells = [
        ("label", str(row.get("label") or "")),
        ("family", family_title(str(row.get("family") or ""))),
        ("family_rank", "" if row.get("family_rank") is None else str(row.get("family_rank"))),
        ("unit", str(row.get("unit") or "")),
        ("par2_mean", _format_number(row.get("par2_mean"))),
        ("published", _format_number(row.get("published"))),
        ("median_calls", _format_number(row.get("median_calls"))),
        ("mean_calls", _format_number(row.get("mean_calls"))),
    ]
    if per_image:
        cells.append(("mean_per_image", _format_number(row.get("mean_per_image"))))
    cells.append(("hessian_calls", _format_number(row.get("hessian_calls"))))
    cells.append(("fd_calls", _format_number(row.get("fd_calls"))))
    cells.append(("failures", failure_cell(row)))
    cells.append(("n", str(row.get("n"))))
    return cells


def family_cells(row: dict) -> list[tuple[str, str]]:
    """Cells on the within-family table."""
    return [
        ("family_rank", "" if row.get("family_rank") is None else str(row.get("family_rank"))),
        ("label", str(row.get("label") or "")),
        ("rank_cost", _format_number(row.get("rank_cost"))),
        ("unit", str(row.get("unit") or "")),
        ("hessian_calls", _format_number(row.get("hessian_calls"))),
        ("fd_calls", _format_number(row.get("fd_calls"))),
        ("n", str(row.get("n"))),
    ]


def _row_cells(row: dict, per_image: bool) -> str:
    headers = _headers({"ranked": [{"mean_per_image": 0}] if per_image else []})
    return "".join(
        f'<td data-label="{_cell(header)}" data-field="{field}">{_cell(value)}</td>'
        for header, (field, value) in zip(headers, ranked_cells(row, per_image))
    )


def _off_harness(row: dict) -> bool:
    return str(row.get("counter") or "unset") in {"", "unset", "client"}


def _joined_names(rows: list[dict]) -> str:
    names: list[str] = []
    for row in rows:
        name = str(row.get("code") or "").strip()
        if not name:
            template = str(row.get("template") or row.get("entry") or "")
            name = template.split("/")[0] if template else "unknown"
        if name not in names:
            names.append(name)
    if not names:
        return "These codes"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _aside_rows(problem: dict) -> list[dict]:
    return list(problem.get("not_comparable") or problem.get("client_counter") or [])


def gap_paragraph(problem: dict, unparsed: list[dict] | None = None) -> str:
    """Why a code is off the ranked table for this problem."""
    bits: list[str] = []
    missing = [row for row in _aside_rows(problem) if _off_harness(row)]
    other = [row for row in _aside_rows(problem) if not _off_harness(row)]
    suite = str(problem.get("id") or "").split("/")[0]
    kind = "band rows" if suite in {"neb", "double-ended-saddle-search"} else "rows"
    if missing:
        bits.append(f"{_joined_names(missing)} {kind} are not yet on the harness counter")
    if other:
        bits.append(f"{_joined_names(other)} {kind} are off the ranked table")
    if problem.get("short") or problem.get("withheld"):
        bits.append("A run short of the full set, or a run that is not the stock method, stays off the ranked table")
    if unparsed:
        bits.append("Files with no per-start harness record stay off the ranked table")
    if not bits and not problem.get("ranked"):
        bits.append("No run is committed for this problem")
    if not bits:
        return ""
    return "".join(f'<p class="bench-gaps">{_cell(bit)}.</p>' for bit in bits)


def _reason_text(row: dict) -> str:
    if row.get("short"):
        return str(row.get("short_reason") or "short of the full set")
    if _off_harness(row):
        return "not yet on the harness counter"
    return f"counter {row.get('counter') or 'unset'}"


def _safe_name(row: dict) -> str:
    """A visible name that does not carry a template id."""
    label = str(row.get("label") or "").strip()
    if label:
        return label
    return str(row.get("code") or "").strip()


def _plot_html(prefix: str, name: str, label: str, display: str) -> str:
    """Lazy WebP thumbnail, PNG fallback, click-through to the full SVG."""
    stem = name[:-4] if name.endswith(".svg") else name
    svg_name = name if name.endswith(".svg") else f"{stem}.svg"
    alt = f"{label} for {display}"
    caption = f"{label}. Select the plot for the full SVG."
    return (
        '<figure class="bench-plot">'
        f'<a href="{_cell(f"{prefix}/{svg_name}")}">'
        "<picture>"
        f'<source srcset="{_cell(f"{prefix}/{stem}.webp")}" type="image/webp">'
        f'<img src="{_cell(f"{prefix}/{stem}.png")}" alt="{_cell(alt)}" '
        'loading="lazy" decoding="async">'
        "</picture>"
        "</a>"
        f"<figcaption>{_cell(caption)}</figcaption>"
        "</figure>"
    )


def plot_names_for(problem: dict) -> set[str]:
    """SVG files ``write_board`` emits for this problem."""
    slug = _slug(str(problem.get("id") or ""))
    names: set[str] = set()
    if problem.get("ranked"):
        names.add(f"{slug}-cost.svg")
    if problem.get("profile"):
        names.add(f"{slug}-profile.svg")
    return names


def _problem_order(problem: dict) -> tuple:
    ident = str(problem.get("id") or "")
    suite = ident.split("/", 1)[0]
    try:
        index = SUITE_ORDER.index(suite)
    except ValueError:
        index = len(SUITE_ORDER)
    return (index, ident.lower())


def summary_codes(document: dict, registry: list[dict]) -> list[str]:
    """Codes that have a full ranked run, in registry order, then any extras."""
    present: set[str] = set()
    for problem in (document.get("problems") or {}).values():
        for row in problem.get("ranked") or []:
            code = str(row.get("code") or "").strip()
            if code:
                present.add(code)
    ordered: list[str] = []
    for pin in registry:
        name = str(pin.get("name") or "").strip()
        if name and name in present and name not in ordered:
            ordered.append(name)
    for name in sorted(present - set(ordered)):
        ordered.append(name)
    return ordered


def best_ranked_by_code(ranked: list[dict]) -> dict[str, dict]:
    """First ranked row of each code. The list is already best-PAR-2 first."""
    best: dict[str, dict] = {}
    for row in ranked:
        code = str(row.get("code") or "").strip()
        if code and code not in best:
            best[code] = row
    return best


def _problem_anchor(problem: dict, templates: bool) -> str:
    ident = str(problem.get("id") or "")
    name = ident.split("/", 1)[1] if "/" in ident else ident
    return f"{name}-templates" if templates else name


def _board_problem_link(problem: dict, templates: bool = False) -> str:
    ident = str(problem.get("id") or "")
    suite, _, name = ident.partition("/")
    anchor = _problem_anchor(problem, templates)
    if not suite or not name:
        return f"#{anchor}"
    return f"generated/suites/{suite}/{name}.html"


def summary_table(problems: list[dict], codes: list[str], link) -> str:
    """One cell per code: the best full harness run, codes side by side.

    ``link(problem, templates=False)`` returns the href for that problem.
    The template link opens the full per-template table.
    """
    lines = [
        '<div class="bench-scroll">',
        '<table data-board-table="summary">',
        "<caption>Best full harness run of each code. Lower PAR-2 is better. "
        "The full template table is on the problem page.</caption>",
        "<thead><tr><th>Problem</th>"
        + "".join(f"<th>{_cell(code)}</th>" for code in codes)
        + "</tr></thead><tbody>",
    ]
    for problem in sorted(problems, key=_problem_order):
        best = best_ranked_by_code(list(problem.get("ranked") or []))
        lines.append(f'<tr data-problem="{_cell(problem["id"])}">')
        title = str(problem.get("title") or problem["id"])
        lines.append(
            f'<th scope="row" data-label="Problem"><a href="{_cell(link(problem))}">{_cell(title)}</a></th>'
        )
        detail = link(problem, templates=True)
        for code in codes:
            row = best.get(code)
            if row is None:
                lines.append(f'<td data-code="{_cell(code)}" data-label="{_cell(code)}">—</td>')
                continue
            template = str(row.get("template") or row.get("entry") or "")
            par2 = dict(ranked_cells(row, False))["par2_mean"]
            lines.append(
                f'<td data-code="{_cell(code)}" data-label="{_cell(code)}" data-template="{_cell(template)}" data-par2="{_cell(par2)}">'
                f'<span class="bench-score">{_cell(par2)}</span>'
                f'<a href="{_cell(detail)}">{_cell(template)}</a>'
                "</td>"
            )
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _winner_html(row: dict) -> str:
    label = str(row.get("label") or "").strip()
    if not label:
        return ""
    score = ""
    kind = ""
    for key, name in (("par2_mean", "PAR-2"), ("published", "Published"), ("rank_cost", "Cost")):
        if row.get(key) is not None:
            score = _format_number(row.get(key))
            kind = name
            break
    parts = ['<div class="winner">', f'<p class="winner-name">{_cell(label)}</p>']
    if score:
        parts.append(f'<p class="winner-score"><small>{_cell(kind)}</small>{_cell(score)}</p>')
    parts.append("</div>")
    return "".join(parts)


def headline_table(problem: dict) -> str:
    """One ranked row: the lowest cost."""
    per_image = _show_per_image(problem)
    headers = _headers(problem)
    lines = [
        '<div class="bench-scroll">',
        f'<table data-board-table="headline" data-problem="{_cell(problem["id"])}">',
        "<thead><tr>" + "".join(f"<th>{_cell(header)}</th>" for header in headers) + "</tr></thead><tbody>",
    ]
    ranked = list(problem.get("ranked") or [])
    if ranked:
        lines.append(f"<tr>{_row_cells(ranked[0], per_image)}</tr>")
    else:
        lines.append(f'<tr><td colspan="{len(headers)}">No ranked row.</td></tr>')
    lines.append("</tbody></table></div>")
    winner = _winner_html(ranked[0]) if ranked else ""
    body = "\n".join(lines)
    return f"{winner}\n{body}" if winner else body


def render_problem_section(
    problem: dict,
    plots: set[str],
    *,
    asset_prefix: str = "_static/board",
    preamble: str = "",
    leaderboard: str = "",
    metric_line: str = "",
    success_note: str = "",
    section_id: str | None = None,
    unparsed: list[dict] | None = None,
    heading: str = "h2",
    show_heading: bool = True,
) -> str:
    """Stat cards, the leaderboard, charts, then the family and overall tables."""
    return _table(
        problem,
        plots,
        asset_prefix=asset_prefix,
        preamble=preamble,
        leaderboard=leaderboard,
        metric_line=metric_line,
        success_note=success_note,
        section_id=section_id,
        unparsed=unparsed,
        heading=heading,
        show_heading=show_heading,
    )


def _caption(per_image: bool) -> str:
    caption = (
        "One row is one code, one version, and one method. A lower cost is better. "
        "PAR-2 counts a finished run as the calls it used and a failed run as twice the cap. "
        "An archived row shows the average it published, "
        "and its PAR-2 applies that rule to the failures the entry records. "
        "Hessians is the mean analytic Hessian count. "
        "FD calls is that count times one force call per coordinate."
    )
    if per_image:
        caption += " The per-image mean divides the total by the number of moving images."
    return caption


def _family_table(problem: dict, family_id: str, rows: list[dict]) -> str:
    headers = ["Rank", "Row", "Cost", "Unit", "Hessians", "FD calls", "Runs"]
    lines = [
        '<div class="bench-scroll">',
        (
            f'<table class="bench-sortable" data-board-table="family" data-family="{_cell(family_id)}" '
            f'data-problem="{_cell(problem["id"])}">'
        ),
        f"<caption>{_cell(family_title(family_id))}. {_cell(_caption(False))}</caption>",
        "<thead><tr>" + "".join(f"<th>{_cell(header)}</th>" for header in headers) + "</tr></thead><tbody>",
    ]
    for row in rows:
        cells = "".join(
            f'<td data-label="{_cell(header)}" data-field="{field}">{_cell(value)}</td>'
            for header, (field, value) in zip(headers, family_cells(row))
        )
        lines.append(f"<tr>{cells}</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def unit_per_image(group: dict) -> bool:
    return any(row.get("mean_per_image") is not None for row in group.get("rows") or [])


def failure_paragraph(problem: dict) -> str:
    """Archived rows the harness failure rule cannot score, each with the reason."""
    rows = [row for row in board_rows(problem) if row.get("failure_note")]
    if not rows:
        return ""
    items = "".join("<li>" + _cell(f"{row.get('label')} ({row['failure_note']})") + "</li>" for row in rows)
    return (
        '<p class="bench-gaps" data-board-note="failures">'
        "These archived rows rank on the value each one published, which leaves out any failure. "
        "The failure rule has no score for them, for the reason given with each.</p>"
        f"<ul>{items}</ul>"
    )


INFERRED_RULE = (
    "A failure class marked inferred comes from a start file that records no class. "
    "A start with an error is crash, a start at or above the cap is cap, "
    "and any other failure is divergence unless its force norm is at or below the stop, "
    "which makes it wrong_stationary."
)


def inferred_rows(problem: dict) -> list[dict]:
    """Shown rows with a failure class the board inferred."""
    return [row for row in board_rows(problem) if row.get("failures_inferred")]


def inferred_paragraph(problem: dict) -> str:
    rows = inferred_rows(problem)
    if not rows:
        return ""
    labels = "; ".join(str(row.get("label") or "") for row in rows)
    return (
        f'<p class="bench-gaps" data-board-note="inferred">{_cell(INFERRED_RULE)} '
        f"Rows with an inferred class: {_cell(labels)}.</p>"
    )


def difference_rows(problem: dict) -> list[dict]:
    """Shown archived rows that ``oracle/differences.toml`` labels."""
    return [row for row in board_rows(problem) + list(problem.get("over_budget") or []) if row.get("difference")]


def difference_text(row: dict) -> str:
    return f"{row.get('label')} ({str(row['difference']).rstrip('.')})"


def difference_paragraph(problem: dict) -> str:
    rows = difference_rows(problem)
    if not rows:
        return ""
    parts = "; ".join(difference_text(row) for row in rows)
    return (
        '<p class="bench-gaps" data-board-note="differences">'
        "These archived rows differ from the table of J. Chem. Theory Comput. 10, 5476 (2014) that prints them, "
        f"and each ranks on its archived values: {_cell(parts)}.</p>"
    )


def budget_per_image(problem: dict) -> bool:
    return any(row.get("mean_per_image") is not None for row in problem.get("over_budget") or [])


def _budget_table(problem: dict) -> str:
    rows = list(problem.get("over_budget") or [])
    if not rows:
        return ""
    per_image = budget_per_image(problem)
    headers = _headers({"ranked": rows})
    reasons = "; ".join(f"{row.get('label')} ({row.get('failure_note')})" for row in rows)
    lines = [
        "<h3>Run under a different budget</h3>",
        "<p>These archived rows record a run above this problem's cap and no failure for it, "
        "so they ran under a larger cap. This table keeps them apart from the ranking above: "
        f"{_cell(reasons)}.</p>",
        '<div class="bench-scroll">',
        f'<table class="bench-sortable" data-board-table="budget" data-problem="{_cell(problem["id"])}">',
        f"<caption>{_cell(_caption(per_image))}</caption>",
        "<thead><tr>" + "".join(f"<th>{_cell(header)}</th>" for header in headers) + "</tr></thead><tbody>",
    ]
    for row in rows:
        lines.append(f"<tr>{_row_cells(row, per_image)}</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _unit_table(problem: dict, group: dict) -> str:
    unit = str(group.get("unit") or "")
    rows = list(group.get("rows") or [])
    per_image = unit_per_image(group)
    headers = _headers({"ranked": rows})
    lines = [
        f"<h3>Ranked in {_cell(unit)}</h3>",
        f"<p>These rows count {_cell(unit)}. The tables above count {_cell(problem.get('unit') or '')}, "
        "so these rows are ranked only among themselves.</p>",
    ]
    source = str(problem.get("unit_source") or "")
    if source:
        lines.append(f'<p class="bench-aside">{_cell(source)}</p>')
    lines.extend(
        [
            '<div class="bench-scroll">',
            (
                f'<table class="bench-sortable" data-board-table="unit" data-unit="{_cell(unit)}" '
                f'data-problem="{_cell(problem["id"])}">'
            ),
            f"<caption>{_cell(_caption(per_image))}</caption>",
            "<thead><tr>" + "".join(f"<th>{_cell(header)}</th>" for header in headers) + "</tr></thead><tbody>",
        ]
    )
    for row in rows:
        lines.append(f"<tr>{_row_cells(row, per_image)}</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _table(
    problem: dict,
    plots: set[str],
    *,
    asset_prefix: str = "_static/board",
    preamble: str = "",
    leaderboard: str = "",
    metric_line: str = "",
    success_note: str = "",
    section_id: str | None = None,
    unparsed: list[dict] | None = None,
    heading: str = "h2",
    show_heading: bool = True,
) -> str:
    per_image = _show_per_image(problem)
    headers = _headers(problem)
    ident = section_id or _slug(problem["id"])
    ranked = list(problem.get("ranked") or [])
    lines = [
        f'<section class="bench" id="{_cell(ident)}" data-problem="{_cell(problem["id"])}">',
    ]
    if show_heading:
        lines.append(f"<{heading}>{_cell(problem['title'])}</{heading}>")
    if preamble:
        lines.append(preamble)
    if metric_line:
        lines.append(metric_line)
    slug = _slug(problem["id"])
    prefix = asset_prefix.rstrip("/")
    from .site_charts import cost_distribution_svg, profile_svg
    from .site_pages import profile_worth_showing

    display = str(problem.get("title") or problem.get("id") or "")
    chart_bits: list[str] = []
    drawn = profile_svg(problem)
    if drawn:
        chart_bits.append(drawn)
    costs = cost_distribution_svg(ranked, display)
    if costs:
        chart_bits.append(costs)
    exports: list[str] = []
    file_links: list[str] = []
    for kind, label in (("profile", "Performance profile"), ("cost", "Cost and success")):
        name = f"{slug}-{kind}.svg"
        if kind == "profile" and not profile_worth_showing(problem):
            if problem.get("profile") or name in plots:
                chart_bits.append(
                    "<p class=\"bench-note\">A performance profile needs at least two curves. "
                    "This problem has fewer, so that plot is omitted.</p>"
                )
            continue
        if name in plots:
            exports.append(_plot_html(prefix, name, label, display))
            file_links.append(f'<a href="{_cell(f"{prefix}/{name}")}">{_cell(label)} SVG</a>')
    if problem.get("profile_gap") and f"{slug}-profile.svg" not in plots:
        chart_bits.append(f'<p class="bench-gaps">{_cell(problem["profile_gap"])}</p>')
    if file_links:
        chart_bits.append('<p class="chart-files">' + " ".join(file_links) + "</p>")
    if exports:
        chart_bits.append('<div class="bench-export">' + "".join(exports) + "</div>")
    from .paper_embed import problem_charts

    root = Path(__file__).resolve().parents[1]
    paper_prefix = prefix[: prefix.rfind("board")] + "paper" if "board" in prefix else prefix
    paper = problem_charts(root, problem["id"], paper_prefix)
    if paper or chart_bits:
        lines.append("<h2>Charts</h2>")
        if paper:
            lines.append(paper)
        if chart_bits:
            lines.append('<div class="lead-charts">')
            lines.extend(chart_bits)
            lines.append("</div>")
    if leaderboard:
        lines.append(leaderboard)
    lines.append(
        "<p class=\"bench-note\">The tables below list every ranked row, by method family and overall.</p>"
    )
    lead = gap_paragraph(problem, unparsed)
    if lead:
        lines.append(lead)
    unscored = failure_paragraph(problem)
    if unscored:
        lines.append(unscored)
    differs = difference_paragraph(problem)
    if differs:
        lines.append(differs)
    guessed = inferred_paragraph(problem)
    if guessed:
        lines.append(guessed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ranked:
        grouped[str(row.get("family") or "")].append(row)
    order = [family_id for family_id, _title in FAMILIES if family_id in grouped]
    order.extend(family_id for family_id in grouped if family_id not in order)
    for family_id in order:
        lines.append(f"<h3>{_cell(family_title(family_id))}</h3>")
        lines.append(_family_table(problem, family_id, grouped[family_id]))
    lines.append("<h3>Overall</h3>")
    lines.append('<div class="bench-scroll">')
    lines.append(
        f'<table class="bench-sortable" data-board-table="ranked" data-problem="{_cell(problem["id"])}">'
    )
    lines.append(f"<caption>{_cell(_caption(per_image))}</caption>")
    lines.append("<thead><tr>" + "".join(f"<th>{_cell(header)}</th>" for header in headers) + "</tr></thead><tbody>")
    for row in ranked:
        lines.append(f"<tr>{_row_cells(row, per_image)}</tr>")
    if not ranked:
        lines.append(f'<tr><td colspan="{len(headers)}">No ranked row.</td></tr>')
    lines.append("</tbody></table></div>")
    if problem.get("other_units"):
        lines.append("<p>A row that counts another unit is ranked in its own table after the overall table.</p>")
    for group in problem.get("other_units") or []:
        lines.append(_unit_table(problem, group))
    budget = _budget_table(problem)
    if budget:
        lines.append(budget)
    held = _aside_rows(problem)
    if held:
        lines.append("<h3>Off the ranked table</h3><ul>")
        for row in held:
            lines.append(
                "<li>"
                f"{_cell(_safe_name(row))} n={row['n']}, {_cell(_reason_text(row))}."
                "</li>"
            )
        lines.append("</ul>")
    pending = list(unparsed or [])
    if pending:
        lines.append("<h3>Not scored</h3><ul>")
        for row in pending:
            label = _safe_name(row) or str(row.get("entry") or row.get("path") or "")
            lines.append(f"<li>{_cell(label)}: {_cell(row.get('note') or '')}</li>")
        lines.append("</ul>")
    if success_note:
        lines.append(success_note)
    lines.append("</section>")
    return "\n".join(lines)


def _drop_unwritten(directory: Path, keep_names: set[str]) -> None:
    """Remove plot files left by an earlier board that this run did not write."""
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_file() and path.name not in keep_names:
            path.unlink()


def _strip_private(document: dict) -> dict:
    public_problems = {}
    for problem_id, problem in document["problems"].items():
        public = {key: value for key, value in problem.items() if not key.startswith("_")}
        if "not_comparable" in public:
            public["client_counter"] = public.pop("not_comparable")
        public_problems[problem_id] = public
    return {
        "schema": document["schema"],
        "ranked_count": document["ranked_count"],
        "problems": public_problems,
        "short": document["short"],
        "withheld": document["withheld"],
        "unparsed": document["unparsed"],
        "unmapped": document["unmapped"],
        "client_counter": document["not_comparable"],
    }


def write_board(root: Path, output: Path, *, site: bool | None = None) -> dict:
    """Write the leaderboard JSON, the plots, and the Board page include.

    With ``site`` true, the plots are also copied to the site and the include
    is the Board page under ``root/docs``. Otherwise every file goes under
    ``output``. ``site`` left as None is true only when ``output`` is the
    checkout's ``results/board``.
    """
    root = Path(root)
    output = Path(output)
    if site is None:
        site = output.resolve() == (root / "results" / "board").resolve()
    document = build_board(root)
    output.mkdir(parents=True, exist_ok=True)
    plot_dir = output / "plots"
    site_plots = root / "docs" / "source" / "_static" / "board"
    sections = []
    written_names: set[str] = set()
    for problem_id, problem in document["problems"].items():
        ranked_rows = problem.get("_ranked_rows") or []
        profile = _profile_for(ranked_rows)
        if profile is not None:
            problem["profile"] = {
                "taus": profile["taus"],
                "rho": profile["rho"],
                "solved": profile["solved"],
                "area_log10": profile["area_log10"],
            }
        written = _write_plots(problem_id, ranked_rows, profile, plot_dir) if ranked_rows else []
        if ranked_rows and profile is None:
            gap = profile_gap(ranked_rows)
            if gap:
                problem["profile_gap"] = gap
        for path in written:
            written_names.add(Path(path).name)
    ordered = sorted(document["problems"].values(), key=_problem_order)
    if ordered:
        from .site_pages import human_title

        loaded = load_all(root / "problems") if (root / "problems").is_dir() else {}
        by_slug: dict[str, list] = {}
        for item in loaded.values():
            by_slug.setdefault(str(item.get("problem") or ""), []).append(item)
        sections.append('<section class="bench" id="headlines">')
        for problem in ordered:
            toml = loaded.get(problem["id"])
            heading = human_title(toml, by_slug) if toml else str(problem.get("title") or problem["id"])
            sections.append(f"<h2>{_cell(heading)}</h2>")
            sections.append(headline_table(problem))
        sections.append("</section>")
    _drop_unwritten(plot_dir, written_names)
    if document["unparsed"]:
        items = [
            f"<li>{_cell(row.get('path') or row.get('entry'))}: {_cell(row.get('note') or '')}</li>"
            for row in document["unparsed"]
        ]
        sections.append("<section id=\"unparsed\"><h2>Not scored</h2><ul>" + "".join(items) + "</ul></section>")
    if document["unmapped"]:
        items = [
            f"<li>{_cell(row.get('entry') or row.get('path'))}: {_cell(row.get('note') or '')}</li>"
            for row in document["unmapped"]
        ]
        sections.append(
            "<section id=\"unmapped\"><h2>Not a problem template</h2><ul>" + "".join(items) + "</ul></section>"
        )
    if site and site_plots != plot_dir and plot_dir.is_dir():
        site_plots.mkdir(parents=True, exist_ok=True)
        for path in plot_dir.iterdir():
            if path.is_file():
                shutil.copy(path, site_plots / path.name)
        _drop_unwritten(site_plots, written_names)
    public = _strip_private(document)
    (output / "index.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    problem_dir = output / "problems"
    problem_dir.mkdir(parents=True, exist_ok=True)
    for problem_id, problem in public["problems"].items():
        (problem_dir / f"{_slug(problem_id)}.json").write_text(
            json.dumps(problem, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    include = root / "docs" / "include" / "v2-board.org" if site else output / "v2-board.org"
    include.parent.mkdir(parents=True, exist_ok=True)
    html = "\n".join(sections) if sections else "<p>No committed results.</p>"
    include.write_text(_org_html(html), encoding="utf-8")
    return public
