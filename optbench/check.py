"""Check archived benchmark.dat rows against problem contracts."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from .dat import normalize
from .problems import load_all


def load_oracle(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = lineno
        rows.append(row)
    return rows


def missing_keys(problem: dict, fields: dict[str, str]) -> list[str]:
    canonical, _renamed = normalize(fields)
    required = problem["metrics"]["required"]
    return [key for key in required if key not in canonical]


def archive_gaps(problems: dict[str, dict], rows: list[dict]) -> list[str]:
    """Ways an archived file departs from its problem contract."""
    gaps: list[str] = []
    for row in rows:
        ident = f"{row['suite']}/{row['problem']}"
        problem = problems.get(ident)
        if problem is None:
            continue
        _canonical, renamed = normalize(row["fields"])
        for old, new in renamed:
            gaps.append(f"spelling {row['path']} {old} -> {new}")
        if problem.get("contract") == "stated":
            for key in missing_keys(problem, row["fields"]):
                gaps.append(f"missing {row['path']} {key}")
        for item in row.get("duplicates") or []:
            gaps.append(
                f"duplicate {row['path']} {item['key']} kept {item['kept']!r} also {item['also']!r}"
            )
    return gaps


def _sets_tuned_true(value: object) -> bool:
    """True when a template document sets the boolean ``tuned`` key."""
    if isinstance(value, dict):
        if value.get("tuned") is True:
            return True
        return any(_sets_tuned_true(item) for item in value.values())
    if isinstance(value, list):
        return any(_sets_tuned_true(item) for item in value)
    return False


def _sets_tuning_tuned(value: object) -> bool:
    """True when a template document sets ``tuning = "tuned"``."""
    if isinstance(value, dict):
        if value.get("tuning") == "tuned":
            return True
        return any(_sets_tuning_tuned(item) for item in value.values())
    if isinstance(value, list):
        return any(_sets_tuning_tuned(item) for item in value)
    return False


def tuned_template_errors(root: Path) -> list[str]:
    """Templates named ``*-tuned``, setting ``tuned = true``, or ``tuning = "tuned"``.

    The name is the file stem or the template id. The label is rejected on
    every id, not only on a ranked row.
    """
    directory = Path(root) / "templates"
    if not directory.is_dir():
        return []
    errors: list[str] = []
    for path in sorted(directory.rglob("*.toml")):
        rel = path.relative_to(root)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            if path.stem.endswith("-tuned"):
                errors.append(f"tuned template {rel} unreadable {exc}")
            continue
        ident = str(data.get("id") or "")
        if (
            path.stem.endswith("-tuned")
            or ident.endswith("-tuned")
            or _sets_tuned_true(data)
            or _sets_tuning_tuned(data)
        ):
            errors.append(f"tuned template {rel}")
    return errors


def tuned_row_errors(document: dict) -> list[str]:
    """Board rows whose template is tuned, on every list."""
    errors: list[str] = []

    def one(row: object, where: str) -> None:
        if not isinstance(row, dict):
            return
        template = str(row.get("template") or "")
        if template.endswith("-tuned") or row.get("tuned") is True or row.get("tuning") == "tuned":
            errors.append(f"tuned row {where} {template or where}")

    for problem_id, problem in (document.get("problems") or {}).items():
        if not isinstance(problem, dict):
            continue
        for bucket in ("ranked", "partial", "history", "not_comparable"):
            for row in problem.get(bucket) or []:
                one(row, f"{problem_id} {bucket}")
    for bucket in ("partial", "history", "not_comparable", "unparsed", "unmapped"):
        for row in document.get(bucket) or []:
            one(row, bucket)
    return errors


def check(problems_dir: Path, oracle_path: Path) -> int:
    """Return 0 when archive gaps match oracle/archive_gaps.txt."""
    problems = load_all(problems_dir)
    rows = load_oracle(oracle_path)
    counts: dict[str, int] = {}
    structural: list[str] = []
    for row in rows:
        ident = f"{row['suite']}/{row['problem']}"
        counts[ident] = counts.get(ident, 0) + 1
        problem = problems.get(ident)
        if problem is None:
            structural.append(f"{ident} has an oracle row and no problem file")
            continue
        if problem.get("contract") == "partial":
            missing = missing_keys(problem, row["fields"])
            for key in missing:
                structural.append(f"partial {row['path']} missing {key}")
    for ident, problem in problems.items():
        expected = problem.get("entry_count")
        got = counts.get(ident, 0)
        if expected is not None and expected != got:
            structural.append(f"{ident} entry_count is {expected}, oracle has {got}")
    gaps = archive_gaps(problems, rows)
    known_path = oracle_path.parent / "archive_gaps.txt"
    known = known_path.read_text(encoding="utf-8").splitlines() if known_path.exists() else []
    print(f"entries {len(rows)}")
    print(f"problems {len(problems)}")
    print(f"archive_gaps {len(gaps)}")
    if structural:
        print(f"structural {len(structural)}")
        for line in structural:
            print(line)
        return 1
    if gaps != known:
        print("archive gaps do not match oracle/archive_gaps.txt")
        for line in gaps:
            if line not in known:
                print(f"+ {line}")
        for line in known:
            if line not in gaps:
                print(f"- {line}")
        return 1
    root = oracle_path.resolve().parent.parent
    tuned = tuned_template_errors(root)
    if tuned:
        print(f"tuned {len(tuned)}")
        for line in tuned:
            print(line)
        return 1
    from .site_check import check_site

    site_errors = check_site(root)
    if site_errors:
        print(f"site {len(site_errors)}")
        for line in site_errors:
            print(line)
        return 1
    print("site tables match the board")
    return 0
