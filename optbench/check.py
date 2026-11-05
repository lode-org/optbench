"""Check archived benchmark.dat rows against problem contracts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .corrections import correction_errors
from .counts import count_errors
from .dat import normalize
from .differences import difference_errors
from .problems import load_all
from .units import unit_errors

# Public text, published board files, and path tokens. Schema fields inside
# campaign records stay as written.
_BANNED_PHRASES = (
    "not rerun",
    "not run",
    "paper row",
    "paper baseline",
    "10% band",
    "wilson interval",
    "inside band",
    "outside band",
)
_BANNED_TOKENS = {"smoke", "legacy", "current", "comparable", "partial"}
_TEXT_SUFFIXES = {".md", ".org", ".rst", ".css", ".yml", ".yaml", ".html"}
_SKIP_DIRS = {".git", ".venv", ".asv", "env", "__pycache__", "build", "doctrees", ".pixi", "node_modules"}


def _banned_hits(text: str) -> list[str]:
    lower = text.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for phrase in _BANNED_PHRASES:
        if phrase in lower and phrase not in seen:
            hits.append(phrase)
            seen.add(phrase)
    for token in re.findall(r"[a-z0-9]+", lower):
        if token.startswith("repro") or token in _BANNED_TOKENS:
            if token not in seen:
                hits.append(token)
                seen.add(token)
    return hits


def _banned_text_files(root: Path) -> list[Path]:
    found: list[Path] = []
    entry = root / "softsaddle" / "entry.toml"
    if entry.is_file():
        found.append(entry)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            suffix = path.suffix.lower()
            relative = path.relative_to(root)
            if suffix in _TEXT_SUFFIXES:
                found.append(path)
            elif suffix == ".json" and relative.parts[:2] == ("results", "board"):
                found.append(path)
    return found


def banned_word_errors(root: Path) -> list[str]:
    """Reject banned words in site text, docs, board files, and path tokens."""
    errors: list[str] = []
    root = Path(root)
    for path in _banned_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        hits = _banned_hits(text)
        if hits:
            errors.append(f"{path.relative_to(root)} contains {', '.join(hits)}")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        relative_dir = Path(dirpath).relative_to(root)
        names = list(dirnames) + filenames
        for name in names:
            relative = (relative_dir / name).as_posix()
            if relative == ".":
                continue
            hits = _banned_hits(relative)
            if hits:
                errors.append(f"{relative} path contains {', '.join(hits)}")
    return errors


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


_HELD_BACK_FILES = ("optbench/board.py", "optbench/findings.py")
_HELD_BACK_MARKERS = (
    "def " + "holdback(",
    "def " + "_apply_omitted(",
    "def " + "_matches_hold(",
    "class " + "Hold:",
    '"hold":',
    "'hold':",
)


def held_back_errors(root: Path) -> list[str]:
    """Reject a list that keeps templates off the ranked table."""
    errors: list[str] = []
    for rel in _HELD_BACK_FILES:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in _HELD_BACK_MARKERS):
            errors.append(f"{rel} keeps templates off the ranked table")
    ledger = root / "oracle" / "findings.jsonl"
    if ledger.is_file():
        for lineno, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            beside = row.get("beside") or {}
            if (isinstance(beside, dict) and ("omitted" in beside or "hold" in beside)) or "hold" in row:
                errors.append(
                    f"oracle/findings.jsonl line {lineno} keeps templates off the ranked table"
                )
                break
    return errors


def check(problems_dir: Path, oracle_path: Path) -> int:
    """Return 0 when archive gaps match oracle/archive_gaps.txt."""
    problems = load_all(problems_dir)
    rows = load_oracle(oracle_path)
    counts: dict[str, int] = {}
    root = oracle_path.resolve().parent.parent
    from .site import _primary

    structural: list[str] = (
        correction_errors(root, rows)
        + difference_errors(root, rows)
        + count_errors(root, rows)
        + unit_errors(problems, _primary)
    )
    for row in rows:
        ident = f"{row['suite']}/{row['problem']}"
        counts[ident] = counts.get(ident, 0) + 1
        problem = problems.get(ident)
        if problem is None:
            structural.append(f"{ident} has an oracle row and no problem file")
            continue
        if problem.get("contract") in {"partial", "shared"}:
            missing = missing_keys(problem, row["fields"])
            for key in missing:
                structural.append(f"contract {row['path']} missing {key}")
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
    from .board import build_board, rank_violations
    from .campaign import list_templates
    from .pins import pin_errors
    from .site_check import check_site

    site_errors = check_site(root)
    document = build_board(root)
    site_errors.extend(rank_violations(document))
    site_errors.extend(pin_errors(root, list_templates(root), document["_scored_rows"]))
    site_errors.extend(held_back_errors(root))
    site_errors.extend(banned_word_errors(root))
    if site_errors:
        print(f"site {len(site_errors)}")
        for line in site_errors:
            print(line)
        return 1
    print("site tables match the board")
    from .findings import findings_errors

    finding_errors = findings_errors(root)
    if finding_errors:
        print(f"findings {len(finding_errors)}")
        for line in finding_errors:
            print(line)
        return 1
    print("findings match the archive rows")
    return 0
