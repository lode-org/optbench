"""Archived rows whose values differ from the table that cites them.

``oracle/differences.toml`` names the entry, the table that prints it, the
archived and printed values, the note the board prints beside the row, and
the evidence for that note. The archived file stays as it is.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

DIFFERENCES = Path("oracle") / "differences.toml"


def load_differences(root: Path) -> list[dict]:
    path = Path(root) / DIFFERENCES
    if not path.is_file():
        return []
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    return [item for item in record.get("difference") or [] if isinstance(item, dict)]


def difference_note(path: str, fields: dict[str, str], differences: list[dict]) -> str:
    """The note for ``path`` while the archive still has the recorded values."""
    for item in differences:
        archived = item.get("archived") or {}
        if str(item.get("path") or "") == path and all(fields.get(key) == str(value) for key, value in archived.items()):
            return str(item.get("note") or "")
    return ""


def difference_errors(root: Path, entries: list[dict]) -> list[str]:
    """Differences whose entry, archived values, note, or evidence is missing."""
    from .dat import normalize

    by_path = {str(entry.get("path") or ""): entry for entry in entries}
    errors: list[str] = []
    seen: set[str] = set()
    for item in load_differences(root):
        path = str(item.get("path") or "")
        where = f"{DIFFERENCES.as_posix()} {path}"
        if path in seen:
            errors.append(f"{where} is listed twice")
        seen.add(path)
        for key in ("table", "note", "evidence"):
            if not str(item.get(key) or "").strip():
                errors.append(f"{where} has no {key}")
        archived = item.get("archived")
        if not isinstance(archived, dict) or not archived:
            errors.append(f"{where} records no archived values")
            archived = {}
        printed = item.get("printed")
        if not isinstance(printed, dict) or not printed:
            errors.append(f"{where} records no printed values")
        elif all(str(printed.get(key)) == str(value) for key, value in archived.items()):
            errors.append(f"{where} prints the archived values")
        entry = by_path.get(path)
        if entry is None:
            errors.append(f"{where} names no oracle entry")
            continue
        fields, _renamed = normalize(entry.get("fields") or {})
        for key, value in sorted(archived.items()):
            if fields.get(key) != str(value):
                errors.append(f"{where} expects {key} {value!r}, the archive has {fields.get(key)!r}")
    return errors
