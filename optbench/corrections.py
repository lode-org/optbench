"""Archived fields that the entry's own run script contradicts.

``oracle/corrections.toml`` names the entry, the field, the archived value,
the value the board reads, and the source. The archived file stays as it is.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

CORRECTIONS = Path("oracle") / "corrections.toml"


def load_corrections(root: Path) -> list[dict]:
    path = Path(root) / CORRECTIONS
    if not path.is_file():
        return []
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    return [item for item in record.get("correction") or [] if isinstance(item, dict)]


def corrected(path: str, fields: dict[str, str], corrections: list[dict]) -> dict[str, str]:
    """``fields`` with each correction for ``path`` applied where the archive still has its value."""
    out = dict(fields)
    for item in corrections:
        field = str(item.get("field") or "")
        if str(item.get("path") or "") == path and out.get(field) == str(item.get("was") or ""):
            out[field] = str(item.get("value") or "")
    return out


def correction_errors(root: Path, entries: list[dict]) -> list[str]:
    """Corrections whose entry, archived value, or source is missing."""
    from .dat import normalize

    by_path = {str(entry.get("path") or ""): entry for entry in entries}
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in load_corrections(root):
        path = str(item.get("path") or "")
        field = str(item.get("field") or "")
        where = f"{CORRECTIONS.as_posix()} {path} {field}"
        if (path, field) in seen:
            errors.append(f"{where} is corrected twice")
        seen.add((path, field))
        if not str(item.get("source") or "").strip():
            errors.append(f"{where} has no source")
        entry = by_path.get(path)
        if entry is None:
            errors.append(f"{where} names no oracle entry")
            continue
        fields, _renamed = normalize(entry.get("fields") or {})
        was = str(item.get("was") or "")
        if fields.get(field) != was:
            errors.append(f"{where} expects {was!r}, the archive has {fields.get(field)!r}")
    return errors
