"""How many runs an archived mean covers.

An hTST ``benchmark.dat`` states its own count on the ``average over N``
line after each mean. Other entries state the count only in the scripts
that wrote the file. ``oracle/counts.toml`` names the entry, the count,
what was counted, and the script that states it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

COUNTS = Path("oracle") / "counts.toml"
_AVERAGE = re.compile(r"average over (\d+)")


def load_counts(root: Path) -> list[dict]:
    path = Path(root) / COUNTS
    if not path.is_file():
        return []
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    return [item for item in record.get("count") or [] if isinstance(item, dict)]


def text_count(text: str, field: str) -> int | None:
    """The ``average over N`` on the line after ``field`` in an archived file."""
    lines = [line.strip() for line in str(text or "").splitlines()]
    for index, line in enumerate(lines[:-1]):
        if line.split(maxsplit=1)[:1] == [field]:
            found = _AVERAGE.fullmatch(lines[index + 1])
            return int(found.group(1)) if found else None
    return None


def entry_count(item: dict, field: str, counts: list[dict]) -> int | None:
    """The entry's own count of the runs behind ``field``, else its count in ``counts``."""
    own = text_count(str(item.get("text") or ""), field)
    if own is not None:
        return own
    path = str(item.get("path") or "")
    for stated in counts:
        if str(stated.get("path") or "") == path:
            return int(stated["n"])
    return None


def count_errors(root: Path, entries: list[dict]) -> list[str]:
    """Counts whose entry, number, or source is missing, or that the entry already states."""
    by_path = {str(entry.get("path") or ""): entry for entry in entries}
    errors: list[str] = []
    seen: set[str] = set()
    for item in load_counts(root):
        path = str(item.get("path") or "")
        where = f"{COUNTS.as_posix()} {path}"
        if path in seen:
            errors.append(f"{where} is counted twice")
        seen.add(path)
        n = item.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            errors.append(f"{where} n is not a positive whole number")
        for key in ("counted", "source"):
            if not str(item.get(key) or "").strip():
                errors.append(f"{where} has no {key}")
        entry = by_path.get(path)
        if entry is None:
            errors.append(f"{where} names no oracle entry")
        elif any(_AVERAGE.fullmatch(line.strip()) for line in str(entry.get("text") or "").splitlines()):
            errors.append(f"{where} also states its own count")
    return errors
