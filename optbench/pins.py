"""Code labels on templates and scored rows against the registry pins."""

from __future__ import annotations

import tomllib
from pathlib import Path

PIN_GAPS = Path("oracle") / "pin_gaps.txt"
_PINNED_SOURCES = frozenset({"campaign", "asv"})


def load_registry(root: Path) -> list[dict]:
    path = Path(root) / "codes" / "registry.toml"
    if not path.is_file():
        return []
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    return [code for code in record.get("code") or [] if isinstance(code, dict)]


def _names(pin: dict) -> set[str]:
    raw = pin.get("aliases") or []
    names = [str(pin.get("id") or ""), str(pin.get("name") or "")]
    names.extend([raw] if isinstance(raw, str) else [str(item) for item in raw])
    return {name.casefold() for name in names if name.strip()}


def same_commit(recorded: str, pinned: str) -> bool:
    """True when one commit is a prefix of the other, so a short sha matches its pin."""
    return bool(recorded and pinned) and (recorded.startswith(pinned) or pinned.startswith(recorded))


def pin_gap(code: str, commit: str, registry: list[dict]) -> str | None:
    """How a code and commit label departs from the registry, or None for a pin.

    Several entries can share a name or an alias. The commit selects the pin.
    """
    named = [pin for pin in registry if code.casefold() in _names(pin)]
    if not named:
        return f"{code} {commit} has no registry entry"
    pinned = [str(pin.get("commit") or "") for pin in named]
    if any(same_commit(commit, value) for value in pinned):
        return None
    listed = ", ".join(sorted({value for value in pinned if value})) or "no commit"
    return f"{code} {commit}, registry pins {listed}"


def template_pin_errors(templates: list[dict], registry: list[dict]) -> list[str]:
    """Templates whose code and commit are not a registry pin."""
    errors: list[str] = []
    for template in templates:
        code = str(template.get("code") or "")
        commit = str(template.get("commit") or "")
        if not code or not commit:
            continue
        gap = pin_gap(code, commit, registry)
        if gap:
            errors.append(f"template {template.get('id')} names {gap}")
    return sorted(errors)


def row_pin_gaps(rows: list[dict], registry: list[dict]) -> list[str]:
    """Scored campaign and asv rows whose recorded label is not a registry pin."""
    gaps: set[str] = set()
    for row in rows:
        if str(row.get("source") or "") not in _PINNED_SOURCES:
            continue
        code = str(row.get("code") or "")
        commit = str(row.get("commit") or "")
        if not code or not commit:
            continue
        gap = pin_gap(code, commit, registry)
        if gap:
            where = str(row.get("path") or "")
            if row.get("source") == "asv":
                where = f"{where} {row.get('entry')}"
            gaps.add(f"{where} names {gap}")
    return sorted(gaps)


def known_pin_gaps(root: Path) -> list[str]:
    path = Path(root) / PIN_GAPS
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() and not line.startswith("#")]


def pin_errors(root: Path, templates: list[dict], rows: list[dict]) -> list[str]:
    """Template labels off the registry, and row labels not listed in ``oracle/pin_gaps.txt``.

    A committed row keeps the label it was recorded with. The list names the
    rows whose label is not the pin, so a new one fails until it is listed.
    """
    registry = load_registry(root)
    errors = template_pin_errors(templates, registry)
    gaps = row_pin_gaps(rows, registry)
    known = known_pin_gaps(root)
    errors.extend(f"pin gap not in {PIN_GAPS.as_posix()}: {line}" for line in gaps if line not in known)
    errors.extend(f"{PIN_GAPS.as_posix()} lists a row that is now pinned: {line}" for line in known if line not in gaps)
    return errors
