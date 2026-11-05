"""The unit each board row counts in.

A problem file's ``[units]`` table names the harness unit, the archive
unit, and the source for both. Without it, both are force calls. A row is
ranked only beside rows in the same unit.
"""

from __future__ import annotations

FORCE_CALLS = "force calls"
FORCE_CALL_FIELDS = frozenset({"force_calls", "barrier_force_calls"})
_KEYS = frozenset({"harness", "archive", "source"})


def _units(problem: dict | None) -> dict:
    units = (problem or {}).get("units")
    return units if isinstance(units, dict) else {}


def harness_unit(problem: dict | None) -> str:
    return str(_units(problem).get("harness") or FORCE_CALLS)


def archive_unit(problem: dict | None) -> str:
    units = _units(problem)
    return str(units.get("archive") or units.get("harness") or FORCE_CALLS)


def unit_source(problem: dict | None) -> str:
    return str(_units(problem).get("source") or "")


def row_unit(problem: dict | None, origin: str) -> str:
    return archive_unit(problem) if origin == "archive" else harness_unit(problem)


def rank_unit(problem: dict | None, rows: list[dict]) -> str:
    """The harness unit when a harness row is ranked, else the archive unit."""
    if rows and all(row.get("origin") == "archive" for row in rows):
        return archive_unit(problem)
    return harness_unit(problem)


def unit_errors(problems: dict[str, dict], primary_of) -> list[str]:
    """Problems whose ranked field is not a force-call count and has no stated unit."""
    errors: list[str] = []
    for ident in sorted(problems):
        problem = problems[ident]
        units = problem.get("units")
        primary = primary_of(problem)
        if units is not None and not isinstance(units, dict):
            errors.append(f"{ident} [units] is not a table")
            continue
        units = units or {}
        unknown = sorted(set(units) - _KEYS)
        if unknown:
            errors.append(f"{ident} [units] has unknown keys {', '.join(unknown)}")
        if units and not str(units.get("source") or "").strip():
            errors.append(f"{ident} [units] has no source")
        if primary and primary not in FORCE_CALL_FIELDS and not units.get("archive"):
            errors.append(f"{ident} ranks {primary} and its [units] table names no archive unit")
    return errors
