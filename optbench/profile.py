"""Dolan-Moré performance profile and PAR-2 across codes.

The profile is the fraction of problems on which a code's cost is at
most ``tau`` times the best finite cost on that problem. An unsolved
problem has an infinite ratio: it stays in the denominator and does not
cross a finite ``tau``. The PAR-2 mean is a different summary of the
same runs and does charge those failures ``2 * cap``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .score import score_v2

DEFAULT_TAUS = (1.0, 2.0, 4.0, 8.0, 16.0)


def performance_profile(
    costs: dict[str, Sequence[float]],
    taus: Sequence[float] = DEFAULT_TAUS,
) -> dict:
    """``costs[code][problem]`` is a finite cost, or non-finite when unsolved."""
    codes = list(costs)
    if not codes:
        raise ValueError("costs is empty")
    width = len(costs[codes[0]])
    if width == 0 or any(len(costs[code]) != width for code in codes):
        raise ValueError("each code needs the same non-empty problem list")
    ordered = [float(tau) for tau in taus]
    if any(tau <= 0.0 for tau in ordered) or ordered != sorted(ordered):
        raise ValueError("taus must be positive and ascending")
    reference: list[float] = []
    for problem in range(width):
        finite = [float(costs[code][problem]) for code in codes if math.isfinite(costs[code][problem])]
        if not finite:
            raise ValueError(f"problem {problem} has no finite cost")
        best = min(finite)
        if best <= 0.0:
            raise ValueError("a reference cost must be positive")
        reference.append(best)
    ratios: dict[str, list[float]] = {}
    rho: dict[str, list[float]] = {}
    solved: dict[str, float] = {}
    area: dict[str, float] = {}
    for code in codes:
        row = []
        for problem in range(width):
            value = float(costs[code][problem])
            row.append(value / reference[problem] if math.isfinite(value) else math.inf)
        ratios[code] = row
        rho[code] = [sum(1 for ratio in row if ratio <= tau) / width for tau in ordered]
        solved[code] = sum(1 for ratio in row if math.isfinite(ratio)) / width
        area[code] = _log_area(ordered, rho[code])
    return {
        "taus": ordered,
        "reference": reference,
        "ratios": ratios,
        "rho": rho,
        "solved": solved,
        "area_log10": area,
    }


def _log_area(taus: Sequence[float], values: Sequence[float]) -> float:
    """Trapezoid integral of ``rho`` against ``log10(tau)``."""
    if len(taus) < 2:
        return 0.0
    total = 0.0
    for left, right, low, high in zip(taus, taus[1:], values, values[1:]):
        span = math.log10(right) - math.log10(left)
        total += 0.5 * (low + high) * span
    return total


def profile_cost(record: dict) -> float:
    """Finite harness cost of a success, otherwise infinity for the profile."""
    try:
        success = record.success
        calls = record.force_calls
    except AttributeError:
        success = record["success"]
        calls = record["force_calls"]
    if not success:
        return math.inf
    return float(calls)


def par2_across(runs: dict[str, list[dict]], cap: int | None) -> dict[str, dict]:
    """PAR-2 summary of each code. Failures stay in that mean."""
    return {code: score_v2(records, cap) for code, records in runs.items()}
