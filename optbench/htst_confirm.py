"""hTST confirmation on search rows.

Each batch restarts job ids at zero. The order inside a batch is the id
order the seed wrote. A sort of the whole file mixes those batches. The
PotServer total is recorded beside that order and is not taken from the ids.
"""

from __future__ import annotations

import math
from pathlib import Path

BARRIERS = [
    0.601, 0.601, 0.620,
    0.986, 0.987, 0.989, 0.986, 0.987, 0.989,
    1.196, 1.196, 1.196, 1.196,
    1.207, 1.207,
]
RATE_REF = 155055901.244
RATE_NEED = 0.999 * RATE_REF
PREFACTOR = 1.2e12
KB = 8.6173324e-5
TEMPERATURE = 700.0


def round3(value: float) -> float:
    """Python 2 round-half-away-from-zero, three decimals."""
    scaled = value * 1000.0
    if scaled >= 0:
        return math.floor(scaled + 0.5) / 1000.0
    return math.ceil(scaled - 0.5) / 1000.0


def creation_order(rows: list[list[str]], batch: int) -> list[list[str]]:
    """Seed order.

    The communicator writes the job-id counter in the scratch directory, and
    the loop clears that directory after each register. Each batch therefore
    restarts its labels at zero. File order inside a batch is completion
    order. Sorting that batch by job id is the order the seed generated.
    """
    if batch <= 0:
        return list(rows)
    ordered: list[list[str]] = []
    for start in range(0, len(rows), batch):
        ordered.extend(sorted(rows[start : start + batch], key=lambda fields: int(fields[0])))
    return ordered


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def score_rows(
    search_rows: list[list[str]],
    process_lines: list[str],
    batch: int,
    potserver: int | None = None,
) -> dict:
    """Barrier and rate windows, plus the PotServer total.

    ``potserver`` is stored as given. Job-id order does not enter it.
    """
    remaining = list(BARRIERS)
    barrier_ids_ref: list[int] = []
    for line in process_lines:
        fields = line.split()
        if not fields or fields[0] == "proc":
            continue
        number = int(fields[0])
        barrier = round3(float(fields[6]))
        if barrier in remaining:
            barrier_ids_ref.append(number)
            remaining.pop(remaining.index(barrier))

    rows = creation_order(search_rows, batch)
    barrier_ids: list[int] = []
    rate_ids: list[int] = []
    search_b = 0
    search_r = 0
    fcs_b = 0
    fcs_r = 0
    rate_sum = 0.0
    barrier_jobs: list[int] = []
    barrier_fcs: list[int] = []
    rate_jobs: list[int] = []
    rate_fcs: list[int] = []
    rate_at: list[float] = []
    consumed: list[float] = []
    histogram: dict[str, int] = {}
    good = 0
    seen = 0
    met_at: dict | None = None
    for fields in rows:
        seen += 1
        search_b += 1
        search_r += 1
        sad = int(fields[4])
        fcs_b += sad
        fcs_r += sad
        result = fields[-1]
        if "-" not in result:
            continue
        kind, ident = result.split("-", 1)
        if kind not in ("good", "repeat"):
            continue
        process = int(ident)
        if kind == "good":
            good += 1
            height = round3(float(fields[2]))
            histogram[f"{height:.3f}"] = histogram.get(f"{height:.3f}", 0) + 1
        if process in barrier_ids_ref and process not in barrier_ids:
            barrier_ids.append(process)
            consumed.append(round3(float(fields[2])))
        if len(barrier_ids_ref) and len(barrier_ids) == len(barrier_ids_ref) and not remaining:
            barrier_jobs.append(search_b)
            barrier_fcs.append(fcs_b)
            barrier_ids = []
            search_b = 0
            fcs_b = 0
        if rate_sum < RATE_NEED and process not in rate_ids:
            rate_ids.append(process)
            rate_sum += PREFACTOR * math.exp(-float(fields[2]) / (KB * TEMPERATURE))
        if rate_sum >= RATE_NEED:
            rate_jobs.append(search_r)
            rate_fcs.append(fcs_r)
            rate_at.append(rate_sum)
            rate_ids = []
            search_r = 0
            fcs_r = 0
            rate_sum = 0.0
        if met_at is None and barrier_jobs and rate_jobs:
            met_at = {
                "search": seen,
                "barrier_jobs": _mean(barrier_jobs),
                "barrier_force_calls": _mean(barrier_fcs),
                "rate_jobs": _mean(rate_jobs),
                "rate_force_calls": _mean(rate_fcs),
                "barrier_windows": len(barrier_jobs),
                "rate_windows": len(rate_jobs),
            }

    return {
        "searches": len(rows),
        "good": good,
        "histogram": histogram,
        "remaining": remaining,
        "barrier_windows": len(barrier_jobs),
        "rate_windows": len(rate_jobs),
        "barrier_jobs": _mean(barrier_jobs),
        "barrier_force_calls": _mean(barrier_fcs),
        "rate_jobs": _mean(rate_jobs),
        "rate_force_calls": _mean(rate_fcs),
        "barrier_jobs_windows": barrier_jobs,
        "barrier_fcs_windows": barrier_fcs,
        "rate_jobs_windows": rate_jobs,
        "rate_fcs_windows": rate_fcs,
        "rates": rate_at,
        "consumed": consumed,
        "rate_progress": rate_sum,
        "met": bool(barrier_jobs) and bool(rate_jobs),
        "met_at": met_at,
        "potserver": potserver,
    }


def _search_rows(text: str) -> list[list[str]]:
    lines = text.splitlines()
    rows = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 6:
            continue
        rows.append(fields)
    return rows


def score_state(state: Path, batch: int = 8, potserver: int | None = None) -> dict:
    """Score one state directory."""
    search_path = state / "search_results.txt"
    process_path = state / "processtable"
    search_text = search_path.read_text(encoding="utf-8", errors="replace") if search_path.is_file() else ""
    process_text = process_path.read_text(encoding="utf-8", errors="replace") if process_path.is_file() else ""
    return score_rows(_search_rows(search_text), process_text.splitlines(), batch, potserver)
