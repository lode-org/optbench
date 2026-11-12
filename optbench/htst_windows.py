"""hTST barrier windows and the benchmark.dat lines for one sample.

A window closes when every reference barrier, rounded to 3 decimals,
has been seen once. A blank identity field is omitted. The archive
parser rejects a key with no value, and an eOn template often has no
tuning string.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# archive/entries/hTST/Pt/eon-dimer-lbfgs/report.py
KB_EV = 8.6173324e-5

HTST = {
    "hTST/Pt": {
        "temperature": 700.0,
        "prefactor": 1.2e12,
        "kb": KB_EV,
        "rate_accuracy": 0.999,
        "rate_ref": 155055901.244,
        "barriers": [
            0.601,
            0.601,
            0.620,
            0.986,
            0.987,
            0.989,
            0.986,
            0.987,
            0.989,
            1.196,
            1.196,
            1.196,
            1.196,
            1.207,
            1.207,
        ],
        "barrier_cycles": 13,
        "rate_cycles": 224,
        "source": "archive/entries/hTST/Pt/eon-dimer-lbfgs/report.py",
    },
}


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def reduce_htst(rows: Sequence[Mapping], spec: Mapping) -> dict:
    """Barrier and rate cycles from one KMC state, in search order.

    ``process_id`` is set only for a search the archive would call good
    or repeat. The first time an id appears, its barrier is rounded to
    three decimals and matched against the reference list. A later
    search with the same id is a repeat.
    """
    remaining = [float(value) for value in spec["barriers"]]
    full = len(remaining)
    ref_ids: list[int] = []
    known: dict[int, float] = {}
    barrier_ids: list[int] = []
    search_b = 0
    calls_b = 0
    barrier_searches: list[int] = []
    barrier_calls: list[int] = []
    rate_ids: list[int] = []
    search_r = 0
    calls_r = 0
    rate_sum = 0.0
    rate_searches: list[int] = []
    rate_calls: list[int] = []
    threshold = float(spec["rate_ref"]) * float(spec["rate_accuracy"])
    temperature = float(spec["temperature"])
    prefactor = float(spec["prefactor"])
    kb = float(spec["kb"])
    for row in rows:
        spent = int(row["force_calls"])
        search_b += 1
        search_r += 1
        calls_b += spent
        calls_r += spent
        pid = row.get("process_id")
        if pid is None:
            continue
        ident = int(pid)
        barrier = float(row["barrier"])
        if ident not in known:
            known[ident] = barrier
            key = round(barrier, 3)
            if key in remaining:
                remaining.remove(key)
                ref_ids.append(ident)
        if ident in ref_ids and ident not in barrier_ids:
            barrier_ids.append(ident)
        if full and len(ref_ids) == full and len(barrier_ids) == len(ref_ids):
            barrier_searches.append(search_b)
            barrier_calls.append(calls_b)
            barrier_ids = []
            search_b = 0
            calls_b = 0
        if rate_sum < threshold and ident not in rate_ids:
            rate_ids.append(ident)
            rate_sum += prefactor * math.exp(-barrier / (kb * temperature))
        if rate_sum >= threshold:
            rate_searches.append(search_r)
            rate_calls.append(calls_r)
            rate_ids = []
            search_r = 0
            calls_r = 0
            rate_sum = 0.0
    return {
        "searches": len(rows),
        "processes": len(known),
        "matched_barriers": len(ref_ids),
        "reference_barriers": full,
        "unmatched_barriers": remaining,
        "barrier_cycles": len(barrier_searches),
        "barrier_jobs": _mean(barrier_searches),
        "barrier_force_calls": _mean(barrier_calls),
        "barrier_call_list": barrier_calls,
        "rate_cycles": len(rate_searches),
        "rate_jobs": _mean(rate_searches),
        "rate_force_calls": _mean(rate_calls),
        "rate_call_list": rate_calls,
    }


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def problem_force_cap(problem: Mapping) -> int | None:
    """Force-call cap from the problem file.

    A template ``failure_cap`` is not a substitute for a number the
    problem omits.
    """
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    if stated in (None, ""):
        return None
    return int(stated)


def htst_dat(
    summary: Mapping,
    *,
    template: Mapping,
    problem: Mapping,
    spec: Mapping,
    searches: int,
    minimize_calls: int,
    unfinished: bool,
    comments: Sequence[str],
) -> str:
    """Archive keys plus the cycle counts.

    ``average over`` is split so the barrier count and the rate count
    both survive. Empty code, commit, algorithm, and tuning values are
    left out.
    """
    lines = [line for line in comments if line]
    if summary["barrier_jobs"] is not None:
        lines.append(f"barrier_jobs {_fmt(float(summary['barrier_jobs']))}")
        lines.append(f"barrier_force_calls {_fmt(float(summary['barrier_force_calls']))}")
        lines.append(f"barrier_average_over {int(summary['barrier_cycles'])}")
    if summary["rate_jobs"] is not None:
        lines.append(f"rate_jobs {_fmt(float(summary['rate_jobs']))}")
        lines.append(f"rate_force_calls {_fmt(float(summary['rate_force_calls']))}")
        lines.append(f"rate_average_over {int(summary['rate_cycles'])}")
    lines.append(f"temperature_K {_fmt(float(spec['temperature']))}")
    lines.append(f"prefactor {_fmt(float(spec['prefactor']))}")
    lines.append(f"rate_ref {_fmt(float(spec['rate_ref']))}")
    lines.append(f"matched_barriers {int(summary['matched_barriers'])}")
    lines.append(f"reference_barriers {int(summary['reference_barriers'])}")
    lines.append(f"processes {int(summary['processes'])}")
    lines.append(f"total_searches {int(searches)}")
    lines.append(f"minimize_calls {int(minimize_calls)}")
    lines.append(f"code {template.get('code', 'eOn')}")
    for key in ("code_version", "commit", "algorithm", "tuning"):
        value = str(template.get(key) or "").strip()
        if value:
            lines.append(f"{key} {value}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append("counter harness")
    lines.append(f"reducer {template.get('reducer', 'htst-cycles')}")
    cap = problem_force_cap(problem)
    if cap is not None:
        lines.append(f"cap {int(cap)}")
    if unfinished:
        lines.append("partial true")
    return "\n".join(lines) + "\n"
