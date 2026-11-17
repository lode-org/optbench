"""One primary failure class for a start.

The order is fixed, so a start lands in one bin:

1. ``crash`` — the optimizer raised, or the record says so.
2. ``divergence`` — a non-finite point, or a step past the ceiling.
3. ``cap`` — the search used the whole force-call budget and did not
   blow up.
4. ``duplicate`` — the saddle repeats one already kept.
5. ``over_barrier`` — the barrier is at or above the problem ceiling.
6. ``disconnected`` — the minimized endpoints are not the reactant.
7. ``wrong_stationary`` — the force stop, the Hessian index, the
   energy, or the Kabsch RMSD rejects the point.

A success has no class. Every failure stays in the PAR-2 mean; the
class says why it is there. ``FAILURES`` is the four bins an asv
track records. ``CATALOG_FAILURES`` are the saddle-catalog misses
that used to fall through to ``divergence``.
"""

from __future__ import annotations

FAILURES = ("crash", "divergence", "cap", "wrong_stationary")
CATALOG_FAILURES = ("duplicate", "over_barrier", "disconnected")
ALL_FAILURES = FAILURES + CATALOG_FAILURES


def classify_failure(reasons: list[str]) -> str | None:
    """Primary class, or ``None`` when ``reasons`` is empty."""
    if not reasons:
        return None
    if "crash" in reasons:
        return "crash"
    if "divergence" in reasons:
        return "divergence"
    if "cap" in reasons:
        return "cap"
    if "duplicate" in reasons:
        return "duplicate"
    if "barrier" in reasons or "over_barrier" in reasons:
        return "over_barrier"
    if "not_connected" in reasons:
        return "disconnected"
    if any(name in reasons for name in ("hessian_index", "energy", "rmsd", "force_norm")):
        return "wrong_stationary"
    return "wrong_stationary"


def failure_of(record: dict, cap: int | None) -> str | None:
    """Class stored on the record, or inferred when an older row omits it."""
    if record.get("success"):
        label = record.get("failure")
        if label not in (None, "", "none"):
            raise ValueError("a success cannot carry a failure class")
        return None
    reasons = [str(reason) for reason in record.get("reasons") or []]
    if reasons:
        return classify_failure(reasons)
    label = record.get("failure")
    if label in ALL_FAILURES:
        return str(label)
    calls = float(record["force_calls"])
    if cap is not None and calls >= cap:
        return "cap"
    return "divergence"


def failure_tally(records: list[dict], cap: int | None) -> dict[str, int]:
    """Count each class. The counts plus nothing else equal ``nfailed``."""
    counts = {name: 0 for name in ALL_FAILURES}
    for record in records:
        label = failure_of(record, cap)
        if label is None:
            continue
        counts[label] += 1
    return counts
