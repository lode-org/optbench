"""Reducers for force-call tables.

The archive mean is awk ``printf "%.0f"``. On halves that rounds to even,
which is what Python ``round`` does for these ties.

The v2 cost is the PAR-2 mean. A failure is charged ``2 * cap`` instead
of being dropped. The Wilson bounds and the bootstrap median sit
beside that mean. See ``docs/design/decision.md``.
"""

from __future__ import annotations

import math
import random
import statistics

from .failures import FAILURES, failure_tally

WILSON_Z = 1.959963984540054
PAR_FACTOR = 2
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261015


def half_even(value: float) -> int:
    """Round to an integer, ties to even."""
    return round(value)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2


def summarize_all(counts: list[int], nfailed: int) -> dict[str, int | float | None]:
    """Mean, min, max, and median of every count, failures included.

    Transition-state rows use this reducer. A start that hit the step cap
    stays in the mean, and ``nfailed`` counts those starts.
    """
    if nfailed < 0:
        raise ValueError("nfailed must be zero or positive")
    if not counts:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "force_calls_median": None,
            "nfailed": nfailed,
        }
    return {
        "force_calls": half_even(sum(counts) / len(counts)),
        "force_calls_min": min(counts),
        "force_calls_max": max(counts),
        "force_calls_median": _median([float(count) for count in counts]),
        "nfailed": nfailed,
    }


def band_total(force_calls: float, *, images: int | None, call_unit: str) -> tuple[float, float | None]:
    """Band total and the per-image quotient of that same total.

    ``call_unit`` is ``total`` or ``per_image``. The per-image value is
    ``None`` when the problem states no image count.
    """
    if call_unit == "total":
        total = float(force_calls)
    elif call_unit == "per_image":
        if not images:
            raise ValueError("per-image count has no image count")
        total = float(force_calls) * images
    else:
        raise ValueError(f"unknown call unit {call_unit}")
    per = None if not images else total / images
    return total, per


def normalize_start(raw: dict, images: int | None, *, per_image_track: bool = False) -> dict:
    """One start whose ``force_calls`` is the harness total across the band.

    A ``force_calls_neb`` row stores the per-image quotient in
    ``force_calls`` and the band counter in ``raw``. An old ASE asv track
    stored the per-image quotient; ``per_image_track`` puts that track back
    on the total. Every other row is already a total. ``per_image`` divides
    the total by ``images``, the same divisor for every code on the problem.
    """
    if not isinstance(raw, dict):
        raise ValueError("start is not a record")
    scaled = dict(raw)
    count_key = str(raw.get("count_key") or "")
    stated = str(raw.get("call_unit") or "")
    try:
        if count_key == "force_calls_neb":
            if raw.get("raw") not in (None, ""):
                total = float(raw["raw"])
                per = None if not images else total / images
            else:
                total, per = band_total(float(raw["force_calls"]), images=images, call_unit="per_image")
        elif stated == "per_image" or per_image_track:
            calls = raw.get("force_calls")
            if calls is None:
                raise ValueError("per-image count has no force calls")
            total, per = band_total(float(calls), images=images, call_unit="per_image")
        else:
            calls = raw.get("force_calls", raw.get("raw", raw.get("per_search")))
            if calls is None:
                return scaled
            total, per = band_total(float(calls), images=images, call_unit="total")
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc) or "per-image count has no image count") from exc
    scaled["force_calls"] = total
    scaled["call_unit"] = "total"
    if per is not None:
        scaled["per_image"] = per
    return scaled


def summarize_divided(
    counts: list[int],
    failed: list[bool],
    *,
    images: int,
    cap: int,
) -> dict[str, int | float | str | None]:
    """Mean of band totals that converged under ``cap``.

    ``cap`` is a total across the band, compared with ``count`` itself.
    ``force_calls`` is that total. ``force_calls_per_image`` divides the
    same kept totals by ``images``.
    """
    if images <= 0:
        raise ValueError("images must be positive")
    if cap <= 0:
        raise ValueError("cap must be positive")
    if len(counts) != len(failed):
        raise ValueError("counts and failed must have the same length")
    ok_raw: list[int] = []
    ok_per: list[float] = []
    nfailed = 0
    for count, flag in zip(counts, failed):
        if flag or count >= cap:
            nfailed += 1
            continue
        ok_raw.append(count)
        ok_per.append(count / images)
    if not ok_per:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "force_calls_median": None,
            "force_calls_raw": None,
            "force_calls_per_image": None,
            "force_calls_per_image_min": None,
            "force_calls_per_image_max": None,
            "force_calls_per_image_median": None,
            "call_unit": "total",
            "nfailed": nfailed,
        }
    total_mean = half_even(sum(ok_raw) / len(ok_raw))
    return {
        "force_calls": total_mean,
        "force_calls_min": min(ok_raw),
        "force_calls_max": max(ok_raw),
        "force_calls_median": _median([float(value) for value in ok_raw]),
        "force_calls_raw": total_mean,
        "force_calls_per_image": half_even(sum(ok_per) / len(ok_per)),
        "force_calls_per_image_min": half_even(min(ok_per)),
        "force_calls_per_image_max": half_even(max(ok_per)),
        "force_calls_per_image_median": _median(ok_per),
        "call_unit": "total",
        "nfailed": nfailed,
    }


def summarize_capped(counts: list[int], cap: int) -> dict[str, int | None]:
    """Mean, min, and max of counts below ``cap``, and how many meet the cap.

    A count at or above ``cap`` is a failure and is left out of the mean.
    The mean is rounded with :func:`half_even`.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    ok = [count for count in counts if count < cap]
    failed = len(counts) - len(ok)
    if not ok:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "nfailed": failed,
        }
    return {
        "force_calls": half_even(sum(ok) / len(ok)),
        "force_calls_min": min(ok),
        "force_calls_max": max(ok),
        "nfailed": failed,
    }


def wilson_interval(successes: int, starts: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for ``successes / starts``.

    ``z`` is the standard-normal quantile. The default is the 95% value.
    """
    if starts <= 0:
        raise ValueError("starts must be positive")
    if successes < 0 or successes > starts:
        raise ValueError("successes must lie between 0 and starts")
    phat = successes / starts
    z2 = z * z
    denom = 1.0 + z2 / starts
    center = (phat + z2 / (2.0 * starts)) / denom
    margin = z * math.sqrt(phat * (1.0 - phat) / starts + z2 / (4.0 * starts * starts)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    # k = 0 and k = n are exactly 0 and 1. Float error can land just inside.
    if successes == 0:
        low = 0.0
    if successes == starts:
        high = 1.0
    return low, high


def penalised_cost(force_calls: float, success: bool, cap: int | None, factor: int = PAR_FACTOR) -> float:
    """PAR cost. A failure, or a success at or above the cap, costs ``factor * cap``."""
    if cap is not None and (not success or force_calls >= cap):
        return float(factor * cap)
    if not success:
        raise ValueError("a failure needs a cap")
    return float(force_calls)


def quantile(values: list[float], fraction: float) -> float:
    """Linear quantile, the same ranks as ``sample_stats``."""
    if not values:
        raise ValueError("values is empty")
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 1:
        return ordered[0]
    position = (count - 1) * fraction
    low = int(position)
    high = min(low + 1, count - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _percentile(sorted_values: list[float], fraction: float) -> float:
    return quantile(sorted_values, fraction)


def bootstrap_interval(
    values: list[float],
    statistic,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval.

    Returns ``(statistic(values), low, high)``. Resamples are drawn with
    replacement from a fixed seed so a published interval can be recomputed.
    """
    if not values:
        raise ValueError("values is empty")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    point = float(statistic(values))
    rng = random.Random(seed)
    count = len(values)
    samples = []
    for _ in range(resamples):
        draw = [values[rng.randrange(count)] for _ in range(count)]
        samples.append(float(statistic(draw)))
    samples.sort()
    low = _percentile(samples, alpha / 2.0)
    high = _percentile(samples, 1.0 - alpha / 2.0)
    return point, low, high


def _iqr(values: list[float]) -> float:
    return quantile(values, 0.75) - quantile(values, 0.25)


def distribution_summary(
    values: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Median, interquartile range, and bootstrap intervals of a cost sample."""
    if not values:
        raise ValueError("values is empty")
    median, median_low, median_high = bootstrap_interval(
        values, statistics.median, resamples=resamples, seed=seed
    )
    iqr, iqr_low, iqr_high = bootstrap_interval(values, _iqr, resamples=resamples, seed=seed)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": median,
        "median_low": median_low,
        "median_high": median_high,
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "iqr": iqr,
        "iqr_low": iqr_low,
        "iqr_high": iqr_high,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def score_v2(
    records: list[dict],
    cap: int | None,
    *,
    factor: int = PAR_FACTOR,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """PAR-2 mean, Wilson success interval, and bootstrap median of one run.

    Each record has ``force_calls`` and ``success``. When ``cap`` is
    ``None`` every record must be a success, and ``par_applied`` is false.
    That is the Pt-island per-search sample: the problem states no cap.
    """
    if not records:
        raise ValueError("records is empty")
    successes = 0
    costs: list[float] = []
    for record in records:
        success = bool(record["success"])
        if success:
            successes += 1
        costs.append(penalised_cost(float(record["force_calls"]), success, cap, factor))
    starts = len(records)
    low, high = wilson_interval(successes, starts)
    summary = distribution_summary(costs, resamples=resamples, seed=seed)
    tally = failure_tally(records, cap)
    failed = starts - successes
    if sum(tally.values()) != failed:
        raise RuntimeError("failure classes dropped a start")
    finished = [float(record["force_calls"]) for record in records if record["success"]]
    summary.update(
        {
            "scoring": "par2-wilson-bootstrap",
            "successes": successes,
            "nfailed": failed,
            "success_rate": successes / starts,
            "wilson_low": low,
            "wilson_high": high,
            "par_factor": factor,
            "cap": cap,
            "par_applied": cap is not None,
            "par2_mean": summary["mean"] if cap is not None else None,
            "includes_failures": True,
            "success_only_mean": (sum(finished) / len(finished)) if finished else None,
            **{f"n_{name}": tally[name] for name in FAILURES},
        }
    )
    return summary
