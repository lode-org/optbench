"""Escape-rate and event-catalogue scores for the aluminum adatom.

The 2012 pages compare a calculated escape rate with a parallel-replica
reference, and they compare a product histogram with the molecular
dynamics standard. Force calls per simulated time is the work spent on
the catalogue divided by the time those events would cover.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

import numpy as np

# Published molecular dynamics branching ratios at 250 K.
# website/branchingratios.tmpl, MD_standard row, 15000 samples.
MD_STANDARD_250K = (
    0.00020,
    0.38127,
    0.23493,
    0.12473,
    0.24273,
    0.00200,
    0.00140,
    0.00280,
    0.00240,
    0.00087,
    0.00193,
    0.00147,
    0.00040,
    0.00113,
    0.00093,
    0.00020,
    0.00007,
    0.00013,
    0.00007,
    0.00007,
    0.00007,
    0.00007,
    0.00007,
    0.00007,
)

# Bond-boost hyperdynamics row on the same page, 5000 samples.
BOND_BOOST_250K = (
    0.00040,
    0.37080,
    0.23620,
    0.13280,
    0.24320,
    0.00160,
    0.00140,
    0.00240,
    0.00180,
    0.00100,
    0.00200,
    0.00200,
    0.00080,
    0.00140,
    0.00020,
    0.00100,
    0.00000,
    0.00020,
    0.00000,
    0.00000,
    0.00000,
    0.00000,
    0.00000,
    0.00000,
)

UNMATCHED = -1

# akmc.py divides the temperature in kelvin by this to get kT in eV.
KELVIN_TO_EV = 1.0 / 11604.5


def old_scheme_confidence(repeats: int) -> float:
    """Old AKMC confidence. It is 0 until a relevant process repeats."""
    if repeats < 1:
        return 0.0
    return max(0.0, 1.0 - 1.0 / float(repeats))


def thermal_limit(lowest_barrier: float, temperature_K: float, window: float) -> float:
    """Highest barrier still inside ``window`` kT of the lowest barrier."""
    if not math.isfinite(lowest_barrier):
        raise ValueError("lowest barrier must be finite")
    if temperature_K <= 0 or not math.isfinite(temperature_K):
        raise ValueError("temperature must be positive")
    if window < 0 or not math.isfinite(window):
        raise ValueError("thermal window must be non-negative")
    return lowest_barrier + temperature_K * KELVIN_TO_EV * window


def select_thermal(
    processes: Sequence[Mapping[str, float]],
    temperature_K: float,
    window: float,
    min_barrier: float,
) -> list[Mapping[str, float]]:
    """Processes inside the thermal window and at or above ``min_barrier``."""
    if min_barrier < 0 or not math.isfinite(min_barrier):
        raise ValueError("minimum barrier must be non-negative")
    if not processes:
        return []
    lowest = min(float(item["barrier"]) for item in processes)
    ceiling = thermal_limit(lowest, temperature_K, window)
    kept = []
    for item in processes:
        barrier = float(item["barrier"])
        rate = float(item["rate"])
        if barrier < min_barrier or barrier > ceiling or rate <= 0:
            continue
        kept.append(item)
    return kept


def percent_uncertainty(rate: float, uncertainty: float) -> float:
    """100 * uncertainty / rate, the percent column on the 2012 pages."""
    if rate == 0 or not math.isfinite(rate) or not math.isfinite(uncertainty):
        raise ValueError("rate and uncertainty must be finite and the rate non-zero")
    return 100.0 * uncertainty / rate


def rate_accuracy(rate: float, reference: float) -> float:
    """One minus the absolute relative error, floored at zero."""
    if reference <= 0 or not math.isfinite(reference) or not math.isfinite(rate) or rate < 0:
        return 0.0
    return max(0.0, 1.0 - abs(rate - reference) / reference)


def _weights(values: Mapping[int, float], keys: set[int]) -> tuple[float, dict[int, float]]:
    raw = {key: max(0.0, float(values.get(key, 0.0))) for key in keys}
    total = sum(raw.values())
    return total, raw


def catalogue_accuracy(found: Mapping[int, float], reference: Mapping[int, float]) -> float:
    """One minus the total variation distance of two product distributions.

    Weights are normalized. A channel that appears on only one side
    still counts. Channel ``-1`` is an unmatched product.
    """
    keys = set(found) | set(reference)
    if not keys:
        return 0.0
    found_total, found_raw = _weights(found, keys)
    ref_total, ref_raw = _weights(reference, keys)
    if found_total <= 0 or ref_total <= 0:
        return 0.0
    variation = 0.5 * sum(
        abs(found_raw[key] / found_total - ref_raw[key] / ref_total) for key in keys
    )
    return max(0.0, 1.0 - variation)


def simulated_time_seconds(rate: float, products: int) -> float:
    """Seconds covered by ``products`` escapes at ``rate`` per second."""
    if products <= 0:
        raise ValueError("products must be positive")
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("rate must be positive")
    return products / rate


def force_calls_per_simulated_time(force_calls: float, simulated_time: float) -> float:
    """Force calls divided by simulated time in seconds."""
    if simulated_time <= 0 or not math.isfinite(simulated_time):
        raise ValueError("simulated time must be positive")
    if force_calls < 0 or not math.isfinite(force_calls):
        raise ValueError("force calls must be finite and non-negative")
    return force_calls / simulated_time


def rate_summary(rates: Sequence[float]) -> dict[str, float | int | None]:
    """Mean rate and the standard error of that mean.

    One rate has no standard error. The 2012 percent uncertainty is
    100 times the standard error divided by the mean.
    """
    values = [float(rate) for rate in rates]
    if not values:
        raise ValueError("rates must be non-empty")
    if any(not math.isfinite(rate) or rate < 0 for rate in values):
        raise ValueError("rates must be finite and non-negative")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {
            "true_rate": mean,
            "uncertainty": None,
            "percent_uncertainty": None,
            "n": 1,
        }
    variance = sum((rate - mean) ** 2 for rate in values) / (len(values) - 1)
    error = math.sqrt(variance) / math.sqrt(len(values))
    return {
        "true_rate": mean,
        "uncertainty": error,
        "percent_uncertainty": percent_uncertainty(mean, error) if mean else None,
        "n": len(values),
    }


def distribution(weights: Sequence[float]) -> dict[int, float]:
    """Channel index to weight. The sequence is product 0, product 1, ..."""
    return {index: float(weight) for index, weight in enumerate(weights)}


def con_spec_version(text: str) -> int:
    """Spec version from a JSON metadata line, or 1 when that line is absent."""
    for line in text.splitlines():
        if "con_spec_version" not in line:
            continue
        try:
            payload = json.loads(line)
            return int(payload["con_spec_version"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return 1
    return 1


def _fixed_flag(flag: int, spec: int) -> bool:
    """True when the atom is not free on every axis.

    Con files before spec 2 use 0 for free and 1 for fixed. Spec 2 and later store
    a three-bit mask of fixed axes, so 0 is free and 7 is fixed in x, y,
    and z. eOn writes that mask on the product geometries.
    """
    if spec < 2:
        if flag not in (0, 1):
            raise ValueError(f"con flag must be 0 or 1, found {flag}")
        return flag == 1
    if flag < 0 or flag > 7:
        raise ValueError(f"con flag must be a 3-bit mask, found {flag}")
    return flag != 0


def parse_con(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates and a fixed mask from a ``.con`` file."""
    spec = con_spec_version(text)
    rows: list[tuple[int, float, float, float, bool]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            flag = int(parts[3])
            atom_id = int(parts[4])
        except ValueError:
            continue
        rows.append((atom_id, x, y, z, _fixed_flag(flag, spec)))
    if not rows:
        raise ValueError("con file has no atoms")
    rows.sort()
    ids = [row[0] for row in rows]
    if ids != list(range(len(ids))):
        raise ValueError("con atom ids are not a dense range from 0")
    points = np.array([[row[1], row[2], row[3]] for row in rows], dtype=float)
    mask = np.array([row[4] for row in rows], dtype=bool)
    return points, mask


def mobile_indices(fixed: np.ndarray) -> np.ndarray:
    """Indexes of atoms that are free to move."""
    return np.flatnonzero(~fixed)


def mobile_rmsd(query: np.ndarray, reference: np.ndarray, mobile: np.ndarray) -> float:
    """Cartesian RMSD of the free atoms after removing their centroids."""
    if len(mobile) == 0:
        raise ValueError("no mobile atoms")
    left = query[mobile]
    right = reference[mobile]
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    delta = left - right
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def match_product(
    query: np.ndarray,
    products: Sequence[np.ndarray],
    mobile: np.ndarray,
    tolerance: float,
) -> int | None:
    """Index of the closest catalogue product inside ``tolerance`` angstroms."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    best: int | None = None
    best_rmsd = tolerance
    for index, product in enumerate(products):
        rmsd = mobile_rmsd(query, product, mobile)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best = index
    return best


def rate_distribution(channels: Sequence[tuple[int | None, float]]) -> dict[int, float]:
    """Sum rates by catalogue channel. ``None`` is the unmatched channel."""
    found: dict[int, float] = {}
    for channel, rate in channels:
        if rate < 0 or not math.isfinite(rate):
            raise ValueError("a channel rate must be finite and non-negative")
        if rate == 0:
            continue
        key = UNMATCHED if channel is None else int(channel)
        found[key] = found.get(key, 0.0) + float(rate)
    return found
