"""Link and novelty tests of a displacement search, as eOn judges them.

Two structures are the same when the largest per-atom minimum-image
displacement is strictly below ``distance_difference``. That is
``Matter::compare`` with atoms told apart by index: it returns
``distance_difference > perAtomNorm``. A search is connected when
exactly one minimized endpoint is the reactant minimum
(``ProcessSearchJob``, status ``STATUS_BAD_NOT_CONNECTED`` otherwise).

A connected saddle repeats a process already in the table when the
barriers differ by at most ``energy_difference`` and the saddle
geometries pass the same per-atom test (``AKMCState.find_repeat``).
Only connected saddles enter the table, as only a good result reaches
``AKMCState.add_process``.

The problem states both tolerances in ``[connection]``. There is no
fallback value, so a problem cannot be judged by a number it never
stated.
"""

from __future__ import annotations

import numpy as np


def tolerances(problem: dict) -> tuple[float, float]:
    """``distance_difference`` (angstrom) and ``energy_difference`` (eV)."""
    block = problem.get("connection")
    if not block or "distance_difference" not in block or "energy_difference" not in block:
        raise ValueError(f"{problem.get('id')} states no [connection] distance_difference and energy_difference")
    return float(block["distance_difference"]), float(block["energy_difference"])


def periodic_lengths(cell, pbc) -> np.ndarray:
    """Box lengths on the periodic axes and zero on the others."""
    lengths = np.zeros(3, dtype=np.float64)
    if cell is None:
        return lengths
    box = np.asarray(cell, dtype=np.float64)
    diagonal = np.diag(box.reshape(3, 3)) if box.size == 9 else box.reshape(-1)[:3]
    flags = [False, False, False] if pbc is None else [bool(flag) for flag in pbc]
    for axis in range(3):
        if flags[axis] and float(diagonal[axis]) > 0.0:
            lengths[axis] = float(diagonal[axis])
    return lengths


def per_atom_max(moved, reference, lengths) -> float:
    """Largest per-atom minimum-image distance between two geometries."""
    left = np.asarray(moved, dtype=np.float64).reshape(-1, 3)
    right = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if left.shape != right.shape:
        raise ValueError(f"geometry has shape {left.shape}, reference has {right.shape}")
    if left.size == 0:
        return 0.0
    delta = left - right
    box = np.asarray(lengths, dtype=np.float64)
    periodic = box > 0.0
    delta[:, periodic] -= box[periodic] * np.round(delta[:, periodic] / box[periodic])
    return float(np.max(np.linalg.norm(delta, axis=1)))


def same_structure(moved, reference, lengths, distance_difference: float) -> bool:
    return float(distance_difference) > per_atom_max(moved, reference, lengths)


def connected(reactant, plus, minus, lengths, distance_difference: float) -> bool:
    """Exactly one minimized endpoint is the reactant."""
    return same_structure(plus, reactant, lengths, distance_difference) ^ same_structure(
        minus, reactant, lengths, distance_difference
    )


class ProcessTable:
    """Connected saddles and their barriers, in the order they were found."""

    def __init__(self, lengths, distance_difference: float, energy_difference: float):
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.distance_difference = float(distance_difference)
        self.energy_difference = float(energy_difference)
        self.saddles: list[tuple[np.ndarray, float]] = []

    def __len__(self) -> int:
        return len(self.saddles)

    def repeat(self, saddle, barrier: float) -> bool:
        point = np.asarray(saddle, dtype=np.float64).reshape(-1, 3)
        for other, other_barrier in self.saddles:
            if abs(float(barrier) - other_barrier) > self.energy_difference:
                continue
            if same_structure(point, other, self.lengths, self.distance_difference):
                return True
        return False

    def add(self, saddle, barrier: float) -> None:
        point = np.array(np.asarray(saddle, dtype=np.float64).reshape(-1, 3), copy=True)
        self.saddles.append((point, float(barrier)))
