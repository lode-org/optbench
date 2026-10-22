"""Pt-island link and novelty tests, as eOn judges a process search.

Two minima are the same structure when the largest per-atom
minimum-image displacement is strictly below ``distance_difference``
(``Matter::compare`` with atoms distinguished by index, which returns
``distance_difference > perAtomNorm``). The eOn default is 0.1 angstrom.
A search is connected when exactly one minimized endpoint is the
reactant minimum.

A connected saddle is a repeat of an earlier one when the barriers
differ by at most ``energy_difference`` (default 0.01 eV) and the two
saddle geometries pass the same per-atom test (``AKMCState.find_repeat``).
"""

from __future__ import annotations

import numpy as np

DISTANCE_DIFFERENCE = 0.1
ENERGY_DIFFERENCE = 0.01


def tolerances(problem: dict | None) -> tuple[float, float]:
    """``distance_difference`` and ``energy_difference`` from ``[connection]``."""
    block = (problem or {}).get("connection") or {}
    return (
        float(block.get("distance_difference", DISTANCE_DIFFERENCE)),
        float(block.get("energy_difference", ENERGY_DIFFERENCE)),
    )


def periodic_lengths(cell, pbc) -> np.ndarray:
    """Box lengths on the periodic axes, zero on the others."""
    lengths = np.zeros(3, dtype=np.float64)
    if cell is None:
        return lengths
    box = np.asarray(cell, dtype=np.float64)
    diagonal = np.diag(box.reshape(3, 3)) if box.size == 9 else box.reshape(-1)[:3]
    flags = [True, True, True] if pbc is None else [bool(flag) for flag in pbc]
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


def same_structure(moved, reference, lengths, tolerance: float = DISTANCE_DIFFERENCE) -> bool:
    return float(tolerance) > per_atom_max(moved, reference, lengths)


def connected(reactant, plus, minus, lengths, tolerance: float = DISTANCE_DIFFERENCE) -> bool:
    """Exactly one minimized endpoint is the reactant."""
    return same_structure(plus, reactant, lengths, tolerance) ^ same_structure(minus, reactant, lengths, tolerance)


class ProcessTable:
    """Connected saddles in the order they were found."""

    def __init__(self, lengths, tolerance: float = DISTANCE_DIFFERENCE, energy_difference: float = ENERGY_DIFFERENCE):
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.tolerance = float(tolerance)
        self.energy_difference = float(energy_difference)
        self.saddles: list[tuple[np.ndarray, float]] = []

    def __len__(self) -> int:
        return len(self.saddles)

    def repeat(self, saddle, barrier: float) -> bool:
        point = np.asarray(saddle, dtype=np.float64).reshape(-1, 3)
        for other, other_barrier in self.saddles:
            if abs(float(barrier) - other_barrier) > self.energy_difference:
                continue
            if same_structure(point, other, self.lengths, self.tolerance):
                return True
        return False

    def add(self, saddle, barrier: float) -> None:
        self.saddles.append((np.array(np.asarray(saddle, dtype=np.float64).reshape(-1, 3), copy=True), float(barrier)))
