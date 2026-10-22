"""eOn per-atom link and novelty tests on the Pt island."""

from __future__ import annotations

import numpy as np
import pytest

from optbench.connection import (
    ProcessTable,
    connected,
    per_atom_max,
    periodic_lengths,
    same_structure,
    tolerances,
)


def _slab(count: int = 175, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, size=(count, 3))


def test_one_atom_hop_is_a_new_structure_under_rmsd_half():
    reactant = _slab()
    product = reactant.copy()
    product[3] += [2.77, 0.0, 0.0]
    rmsd = float(np.sqrt(np.mean(np.sum((product - reactant) ** 2, axis=1))))
    assert rmsd < 0.5
    lengths = np.zeros(3)
    assert not same_structure(product, reactant, lengths)
    assert connected(reactant, reactant + 0.01, product, lengths)
    assert not connected(reactant, reactant, reactant + 0.01, lengths)
    assert not connected(reactant, product, product, lengths)


def test_the_test_is_strict_at_the_tolerance():
    reference = np.zeros((2, 3))
    moved = reference.copy()
    moved[0, 0] = 0.1
    assert per_atom_max(moved, reference, np.zeros(3)) == pytest.approx(0.1)
    assert not same_structure(moved, reference, np.zeros(3), 0.1)
    assert same_structure(moved, reference, np.zeros(3), 0.1000001)


def test_minimum_image_on_periodic_axes_only():
    cell = np.diag([10.0, 12.0, 30.0])
    lengths = periodic_lengths(cell, [True, True, False])
    assert lengths.tolist() == [10.0, 12.0, 0.0]
    assert periodic_lengths(cell.reshape(-1).tolist(), [True, True, False]).tolist() == [10.0, 12.0, 0.0]
    reference = np.array([[0.05, 0.05, 0.05]])
    wrapped = np.array([[9.99, 11.99, 0.05]])
    assert per_atom_max(wrapped, reference, lengths) == pytest.approx(np.hypot(0.06, 0.06))
    lifted = np.array([[0.05, 0.05, 29.99]])
    assert per_atom_max(lifted, reference, lengths) == pytest.approx(29.94)


def test_repeat_needs_the_barrier_and_the_geometry():
    saddle = _slab()
    table = ProcessTable(np.zeros(3), 0.1, 0.01)
    table.add(saddle, 0.80)
    assert len(table) == 1
    assert table.repeat(saddle + 0.02, 0.805)
    assert not table.repeat(saddle + 0.02, 0.82)
    other = saddle.copy()
    other[5] += [1.4, 0.0, 0.0]
    assert not table.repeat(other, 0.80)


def test_tolerances_read_the_problem_block():
    assert tolerances({}) == (0.1, 0.01)
    assert tolerances({"connection": {"distance_difference": 0.2, "energy_difference": 0.05}}) == (0.2, 0.05)
