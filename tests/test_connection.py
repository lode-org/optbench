"""The eOn link and repeat tests on the saddle-search/pt-island geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import resolve_problem
from optbench.connection import (
    ProcessTable,
    connected,
    per_atom_max,
    periodic_lengths,
    same_structure,
    tolerances,
)
from optbench.rgsaddle_minmode import _stored
from optbench.surfaces import load_con

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def island():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    positions, _fixed, cell = load_con(_stored(ROOT, "website/saddle-search/pt-island/pt-island.con"))
    lengths = periodic_lengths(cell, problem["potential"]["pbc"])
    distance_difference, energy_difference = tolerances(problem)
    hop = np.array([float(problem["potential"]["r0"]), 0.0, 0.0])
    return positions, lengths, distance_difference, energy_difference, hop


def _rmsd(moved, reference):
    delta = np.asarray(moved) - np.asarray(reference)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def test_the_problem_states_the_eon_defaults():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    assert tolerances(problem) == (0.1, 0.01)
    with pytest.raises(ValueError):
        tolerances({"id": "no-connection"})


def test_one_atom_hop_is_a_new_structure(island):
    positions, lengths, distance_difference, _energy, hop = island
    product = positions.copy()
    product[0] += hop
    assert _rmsd(product, positions) < 0.5
    assert per_atom_max(product, positions, lengths) == pytest.approx(hop[0])
    assert not same_structure(product, positions, lengths, distance_difference)
    assert connected(positions, positions.copy(), product, lengths, distance_difference)


def test_no_endpoint_back_is_not_connected(island):
    positions, lengths, distance_difference, _energy, hop = island
    plus = positions.copy()
    plus[0] += hop
    minus = positions.copy()
    minus[:7] += 1.5 * hop
    assert (_rmsd(plus, positions) <= 0.5) ^ (_rmsd(minus, positions) <= 0.5)
    assert not connected(positions, plus, minus, lengths, distance_difference)


def test_both_endpoints_back_is_not_connected(island):
    positions, lengths, distance_difference, _energy, _hop = island
    nudged = positions.copy()
    nudged[:7] += 0.5 * distance_difference
    assert not connected(positions, positions.copy(), nudged, lengths, distance_difference)


def test_the_comparison_is_strict_and_uses_the_minimum_image():
    lengths = periodic_lengths(np.diag([10.0, 10.0, 30.0]), [True, True, False])
    origin = np.zeros((1, 3))
    assert not same_structure(origin + [0.1, 0.0, 0.0], origin, lengths, 0.1)
    assert same_structure(origin + [10.0, 0.0, 0.0], origin, lengths, 0.1)
    assert not same_structure(origin + [0.0, 0.0, 30.0], origin, lengths, 0.1)


def test_a_repeat_needs_the_barrier_and_the_structure(island):
    positions, lengths, distance_difference, energy_difference, hop = island
    table = ProcessTable(lengths, distance_difference, energy_difference)
    table.add(positions, 0.5)
    assert table.repeat(positions, 0.5 + 0.5 * energy_difference)
    assert not table.repeat(positions, 0.5 + 2.0 * energy_difference)
    other = positions.copy()
    other[0] += hop
    assert not table.repeat(other, 0.5)
    assert len(table) == 1
