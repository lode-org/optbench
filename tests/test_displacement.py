"""The eOn listed-atom displacement on the saddle-search/pt-island geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import load_template, resolve_problem
from optbench.connection import periodic_lengths
from optbench.displacement import displace, neighbours
from optbench.rgsaddle_minmode import _stored
from optbench.surfaces import load_con

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def island():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    positions, fixed, cell = load_con(_stored(ROOT, "website/saddle-search/pt-island/pt-island.con"))
    lengths = periodic_lengths(cell, problem["potential"]["pbc"])
    search = load_template(ROOT, "rgsaddle/saddle-dimer-lbfgs")["ini"]["Search"]
    listed = [int(part) for part in str(search["displace_atoms"]).split(",")]
    return positions, ~fixed, lengths, listed, float(search["displace_radius"]), float(search["displace_magnitude"])


def _distance(points, atom, lengths):
    delta = points - points[atom]
    periodic = lengths > 0.0
    delta[:, periodic] -= lengths[periodic] * np.round(delta[:, periodic] / lengths[periodic])
    return np.linalg.norm(delta, axis=1)


def test_the_templates_state_the_archived_radius():
    for path in sorted((ROOT / "templates" / "rgsaddle").glob("saddle-*.toml")):
        search = load_template(ROOT, f"rgsaddle/{path.stem}")["ini"]["Search"]
        assert float(search["displace_radius"]) == 3.3
        assert search["displace_atoms"] == "0,1,2,3,4,5,6"
        assert float(search["displace_magnitude"]) == 0.1


def test_neighbours_are_the_atoms_inside_the_radius(island):
    positions, _free, lengths, listed, radius, _magnitude = island
    near = neighbours(positions, lengths, radius, listed)
    for atom in listed:
        distance = _distance(positions, atom, lengths)
        inside = {int(other) for other in np.flatnonzero(distance < radius)} - {atom}
        assert set(near[atom]) == inside
        assert inside


def test_a_displacement_moves_the_epicenter_and_its_free_neighbours(island):
    positions, free, lengths, listed, radius, magnitude = island
    near = neighbours(positions, lengths, radius, listed)
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        epicenter, shift, mode = displace(rng, free, listed, near, magnitude)
        seen.add(epicenter)
        group = [atom for atom in [epicenter, *near[epicenter]] if free[atom]]
        moved = set(np.flatnonzero(np.any(shift != 0.0, axis=1)).tolist())
        assert moved == set(group)
        assert set(np.flatnonzero(np.any(mode != 0.0, axis=1)).tolist()) == set(group)
        assert not np.any(shift[~free])
        assert np.linalg.norm(mode) == pytest.approx(1.0)
        assert abs(float(np.vdot(mode, shift)) / np.linalg.norm(shift)) < 1.0
    assert seen == set(listed)


def test_the_mode_seed_is_a_second_draw(island):
    positions, free, lengths, listed, radius, magnitude = island
    near = neighbours(positions, lengths, radius, listed)
    epicenter, shift, mode = displace(np.random.default_rng(7), free, listed, near, magnitude)
    replay = np.random.default_rng(7)
    assert int(listed[int(replay.integers(0, len(listed)))]) == epicenter
    group = [atom for atom in [epicenter, *near[epicenter]] if free[atom]]
    first = np.zeros_like(shift)
    second = np.zeros_like(shift)
    for atom in group:
        first[atom] = replay.normal(scale=magnitude, size=3)
    for atom in group:
        second[atom] = replay.normal(scale=magnitude, size=3)
    assert np.array_equal(shift, first)
    assert np.array_equal(mode, second / np.linalg.norm(second))
