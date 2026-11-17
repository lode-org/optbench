"""The geomeTRIC runner on clusters and a dimer with closed-form answers.

The batched distance, angle and dihedral kernels are checked on shapes with
known values: a 3-4-5 triangle, right and straight angles, and a dihedral
whose value is the angle it was built with. Their derivatives are checked
against central differences. The functions that patch the library are run on
stand-in classes that have only the names and signatures they touch, so the
patching logic is covered without the library. Four atoms at the corners of a
regular tetrahedron at the pair minimum of ``4 (r**-12 - r**-6)`` have energy
-6, and two Morse atoms ``r0`` apart in a large cell have no force and a
stretching curvature of ``4 alpha**2 D0``.

geomeTRIC is not in the lockfile. Tests that run the library skip when it
cannot be imported and read the version it reports instead of stating one.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import signal
import statistics
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from optbench.counter import ForceCallCap
from optbench.potentials import MORSE_ALPHA, MORSE_CUTOFF, MORSE_D0, MORSE_R0, lennard_jones
from optbench.runners import geometric_min, rgmin_min
from optbench.score import BOOTSTRAP_SEED
from optbench.v2 import HOST_LABEL_ENV

LIBRARY_INSTALLED = importlib.util.find_spec("geometric") is not None
R_MIN = 2.0 ** (1.0 / 6.0)
CAP = 400
STUB_REVISION = "0123456789abcdef0123456789abcdef01234567"
GENERIC = np.array(
    [
        [0.1, 0.0, 0.2],
        [1.3, 0.2, -0.1],
        [1.9, 1.1, 0.4],
        [3.0, 1.0, 1.2],
        [2.2, -0.5, 1.5],
    ]
)


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _tetrahedron(edge: float = R_MIN, shake: float = 0.0) -> list[float]:
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) * edge / math.sqrt(8.0)
    return [float(value + shake * math.sin(1.7 * index + 0.3)) for index, value in enumerate(corners.reshape(-1))]


def _spectrum(edge: float) -> list[float]:
    stiffness, tension = _curvature(edge), _slope(edge) / edge
    return sorted([stiffness + 3.0 * tension] * 2 + [2.0 * stiffness + 2.0 * tension] * 3 + [4.0 * stiffness])


def _dimer(stretch: float = 0.0) -> list[float]:
    return [5.0, 5.0, 5.0, 5.0 + MORSE_R0 + stretch, 5.0, 5.0]


def _dimer_energy() -> float:
    cut = math.exp(-MORSE_ALPHA * (MORSE_CUTOFF - MORSE_R0))
    return -MORSE_D0 - MORSE_D0 * (cut * cut - 2.0 * cut)


def _central(values, xyz: np.ndarray, rows: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    out = np.zeros((len(rows), xyz.shape[0], 3))
    for atom in range(xyz.shape[0]):
        for axis in range(3):
            plus, minus = xyz.copy(), xyz.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            out[:, atom, axis] = (values(plus, rows)[0] - values(minus, rows)[0]) / (2.0 * step)
    return out


@pytest.fixture(autouse=True)
def _label(monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")


@pytest.fixture(autouse=True)
def _watchdog():
    """A search that never reaches a stop must fail the test, not hold the suite."""

    def expired(signum, frame):
        raise TimeoutError("the test ran past its time limit")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_distances_come_with_unit_vectors_that_cancel_over_a_pair():
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    lengths, deriv = geometric_min._distance_batch(xyz, pairs)
    assert lengths.tolist() == [5.0, 12.0, 13.0]
    assert deriv.shape == (3, 3, 3)
    assert deriv[0, 0].tolist() == pytest.approx([-0.6, -0.8, 0.0])
    assert deriv[0, 1].tolist() == pytest.approx([0.6, 0.8, 0.0])
    assert deriv[0, 2].tolist() == [0.0, 0.0, 0.0]
    assert np.abs(deriv.sum(axis=1)).max() < 1.0e-15
    assert deriv == pytest.approx(_central(geometric_min._distance_batch, xyz, pairs), abs=1e-8)


def test_a_right_angle_changes_as_the_arm_across_it_turns_away_from_the_other():
    xyz = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    values, deriv = geometric_min._angle_batch(xyz, np.array([[0, 1, 2]]))
    assert values[0] == pytest.approx(math.pi / 2.0, abs=1e-15)
    assert deriv[0, 0].tolist() == pytest.approx([0.0, -0.5, 0.0])
    assert deriv[0, 2].tolist() == pytest.approx([-1.0 / 3.0, 0.0, 0.0])
    assert deriv[0, 1].tolist() == pytest.approx([1.0 / 3.0, 0.5, 0.0])


def test_the_corners_of_an_equilateral_triangle_and_of_a_bent_chain_have_the_angles_that_made_them():
    height = math.sqrt(3.0) / 2.0
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, height, 0.0]])
    triples = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    values, _ = geometric_min._angle_batch(triangle, triples)
    assert values.tolist() == pytest.approx([math.pi / 3.0] * 3)
    bent = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0), 0.0]])
    values, _ = geometric_min._angle_batch(bent, np.array([[0, 1, 2]]))
    assert values[0] == pytest.approx(2.0 * math.pi / 3.0)


def test_angle_derivatives_match_central_differences_on_a_generic_shape():
    triples = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]])
    values, deriv = geometric_min._angle_batch(GENERIC, triples)
    assert np.all((values > 0.0) & (values < math.pi))
    assert deriv == pytest.approx(_central(geometric_min._angle_batch, GENERIC, triples), abs=1e-7)
    assert np.abs(deriv.sum(axis=1)).max() < 1.0e-12


@pytest.mark.parametrize(
    ("axis", "straight"),
    [
        ((1.0, 0.0, 0.0), True),
        ((1.0, 0.0, 0.0), False),
        ((1.0, -1.0, 1.0), True),
        ((-1.0, 1.0, 1.0), True),
        ((0.0, 0.0, 1.0), True),
        ((0.0, 0.0, 1.0), False),
    ],
)
def test_a_straight_or_folded_angle_is_pi_or_zero_with_derivatives_across_the_line(axis, straight):
    unit = np.asarray(axis) / np.linalg.norm(axis)
    first, last = 1.5, 2.5
    xyz = np.array([-first * unit if straight else first * unit, [0.0, 0.0, 0.0], last * unit])
    values, deriv = geometric_min._angle_batch(xyz, np.array([[0, 1, 2]]))
    assert values[0] == (math.pi if straight else 0.0)
    assert np.isfinite(deriv).all()
    assert abs(float(deriv[0, 0] @ unit)) < 1.0e-12 and abs(float(deriv[0, 2] @ unit)) < 1.0e-12
    assert np.linalg.norm(deriv[0, 0]) == pytest.approx(1.0 / first)
    assert np.linalg.norm(deriv[0, 2]) == pytest.approx(1.0 / last)
    assert np.abs(deriv.sum(axis=1)).max() < 1.0e-12


def _chain(angle: float) -> np.ndarray:
    """Four atoms along z with the last one turned by ``angle`` about the axis."""
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [math.cos(angle), math.sin(angle), 1.0]]
    )


@pytest.mark.parametrize("angle", [0.0, math.pi / 3.0, math.pi / 2.0, 2.0 * math.pi / 3.0, math.pi, -math.pi / 3.0])
def test_a_dihedral_is_the_angle_the_last_atom_was_turned_through(angle):
    values, _ = geometric_min._dihedral_batch(_chain(angle), np.array([[0, 1, 2, 3]]))
    assert values[0] == pytest.approx(angle, abs=1e-12)


def test_a_dihedral_changes_sign_in_a_mirror_and_keeps_its_derivative_across_the_axis():
    quads = np.array([[0, 1, 2, 3]])
    turned = _chain(1.0)
    mirrored = turned * np.array([1.0, -1.0, 1.0])
    first, deriv = geometric_min._dihedral_batch(turned, quads)
    second, mirrored_deriv = geometric_min._dihedral_batch(mirrored, quads)
    assert first[0] == pytest.approx(-second[0])
    assert mirrored_deriv[0] == pytest.approx(-deriv[0] * np.array([1.0, -1.0, 1.0]))
    assert np.abs(deriv[0, :, 2]).max() < 1.0e-12


def test_dihedral_derivatives_match_central_differences_and_cancel_over_the_shape():
    quads = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4]])
    values, deriv = geometric_min._dihedral_batch(GENERIC, quads)
    assert np.all(np.abs(values) < math.pi)
    assert deriv == pytest.approx(_central(geometric_min._dihedral_batch, GENERIC, quads), abs=1e-6)
    assert np.abs(deriv.sum(axis=1)).max() < 1.0e-12


def test_a_dihedral_with_a_straight_arm_has_no_derivative_on_that_side_and_none_is_not_a_number():
    straight_first = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 1.0, 0.3]])
    values, deriv = geometric_min._dihedral_batch(straight_first, np.array([[0, 1, 2, 3]]))
    assert np.isfinite(values).all() and np.isfinite(deriv).all()
    assert deriv[0, 0].tolist() == [0.0, 0.0, 0.0]
    assert np.linalg.norm(deriv[0, 3]) > 0.0
    straight_last = np.array([[0.0, 1.0, 0.3], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _, deriv = geometric_min._dihedral_batch(straight_last, np.array([[0, 1, 2, 3]]))
    assert np.isfinite(deriv).all()
    assert deriv[0, 3].tolist() == [0.0, 0.0, 0.0]
    assert np.linalg.norm(deriv[0, 0]) > 0.0


def test_a_turn_that_crosses_the_cut_is_wrapped_to_the_shorter_way_round():
    wrapped = geometric_min._wrap_periodic(
        np.array([3.0 * math.pi, -3.0 * math.pi, 0.5, math.pi + 0.1, -math.pi - 0.1, 6.0])
    )
    assert wrapped.tolist() == pytest.approx(
        [math.pi, -math.pi, 0.5, -math.pi + 0.1, math.pi - 0.1, 6.0 - 2.0 * math.pi]
    )


class Distance:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def value(self, xyz):
        points = np.asarray(xyz).reshape(-1, 3)
        return float(np.linalg.norm(points[self.a] - points[self.b]))

    def derivative(self, xyz):
        return _numeric(self.value, xyz)

    def calcDiff(self, xyz1, xyz2):
        return self.value(xyz1) - self.value(xyz2)

    def __eq__(self, other):
        return type(other) is type(self) and {self.a, self.b} == {other.a, other.b}

    __hash__ = None


class Angle:
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def value(self, xyz):
        points = np.asarray(xyz).reshape(-1, 3)
        u, v = points[self.a] - points[self.b], points[self.c] - points[self.b]
        return float(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)))

    def derivative(self, xyz):
        return _numeric(self.value, xyz)

    def calcDiff(self, xyz1, xyz2):
        return self.value(xyz1) - self.value(xyz2)

    def __eq__(self, other):
        return type(other) is type(self) and (self.a, self.b, self.c) == (other.a, other.b, other.c)

    __hash__ = None


class Dihedral:
    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a, b, c, d

    def value(self, xyz):
        p = np.asarray(xyz).reshape(-1, 3)
        one, two, three = p[self.b] - p[self.a], p[self.c] - p[self.b], p[self.d] - p[self.c]
        return float(
            np.arctan2(
                np.linalg.norm(two) * one @ np.cross(two, three),
                np.cross(two, three) @ np.cross(one, two),
            )
        )

    def derivative(self, xyz):
        return _numeric(self.value, xyz)

    def calcDiff(self, xyz1, xyz2):
        turn = self.value(xyz1) - self.value(xyz2)
        return (turn + math.pi) % (2.0 * math.pi) - math.pi

    def __eq__(self, other):
        return type(other) is type(self) and (self.a, self.b, self.c, self.d) == (other.a, other.b, other.c, other.d)

    __hash__ = None


class CartesianX:
    def __init__(self, a):
        self.a = a

    def value(self, xyz):
        return float(np.asarray(xyz).reshape(-1)[3 * self.a])

    def derivative(self, xyz):
        out = np.zeros((len(np.asarray(xyz).reshape(-1)) // 3, 3))
        out[self.a, 0] = 1.0
        return out

    def calcDiff(self, xyz1, xyz2):
        return self.value(xyz1) - self.value(xyz2)

    def __eq__(self, other):
        return type(other) is type(self) and self.a == other.a

    __hash__ = None


def _numeric(function, xyz, step: float = 1.0e-6):
    flat = np.asarray(xyz, dtype=float).reshape(-1)
    out = np.zeros(flat.size)
    for slot in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[slot] += step
        minus[slot] -= step
        out[slot] = (function(plus) - function(minus)) / (2.0 * step)
    return out.reshape(-1, 3)


@pytest.fixture
def library(monkeypatch):
    """Stand-ins for the geomeTRIC classes the installers patch."""
    seen = {"calculate": 0, "derivatives": 0, "calcDiff": 0, "delete": 0, "topology": []}

    class PrimitiveInternalCoordinates:
        def __init__(self, internals=()):
            self.Internals = list(internals)

        def calculate(self, xyz):
            seen["calculate"] += 1
            return np.array([item.value(xyz) for item in self.Internals])

        def derivatives(self, xyz):
            seen["derivatives"] += 1
            return np.array([item.derivative(xyz) for item in self.Internals])

        def calcDiff(self, xyz1, xyz2):
            seen["calcDiff"] += 1
            return np.array([item.calcDiff(xyz1, xyz2) for item in self.Internals])

        def add(self, dof):
            if dof not in self.Internals:
                self.Internals.append(dof)

        def delete(self, dof):
            seen["delete"] += 1
            self.Internals = [item for item in self.Internals if item != dof]

    class Molecule:
        def __init__(self, read_bonds=False):
            self.top_settings = {"read_bonds": read_bonds}

        def build_topology(self, force_bonds=True, bond_order=0.0, metal_bo_factor=0.5, **kwargs):
            seen["topology"].append((force_bonds, bond_order, metal_bo_factor, dict(kwargs)))

    package = types.ModuleType("geometric")
    package.__path__ = []
    internal = types.ModuleType("geometric.internal")
    internal.PrimitiveInternalCoordinates = PrimitiveInternalCoordinates
    molecule = types.ModuleType("geometric.molecule")
    molecule.Molecule = Molecule
    monkeypatch.setitem(sys.modules, "geometric", package)
    monkeypatch.setitem(sys.modules, "geometric.internal", internal)
    monkeypatch.setitem(sys.modules, "geometric.molecule", molecule)
    return types.SimpleNamespace(
        coordinates=PrimitiveInternalCoordinates, molecule=Molecule, seen=seen
    )


def test_primitives_are_grouped_by_kind_with_their_slots_and_atoms():
    ic = types.SimpleNamespace(
        Internals=[Distance(0, 1), CartesianX(2), Angle(0, 1, 2), Dihedral(0, 1, 2, 3), Distance(2, 3)]
    )
    grouped = geometric_min._groups(ic)
    assert sorted(grouped) == ["Angle", "Dihedral", "Distance"]
    slots, pairs = grouped["Distance"]
    assert slots.tolist() == [0, 4] and pairs.tolist() == [[0, 1], [2, 3]]
    assert grouped["Angle"][0].tolist() == [2] and grouped["Angle"][1].tolist() == [[0, 1, 2]]
    assert grouped["Dihedral"][0].tolist() == [3] and grouped["Dihedral"][1].tolist() == [[0, 1, 2, 3]]
    assert geometric_min._groups(types.SimpleNamespace(Internals=[CartesianX(0)])) == {}
    assert geometric_min._batchable(ic) is True
    assert geometric_min._batchable(types.SimpleNamespace(Internals=[CartesianX(0)])) is False
    assert geometric_min._batchable(types.SimpleNamespace(Internals=[])) is False


def _mixed(library) -> tuple:
    internals = [Distance(0, 1), CartesianX(2), Angle(0, 1, 2), Dihedral(0, 1, 2, 3), Distance(3, 4), Angle(2, 3, 4)]
    return library.coordinates(internals), GENERIC.reshape(-1)


def test_the_batched_blocks_give_what_the_per_primitive_methods_gave_in_the_same_order(library):
    ic, xyz = _mixed(library)
    reference_values = ic.calculate(xyz)
    reference_deriv = ic.derivatives(xyz)
    shifted = xyz + 0.01 * np.sin(np.arange(xyz.size))
    reference_diff = ic.calcDiff(shifted, xyz)
    geometric_min.install_batch_primitives()
    before = dict(library.seen)
    assert ic.calculate(xyz) == pytest.approx(reference_values, abs=1e-12)
    assert ic.derivatives(xyz) == pytest.approx(reference_deriv, abs=1e-6)
    assert ic.calcDiff(shifted, xyz) == pytest.approx(reference_diff, abs=1e-12)
    assert library.seen["calculate"] == before["calculate"]
    assert library.seen["derivatives"] == before["derivatives"] and library.seen["calcDiff"] == before["calcDiff"]


def test_a_dihedral_that_crosses_the_cut_between_two_shapes_is_wrapped_in_the_batched_difference(library):
    ic = library.coordinates([Dihedral(0, 1, 2, 3), Distance(0, 1)])
    geometric_min.install_batch_primitives()
    near_pi = _chain(3.0).reshape(-1)
    past_pi = _chain(-3.0).reshape(-1)
    diff = ic.calcDiff(near_pi, past_pi)
    assert diff[0] == pytest.approx(6.0 - 2.0 * math.pi)
    assert diff[1] == pytest.approx(0.0, abs=1e-15)


SUBSETS = {
    "distances only": [Distance(0, 1), Distance(3, 4)],
    "angles only": [Angle(0, 1, 2), Angle(2, 3, 4)],
    "dihedrals only": [Dihedral(0, 1, 2, 3), Dihedral(1, 2, 3, 4)],
    "no distances": [Angle(0, 1, 2), CartesianX(1), Dihedral(0, 1, 2, 3)],
    "no angles": [Distance(0, 1), Dihedral(0, 1, 2, 3), CartesianX(0)],
    "no dihedrals": [Distance(0, 1), Angle(0, 1, 2), CartesianX(3)],
}


@pytest.mark.parametrize("name", list(SUBSETS))
def test_a_set_with_only_some_kinds_of_primitive_is_batched_to_the_same_blocks(library, name):
    ic = library.coordinates(list(SUBSETS[name]))
    xyz = GENERIC.reshape(-1)
    shifted = xyz + 0.01 * np.sin(np.arange(xyz.size))
    reference = (ic.calculate(xyz), ic.derivatives(xyz), ic.calcDiff(shifted, xyz))
    geometric_min.install_batch_primitives()
    assert ic.calculate(xyz) == pytest.approx(reference[0], abs=1e-12)
    assert ic.derivatives(xyz) == pytest.approx(reference[1], abs=1e-6)
    assert ic.calcDiff(shifted, xyz) == pytest.approx(reference[2], abs=1e-12)


def test_a_set_with_nothing_to_batch_goes_to_the_original_methods(library):
    ic = library.coordinates([CartesianX(0), CartesianX(1)])
    xyz = GENERIC.reshape(-1)
    geometric_min.install_batch_primitives()
    assert ic.calculate(xyz).tolist() == [float(xyz[0]), float(xyz[3])]
    assert ic.derivatives(xyz).shape == (2, 5, 3)
    assert ic.calcDiff(xyz + 1.0, xyz).tolist() == pytest.approx([1.0, 1.0])
    assert (library.seen["calculate"], library.seen["derivatives"], library.seen["calcDiff"]) == (1, 1, 1)


def test_installing_the_batches_twice_does_not_wrap_them_again(library):
    geometric_min.install_batch_primitives()
    once = (library.coordinates.calculate, library.coordinates.derivatives, library.coordinates.calcDiff)
    geometric_min.install_batch_primitives()
    assert (library.coordinates.calculate, library.coordinates.derivatives, library.coordinates.calcDiff) == once


def test_a_primitive_that_is_already_in_the_set_is_not_added_again(library):
    geometric_min.install_fast_add()
    ic = library.coordinates()
    for dof in (Distance(0, 1), Distance(1, 0), Distance(0, 2), Angle(0, 1, 2), Angle(0, 1, 2), CartesianX(1)):
        ic.add(dof)
    assert [type(item).__name__ for item in ic.Internals] == ["Distance", "Distance", "Angle", "CartesianX"]
    assert ic.Internals[0] == Distance(1, 0)
    keys = set(ic._buckets)
    assert (Distance, frozenset({0, 1})) in keys and (Distance, frozenset({0, 2})) in keys
    assert (Angle, frozenset({0, 1, 2})) in keys and (CartesianX, frozenset({1})) in keys


def test_primitives_that_name_their_atoms_in_a_list_or_not_by_index_still_get_a_bucket(library):
    class Rotation:
        def __init__(self, a):
            self.a = a

        def _key(self):
            return tuple(sorted(self.a)) if isinstance(self.a, (list, tuple)) else self.a

        def __eq__(self, other):
            return type(other) is type(self) and self._key() == other._key()

        __hash__ = None

    geometric_min.install_fast_add()
    ic = library.coordinates()
    ic.add(Rotation([2, 0, 1]))
    ic.add(Rotation((0, 1, 2)))
    ic.add(Rotation("water"))
    ic.add(Rotation([0, 1, 2]))
    assert [item.a for item in ic.Internals] == [[2, 0, 1], "water"]
    assert (Rotation, frozenset({0, 1, 2})) in ic._buckets
    assert (Rotation, frozenset({repr("water")})) in ic._buckets


def test_a_deleted_primitive_can_be_added_again(library):
    geometric_min.install_fast_add()
    ic = library.coordinates()
    ic.add(Distance(0, 1))
    ic.add(Distance(1, 2))
    ic.delete(Distance(0, 1))
    assert library.seen["delete"] == 1
    assert [(item.a, item.b) for item in ic.Internals] == [(1, 2)]
    ic.add(Distance(0, 1))
    assert sorted((item.a, item.b) for item in ic.Internals) == [(0, 1), (1, 2)]
    ic.delete(Distance(5, 6))
    assert len(ic.Internals) == 2
    empty = library.coordinates()
    empty.delete(Distance(0, 1))
    assert empty.Internals == []


def test_installing_the_hashed_add_twice_does_not_wrap_the_delete_again(library):
    geometric_min.install_fast_add()
    first = library.coordinates.delete
    geometric_min.install_fast_add()
    assert library.coordinates.delete is first
    ic = library.coordinates()
    ic.add(Distance(0, 1))
    ic.delete(Distance(0, 1))
    assert library.seen["delete"] == 1


def test_a_rebuild_keeps_the_bonds_that_were_read_and_discovers_them_otherwise(library):
    geometric_min.install_keep_read_bonds()
    kept = library.molecule(read_bonds=True)
    kept.build_topology(force_bonds=True, bond_order=0.3, extra=1)
    open_list = library.molecule(read_bonds=False)
    open_list.build_topology(metal_bo_factor=0.9)
    assert library.seen["topology"] == [
        (False, 0.3, 0.5, {"extra": 1}),
        (True, 0.0, 0.9, {}),
    ]


def test_installing_the_bond_keeper_twice_does_not_wrap_it_again(library):
    geometric_min.install_keep_read_bonds()
    first = library.molecule.build_topology
    geometric_min.install_keep_read_bonds()
    assert library.molecule.build_topology is first


def test_the_nearest_neighbours_of_points_on_a_line_are_the_bonds_between_them():
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0], [4.5, 0.0, 0.0]]
    assert geometric_min.knearest(line, 1, None) == [(0, 1), (1, 2), (2, 3)]
    assert geometric_min.knearest(line, 2, None) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    assert geometric_min.knearest(line, 10, None) == [(i, j) for i in range(4) for j in range(i + 1, 4)]
    assert geometric_min.knearest(np.asarray(line).reshape(-1), 1, None) == [(0, 1), (1, 2), (2, 3)]


def test_no_neighbours_are_asked_for_of_one_point_or_of_none():
    assert geometric_min.knearest([[0.0, 0.0, 0.0]], 2, None) == []
    assert geometric_min.knearest([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0, None) == []
    assert geometric_min.knearest([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], -1, None) == []


def test_a_box_makes_two_atoms_either_side_of_the_edge_neighbours_and_no_box_does_not():
    atoms = [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert geometric_min.knearest(atoms, 1, [10.0, 10.0, 10.0]) == [(0, 1), (0, 2)]
    assert geometric_min.knearest(atoms, 1, None) == [(0, 2), (1, 2)]
    assert geometric_min.knearest(atoms, 1, [10.0, 10.0]) == [(0, 2), (1, 2)]


def test_one_element_name_is_repeated_and_a_list_must_have_one_per_atom():
    assert geometric_min._element_names(3, "H") == ["H", "H", "H"]
    assert geometric_min._element_names(2, ["H", "Pt"]) == ["H", "Pt"]
    assert geometric_min._element_names(2, ("Pt", "Pt")) == ["Pt", "Pt"]
    with pytest.raises(ValueError, match="3 elements for 4 atoms"):
        geometric_min._element_names(4, ["H", "H", "H"])


def test_a_problem_names_its_potential_or_is_refused():
    assert geometric_min._kind("minimization/lj38") == "lj"
    assert geometric_min._kind("minimization/lj38-noise") == "lj"
    assert geometric_min._kind("minimization/morse-bulk") == "morse"
    with pytest.raises(ValueError, match="minimization/other has no geomeTRIC potential"):
        geometric_min._kind("minimization/other")


def test_the_section_gives_cartesian_no_neighbours_and_tric_two_unless_it_says_otherwise():
    base = {"id": "geometric/test"}
    assert geometric_min._section({**base, "geometric": {"coordsys": "cart", "element": "H"}}) == {
        "coordsys": "cart",
        "element": "H",
        "neighbors": 0,
    }
    assert geometric_min._section({**base, "geometric": {"coordsys": "tric", "element": "Pt"}})["neighbors"] == 2
    assert geometric_min._section({**base, "geometric": {"coordsys": "tric", "element": "H", "neighbors": 3}})["neighbors"] == 3
    assert geometric_min._section({**base, "solver": {"coordsys": "cart", "element": "H", "neighbors": 1}})["neighbors"] == 1
    with pytest.raises(ValueError, match="geometric/test needs geometric.coordsys cart or tric"):
        geometric_min._section({**base, "geometric": {"coordsys": "dlc", "element": "H"}})
    with pytest.raises(ValueError, match="geometric/test needs geometric.coordsys cart or tric"):
        geometric_min._section(base)
    with pytest.raises(ValueError, match="geometric/test needs geometric.element"):
        geometric_min._section({**base, "geometric": {"coordsys": "cart"}})


def test_a_progress_line_is_printed_only_when_asked_and_when_there_is_a_gradient(monkeypatch, capsys):
    engine = types.SimpleNamespace(problem_gradient=np.array([3.0, 4.0, 0.0]))
    optimizer = types.SimpleNamespace(trust=0.25, IC=Distance(0, 1))
    job = {"start": "initial_4"}
    monkeypatch.delenv("OPTBENCH_GEOMETRIC_PROGRESS", raising=False)
    geometric_min._report_step(job, 7, engine, optimizer)
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("OPTBENCH_GEOMETRIC_PROGRESS", "1")
    geometric_min._report_step(job, 7, engine, optimizer)
    assert capsys.readouterr().out == "initial_4 step=7 norm=5 trust=0.25 ic=Distance\n"
    engine.problem_gradient = None
    geometric_min._report_step(job, 8, engine, optimizer)
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("OPTBENCH_GEOMETRIC_PROGRESS", "0")
    engine.problem_gradient = np.array([1.0])
    geometric_min._report_step(job, 9, engine, optimizer)
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("OPTBENCH_GEOMETRIC_PROGRESS", "1")
    engine.problem_gradient = np.array([0.1234567891, 0.0, 0.0])
    optimizer.trust = 0.123456
    geometric_min._report_step(job, 10, engine, optimizer)
    assert capsys.readouterr().out == "initial_4 step=10 norm=0.123457 trust=0.1235 ic=Distance\n"


def test_the_library_loggers_are_set_to_report_errors_only():
    names = ("geometric", "geometric.nifty")
    before = {name: logging.getLogger(name).level for name in names}
    try:
        for name in names:
            logging.getLogger(name).setLevel(logging.DEBUG)
        geometric_min._silence()
        assert [logging.getLogger(name).level for name in names] == [logging.ERROR, logging.ERROR]
    finally:
        for name, level in before.items():
            logging.getLogger(name).setLevel(level)


def test_the_run_pins_one_thread_for_the_numerical_libraries(monkeypatch):
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    for name in names:
        monkeypatch.setenv(name, "8")
    geometric_min._init_worker()
    assert [geometric_min.os.environ[name] for name in names] == ["1"] * len(names)


def test_float_fields_print_whole_numbers_without_a_point_and_others_to_twelve_digits():
    assert geometric_min._format_field(12.0) == "12"
    assert geometric_min._format_field(-6.0) == "-6"
    assert geometric_min._format_field(2.0e12) == "2000000000000"
    assert geometric_min._format_field(0.123456789012345) == "0.123456789012"
    assert geometric_min._format_field(7) == "7"
    assert geometric_min._format_field("a b") == "a b"


def test_a_dat_file_lists_the_columns_the_coordinate_systems_and_the_notes():
    template = {
        "id": "geometric/test",
        "code": "geomeTRIC",
        "algorithm": "TRIC",
        "problem": "minimization/lj38",
        "tuning": "default",
        "note": "first\nsecond",
    }
    summary = {"force_calls": 25, "force_calls_min": 24, "force_calls_max": 27, "nfailed": 0, "median": 25, "partial": True,
               "energy_min": -6.0, "energy_max": -5.5}
    metrics = {"par2_mean": 25.3333333333, "wilson_low": 0.5, "wilson_high": 1.0, "n": 3, "successes": 3, "nfailed": 0}
    text = geometric_min._dat(
        summary, template, metrics, "abc", "1.1.1", {"DelocalizedInternalCoordinates": 1, "CartesianCoordinates": 2}
    )
    lines = text.splitlines()
    assert "# final coordinate system: CartesianCoordinates 2, DelocalizedInternalCoordinates 1" in lines
    assert "# first second" in lines
    assert any(line.startswith("# LJ38 index removes the six rigid modes") for line in lines)
    assert [line for line in lines if not line.startswith("#")] == [
        "force_calls 25",
        "force_calls_min 24",
        "force_calls_max 27",
        "nfailed 0",
        "median 25",
        "code geomeTRIC",
        "code_version 1.1.1",
        "commit abc",
        "algorithm TRIC",
        "template geometric/test",
        "tuning default",
        "counter harness",
        "problem minimization/lj38",
        "par2_mean 25.3333333333",
        "wilson_low 0.5",
        "wilson_high 1",
        "v2_nfailed 0",
        "v2_successes 3",
        "n 3",
        "partial true",
        "energy_min -6",
        "energy_max -5.5",
    ]
    bare = geometric_min._dat({}, {"id": "geometric/test", "problem": "minimization/morse-bulk"}, {}, "abc", "1", {})
    assert not any(line.startswith("# LJ38") or line.startswith("# final coordinate") for line in bare.splitlines())
    assert "partial" not in bare


def _optimizer(point, stopped=True, error="", steps=7, rebuilds=2, system="CartesianCoordinates", probe=None):
    def optimize(job, evaluate, stop):
        if probe is not None:
            probe["job"] = job
        value, gradient = evaluate(job["coords"])
        if probe is not None:
            probe["evaluated"] = (value, gradient)
            probe["stop"] = stop(gradient)
        return {
            "point": None if point is None else list(point),
            "stopped": stopped,
            "error": error,
            "steps": steps,
            "rebuilds": rebuilds,
            "coordinate_system": system,
            "bonds": 0,
        }

    return optimize


def _job(**overrides) -> dict:
    job = {
        "index": 2,
        "start": "initial_2",
        "coords": _tetrahedron(),
        "box": None,
        "kind": "lj",
        "coordsys": "cart",
        "neighbors": 0,
        "element": "H",
        "gtol": 1.0e-6,
        "cap": CAP,
    }
    job.update(overrides)
    return job


def test_a_stopped_search_at_the_tetrahedron_is_a_success_with_the_closed_form_spectrum(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(_tetrahedron(), probe=seen))
    record = geometric_min._minimize_job(_job())
    assert record["success"] is True
    assert (record["failure"], record["reasons"], record["converged"]) == (None, [], True)
    assert record["energy"] == pytest.approx(-6.0, abs=1e-12)
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx(_spectrum(R_MIN), rel=1e-6)
    assert (record["force_calls"], record["verify_calls"], record["hessian_calls"]) == (1, 0, 0)
    assert (record["start"], record["index"], record["expected_index"]) == ("initial_2", 2, 0)
    assert (record["schema"], record["counter"], record["comparable"]) == ("optbench.start/v2", "harness", True)
    assert (record["geometric_steps"], record["rebuilds"], record["coordinate_system"]) == (7, 2, "CartesianCoordinates")
    assert record["seconds"] >= 0.0 and "error" not in record
    assert seen["job"]["coordsys"] == "cart" and seen["stop"] is True
    assert seen["evaluated"][0] == pytest.approx(-6.0, abs=1e-12)


def test_the_search_is_given_the_gradient_it_has_to_judge_in_problem_units(monkeypatch):
    seen: dict = {}
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=False, probe=seen))
    geometric_min._minimize_job(_job(coords=point))
    energy, gradient = lennard_jones(point)
    assert seen["evaluated"][0] == pytest.approx(energy)
    assert np.asarray(seen["evaluated"][1]) == pytest.approx(gradient)
    assert seen["stop"] is False


def test_a_force_norm_exactly_at_the_stop_is_stopped_and_one_above_it_is_not(monkeypatch):
    point = _tetrahedron(shake=0.03)
    norm = float(np.linalg.norm(lennard_jones(point)[1]))
    seen: dict = {}
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, probe=seen))
    geometric_min._minimize_job(_job(coords=point, gtol=norm))
    assert seen["stop"] is True
    geometric_min._minimize_job(_job(coords=point, gtol=norm * (1.0 - 1.0e-9)))
    assert seen["stop"] is False


def test_a_search_still_above_the_force_stop_under_the_cap_is_a_divergence(monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=False))
    record = geometric_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["reasons"], record["converged"]) == ("divergence", ["force_norm"], False)
    assert record["force_norm"] > 1.0e-6


def test_a_driver_that_claims_a_stop_away_from_the_force_stop_is_not_converged(monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=True))
    record = geometric_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["reasons"], record["converged"]) == ("divergence", ["force_norm"], False)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(_tetrahedron(), stopped=True, error="ForceCallCap: force-call cap 4"))
    record = geometric_min._minimize_job(_job())
    assert (record["failure"], record["converged"]) == ("cap", False)


def test_reaching_the_force_stop_on_the_last_permitted_call_is_a_cap_failure(monkeypatch):
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(_tetrahedron()))
    record = geometric_min._minimize_job(_job(cap=1))
    assert record["force_calls"] == 1
    assert (record["failure"], record["reasons"], record["converged"]) == ("cap", ["cap"], False)
    assert geometric_min._minimize_job(_job(cap=2))["success"] is True


def test_a_search_that_spent_the_budget_is_a_cap_failure_whatever_else_it_reports(monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=False, error="ForceCallCap: force-call cap 4"))
    record = geometric_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["error"]) == ("cap", "ForceCallCap: force-call cap 4")
    assert record["hessian_index"] is not None
    for error in ("force-call cap 4", "ForceCallCap"):
        monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=False, error=error))
        record = geometric_min._minimize_job(_job(coords=point))
        assert (record["failure"], record["hessian_index"] is not None) == ("cap", True)


def test_a_search_that_stopped_at_a_cap_of_one_has_no_second_call_to_verify_with(monkeypatch):
    point = _tetrahedron(shake=0.03)

    def optimize(job, evaluate, stop):
        evaluate(job["coords"])
        return {"point": [value + 0.01 for value in point], "stopped": False, "error": "", "steps": 0}

    monkeypatch.setattr(geometric_min, "optimize", optimize)
    record = geometric_min._minimize_job(_job(coords=point, cap=1))
    assert (record["force_calls"], record["verify_calls"]) == (1, 1)
    assert record["failure"] == "cap"


def test_an_error_from_the_search_is_a_crash_and_leaves_no_index(monkeypatch):
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=False, error="ValueError: lost"))
    record = geometric_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["error"]) == ("crash", "ValueError: lost")
    assert record["hessian_index"] is None and record["energy"] is not None


def test_a_search_that_raises_is_a_crash_with_no_point_and_the_calls_it_made(monkeypatch):
    def optimize(job, evaluate, stop):
        evaluate(job["coords"])
        raise RuntimeError("coordinate rebuild failed")

    monkeypatch.setattr(geometric_min, "optimize", optimize)
    record = geometric_min._minimize_job(_job(coords=_tetrahedron(shake=0.03)))
    assert (record["failure"], record["error"]) == ("crash", "RuntimeError: coordinate rebuild failed")
    assert record["force_calls"] == 1 and record["energy"] is None
    assert record["geometric_steps"] is None and record["coordinate_system"] == ""


def test_a_search_with_no_point_that_hit_the_cap_is_counted_with_its_calls(monkeypatch):
    def optimize(job, evaluate, stop):
        raise ForceCallCap("force-call cap 0")

    monkeypatch.setattr(geometric_min, "optimize", optimize)
    record = geometric_min._minimize_job(_job(cap=0))
    assert record["error"] == "ForceCallCap: force-call cap 0"
    assert record["force_calls"] == 0 and record["energy"] is None


def test_a_final_point_the_potential_cannot_evaluate_is_a_crash_with_no_energy(monkeypatch):
    odd = [0.0, 0.0, 0.0, 1.0, 0.0]
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(odd))
    record = geometric_min._minimize_job(_job())
    assert (record["failure"], record["converged"]) == ("crash", False)
    assert record["error"].startswith("ValueError")
    assert (record["energy"], record["force_norm"], record["hessian_index"]) == (None, None, None)
    assert record["force_calls"] == 1
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(odd, stopped=False, error="RuntimeError: first"))
    assert geometric_min._minimize_job(_job())["error"] == "RuntimeError: first"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("where", ["one coordinate", "every coordinate"])
def test_a_final_point_that_is_not_finite_is_a_divergence_with_no_energy_force_or_index(monkeypatch, bad, where):
    point = _tetrahedron()
    if where == "one coordinate":
        point[3] = bad
    else:
        point = [bad] * len(point)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=False))
    with np.errstate(all="ignore"):
        record = geometric_min._minimize_job(_job())
    assert (record["failure"], record["reasons"], record["converged"]) == ("divergence", ["divergence"], False)
    assert (record["energy"], record["force_norm"], record["hessian_index"]) == (None, None, None)
    assert record["min_eigenvalues"] == []
    assert (record["force_calls"], record["verify_calls"]) == (1, 0)
    assert "error" not in record


def test_a_hessian_that_cannot_be_diagonalised_is_a_crash_that_keeps_the_energy_and_force(monkeypatch):
    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(rgmin_min, "_index_at", broken)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(_tetrahedron()))
    record = geometric_min._minimize_job(_job())
    assert (record["failure"], record["error"]) == ("crash", "LinAlgError: Eigenvalues did not converge")
    assert record["energy"] == pytest.approx(-6.0, abs=1e-12) and record["hessian_index"] is None
    point = _tetrahedron(shake=0.03)
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(point, stopped=False, error="ForceCallCap: force-call cap 4"))
    record = geometric_min._minimize_job(_job(coords=point))
    assert (record["failure"], record["error"]) == ("cap", "ForceCallCap: force-call cap 4")
    assert record["hessian_index"] is None


def test_a_stationary_square_is_a_wrong_stationary_point(monkeypatch):
    low, high = 1.0, 1.2
    for _ in range(200):
        middle = (low + high) / 2.0
        slope = 4.0 * _slope(middle) + 2.0 * math.sqrt(2.0) * _slope(math.sqrt(2.0) * middle)
        low, high = (middle, high) if slope < 0.0 else (low, middle)
    half = (low + high) / 4.0
    square = [-half, -half, 0.0, half, -half, 0.0, half, half, 0.0, -half, half, 0.0]
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(square))
    record = geometric_min._minimize_job(_job(coords=square))
    assert (record["failure"], record["reasons"], record["converged"]) == ("wrong_stationary", ["hessian_index"], True)
    assert record["hessian_index"] == 2


def test_the_morse_dimer_at_its_pair_minimum_is_a_success_with_one_stretching_curvature(monkeypatch):
    box = [20.0, 20.0, 20.0]
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(_dimer(), system="DelocalizedInternalCoordinates"))
    record = geometric_min._minimize_job(_job(kind="morse", coords=_dimer(), box=box, element="Pt", coordsys="tric"))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(_dimer_energy(), abs=1e-12)
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx([0.0] * 5 + [4.0 * MORSE_ALPHA**2 * MORSE_D0], abs=1e-8)
    assert record["coordinate_system"] == "DelocalizedInternalCoordinates"
    monkeypatch.setattr(geometric_min, "optimize", _optimizer(_dimer(0.05), stopped=False))
    stretched = geometric_min._minimize_job(_job(kind="morse", coords=_dimer(0.05), box=box))
    assert stretched["failure"] == "divergence" and stretched["force_norm"] > 1.0e-3


def _con(coords: list[float], label: str = "H") -> str:
    rows = [coords[offset : offset + 3] for offset in range(0, len(coords), 3)]
    head = ["a label", "0.0000 TIME", "40.0 40.0 40.0", "90.0 90.0 90.0", "0 0", "0 0 0"]
    atoms = [f"{x!r} {y!r} {z!r} 0 {index}" for index, (x, y, z) in enumerate(rows)]
    return "\n".join([*head, "1", str(len(rows)), "1.008", label, "Coordinates of Component 1", *atoms, ""])


def _starts(folder: Path, shakes=(0.01, 0.03, 0.05)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for index, shake in enumerate(shakes):
        (folder / f"initial_{index}.con").write_text(_con(_tetrahedron(shake=shake)), encoding="utf-8")
    return folder


def _problem(**sample) -> dict:
    problem = {"id": "minimization/lj38", "convergence": {"force_norm": 1.0e-6, "max_force_calls": CAP}}
    if sample:
        problem["sample"] = sample
    return problem


def _template(**changes) -> dict:
    template = {
        "id": "geometric/test",
        "engine": "geometric",
        "problem": "minimization/lj38",
        "code": "geomeTRIC",
        "algorithm": "TRIC",
        "tuning": "default",
        "commit": STUB_REVISION,
        "code_version": "9.8.7",
        "geometric": {"coordsys": "tric", "element": "H"},
    }
    template.update(changes)
    return template


@pytest.fixture
def stub_library(monkeypatch):
    stub = types.ModuleType("geometric")
    stub.__version__ = "9.8.7"
    stub.__git_revision__ = STUB_REVISION
    monkeypatch.setitem(sys.modules, "geometric", stub)
    return stub


def _run(tmp_path: Path, template: dict | None = None, problem: dict | None = None, **options) -> dict:
    options.setdefault("workers", 1)
    options.setdefault("geometries", tmp_path / "cons")
    options.setdefault("output", tmp_path / "out" / "bench.dat")
    options.setdefault("work", tmp_path / "work")
    return geometric_min.run_template(tmp_path, problem or _problem(), template or _template(), **options)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _by_start(monkeypatch, outcomes: dict[str, dict]):
    """A search whose result depends on the start it is given."""

    def optimize(job, evaluate, stop):
        evaluate(job["coords"])
        result = {"point": job["coords"], "stopped": True, "error": "", "steps": 3, "rebuilds": 0,
                  "coordinate_system": "DelocalizedInternalCoordinates"}
        result.update(outcomes.get(job["start"], {}))
        return result

    monkeypatch.setattr(geometric_min, "optimize", optimize)


def test_a_run_over_three_starts_scores_each_and_writes_every_file(monkeypatch, tmp_path, stub_library, capsys):
    exact = _tetrahedron()
    _by_start(monkeypatch, {"initial_0": {"point": exact}, "initial_1": {"point": exact}, "initial_2": {"point": exact}})
    _starts(tmp_path / "cons")
    summary = _run(tmp_path, problem=_problem(starts=3))
    output = tmp_path / "out" / "bench.dat"
    rows = _rows(output.with_suffix(".starts.jsonl"))
    assert [row["start"] for row in rows] == ["initial_0", "initial_1", "initial_2"]
    assert all(row["success"] and "index" not in row for row in rows)
    assert (summary["starts"], summary["template"], summary["problem"]) == (3, "geometric/test", "minimization/lj38")
    assert (summary["nfailed"], summary["v2_nfailed"], summary["partial"]) == (0, 0, False)
    assert summary["coordinate_systems"] == {"DelocalizedInternalCoordinates": 3}
    assert summary["par2_mean"] == pytest.approx(statistics.mean(row["force_calls"] for row in rows))
    assert not output.with_suffix(".partial.jsonl").exists()
    document = json.loads(output.with_suffix(".v2.json").read_text(encoding="utf-8"))
    run = document["runs"][0]
    assert (document["schema"], document["problem"], document["status"]) == ("optbench.result/v2", "minimization/lj38", "archive")
    assert (document["scoring"], document["par_factor"]) == ("par2-wilson-bootstrap", 2)
    assert (document["bootstrap_resamples"], document["bootstrap_seed"]) == (2000, BOOTSTRAP_SEED)
    assert (run["entry"], run["code_version"], run["commit"], run["cap"]) == ("bench", "9.8.7", STUB_REVISION, CAP)
    assert (run["code"], run["algorithm"], run["tuning"], run["present_source"]) == ("geomeTRIC", "TRIC", "default", True)
    assert run["parameters"] == {"coordsys": "tric", "element": "H", "neighbors": 2, "tmin": 1.0e-4}
    assert run["metrics"]["n"] == 3 and run["metrics"]["successes"] == 3
    assert run["provenance"]["machine"]["node"] == "workstation-a"
    text = output.read_text(encoding="utf-8")
    assert "# final coordinate system: DelocalizedInternalCoordinates 3" in text.splitlines()
    fields = dict(line.split(" ", 1) for line in text.splitlines() if not line.startswith("#"))
    assert (fields["template"], fields["commit"], fields["code_version"], fields["n"]) == ("geometric/test", STUB_REVISION, "9.8.7", "3")
    assert (fields["problem"], fields["tuning"], fields["counter"]) == ("minimization/lj38", "default", "harness")
    assert "geometric/test" not in capsys.readouterr().out


def test_each_start_reaches_the_search_with_its_labels_and_the_settings_of_the_template(monkeypatch, tmp_path, stub_library):
    jobs: list[dict] = []

    def optimize(job, evaluate, stop):
        jobs.append(job)
        evaluate(job["coords"])
        return {"point": job["coords"], "stopped": False, "error": "", "steps": 0, "rebuilds": 0,
                "coordinate_system": ""}

    monkeypatch.setattr(geometric_min, "optimize", optimize)
    _starts(tmp_path / "cons", shakes=(0.01, 0.03))
    _run(tmp_path, _template(geometric={"coordsys": "tric", "element": "Pt", "neighbors": 3}))
    assert [(job["index"], job["start"]) for job in jobs] == [(0, "initial_0"), (1, "initial_1")]
    first = jobs[0]
    assert first["coords"] == _tetrahedron(shake=0.01)
    assert first["symbols"] == ["H"] * 4
    assert first["box"] is None
    assert (first["kind"], first["coordsys"], first["neighbors"], first["element"]) == ("lj", "tric", 3, "Pt")
    assert (first["gtol"], first["cap"]) == (1.0e-6, CAP)


def test_a_morse_start_reaches_the_search_with_the_cell_of_its_file(monkeypatch, tmp_path, stub_library):
    jobs: list[dict] = []

    def optimize(job, evaluate, stop):
        jobs.append(job)
        evaluate(job["coords"])
        return {"point": job["coords"], "stopped": False, "error": "", "steps": 0, "rebuilds": 0,
                "coordinate_system": ""}

    monkeypatch.setattr(geometric_min, "optimize", optimize)
    _starts(tmp_path / "cons", shakes=(0.01,))
    problem = {"id": "minimization/morse-bulk", "convergence": {"force_norm": 1.0e-6, "max_force_calls": CAP}}
    _run(tmp_path, _template(problem="minimization/morse-bulk"), problem)
    assert jobs[0]["kind"] == "morse" and jobs[0]["box"] == [40.0, 40.0, 40.0]


def test_the_provenance_names_the_library_when_the_template_does_not(monkeypatch, tmp_path, stub_library):
    _by_start(monkeypatch, {})
    _starts(tmp_path / "cons", shakes=(0.01,))
    template = _template()
    del template["code"]
    _run(tmp_path, template)
    run = json.loads((tmp_path / "out" / "bench.v2.json").read_text(encoding="utf-8"))["runs"][0]
    assert run["code"] is None and run["provenance"]["code"] == "geomeTRIC"


def test_a_template_without_an_algorithm_is_named_by_its_coordinate_system(monkeypatch, tmp_path, stub_library):
    _by_start(monkeypatch, {})
    _starts(tmp_path / "cons", shakes=(0.01,))
    template = _template()
    del template["algorithm"]
    _run(tmp_path, template)
    document = json.loads((tmp_path / "out" / "bench.v2.json").read_text(encoding="utf-8"))
    assert document["runs"][0]["algorithm"] == "tric"


def test_a_start_that_crashes_a_start_that_misses_the_cap_and_one_that_converges_are_scored_together(
    monkeypatch, tmp_path, stub_library
):
    exact = _tetrahedron()
    shaken = _tetrahedron(shake=0.05)
    _by_start(
        monkeypatch,
        {
            "initial_0": {"point": exact},
            "initial_1": {"point": shaken, "stopped": False, "error": "ForceCallCap: force-call cap 400"},
            "initial_2": {"point": None, "stopped": False, "error": "RuntimeError: lost", "coordinate_system": ""},
        },
    )
    _starts(tmp_path / "cons")
    summary = _run(tmp_path)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert [row["failure"] for row in rows] == [None, "cap", "crash"]
    assert (summary["nfailed"], summary["v2_nfailed"]) == (2, 2)
    assert summary["force_calls"] == 1 and summary["force_calls_min"] == summary["force_calls_max"] == 1
    assert summary["median"] == 1
    assert summary["par2_mean"] == pytest.approx((1 + 2 * CAP + 2 * CAP) / 3)
    assert summary["energy_min"] < summary["energy_max"]
    assert summary["coordinate_systems"] == {"DelocalizedInternalCoordinates": 2, "unknown": 1}


def test_a_start_that_ends_at_no_point_does_not_set_the_energy_range_of_the_run(monkeypatch, tmp_path, stub_library):
    exact = _tetrahedron()
    _by_start(
        monkeypatch,
        {
            "initial_0": {"point": exact},
            "initial_1": {"point": [math.nan] * len(exact), "stopped": False},
            "initial_2": {"point": exact},
        },
    )
    _starts(tmp_path / "cons")
    with np.errstate(all="ignore"):
        summary = _run(tmp_path)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert [row["failure"] for row in rows] == [None, "divergence", None]
    assert rows[1]["energy"] is None
    assert summary["energy_min"] == pytest.approx(-6.0, abs=1e-12)
    assert summary["energy_max"] == pytest.approx(-6.0, abs=1e-12)


def test_starts_that_all_fail_leave_no_median_and_cost_twice_the_cap(monkeypatch, tmp_path, stub_library):
    _by_start(monkeypatch, {name: {"stopped": False} for name in ("initial_0", "initial_1", "initial_2")})
    _starts(tmp_path / "cons")
    summary = _run(tmp_path)
    assert (summary["nfailed"], summary["par2_mean"]) == (3, 2 * CAP)
    assert (summary["force_calls"], summary["median"]) == (None, None)


def test_a_limit_runs_only_the_first_starts_and_marks_the_run_partial_against_the_sample(monkeypatch, tmp_path, stub_library):
    _by_start(monkeypatch, {})
    _starts(tmp_path / "cons")
    summary = _run(tmp_path, problem=_problem(starts=3), limit=2)
    output = tmp_path / "out" / "bench.dat"
    assert summary["starts"] == 2 and summary["partial"] is True
    assert [row["start"] for row in _rows(output.with_suffix(".starts.jsonl"))] == ["initial_0", "initial_1"]
    assert output.with_suffix(".partial.jsonl").is_file()
    assert "partial true" in output.read_text(encoding="utf-8").splitlines()


def _recording(monkeypatch) -> list[str]:
    ran: list[str] = []

    def optimize(job, evaluate, stop):
        ran.append(job["start"])
        evaluate(job["coords"])
        return {"point": _tetrahedron(), "stopped": True, "error": "", "steps": 1, "rebuilds": 0,
                "coordinate_system": "CartesianCoordinates"}

    monkeypatch.setattr(geometric_min, "optimize", optimize)
    return ran


def test_a_resumed_run_with_no_partial_file_runs_every_start(monkeypatch, tmp_path, stub_library):
    ran = _recording(monkeypatch)
    _starts(tmp_path / "cons")
    summary = _run(tmp_path, resume=True)
    assert ran == ["initial_0", "initial_1", "initial_2"]
    assert summary["starts"] == 3


def test_a_resumed_run_skips_blank_and_torn_lines_in_the_partial_file(monkeypatch, tmp_path, stub_library):
    ran = _recording(monkeypatch)
    _starts(tmp_path / "cons")
    _run(tmp_path, problem=_problem(starts=3), limit=2)
    with (tmp_path / "out" / "bench.partial.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('\n   \n{"start": "initial_2", "ind')
    ran.clear()
    summary = _run(tmp_path, problem=_problem(starts=3), resume=True)
    assert ran == ["initial_2"]
    assert (summary["starts"], summary["partial"]) == (3, False)


def test_a_resumed_run_keeps_the_starts_already_done_and_runs_only_the_rest(monkeypatch, tmp_path, stub_library):
    ran = _recording(monkeypatch)
    _starts(tmp_path / "cons")
    _run(tmp_path, problem=_problem(starts=3), limit=2)
    assert ran == ["initial_0", "initial_1"]
    ran.clear()
    summary = _run(tmp_path, problem=_problem(starts=3), resume=True)
    assert ran == ["initial_2"]
    assert (summary["starts"], summary["partial"]) == (3, False)
    assert not (tmp_path / "out" / "bench.partial.jsonl").exists()
    assert [row["start"] for row in _rows(tmp_path / "out" / "bench.starts.jsonl")] == ["initial_0", "initial_1", "initial_2"]
    ran.clear()
    _run(tmp_path, problem=_problem(starts=3), limit=1)
    _run(tmp_path, problem=_problem(starts=3), limit=2)
    assert ran == ["initial_0", "initial_0", "initial_1"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"code_version": "1.0"}, "geometric 9.8.7 does not match template 1.0"),
        ({"commit": "f" * 40}, f"geometric revision {STUB_REVISION} does not match template {'f' * 40}"),
        ({"commit": ""}, "geometric/test has no commit"),
        ({"problem": "minimization/morse-bulk"}, "is for minimization/morse-bulk, not minimization/lj38"),
        ({"geometric": {"coordsys": "cart"}}, "needs geometric.element"),
        ({"geometric": {"coordsys": "dlc", "element": "H"}}, "needs geometric.coordsys cart or tric"),
    ],
)
def test_a_template_that_does_not_fit_the_library_or_the_problem_is_refused_before_any_start_runs(
    tmp_path, stub_library, changes, message
):
    with pytest.raises((RuntimeError, ValueError), match=message):
        _run(tmp_path, _template(**changes))
    assert not (tmp_path / "out").exists()


def test_a_problem_with_no_potential_is_refused_before_any_start_runs(tmp_path, stub_library):
    problem = {"id": "minimization/other", "convergence": {"force_norm": 1.0, "max_force_calls": 5}}
    with pytest.raises(ValueError, match="minimization/other has no geomeTRIC potential"):
        _run(tmp_path, _template(problem="minimization/other"), problem)
    assert not (tmp_path / "out").exists()


def test_a_library_with_no_revision_is_accepted_whatever_commit_the_template_names(monkeypatch, tmp_path, stub_library):
    del stub_library.__git_revision__
    _by_start(monkeypatch, {})
    _starts(tmp_path / "cons", shakes=(0.01,))
    assert _run(tmp_path, _template(commit="f" * 40))["starts"] == 1


def test_a_template_with_no_pinned_version_is_not_checked_against_the_library(monkeypatch, tmp_path, stub_library):
    _by_start(monkeypatch, {})
    _starts(tmp_path / "cons", shakes=(0.01,))
    template = _template()
    del template["code_version"]
    assert _run(tmp_path, template)["starts"] == 1
    document = json.loads((tmp_path / "out" / "bench.v2.json").read_text(encoding="utf-8"))
    assert document["runs"][0]["code_version"] == "9.8.7"


def test_a_library_that_cannot_be_imported_by_the_search_is_a_crash_on_every_start(tmp_path, stub_library):
    if LIBRARY_INSTALLED:
        pytest.skip("the library is installed, so the search can import it")
    _starts(tmp_path / "cons", shakes=(0.01, 0.03))
    summary = _run(tmp_path)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert [row["failure"] for row in rows] == ["crash", "crash"]
    assert all("geometric" in row["error"] for row in rows)
    assert (summary["nfailed"], summary["par2_mean"]) == (2, 2 * CAP)


def test_workers_that_cannot_import_the_library_record_a_crash_for_every_start(tmp_path, stub_library, capsys):
    if LIBRARY_INSTALLED:
        pytest.skip("the library is installed, so a worker can import it")
    _starts(tmp_path / "cons", shakes=(0.01, 0.03, 0.05))
    summary = _run(tmp_path, workers=2)
    rows = _rows(tmp_path / "out" / "bench.starts.jsonl")
    assert [row["start"] for row in rows] == ["initial_0", "initial_1", "initial_2"]
    assert [row["failure"] for row in rows] == ["crash"] * 3
    assert summary["nfailed"] == 3
    assert capsys.readouterr().out.splitlines()[-1] == "geometric/test 3/3"
    assert (tmp_path / "out" / "bench.partial.jsonl").exists() is False


def _installed() -> types.ModuleType:
    pytest.importorskip("geometric.engine")
    return pytest.importorskip("geometric")


def _real_job(coords, coordsys="cart", **overrides) -> dict:
    return _job(coords=list(coords), coordsys=coordsys, neighbors=2 if coordsys == "tric" else 0, **overrides)


@pytest.mark.parametrize("coordsys", ["cart", "tric"])
@pytest.mark.parametrize("shake", [0.03, 0.08])
def test_the_installed_library_settles_a_shaken_tetrahedron_on_the_closed_form_minimum(coordsys, shake):
    _installed()
    record = geometric_min._minimize_job(_real_job(_tetrahedron(shake=shake), coordsys))
    assert record["success"] is True
    assert record["energy"] == pytest.approx(-6.0, abs=1e-8)
    assert record["hessian_index"] == 0
    assert record["min_eigenvalues"] == pytest.approx(_spectrum(R_MIN), rel=1e-3)
    assert 1 < record["force_calls"] < CAP
    assert record["coordinate_system"] == (
        "CartesianCoordinates" if coordsys == "cart" else "DelocalizedInternalCoordinates"
    )


@pytest.mark.parametrize("coordsys", ["cart", "tric"])
def test_the_installed_library_takes_no_step_from_the_minimum_itself(coordsys):
    _installed()
    record = geometric_min._minimize_job(_real_job(_tetrahedron(), coordsys))
    assert record["success"] is True
    assert record["force_calls"] == 1 and record["geometric_steps"] == 0


def test_the_installed_library_is_stopped_by_the_budget_at_a_point_it_has_reached():
    _installed()
    record = geometric_min._minimize_job(_real_job(_tetrahedron(shake=0.08), "cart", cap=4))
    assert record["failure"] == "cap"
    assert record["force_calls"] == 4
    assert record["error"].startswith("ForceCallCap: force-call cap 4")
    assert record["energy"] is not None


def test_the_installed_library_gets_the_start_in_problem_units_and_the_gradient_comes_back_in_them():
    _installed()
    seen: dict = {"coords": [], "stops": []}

    def evaluate(flat):
        seen["coords"].append(np.asarray(flat, dtype=float).copy())
        energy, gradient = lennard_jones(flat)
        seen["last"] = np.asarray(gradient, dtype=float)
        return energy, gradient

    def stop(gradient):
        seen["stops"].append((np.asarray(gradient, dtype=float).copy(), seen["last"].copy()))
        return float(np.linalg.norm(gradient)) <= 1.0e-6

    start = _tetrahedron(shake=0.05)
    report = geometric_min.optimize(_real_job(start, "cart"), evaluate, stop)
    assert seen["coords"][0] == pytest.approx(start, abs=1e-12)
    assert all(np.allclose(given, evaluated, rtol=0.0, atol=1e-12) for given, evaluated in seen["stops"])
    assert report["stopped"] is True and report["error"] == ""
    assert report["point"] is not None and report["coordinate_system"] == "CartesianCoordinates"
    assert lennard_jones(report["point"])[0] == pytest.approx(-6.0, abs=1e-8)


def test_the_installed_library_brings_a_stretched_morse_dimer_back_to_its_pair_distance():
    _installed()
    box = [20.0, 20.0, 20.0]
    record = geometric_min._minimize_job(
        _job(kind="morse", coords=_dimer(0.1), box=box, element="Pt", coordsys="cart", neighbors=0)
    )
    assert record["success"] is True
    assert record["energy"] == pytest.approx(_dimer_energy(), abs=1e-9)
    assert record["min_eigenvalues"][-1] == pytest.approx(4.0 * MORSE_ALPHA**2 * MORSE_D0, rel=1e-4)


def test_the_batched_kernels_agree_with_the_installed_library_on_generic_and_degenerate_shapes():
    _installed()
    from geometric.internal import Angle, Dihedral

    shapes = {
        "generic": GENERIC,
        "straight first arm": np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 1.0, 0.3]]),
        "straight last arm": np.array([[0.0, 1.0, 0.3], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cis": _chain(0.0),
        "trans": _chain(math.pi),
    }
    for name, xyz in shapes.items():
        flat = xyz.reshape(-1)
        value, deriv = geometric_min._dihedral_batch(xyz, np.array([[0, 1, 2, 3]]))
        library = Dihedral(0, 1, 2, 3)
        assert value[0] == pytest.approx(library.value(flat), abs=1e-12), name
        assert deriv[0] == pytest.approx(library.derivative(flat), abs=1e-10), name
        value, deriv = geometric_min._angle_batch(xyz, np.array([[0, 1, 2]]))
        library = Angle(0, 1, 2)
        assert value[0] == pytest.approx(library.value(flat), abs=1e-12), name
        assert deriv[0] == pytest.approx(library.derivative(flat), abs=1e-10), name


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
@pytest.mark.parametrize("straight", [True, False])
def test_the_batched_angle_agrees_with_the_installed_library_on_a_line_where_the_library_is_not_finite_too(axis, straight):
    _installed()
    from geometric.internal import Angle

    unit = np.asarray(axis) / np.linalg.norm(axis)
    xyz = np.array([-1.5 * unit if straight else 1.5 * unit, [0.0, 0.0, 0.0], 2.5 * unit])
    library = Angle(0, 1, 2)
    with np.errstate(all="ignore"):
        value, deriv = geometric_min._angle_batch(xyz, np.array([[0, 1, 2]]))
        expected = library.derivative(xyz.reshape(-1))
    assert value[0] == library.value(xyz.reshape(-1))
    assert np.allclose(deriv[0], expected, atol=1e-12, equal_nan=True)


def _installed_template(coordsys: str) -> tuple[dict, str]:
    library = _installed()
    revision = str(getattr(library, "__git_revision__", "") or STUB_REVISION)
    return (
        _template(code_version=str(library.__version__), commit=revision, geometric={"coordsys": coordsys, "element": "H"}),
        revision,
    )


@pytest.mark.parametrize("coordsys", ["cart", "tric"])
def test_a_run_with_the_installed_library_gives_one_worker_and_two_the_same_rows(tmp_path, coordsys):
    template, revision = _installed_template(coordsys)
    _starts(tmp_path / "cons")
    keep = ("start", "force_calls", "success", "energy", "hessian_index", "coordinate_system", "geometric_steps")
    for workers in (1, 2):
        summary = _run(tmp_path, template, output=tmp_path / f"w{workers}" / "bench.dat", workers=workers)
        assert summary["nfailed"] == 0 and summary["starts"] == 3
    one = _rows(tmp_path / "w1" / "bench.starts.jsonl")
    two = _rows(tmp_path / "w2" / "bench.starts.jsonl")
    assert [{key: row[key] for key in keep} for row in one] == [{key: row[key] for key in keep} for row in two]
    assert all(row["success"] for row in one)
    document = json.loads((tmp_path / "w1" / "bench.v2.json").read_text(encoding="utf-8"))
    assert document["runs"][0]["commit"] == revision
