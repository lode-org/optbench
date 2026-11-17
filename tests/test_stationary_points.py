"""Hessian index of a finished search, on Lennard-Jones geometries with known curvature.

Three code paths decide whether a search ended on the stationary point it
claims: the eOn harness reads a ``.con`` frame, the ASE runners difference
the forces of a calculator, and the band runner counts the eigenvalues of a
Hessian it already holds. They have to agree on the same geometry, and each
has to leave the rigid-body modes out of the count when it should.

Every expected curvature below comes from the pair derivatives of
``u(r) = 4 (r**-12 - r**-6)``, written out here, not from the code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optbench.eon_harness import lj40, saddle_index
from optbench.pes import lennard_jones
from optbench.rgsaddle_band import _project_rigid as band_project
from optbench.rgsaddle_band import vibrational_index as band_index
from optbench.stationary import (
    _rigid_basis,
    jacobi_eigenvalues,
    judge_stationary,
    kabsch_rmsd,
    negative_index,
    numerical_hessian,
    vibrational_eigenvalues,
)

RMIN = 2.0 ** (1.0 / 6.0)
TRIANGLE = np.array(
    [[0.0, 0.0, 0.0], [RMIN, 0.0, 0.0], [RMIN / 2.0, RMIN * math.sqrt(3.0) / 2.0, 0.0]]
)
TETRAHEDRON = np.vstack(
    [TRIANGLE, [[RMIN / 2.0, RMIN * math.sqrt(3.0) / 6.0, RMIN * math.sqrt(2.0 / 3.0)]]]
)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


def _balanced_chain() -> np.ndarray:
    """Three collinear atoms at the spacing where the outer atoms feel no net force."""
    low, high = 1.0, RMIN
    for _ in range(80):
        mid = 0.5 * (low + high)
        if _slope(mid) + _slope(2.0 * mid) < 0.0:
            low = mid
        else:
            high = mid
    spacing = 0.5 * (low + high)
    return np.array([[-spacing, 0.0, 0.0], [0.0, 0.0, 0.0], [spacing, 0.0, 0.0]])


def _con(positions, fixed=()) -> str:
    head = [
        "stationary point under test",
        "0 0 0",
        "30.000000 30.000000 30.000000",
        "90.000000 90.000000 90.000000",
        "0 0",
        "0 0 0",
        "1",
        str(len(positions)),
        "39.948000",
        "Ar",
        "Coordinates of Component 1",
    ]
    atoms = [
        f"{x:.10f} {y:.10f} {z:.10f} {int(index in fixed)} {index}"
        for index, (x, y, z) in enumerate(positions)
    ]
    return "\n".join([*head, *atoms, ""])


def _atoms(positions, fixed=()):
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.constraints import FixAtoms

    atoms = Atoms(f"Ar{len(positions)}", positions=np.asarray(positions, dtype=float))
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=40.0, smooth=False)
    if fixed:
        atoms.set_constraint(FixAtoms(indices=list(fixed)))
    return atoms


def _ase_index(atoms) -> int:
    from optbench.ase_sella import vibrational_index

    return vibrational_index(atoms)


def _hessian(positions) -> np.ndarray:
    def gradient(flat):
        return lennard_jones(np.asarray(flat, dtype=float).reshape(-1, 3))[1].ravel()

    return np.array(numerical_hessian(gradient, np.asarray(positions).ravel().tolist()))


@pytest.mark.parametrize(("half_gap", "index"), [(1.5, 1), (1.2, 0), (1.0, 2)])
def test_a_mobile_atom_between_two_pinned_ones_has_the_curvature_the_pair_gives(half_gap, index):
    """By symmetry the force at the midpoint is zero.

    Along the axis the curvature is 2 u''(a), negative past the inflection at
    (26/7)**(1/6) = 1.2445. Across it the curvature is 2 u'(a)/a, positive past
    the pair minimum at 1.1225.
    """
    positions = np.array([[-half_gap, 0.0, 0.0], [0.0, 0.0, 0.0], [half_gap, 0.0, 0.0]])
    found = saddle_index(lj40, _con(positions, fixed=(0, 2)), periodic=False)
    assert found["hessian_index"] == index
    assert found["force_norm"] == pytest.approx(0.0, abs=1e-9)
    assert _ase_index(_atoms(positions, fixed=(0, 2))) == index


def test_the_force_norm_counts_the_mobile_atom_only():
    half_gap, drift = 1.5, 0.05
    positions = np.array([[-half_gap, 0.0, 0.0], [0.0, drift, 0.0], [half_gap, 0.0, 0.0]])
    radius = math.hypot(half_gap, drift)
    mobile = 2.0 * abs(_slope(radius)) * drift / radius
    found = saddle_index(lj40, _con(positions, fixed=(0, 2)), periodic=False)
    assert found["force_norm"] == pytest.approx(mobile, rel=1e-9)
    assert found["hessian_index"] == 1


def test_a_con_with_every_atom_pinned_has_no_index_to_report():
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="every atom is fixed"):
        saddle_index(lj40, _con(positions, fixed=(0, 1)), periodic=False)
    assert _ase_index(_atoms(positions, fixed=(0, 1))) == 0


def test_the_ase_index_leaves_the_geometry_and_the_forces_as_it_found_them():
    positions = np.array([[-1.5, 0.0, 0.0], [0.0, 0.03, 0.0], [1.5, 0.0, 0.0]])
    atoms = _atoms(positions, fixed=(0, 2))
    before = atoms.get_positions().copy()
    forces = atoms.get_forces().copy()
    _ase_index(atoms)
    assert np.array_equal(atoms.get_positions(), before)
    assert np.allclose(atoms.get_forces(), forces, rtol=0.0, atol=1e-12)


def test_the_index_calls_are_verification_and_do_not_spend_the_search_budget():
    from optbench.ase_sella import _verify_index
    from optbench.counter import attach_ase_counter

    positions = np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    atoms = _atoms(positions, fixed=(0, 2))
    attach_ase_counter(atoms.calc, cap=1)
    atoms.get_forces()
    state = atoms.calc.optbench_counter
    assert state["calls"] == 1
    assert _verify_index(atoms, 1) == 1
    assert state["calls"] == 1
    assert state["verify_calls"] >= 6
    assert state["verifying"] is False


def test_a_verification_already_in_progress_stays_in_progress():
    from optbench.ase_sella import _verify_index
    from optbench.counter import attach_ase_counter

    positions = np.array([[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    atoms = _atoms(positions, fixed=(0, 2))
    attach_ase_counter(atoms.calc)
    atoms.calc.optbench_counter["verifying"] = True
    assert _verify_index(atoms, 0) == 0
    assert atoms.calc.optbench_counter["verifying"] is True
    assert atoms.calc.optbench_counter["calls"] == 0


def test_a_calculator_without_a_counter_is_verified_all_the_same():
    from optbench.ase_sella import _verify_index

    positions = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _verify_index(_atoms(positions, fixed=(0, 2)), 2) == 2


@pytest.mark.parametrize(
    ("positions", "index"),
    [(TRIANGLE, 0), (TETRAHEDRON, 0), (_balanced_chain(), 2)],
    ids=["triangle", "tetrahedron", "balanced chain"],
)
def test_free_clusters_have_the_index_their_geometry_gives(positions, index):
    """The pair minimum is the minimum of the triangle and the tetrahedron.

    A compressed collinear chain buckles, so its two bends are negative.
    """
    found = saddle_index(lj40, _con(positions), periodic=False)
    assert found["hessian_index"] == index
    assert found["force_norm"] < 1e-7
    assert _ase_index(_atoms(positions)) == index
    hessian = _hessian(positions)
    assert band_index(hessian, positions, True) == index
    values = vibrational_eigenvalues(hessian, positions, rotations=True)
    assert negative_index(values, 1e-6) == index


def test_rigid_leakage_at_a_displaced_geometry_is_not_a_negative_mode():
    """Away from a stationary point the rotations pick up curvature of either sign.

    Pushing one atom of the triangle 0.02 along x leaves a force of about 1, and
    the unprojected Hessian shows two negative eigenvalues that belong to the
    rotations. The vibrations are all positive.
    """
    moved = TRIANGLE.copy()
    moved[2, 0] += 0.02
    hessian = _hessian(moved)
    assert negative_index(np.linalg.eigvalsh(hessian), 1e-6) == 2

    assert saddle_index(lj40, _con(moved), periodic=False)["hessian_index"] == 0
    assert _ase_index(_atoms(moved)) == 0
    assert band_index(hessian, moved, True) == 0
    assert negative_index(vibrational_eigenvalues(hessian, moved, rotations=True), 1e-6) == 0


def test_a_periodic_cell_drops_the_translations_and_keeps_the_rotations():
    moved = TRIANGLE.copy()
    moved[2, 0] += 0.02
    hessian = _hessian(moved)
    assert saddle_index(lj40, _con(moved), periodic=True)["hessian_index"] == 2
    assert band_index(hessian, moved, False, rotations=0) == 2
    kept = vibrational_eigenvalues(hessian, moved, rotations=False)
    assert len(kept) == 6
    assert negative_index(kept, 1e-6) == 2


def test_a_compressed_dimer_has_one_vibration_and_two_leaked_rotations():
    """At r = 1 the pair is far from its minimum: u'(1) = -24 and u''(1) = 456.

    The stretch has curvature 2 u'' = 912. The two rotations have curvature
    2 u'/r = -48 each, which is rigid-body leakage and not a vibration.
    """
    dimer = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    hessian = _hessian(dimer)
    assert np.linalg.eigvalsh(hessian)[:2] == pytest.approx([-48.0, -48.0], rel=1e-4)
    assert negative_index(np.linalg.eigvalsh(hessian), 1e-6) == 2

    vibrations = vibrational_eigenvalues(hessian, dimer, rotations=True)
    assert vibrations == pytest.approx([2.0 * _curvature(1.0)], rel=1e-4)
    assert band_index(hessian, dimer, True) == 0
    assert saddle_index(lj40, _con(dimer), periodic=False)["hessian_index"] == 0
    assert _ase_index(_atoms(dimer)) == 0
    assert saddle_index(lj40, _con(dimer), periodic=True)["hessian_index"] == 2


def test_a_fixed_anchor_keeps_the_free_atom_hessian_whole():
    """With atom 0 pinned, the free atom is a 3 by 3 Hessian: u'', u'/r, u'/r."""
    dimer = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _curvature(1.0) > 0.0 > _slope(1.0)
    assert saddle_index(lj40, _con(dimer, fixed=(0,)), periodic=False)["hessian_index"] == 2
    assert _ase_index(_atoms(dimer, fixed=(0,))) == 2


@pytest.mark.parametrize("projector", ["ase", "band"])
def test_rigid_projection_removes_a_translation_and_a_rotation_but_not_the_stretch(projector):
    pytest.importorskip("ase")
    from optbench.ase_sella import _project_rigid as ase_project

    project = ase_project if projector == "ase" else band_project
    dimer = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])

    def unit(vector):
        return np.asarray(vector, dtype=float) / np.linalg.norm(vector)

    shear = unit([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    spin = unit([0.0, 0.55, 0.0, 0.0, -0.55, 0.0])
    stretch = unit([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def count(hessian, rotations):
        return negative_index(np.linalg.eigvalsh(project(hessian, dimer, rotations)), 1e-6)

    assert count(-np.outer(shear, shear), 3) == 0
    assert count(-np.outer(spin, spin), 3) == 0
    assert count(-np.outer(spin, spin), 0) == 1
    assert count(-np.outer(stretch, stretch), 3) == 1
    assert count(-np.outer(stretch, stretch), 0) == 1


def test_a_single_atom_has_no_rigid_projection_to_make_in_the_band_runner():
    hessian = -np.eye(3)
    assert band_project(hessian, np.zeros((1, 3))) is hessian


def test_the_rigid_basis_of_a_dimer_drops_the_rotation_about_its_axis():
    dimer = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    assert _rigid_basis(dimer, rotations=True).shape == (6, 5)
    assert _rigid_basis(dimer, rotations=False).shape == (6, 3)
    assert _rigid_basis(TRIANGLE, rotations=True).shape == (9, 6)
    assert _rigid_basis(np.zeros((1, 3)), rotations=True).shape == (3, 3)
    assert _rigid_basis(np.zeros((0, 3)), rotations=True).shape == (0, 0)


def test_the_rigid_basis_is_orthonormal_and_spans_every_small_rigid_move():
    basis = _rigid_basis(TETRAHEDRON, rotations=True)
    assert np.allclose(basis.T @ basis, np.eye(6), atol=1e-12)
    axis = np.array([0.3, -0.5, 0.8])
    shift = np.array([0.2, 0.1, -0.4])
    centered = TETRAHEDRON - TETRAHEDRON.mean(axis=0)
    change = (np.cross(axis, centered) + shift).ravel()
    inside = basis @ (basis.T @ change)
    assert np.linalg.norm(change - inside) < 1e-12 * np.linalg.norm(change)
    bend = np.zeros(12)
    bend[3] = 1.0
    leftover = bend - basis @ (basis.T @ bend)
    assert np.linalg.norm(leftover) > 0.5


def test_the_vibrational_spectrum_has_three_n_minus_the_rigid_modes():
    triangle = _hessian(TRIANGLE)
    assert len(vibrational_eigenvalues(triangle, TRIANGLE, rotations=True)) == 3
    assert len(vibrational_eigenvalues(triangle, TRIANGLE, rotations=False)) == 6
    flat = vibrational_eigenvalues(triangle, TRIANGLE.ravel(), rotations=True)
    assert flat == vibrational_eigenvalues(triangle, TRIANGLE, rotations=True)
    assert vibrational_eigenvalues(np.zeros((3, 3)), np.zeros((1, 3)), rotations=True) == []
    assert vibrational_eigenvalues(np.zeros((0, 0)), np.zeros((0, 3)), rotations=True) == []


def test_a_hessian_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="hessian shape does not match the coordinates"):
        vibrational_eigenvalues(np.zeros((6, 6)), TRIANGLE, rotations=True)


def test_a_matrix_that_is_not_square_is_refused():
    with pytest.raises(ValueError, match="matrix must be square"):
        jacobi_eigenvalues([[1.0, 2.0], [2.0]])
    with pytest.raises(ValueError, match="matrix must be square"):
        jacobi_eigenvalues([])


def test_a_jacobi_run_cut_short_still_conserves_the_trace_and_orders_its_values():
    rng = np.random.default_rng(3)
    factor = rng.normal(size=(5, 5))
    matrix = factor + factor.T
    values, vectors = jacobi_eigenvalues(matrix.tolist(), sweeps=1)
    assert values == sorted(values)
    assert sum(values) == pytest.approx(float(np.trace(matrix)), abs=1e-9)
    basis = np.array(vectors)
    assert np.allclose(basis.T @ basis, np.eye(5), atol=1e-9)
    full, _ = jacobi_eigenvalues(matrix.tolist())
    assert full == pytest.approx(np.linalg.eigvalsh(matrix).tolist(), abs=1e-9)


def test_kabsch_aligns_a_planar_set_that_was_flipped_over():
    plane = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 1.5, 0.0], [-0.5, 1.0, 0.0]]
    flipped = [[x + 4.0, -y + 1.0, -z] for x, y, z in plane]
    assert kabsch_rmsd(flipped, plane) == pytest.approx(0.0, abs=1e-8)
    bent = [[x, y, z] for x, y, z in plane]
    bent[3] = [-0.5, 1.0, 0.4]
    assert kabsch_rmsd(bent, plane) > 0.05


def test_kabsch_aligns_a_collinear_set_to_the_best_end_to_end_orientation():
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.0, 0.0]]
    turned = [[7.0, x, 2.0] for x, _y, _z in line]
    assert kabsch_rmsd(turned, line) == pytest.approx(0.0, abs=1e-8)

    stretched = [[7.0, x + (0.1 if index == 3 else 0.0), 2.0] for index, (x, _y, _z) in enumerate(line)]
    ref = np.array([point[0] for point in line])
    ref -= ref.mean()
    mov = np.array([point[1] for point in stretched])
    mov -= mov.mean()
    expected = min(
        math.sqrt(float(np.mean((mov - ref) ** 2))),
        math.sqrt(float(np.mean((-mov - ref) ** 2))),
    )
    assert kabsch_rmsd(stretched, line) == pytest.approx(expected, abs=1e-8)
    assert expected > 0.01


def test_kabsch_sees_no_difference_between_coincident_points_anywhere():
    assert kabsch_rmsd([[1.0, 1.0, 1.0]] * 3, [[5.0, -2.0, 0.0]] * 3) == pytest.approx(0.0, abs=1e-12)


def test_kabsch_refuses_sets_that_cannot_be_compared():
    with pytest.raises(ValueError, match="same non-zero length"):
        kabsch_rmsd([], [])
    with pytest.raises(ValueError, match="same non-zero length"):
        kabsch_rmsd([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="x, y, z"):
        kabsch_rmsd([[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_a_minimum_is_judged_against_index_zero_and_a_saddle_is_refused_for_it():
    kept = judge_stationary(
        kind="minimum",
        eigenvalues=[0.3, 2.0, 40.0],
        energy=-3.0,
        coords=[row.tolist() for row in TRIANGLE],
        reference_energy=-3.0,
        reference_coords=[row.tolist() for row in TRIANGLE],
        energy_tol=1e-6,
        rmsd_tol=1e-6,
    )
    assert kept["ok"] is True
    assert kept["expected_index"] == 0
    assert kept["reasons"] == []

    saddle = judge_stationary(
        kind="minimum",
        eigenvalues=[-0.3, 2.0, 40.0],
        energy=-2.0,
        coords=[row.tolist() for row in TRIANGLE],
        reference_energy=-3.0,
        reference_coords=[row.tolist() for row in TRIANGLE],
        energy_tol=1e-6,
        rmsd_tol=1e-6,
    )
    assert saddle["ok"] is False
    assert saddle["hessian_index"] == 1
    assert saddle["reasons"] == ["hessian_index", "energy"]
    assert saddle["energy_error"] == pytest.approx(1.0)


def test_an_unknown_kind_of_stationary_point_is_refused():
    with pytest.raises(ValueError, match="kind must be saddle or minimum"):
        judge_stationary(
            kind="maximum",
            eigenvalues=[-1.0],
            energy=0.0,
            coords=[[0.0, 0.0, 0.0]],
            reference_energy=0.0,
            reference_coords=[[0.0, 0.0, 0.0]],
            energy_tol=1.0,
            rmsd_tol=1.0,
        )
