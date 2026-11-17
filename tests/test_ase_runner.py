"""ASE runners on stationary points whose answers are known in closed form.

A displaced Lennard-Jones tetrahedron relaxes to the energy of six pair
minima, and its Hessian has no negative mode once the rigid ones are gone.
A free atom between two pinned atoms 3 apart sits on a saddle of index 1:
the curvature along the axis is twice the pair curvature at 1.5, and the
axial wells lie on either side of it. Pair derivatives come from
``u(r) = 4 (r**-12 - r**-6)``, written out here, not from the code under test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms

from optbench import ase_sella
from optbench.ase_lj import LennardJones
from optbench.ase_sella import (
    CallCounter,
    ForceCap,
    _attach_index,
    _cached_l2,
    _cap_error,
    _displace_free,
    _fmax,
    _free_mask,
    _inner_calculator,
    _morse_arrays,
    _reactant_mode,
    _read_atoms,
    _run_start,
)
from optbench.campaign import geometry_archive, resolve_problem
from optbench.pes import evaluate, load_pairs

pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")

RMIN = 2.0 ** (1.0 / 6.0)
EDGE = RMIN
TETRAHEDRON = np.array(
    [
        [0.0, 0.0, 0.0],
        [EDGE, 0.0, 0.0],
        [EDGE / 2.0, EDGE * math.sqrt(3.0) / 2.0, 0.0],
        [EDGE / 2.0, EDGE * math.sqrt(3.0) / 6.0, EDGE * math.sqrt(2.0 / 3.0)],
    ]
)
DISPLACEMENT = np.array(
    [
        [0.08, -0.05, 0.03],
        [-0.06, 0.07, -0.02],
        [0.04, 0.05, 0.09],
        [-0.05, -0.08, 0.04],
    ]
)
POTENTIAL = {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "rc": 40.0, "pbc": [False, False, False]}
PINNED = (0, 2)


def _slope(r: float) -> float:
    return 4.0 * (-12.0 * r**-13 + 6.0 * r**-7)


def _curvature(r: float) -> float:
    return 4.0 * (156.0 * r**-14 - 42.0 * r**-8)


def _well() -> float:
    """Position on the axis where the free atom between the pinned pair feels no force."""
    low, high = -1.0, -0.05
    for _ in range(100):
        mid = 0.5 * (low + high)
        force = -_slope(mid + 1.5) + _slope(1.5 - mid)
        if force > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _xyz(path: Path, positions) -> Path:
    rows = "\n".join(f"Ar {x:.14f} {y:.14f} {z:.14f}" for x, y, z in positions)
    path.write_text(f"{len(positions)}\nfixture\n{rows}\n", encoding="utf-8")
    return path


def _con(path: Path, positions, fixed=()) -> Path:
    head = [
        "pinned pair around one free atom",
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
        f"{x:.14f} {y:.14f} {z:.14f} {int(index in fixed)} {index}"
        for index, (x, y, z) in enumerate(positions)
    ]
    path.write_text("\n".join([*head, *atoms, ""]), encoding="utf-8")
    return path


def _trio(free_atom) -> list[list[float]]:
    return [[-1.5, 0.0, 0.0], list(free_atom), [1.5, 0.0, 0.0]]


def _payload(task: str, paths, **overrides) -> dict:
    payload = {
        "start": "fixture",
        "task": task,
        "paths": [str(path) for path in paths],
        "potential": POTENTIAL,
        "optimizer": {"name": "BFGS", "parameters": {}},
        "neb": None,
        "sella": None,
        "sella_search": None,
        "reactant_positions": None,
        "seed": 0,
        "tol": 1e-3,
        "norm": "l2",
        "max_steps": 1000,
        "raw_cap": 400,
        "snapshot": False,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def evaluations(monkeypatch):
    """Counts every potential evaluation, independently of the harness counter."""
    seen = {"count": 0}

    class Counting(LennardJones):
        def calculate(self, *args, **kwargs):
            seen["count"] += 1
            return super().calculate(*args, **kwargs)

    def inner(potential):
        return Counting(
            epsilon=float(potential["epsilon"]),
            sigma=float(potential["sigma"]),
            rc=float(potential["rc"]),
        )

    monkeypatch.setattr(ase_sella, "_inner_calculator", inner)
    return seen


@pytest.fixture
def displaced(tmp_path) -> Path:
    return _xyz(tmp_path / "displaced.xyz", TETRAHEDRON + DISPLACEMENT)


@pytest.mark.parametrize("name", ["BFGS", "LBFGS", "FIRE", "BFGSLineSearch", "LBFGSLineSearch"])
def test_a_displaced_tetrahedron_relaxes_to_six_pair_minima(displaced, name):
    row = _run_start(_payload("minimize", [displaced], optimizer={"name": name, "parameters": {}}))
    assert row["success"] is True
    assert row["failure"] is None
    assert row["reasons"] == []
    assert row["converged"] is True
    assert row["energy"] == pytest.approx(-6.0, abs=1e-5)
    assert row["hessian_index"] == 0
    assert row["expected_index"] == 0
    assert row["force_norm"] <= 1e-3
    assert 0 < row["force_calls"] < 400
    assert row["schema"] == "optbench.start/v2"
    assert row["counter"] == "harness"
    assert row["wall_seconds"] >= 0.0


def test_search_and_verification_calls_add_up_to_the_evaluations_of_the_potential(displaced, evaluations):
    row = _run_start(_payload("minimize", [displaced]))
    assert row["success"] is True
    assert row["force_calls"] + row["verify_calls"] == evaluations["count"]
    assert row["verify_calls"] >= 2 * 3 * 4


def test_a_start_that_already_meets_the_tolerance_costs_one_call(tmp_path):
    row = _run_start(_payload("minimize", [_xyz(tmp_path / "relaxed.xyz", TETRAHEDRON)]))
    assert row["success"] is True
    assert row["force_calls"] == 1
    assert row["force_norm"] < 1e-9
    assert row["energy"] == pytest.approx(-6.0, abs=1e-8)


@pytest.mark.parametrize("cap", [3, 6])
def test_a_run_that_hits_the_cap_has_used_exactly_the_cap_and_is_not_probed_for_curvature(displaced, evaluations, cap):
    row = _run_start(_payload("minimize", [displaced], raw_cap=cap))
    assert row["success"] is False
    assert row["failure"] == "cap"
    assert row["reasons"] == ["cap", "force_norm"]
    assert row["force_calls"] == cap
    assert evaluations["count"] == cap
    assert row["verify_calls"] == 0
    assert "hessian_index" not in row
    assert row["force_norm"] > 1e-3


def test_running_out_of_steps_before_the_tolerance_is_a_divergence_and_not_a_cap(displaced):
    row = _run_start(_payload("minimize", [displaced], max_steps=2))
    assert row["success"] is False
    assert row["failure"] == "divergence"
    assert row["reasons"] == ["force_norm"]
    assert row["force_calls"] <= 3
    assert row["converged"] is False
    assert row["verify_calls"] == 0


def test_a_minimizer_that_stops_on_a_saddle_has_found_the_wrong_stationary_point(tmp_path):
    saddle = _con(tmp_path / "saddle.con", _trio((0.0, 0.0, 0.0)), fixed=PINNED)
    row = _run_start(_payload("minimize", [saddle]))
    assert row["success"] is False
    assert row["failure"] == "wrong_stationary"
    assert row["reasons"] == ["hessian_index"]
    assert row["hessian_index"] == 1
    assert row["force_calls"] == 1
    assert row["verify_calls"] == 2 * 3 + 1
    pair = 2.0 * 4.0 * (1.5**-12 - 1.5**-6) + 4.0 * (3.0**-12 - 3.0**-6)
    assert row["energy"] == pytest.approx(pair, abs=1e-6)


def test_the_fmax_stop_reads_the_largest_atom_force_and_the_l2_stop_the_total(displaced):
    total = _run_start(_payload("minimize", [displaced], norm="l2"))
    largest = _run_start(_payload("minimize", [displaced], norm="fmax"))
    assert total["success"] and largest["success"]
    assert largest["force_norm"] <= 1e-3
    assert largest["force_calls"] <= total["force_calls"]


def test_the_largest_atom_force_and_the_total_force_ignore_what_a_pinned_atom_feels(tmp_path):
    path = _con(tmp_path / "off-axis.con", _trio((0.2, 0.0, 0.0)), fixed=PINNED)
    atoms = _read_atoms(path)
    atoms.calc = _inner_calculator(POTENTIAL)
    assert _fmax(atoms) is None
    assert _cached_l2(atoms.calc, atoms) is None
    atoms.get_forces()
    free = -_slope(1.7) + _slope(1.3)
    assert abs(free) > 0.1
    assert _fmax(atoms) == pytest.approx(abs(free), rel=1e-9)
    assert _cached_l2(atoms.calc, atoms) == pytest.approx(abs(free), rel=1e-9)
    raw = np.linalg.norm(atoms.calc.results["forces"])
    assert raw > abs(free)


def test_a_pinned_pair_is_free_only_where_the_atom_may_move(tmp_path):
    atoms = _read_atoms(_con(tmp_path / "pinned.con", _trio((0.1, 0.0, 0.0)), fixed=PINNED))
    assert _free_mask(atoms) == [False, True, False]
    assert _free_mask(Atoms("Ar2", positions=[[0, 0, 0], [1, 0, 0]])) == [True, True]


def test_a_seeded_displacement_moves_only_free_atoms_and_repeats(tmp_path):
    path = _con(tmp_path / "pinned.con", _trio((0.1, 0.0, 0.0)), fixed=PINNED)
    first, again, other = (_read_atoms(path) for _ in range(3))
    reference = first.positions.copy()
    _displace_free(first, seed=4, std=0.1)
    _displace_free(again, seed=4, std=0.1)
    _displace_free(other, seed=5, std=0.1)
    assert np.array_equal(first.positions, again.positions)
    assert not np.array_equal(first.positions, other.positions)
    assert np.array_equal(first.positions[[0, 2]], reference[[0, 2]])
    assert not np.array_equal(first.positions[1], reference[1])


def test_the_reactant_mode_is_a_unit_vector_over_the_free_atoms_only(tmp_path):
    atoms = _read_atoms(_con(tmp_path / "pinned.con", _trio((0.3, 0.4, 0.0)), fixed=PINNED))
    reference = np.array(_trio((0.0, 0.0, 0.0)))
    mode = _reactant_mode(atoms, reference)
    assert mode.shape == (3,)
    assert mode.tolist() == pytest.approx([0.6, 0.8, 0.0])
    assert _reactant_mode(atoms, None) is None
    assert _reactant_mode(atoms, reference[:2]) is None
    assert _reactant_mode(atoms, atoms.positions.copy()) is None


@pytest.mark.parametrize("seed", [0, 1])
def test_the_dimer_finds_the_axial_saddle_between_two_pinned_atoms(tmp_path, seed):
    start = _con(tmp_path / "start.con", _trio((0.12, 0.07, 0.03)), fixed=PINNED)
    row = _run_start(_payload("dimer", [start], optimizer={"name": "Dimer", "parameters": {}}, seed=seed, snapshot=True))
    assert row["success"] is True
    assert row["saddle"] is True
    assert row["hessian_index"] == 1
    assert row["expected_index"] == 1
    assert row["curvature"] == pytest.approx(2.0 * _curvature(1.5), rel=1e-3)
    assert row["force_norm"] <= 1e-3
    assert 0 < row["force_calls"] < 200
    assert row["verify_calls"] == 2 * 3 + 1
    assert row["parameters"]["mask"] == "free"
    assert row["parameters"]["order"] == 1


def test_a_dimer_that_starts_on_the_saddle_is_done_in_a_few_calls(tmp_path):
    saddle = _con(tmp_path / "saddle.con", _trio((0.0, 0.0, 0.0)), fixed=PINNED)
    row = _run_start(_payload("dimer", [saddle], optimizer={"name": "Dimer", "parameters": {}}))
    assert row["success"] is True
    assert row["force_norm"] == pytest.approx(0.0, abs=1e-9)
    assert row["force_calls"] <= 10


def test_a_dimer_with_no_budget_left_is_a_cap_and_never_a_saddle(tmp_path):
    start = _con(tmp_path / "start.con", _trio((0.12, 0.07, 0.03)), fixed=PINNED)
    row = _run_start(_payload("dimer", [start], optimizer={"name": "Dimer", "parameters": {}}, raw_cap=8))
    assert row["success"] is False
    assert row["failure"] == "cap"
    assert row["saddle"] is False
    assert row["force_calls"] == 8
    assert row["verify_calls"] == 0


@pytest.fixture
def wells(tmp_path):
    x = _well()
    assert x == pytest.approx(-0.3723, abs=1e-4)
    left = _con(tmp_path / "a.con", _trio((x, 0.3, 0.1)), fixed=PINNED)
    right = _con(tmp_path / "b.con", _trio((-x, -0.3, -0.1)), fixed=PINNED)
    return left, right


def _neb(**overrides) -> dict:
    spec = {"layout": "minima", "moving_images": 3, "spring": 0.1, "climb": True, "method": "improvedtangent"}
    spec.update(overrides)
    return spec


@pytest.mark.parametrize("interpolate", ["linear", "idpp", "geodesic"])
def test_a_band_between_the_wells_settles_and_its_total_counts_every_image(wells, evaluations, interpolate):
    payload = _payload(
        "neb",
        wells,
        optimizer={"name": "FIRE", "parameters": {}},
        neb=_neb(interpolate=interpolate),
        norm="fmax",
        tol=0.01,
        raw_cap=2000,
        snapshot=True,
    )
    row = _run_start(payload)
    assert row["success"] is True
    assert row["failure"] is None
    assert row["force_norm"] <= 0.01
    assert row["force_calls"] >= 3 + 2
    assert row["force_calls"] == evaluations["count"]
    assert row["verify_calls"] == 0
    assert "hessian_index" not in row
    assert row["parameters"]["interpolate"] == interpolate
    assert row["parameters"]["moving_images"] == 3
    assert row["parameters"]["spring"] == 0.1
    assert row["parameters"]["method"] == "improvedtangent"


def test_a_band_that_starts_on_the_path_is_converged_after_one_pass_over_its_images(tmp_path):
    x = _well()
    left = _con(tmp_path / "a.con", _trio((x, 0.0, 0.0)), fixed=PINNED)
    right = _con(tmp_path / "b.con", _trio((-x, 0.0, 0.0)), fixed=PINNED)
    row = _run_start(
        _payload(
            "neb",
            [left, right],
            optimizer={"name": "FIRE", "parameters": {}},
            neb=_neb(),
            norm="fmax",
            tol=0.01,
            raw_cap=2000,
        )
    )
    assert row["success"] is True
    assert row["force_calls"] == 3 + 2
    assert row["force_norm"] < 1e-9


def test_a_band_with_no_budget_left_is_a_cap_and_a_climb_after_is_reported_as_it_was_asked(wells):
    row = _run_start(
        _payload(
            "neb",
            wells,
            optimizer={"name": "FIRE", "parameters": {}},
            neb=_neb(climb_after=0.5),
            norm="fmax",
            tol=1e-6,
            raw_cap=12,
            snapshot=True,
        )
    )
    assert row["success"] is False
    assert row["failure"] == "cap"
    assert row["force_calls"] == 12
    assert row["parameters"]["climb"] is False
    assert row["parameters"]["climb_after"] == 0.5


def test_a_geodesic_guess_keeps_the_pinned_atoms_and_the_endpoints_where_they_were(wells):
    from optbench.ase_sella import _make_band

    payload = _payload(
        "neb",
        wells,
        optimizer={"name": "FIRE", "parameters": {}},
        neb=_neb(moving_images=4, interpolate="geodesic"),
    )
    band, images = _make_band(payload, CallCounter(100))
    assert len(images) == 6
    first = _read_atoms(wells[0]).positions
    last = _read_atoms(wells[1]).positions
    assert np.allclose(images[0].positions, first)
    assert np.allclose(images[-1].positions, last)
    for image in images:
        assert np.allclose(image.positions[[0, 2]], first[[0, 2]])
    middle = np.array([image.positions[1] for image in images])
    assert np.all(np.diff(middle[:, 0]) > 0.0)


@pytest.mark.parametrize(
    ("task", "optimizer", "error"),
    [
        ("nonsense", {"name": "BFGS", "parameters": {}}, "ValueError: nonsense"),
        ("minimize", {"name": "Bogus", "parameters": {}}, "KeyError: 'Bogus'"),
    ],
    ids=["unknown-task", "unknown-optimizer"],
)
def test_a_run_that_cannot_start_is_a_crash_with_the_exception_named(displaced, task, optimizer, error):
    row = _run_start(_payload(task, [displaced], optimizer=optimizer))
    assert row["success"] is False
    assert row["failure"] == "crash"
    assert row["error"] == error
    assert row["reasons"] == ["crash", "force_norm"]
    assert row["force_calls"] == 0
    assert row["converged"] is False


def test_a_geometry_that_is_not_there_is_a_crash_and_not_an_abort(tmp_path):
    row = _run_start(_payload("minimize", [tmp_path / "absent.xyz"]))
    assert row["failure"] == "crash"
    assert row["error"].startswith("FileNotFoundError")


def test_a_hessian_that_cannot_be_taken_fails_the_row_and_names_why(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("calculator went away")

    monkeypatch.setattr(ase_sella, "_verify_index", broken)
    row = {"failed": False}
    _attach_index(row, None, 0)
    assert row["failed"] is True
    assert row["error"] == "Hessian RuntimeError: calculator went away"
    assert row["expected_index"] == 0
    assert "hessian_index" not in row


def test_a_hessian_index_that_differs_from_the_expected_one_fails_the_row(tmp_path):
    atoms = _read_atoms(_con(tmp_path / "saddle.con", _trio((0.0, 0.0, 0.0)), fixed=PINNED))
    atoms.calc = _inner_calculator(POTENTIAL)
    row = {"failed": False}
    _attach_index(row, atoms, 0)
    assert row["hessian_index"] == 1
    assert row["failed"] is True
    row = {"failed": False}
    _attach_index(row, atoms, 1)
    assert row["hessian_index"] == 1
    assert row["failed"] is False


def test_the_counter_reads_calls_then_cap_and_nothing_else():
    counter = CallCounter(7)
    assert (counter[0], counter[1]) == (0, 7)
    with pytest.raises(IndexError):
        counter[2]


def test_only_a_cap_is_taken_for_a_cap():
    assert _cap_error(ForceCap())
    assert _cap_error(RuntimeError("force-call cap"))
    assert not _cap_error(RuntimeError("something else"))
    assert not _cap_error(ValueError("force-call cap"))


def test_symbols_that_are_not_elements_still_load_as_atoms_and_short_files_say_what_is_missing(tmp_path):
    odd = tmp_path / "odd.xyz"
    odd.write_text("2\ncomment\nQ 0 0 0\nQ 1 0 0\n", encoding="utf-8")
    atoms = _read_atoms(odd)
    assert len(atoms) == 2
    assert atoms.positions.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    blank = tmp_path / "blank.xyz"
    blank.write_text("2\n\nAr 0 0 0\nAr 1 0 0\n", encoding="utf-8")
    assert _read_atoms(blank).positions.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    for name, text, message in (
        ("short.xyz", "3\ncomment\nAr 0 0 0\nAr 1 0 0\n", "short.xyz has 2 coordinates, expected 3"),
        ("bare.xyz", "2\n", "bare.xyz needs a count, a comment, and coordinates"),
        ("ragged.xyz", "2\ncomment\nAr 0 0 0\nAr 1 0\n", "ragged.xyz coordinate needs element x y z"),
    ):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            _read_atoms(path)


def _pair(distance: float, box: float | None = None):
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    cell = np.diag([box or 100.0] * 3)
    pbc = np.array([box is not None] * 3)
    return positions, cell, pbc


def test_a_shifted_morse_energy_vanishes_at_the_cutoff_and_a_smooth_one_loses_its_force_too():
    constants = (0.7102, 1.6047, 2.897, 7.0)
    positions, cell, pbc = _pair(7.0 - 1e-9)
    shifted_energy, shifted_forces = _morse_arrays(positions, cell, pbc, *constants, "shift")
    smooth_energy, smooth_forces = _morse_arrays(positions, cell, pbc, *constants, "smooth")
    assert shifted_energy == pytest.approx(0.0, abs=1e-6)
    assert smooth_energy == pytest.approx(0.0, abs=1e-6)
    assert abs(shifted_forces[1, 0]) > 1e-4
    assert abs(smooth_forces[1, 0]) < 1e-6
    beyond, beyond_forces = _morse_arrays(*_pair(7.1), *constants, "shift")
    assert beyond == 0.0
    assert not beyond_forces.any()
    with pytest.raises(ValueError, match="cubic"):
        _morse_arrays(positions, cell, pbc, *constants, "cubic")


def _cutoff_energy(constants) -> float:
    depth, alpha, r0, cutoff = constants
    x = math.exp(-alpha * (cutoff - r0))
    return depth * (x * x - 2.0 * x)


def test_morse_forces_are_equal_and_opposite_and_the_well_sits_at_r0():
    constants = (0.7102, 1.6047, 2.897, 7.0)
    energy, forces = _morse_arrays(*_pair(2.897), *constants, "shift")
    assert energy == pytest.approx(-0.7102 - _cutoff_energy(constants), abs=1e-9)
    assert np.abs(forces).max() < 1e-12
    _energy, pulled = _morse_arrays(*_pair(3.4), *constants, "shift")
    assert pulled[0, 0] > 0.0
    assert pulled[0, 0] == pytest.approx(-pulled[1, 0])
    x = math.exp(-1.6047 * (3.4 - 2.897))
    assert pulled[0, 0] == pytest.approx(2.0 * 1.6047 * 0.7102 * x * (1.0 - x), rel=1e-9)


def test_a_periodic_pair_feels_the_nearest_image_and_an_open_pair_does_not():
    constants = (0.7102, 1.6047, 2.897, 4.0)
    open_energy, open_forces = _morse_arrays(*_pair(7.103), *constants, "shift")
    assert open_energy == 0.0
    assert not open_forces.any()
    wrapped, wrapped_forces = _morse_arrays(*_pair(7.103, box=10.0), *constants, "shift")
    assert wrapped == pytest.approx(-0.7102 - _cutoff_energy(constants), abs=1e-9)
    assert np.abs(wrapped_forces).max() < 1e-9


def test_an_unknown_potential_kind_is_refused_and_a_morse_calculator_is_the_pair_function():
    with pytest.raises(ValueError, match="buckingham"):
        _inner_calculator({"kind": "buckingham"})
    potential = {"kind": "morse", "D": 0.7102, "alpha": 1.6047, "r0": 2.897, "rc": 7.0, "form": "shift"}
    atoms = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]], cell=[30.0, 30.0, 30.0], pbc=False)
    atoms.calc = _inner_calculator(potential)
    energy, forces = _morse_arrays(
        atoms.positions, atoms.cell.array, atoms.pbc, 0.7102, 1.6047, 2.897, 7.0, "shift"
    )
    assert atoms.get_potential_energy() == pytest.approx(energy)
    assert np.allclose(atoms.get_forces(), forces)


def test_the_runner_morse_matches_the_harness_potential_on_a_displaced_platinum_slab():
    root = Path(__file__).resolve().parents[1]
    problem = resolve_problem(root, "neb/Pt-heptamer")
    pair = load_pairs(geometry_archive(root, problem), problem["geometry"]["kind"], 13)[0]
    ripple = 0.05 * np.sin(np.arange(pair.reactant.size, dtype=float)).reshape(pair.reactant.shape)
    coords = pair.reactant + ripple
    energy, gradient = evaluate("morse_pt", coords, pair.box)
    runner_energy, runner_forces = _morse_arrays(
        coords, np.diag(pair.box), np.array([True, True, True]), 0.7102, 1.6047, 2.897, 9.5, "shift"
    )
    assert runner_energy == pytest.approx(energy, rel=1e-12)
    assert np.allclose(runner_forces, -gradient, atol=1e-10)
