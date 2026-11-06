"""Global-optimisation kernels, Cambridge minima, and the energy success rule."""

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from optbench.ccd import (
    LJ_SIZES,
    TIP4P_SIZES,
    binary_minimum,
    lennard_jones_minimum,
    tip4p_minimum,
)
from optbench.globalopt import energy_tolerance, force_cap, judge_globalopt, reference_energy
from optbench.potentials import binary_lennard_jones, lennard_jones
from optbench.problems import load_all
from optbench.score import score_v2
from optbench.tip4p import reference_sites, site_energy_forces, sites_from_ohh, tip4p, tip4p_ohh

ROOT = Path(__file__).resolve().parents[1]


def _fd_gradient(energy_fn, x, step=1e-6):
    gradient = np.zeros_like(x, dtype=float)
    for index in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (energy_fn(plus) - energy_fn(minus)) / (2.0 * step)
    return gradient


def test_binary_dimer_sits_at_the_arithmetic_well():
    sigma = 0.5 * (1.0 + 1.3)
    distance = sigma * 2.0 ** (1.0 / 6.0)
    coords = np.array([0.0, 0.0, 0.0, distance, 0.0, 0.0])
    energy, gradient = binary_lennard_jones(coords, [0, 1])
    assert energy == pytest.approx(-1.0)
    assert np.linalg.norm(gradient) == pytest.approx(0.0, abs=1e-8)


def test_binary_gradient_matches_finite_differences():
    rng = np.random.default_rng(4)
    positions = rng.normal(size=(5, 3))
    positions[:, 0] += np.arange(5) * 1.4
    species = np.array([0, 0, 1, 1, 0])
    flat = positions.reshape(-1)

    def energy(point):
        value, _gradient = binary_lennard_jones(point, species, cutoff=20.0)
        return value

    _value, gradient = binary_lennard_jones(flat, species, cutoff=20.0)
    assert gradient == pytest.approx(_fd_gradient(energy, flat), rel=1e-5, abs=1e-6)


def test_binary_cambridge_minimum_matches_the_header_and_is_stationary():
    record = binary_minimum()
    energy, gradient = binary_lennard_jones(record["positions"], record["species"])
    assert energy == pytest.approx(record["energy"], abs=1e-9, rel=0.0)
    assert np.linalg.norm(gradient) < 1e-3
    # The points file fits inside the archived LAMMPS cutoff of 20.
    cut_energy, cut_gradient = binary_lennard_jones(
        record["positions"], record["species"], cutoff=20.0
    )
    assert cut_energy == pytest.approx(energy, abs=1e-12, rel=0.0)
    assert cut_gradient == pytest.approx(gradient, abs=1e-12, rel=0.0)
    flat = record["positions"].reshape(-1)
    species = record["species"]

    def energy_fn(point):
        value, _gradient = binary_lennard_jones(point, species)
        return value

    sample = np.array([0, 1, 40, 80, 120, 200])
    finite = _fd_gradient(energy_fn, flat)[sample]
    assert gradient[sample] == pytest.approx(finite, rel=1e-4, abs=1e-6)


def test_lennard_jones_cambridge_minima_match_the_table():
    for size in LJ_SIZES:
        record = lennard_jones_minimum(size)
        energy, gradient = lennard_jones(record["positions"])
        assert energy == pytest.approx(float(record["energy"]), abs=1e-6)
        # Printed coordinates leave a residual well under a distorted contact.
        assert np.max(np.abs(gradient)) < 2e-3


def test_tip4p_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    centres = np.array([[0.0, 0.0, 0.0], [5.5, 0.4, -0.2], [-1.0, 6.0, 0.8]])
    axes = rng.normal(scale=0.6, size=(3, 3))
    flat = np.concatenate([centres, axes], axis=1).reshape(-1)

    def energy(point):
        value, _gradient = tip4p(point)
        return value

    _value, gradient = tip4p(flat)
    assert gradient == pytest.approx(_fd_gradient(energy, flat, step=1e-7), rel=1e-5, abs=1e-6)

    zero = flat.copy()
    zero[3:6] = 0.0
    _zero_energy, zero_gradient = tip4p(zero)

    def zero_energy(point):
        value, _gradient = tip4p(point)
        return value

    assert zero_gradient == pytest.approx(_fd_gradient(zero_energy, zero, step=1e-7), rel=1e-5, abs=1e-6)


def test_tip4p_site_forces_match_finite_differences():
    rng = np.random.default_rng(8)
    placed = sites_from_ohh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.75, 0.6],
                [0.0, -0.75, 0.6],
                [4.8, 0.2, 0.1],
                [5.2, 0.9, 0.4],
                [4.4, -0.2, 0.7],
            ]
        )
    )
    # Shake the second molecule off the reference geometry's exact bonds.
    placed[1, 1] += rng.normal(scale=0.02, size=3)
    flat = placed.reshape(-1)

    def energy(point):
        value, _gradient = site_energy_forces(point.reshape(-1, 4, 3))
        return value

    _value, gradient = site_energy_forces(placed)
    assert gradient.reshape(-1) == pytest.approx(_fd_gradient(energy, flat, step=1e-7), rel=1e-4, abs=1e-5)


def test_tip4p_cambridge_minima_match_the_headers_and_are_stationary():
    for size in TIP4P_SIZES:
        record = tip4p_minimum(size)
        placed = sites_from_ohh(record["positions"])
        site_energy, _site_gradient = site_energy_forces(placed)
        rigid_energy, rigid_gradient = tip4p_ohh(record["positions"])
        published = float(record["energy"])
        page = float(record["page_energy"])
        assert site_energy == pytest.approx(published, abs=3e-3)
        assert rigid_energy == pytest.approx(site_energy, abs=1e-3)
        assert round(published, 5) == pytest.approx(page, abs=0.0)
        assert np.linalg.norm(rigid_gradient) < 1e-2
        # Net force and torque sit at the print precision of the xyz file.
        blocks = rigid_gradient.reshape(-1, 6)
        assert np.max(np.abs(blocks[:, :3])) < 1e-3
        assert np.max(np.abs(blocks[:, 3:])) < 1e-3


def test_reference_sites_keep_the_gmin_bond_geometry():
    sites = reference_sites()
    centre = np.sum(sites * np.array([16.0, 1.0, 1.0, 0.0])[:, None], axis=0)
    assert centre == pytest.approx(0.0, abs=1e-12)
    oxygen, hydrogen_1, hydrogen_2, m_site = sites
    assert np.linalg.norm(hydrogen_1 - oxygen) == pytest.approx(0.9572, abs=1e-12)
    assert np.linalg.norm(hydrogen_2 - oxygen) == pytest.approx(0.9572, abs=1e-12)
    bond_1 = hydrogen_1 - oxygen
    bond_2 = hydrogen_2 - oxygen
    cosine = np.dot(bond_1, bond_2) / (np.linalg.norm(bond_1) * np.linalg.norm(bond_2))
    assert math.degrees(math.acos(float(cosine))) == pytest.approx(104.52, abs=1e-8)
    bisector = bond_1 / np.linalg.norm(bond_1) + bond_2 / np.linalg.norm(bond_2)
    bisector /= np.linalg.norm(bisector)
    assert m_site == pytest.approx(oxygen + 0.15 * bisector, abs=1e-12)


def test_judge_globalopt_is_one_sided_and_orders_the_classes():
    reference = -397.492331
    tolerance = 0.001
    cap = 100
    assert judge_globalopt(reference, 10, reference=reference, tolerance=tolerance, cap=cap)["success"]
    edge = judge_globalopt(reference + tolerance, 10, reference=reference, tolerance=tolerance, cap=cap)
    assert edge["success"]
    lower = judge_globalopt(reference - 1.0, 10, reference=reference, tolerance=tolerance, cap=cap)
    assert lower["success"]
    high = judge_globalopt(reference + 0.01, 10, reference=reference, tolerance=tolerance, cap=cap)
    assert high["failure"] == "wrong_stationary"
    capped = judge_globalopt(reference + 0.01, 100, reference=reference, tolerance=tolerance, cap=cap)
    assert capped["failure"] == "cap"
    at_cap = judge_globalopt(reference, cap, reference=reference, tolerance=tolerance, cap=cap)
    assert at_cap["success"] is False
    assert at_cap["failure"] == "cap"
    blown = judge_globalopt(float("nan"), cap, reference=reference, tolerance=tolerance, cap=cap)
    assert blown["failure"] == "divergence"
    missing = judge_globalopt(None, 3, reference=reference, tolerance=tolerance, cap=cap)
    assert missing["failure"] == "divergence"


def test_basin_hopping_displaces_the_quenched_geometry():
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import FIRE

    from optbench.runners.ase_basinhopping import quenched_basin_hopping

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = np.asarray(self.atoms.get_positions(), dtype=float)
            self.results["energy"] = float(np.sum(pos * pos))
            self.results["forces"] = -2.0 * pos

    origins = []
    hopper_type = quenched_basin_hopping()

    class Recording(hopper_type):
        def move(self, ro):
            origins.append(np.array(ro, dtype=float, copy=True))
            return super().move(ro)

    atoms = Atoms("H", positions=[[1.5, -0.4, 0.2]])
    atoms.calc = Harmonic()
    np.random.seed(7)
    hopper = Recording(
        atoms,
        temperature=0.8,
        optimizer=FIRE,
        fmax=1e-6,
        dr=0.4,
        logfile=None,
        trajectory=None,
        optimizer_logfile=None,
        local_minima_trajectory=None,
        adjust_cm=False,
    )
    hopper.run(4)
    assert len(origins) == 4
    assert max(float(np.max(np.abs(origin))) for origin in origins) < 1e-3


def test_cap_keeps_the_best_finished_minimum():
    pytest.importorskip("ase")
    from optbench.campaign import geometry_archive
    from optbench.problems import load_problem
    from optbench.runners.ase_basinhopping import load_starts, search_one

    problem = load_problem(ROOT / "problems" / "globalopt" / "lj75.toml")
    starts = {index: positions for index, positions in load_starts(geometry_archive(ROOT, problem))}
    result = search_one(starts[1], index=1, root=ROOT, cap=900)
    assert result["failure"] == "cap"
    assert result["force_calls"] == 900
    assert result["success"] is False
    # The first quench of coords.1 finishes near call 795. The cap then lands
    # inside the next quench, so the stored energy is that finished minimum.
    assert result["energy"] == pytest.approx(-379.284523, abs=1e-4)


def test_campaign_file_names_keep_the_ase_version(tmp_path: Path):
    from optbench.runners.ase_basinhopping import write_campaign

    record = {
        "schema": "optbench.start/v2",
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "cap",
        "reasons": ["energy", "cap"],
        "force_calls": 10,
        "energy": -1.0,
        "start": "coords.1",
    }
    dat, jsonl = write_campaign(tmp_path, [record], partial=True, cap=10)
    assert dat.name == "ase-3.29.0-lj75-basinhopping.dat"
    assert jsonl.name == "ase-3.29.0-lj75-basinhopping.starts.jsonl"
    assert dat.read_text(encoding="utf-8").splitlines()[0].startswith("# ASE 3.29.0")


def test_resume_skips_starts_already_in_the_jsonl(tmp_path: Path):
    from optbench.runners.ase_basinhopping import load_campaign_records, write_campaign

    kept = {
        "schema": "optbench.start/v2",
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "cap",
        "reasons": ["energy", "cap"],
        "force_calls": 10,
        "energy": -2.0,
        "start": "coords.4",
    }
    _dat, jsonl = write_campaign(tmp_path, [kept], partial=True, cap=10)
    loaded = load_campaign_records(jsonl)
    assert [record["start"] for record in loaded] == ["coords.4"]
    assert load_campaign_records(tmp_path / "missing.jsonl") == []


def test_ase_basin_hopping_accepts_the_lj75_cambridge_minimum():
    pytest.importorskip("ase")
    from optbench.runners.ase_basinhopping import search_one

    record = lennard_jones_minimum(75)
    result = search_one(record["positions"], index=1, root=ROOT, hops=1)
    assert result["success"]
    assert result["failure"] is None
    assert result["force_calls"] < 20
    assert result["counter"] == "harness"
    assert result["template"] == "ase-globalopt/lj75-basinhopping"


def test_scored_globalopt_records_keep_every_failure_in_par2():
    cap = 50
    records = [
        judge_globalopt(-10.0, 4, reference=-10.0, tolerance=0.001, cap=cap),
        judge_globalopt(-9.0, 8, reference=-10.0, tolerance=0.001, cap=cap),
        judge_globalopt(-10.0, 50, reference=-10.0, tolerance=0.001, cap=cap),
    ]
    for record in records:
        record["schema"] = "optbench.start/v2"
        record["counter"] = "harness"
        record["comparable"] = True
    summary = score_v2(records, cap)
    assert summary["n"] == 3
    assert summary["successes"] == 1
    assert summary["nfailed"] == 2
    assert summary["n_wrong_stationary"] == 1
    assert summary["n_cap"] == 1
    assert summary["par2_mean"] == pytest.approx((4 + 2 * cap + 2 * cap) / 3)


def test_problem_files_use_the_fetched_cambridge_energies():
    problems = load_all(ROOT / "problems")
    expected = {}
    for size in LJ_SIZES:
        record = lennard_jones_minimum(size)
        expected[f"globalopt/lj{size}"] = float(record["energy"])
    expected["globalopt/lj100-binary"] = float(binary_minimum()["energy"])
    for size in TIP4P_SIZES:
        record = tip4p_minimum(size)
        expected[f"globalopt/tip4p{size}"] = float(record["energy"])
    for ident, energy in expected.items():
        problem = problems[ident]
        assert reference_energy(problem) == energy
        assert energy_tolerance(problem) == 0.001
        assert int(problem["sample"]["starts"]) > 0
        assert problem["input"]["archive_path"]
        assert force_cap(problem) > 0
    binary = problems["globalopt/lj100-binary"]
    assert binary["potential"]["n_A"] == 42
    assert binary["potential"]["n_B"] == 58
    assert round(reference_energy(binary), 6) == pytest.approx(binary["potential"]["energy_minimum"])
    assert problems["globalopt/lj75"]["sample"]["starts"] == 200
    assert problems["globalopt/lj38"]["convergence"]["max_force_calls"] == 2000000
    assert problems["globalopt/lj75"]["convergence"]["max_force_calls"] == 1000000
    water = problems["globalopt/tip4p10"]
    assert "TIP4P-water.html" in water["reference"]["page"]
    assert "tables.150.html" in problems["globalopt/lj75"]["reference"]["page"]
    assert "NEW_BINARY_LJ_MINIMA/1.3/100" in problems["globalopt/lj100-binary"]["reference"]["points"]
