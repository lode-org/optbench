from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from optbench import gpridge_dimer as native
from optbench.campaign import load_template, resolve_problem
from optbench.engines.common import registry_pin, require_pin

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    problem = resolve_problem(ROOT, "tsopt/cuh2")
    files = native.checked_files(ROOT / problem["input"]["directory"],
                                 problem["input"]["sha256"])
    return problem, files


@pytest.mark.parametrize("displacement", [0.02, 0.3, 1.0, 3.0])
def test_archived_displacements_keep_the_complete_fixed_copper_slab(displacement):
    problem, files = fixture()
    atoms, orientation = native.starting_atoms(files, problem, displacement)
    conf = native.legacy_matrix(files["CuH2-init_PES_point.dat"], "conf")
    reference = native.legacy_matrix(files["displacedH2.dat"], "R_sp")
    start = native.legacy_matrix(files["orients_CuH2.dat"], "orient_start")
    assert atoms.numbers.tolist() == [29] * 216 + [1, 1]
    assert atoms.constraints[0].index.tolist() == list(range(216))
    np.testing.assert_array_equal(atoms.positions[:-2], conf[:-2, :3])
    np.testing.assert_allclose(atoms.positions[-2:].reshape(1, 6),
                               reference + displacement * start, rtol=0, atol=1e-14)
    np.testing.assert_allclose(atoms.cell.lengths(), [15.3456, 21.702, 100])
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_array_equal(orientation[:-2], np.zeros((216, 3)))
    assert np.linalg.norm(orientation) == pytest.approx(1.)


@pytest.mark.parametrize("policy,count,radius", [
    ("auto", 0, False), ("radius", 0, True), ("k2", 2, False), ("k16", 16, False),
])
def test_dimer_policy_templates_keep_the_same_physical_problem(tmp_path, policy, count, radius):
    template = load_template(ROOT, f"gpridge/cuh2-gp-dimer-{policy}")
    require_pin(template, registry_pin(ROOT, "gpridge"))
    argv = native.command("gprd", tmp_path, template["gpridge"])
    assert argv[argv.index("--potential") + 1] == "cuh2"
    assert argv[argv.index("--convergence-force-norm") + 1] == "linf"
    assert argv[argv.index("--dimer-project-rotations") + 1] == "false"
    assert argv[argv.index("--gp-nonperiodic") + 1] == "false"
    assert argv[argv.index("--actdist-fro") + 1] == "5.0"
    assert argv[argv.index("--nn-frozen") + 1] == str(count)
    assert ("--radius-only" in argv) == radius
    assert "--gp-frozen-activation-distance" not in argv
    assert "--oracle-stencil" not in argv
    options = dict(template["gpridge"])
    options.pop("frozen_neighbors")
    options.pop("frozen_selection")
    baseline = dict(load_template(ROOT, "gpridge/cuh2-gp-dimer-auto")["gpridge"])
    baseline.pop("frozen_neighbors")
    baseline.pop("frozen_selection")
    assert options == baseline


class Quadratic(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, diagonal):
        super().__init__()
        self.diagonal = np.asarray(diagonal).reshape(2, 3)
        self.calls = 0
        self.substrates = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.substrates.append(atoms.positions[:-2].copy())
        forces = np.zeros_like(atoms.positions)
        forces[-2:] = -self.diagonal * atoms.positions[-2:]
        self.results = {"energy": -.5 * float(np.sum(forces * atoms.positions)),
                        "forces": forces}


def slab():
    atoms = Atoms("CuHH", positions=[[3, 2, 1], [0, 0, 0], [0, 0, 0]],
                  cell=[15, 22, 100], pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


def test_validation_counts_all_six_physical_modes_and_preserves_the_substrate():
    atoms = slab()
    calc = Quadratic([-2, 3, 4, 5, 6, 7])
    result = native.verify_positions(atoms, atoms.positions.copy(), calc, .01)
    assert result["reasons"] == []
    assert result["hessian_index"] == 1
    assert result["verify_calls"] == calc.calls == 13
    np.testing.assert_allclose(result["hessian_eigenvalues"], [-2, 3, 4, 5, 6, 7])
    for positions in calc.substrates:
        np.testing.assert_array_equal(positions, atoms.positions[:-2])


@pytest.mark.parametrize("diagonal,reason", [
    ([2, 3, 4, 5, 6, 7], "hessian_index"),
    ([-2, -3, 4, 5, 6, 7], "hessian_index"),
    ([-2, 0, 4, 5, 6, 7], "unresolved_mode"),
])
def test_validation_rejects_wrong_or_unresolved_stationary_points(diagonal, reason):
    atoms = slab()
    result = native.verify_positions(atoms, atoms.positions.copy(), Quadratic(diagonal), .01)
    assert reason in result["reasons"]
    assert result["verify_calls"] == 13


def test_force_failure_stops_before_the_hessian_and_counts_the_evaluation():
    atoms = slab()
    positions = atoms.positions.copy()
    positions[-1, 0] = .1
    result = native.verify_positions(atoms, positions, Quadratic([-2, 3, 4, 5, 6, 7]), .01)
    assert result["reasons"] == ["force_norm"]
    assert result["verify_calls"] == 1
    assert result["hessian_index"] is None


def test_final_positions_retain_frozen_atoms_and_reject_invalid_native_results():
    initial = slab().positions
    log = "Final midpoint coordinates (2 atoms):\n1 2 3\n4 5 6\nEnd of final midpoint coordinates"
    result = native.final_positions(log, initial)
    np.testing.assert_array_equal(result[0], initial[0])
    np.testing.assert_array_equal(result[-2:], [[1, 2, 3], [4, 5, 6]])
    for malformed in ("", log.replace("1 2 3", "nan 2 3"), log.replace("4 5 6\n", "")):
        with pytest.raises(ValueError):
            native.final_positions(malformed, initial)


def test_fixture_hashes_reject_mutation_and_external_symlinks(tmp_path):
    fixture = tmp_path / "file"
    fixture.write_text("pinned")
    hashes = {"file": native.digest(fixture)}
    assert native.checked_files(tmp_path, hashes) == {"file": str(fixture)}
    fixture.write_text("different")
    with pytest.raises(ValueError, match="checksum"):
        native.checked_files(tmp_path, hashes)
    outside = tmp_path / "outside"
    outside.write_text("data")
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="checksum"):
        native.checked_files(root, {"linked": native.digest(outside)})


def test_fixed_atom_movement_is_rejected_before_oracle_evaluation():
    atoms = slab()
    positions = atoms.positions.copy()
    positions[0, 0] += .1
    calc = Quadratic([-2, 3, 4, 5, 6, 7])
    with pytest.raises(ValueError, match="fixed Cu"):
        native.verify_positions(atoms, positions, calc, .01)
    assert calc.calls == 0


def test_timed_out_search_keeps_a_failed_record_without_fabricating_calls(tmp_path, monkeypatch):
    import json

    def inputs(directory, job):
        directory.mkdir()
        for name in ("pos.con", "direction.dat", "params.bin"):
            (directory / name).write_text("fixture")
        return slab()

    def timeout(argv, directory, env, stream, seconds):
        stream.write("unfinished outer loop\n")
        return 124

    problem, files = fixture()
    options = load_template(ROOT, "gpridge/cuh2-gp-dimer-k2")["gpridge"]
    job = {"name": "displacement-3", "displacement": 3., "directory": str(tmp_path / "case"),
           "files": files, "sources": {}, "problem": problem,
           "rpc_sources": {}, "rpc_client": problem["rpc_client"],
           "runtime": {"binary": "gprd"}, "oracle": {"binary": "potserv"}, "options": options}
    monkeypatch.setattr(native, "write_inputs", inputs)
    monkeypatch.setattr(native, "run_search", timeout)
    result = native._execute(job)
    assert result["success"] is False
    assert result["counter"] == problem["scoring"]["v2"]["counter"]
    assert result["count_available"] is False
    assert result["returncode"] == 124
    assert {"timeout", "not_converged", "missing_count"}.issubset(result["reasons"])
    assert json.loads((tmp_path / "case/result.json").read_text()) == result
    command = json.loads((tmp_path / "case/command.json").read_text())
    assert command["displacement"] == 3.
    assert command["source_input_sha256"] == problem["input"]["sha256"]


def test_failed_verification_retains_completed_evaluations_and_request_count():
    class FailsOnThird(Quadratic):
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            if self.calls == 2:
                raise TimeoutError("oracle request deadline")
            super().calculate(atoms, properties, system_changes)

    atoms = slab()
    progress = {}
    calc = FailsOnThird([-2, 3, 4, 5, 6, 7])
    with pytest.raises(TimeoutError):
        native.verify_positions(atoms, atoms.positions.copy(), calc, .01, progress)
    assert progress == {"completed": 2, "requested": 3}
    assert calc.calls == 2


def test_rpc_client_pin_is_independent_and_rejects_revision_schema_or_source_changes(tmp_path):
    import json

    client = tmp_path / "calculator.py"
    client.write_text("bounded calculator")
    schema = tmp_path / "Potentials.capnp"
    schema.write_text("compatible server")
    contract = {"source_repository": "https://github.com/OmniPotentRPC/rgpot",
                "source_revision": "a" * 40, "sha256": {"calculator.py": native.digest(client)},
                "server_schema_sha256": native.digest(schema)}
    provenance = tmp_path / "source-provenance.json"
    provenance.write_text(json.dumps({"source_revision": "a" * 40}))
    files, identity = native.checked_rpc_client(tmp_path, contract, str(schema))
    assert files == {"calculator.py": str(client)}
    assert identity == {**contract, "source_root": str(tmp_path),
                        "source_provenance_sha256": native.digest(provenance)}
    provenance.write_text(json.dumps({"source_revision": "b" * 40}))
    with pytest.raises(ValueError, match="revision"):
        native.checked_rpc_client(tmp_path, contract, str(schema))
    provenance.write_text(json.dumps({"source_revision": "a" * 40}))
    schema.write_text("incompatible server")
    with pytest.raises(ValueError, match="server schema"):
        native.checked_rpc_client(tmp_path, contract, str(schema))
    schema.write_text("compatible server")
    client.write_text("unbounded calculator")
    with pytest.raises(ValueError, match="checksum"):
        native.checked_rpc_client(tmp_path, contract, str(schema))
