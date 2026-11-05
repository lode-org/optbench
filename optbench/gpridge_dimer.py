"""Native Cu-H2 dimer searches with an independent Cap'n Proto oracle check."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
import tomllib
from pathlib import Path

import numpy as np

from .campaign import load_template, resolve_problem
from .engines.common import registry_pin, require_pin, run_jobs, write_campaign
from .failures import classify_failure
from .gpridge_band import digest, frozen_arguments, run_search, runtime
from .rgsaddle_band import vibrational_index
from .stationary import curvature_tolerance

# Nocedal and Wright, Numerical Optimization, 2nd ed., Springer, 2006,
# section 8.1, equation 8.7, derive the cube-root step for a
# central-difference gradient from function values. Truncation and
# roundoff balance at the cube root of machine epsilon, times a typical
# magnitude of the variable. Each column of this Hessian is a central
# difference of forces, and the same balance sets its step. Machine
# epsilon is numpy finfo(float64).eps, about 2.2e-16. The coordinates
# are stored in Angstroms, so the length scale is one Angstrom. That
# length is an explicit choice.
_COORDINATE_SCALE_ANGSTROM = 1.0


def hydrogen_hessian_step() -> float:
    """Central-difference step, in Angstrom, for the six hydrogen modes."""
    machine_epsilon = float(np.finfo(np.float64).eps)
    return _COORDINATE_SCALE_ANGSTROM * machine_epsilon ** (1.0 / 3.0)


def checked_files(directory: Path, hashes: dict) -> dict[str, str]:
    directory = directory.resolve(strict=True)
    result = {}
    for name, expected in hashes.items():
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"input checksum differs: {name}")
        result[name] = str(path)
    return result


def checked_rpc_client(directory: Path, contract: dict, server_schema: str) -> tuple[dict, dict]:
    """Pin the maintained client independently of the numerical executable."""
    directory = directory.resolve(strict=True)
    provenance = directory / "source-provenance.json"
    found = json.loads(provenance.read_text())["source_revision"]
    if found != contract["source_revision"]:
        raise ValueError("rgpot RPC client revision differs from the problem contract")
    sources = checked_files(directory, contract["sha256"])
    if digest(Path(server_schema)) != contract["server_schema_sha256"]:
        raise ValueError("rgpot server schema differs from the compatible pinned contract")
    identity = {"source_repository": contract["source_repository"],
                "source_revision": found, "source_root": str(directory),
                "source_provenance_sha256": digest(provenance),
                "sha256": contract["sha256"],
                "server_schema_sha256": contract["server_schema_sha256"]}
    return sources, identity


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_matrix(path: str, label: str) -> np.ndarray:
    tokens = Path(path).read_text().split()
    index = 0
    while index < len(tokens):
        name = tokens[index]
        rows, columns = int(tokens[index + 1]), int(tokens[index + 2])
        count = rows * columns
        values = np.asarray(tokens[index + 3:index + 3 + count], dtype=float)
        if rows < 1 or columns < 1 or len(values) != count or not np.isfinite(values).all():
            raise ValueError(f"invalid archived matrix: {path}")
        if name == label:
            return values.reshape(rows, columns)
        index += 3 + count
    raise ValueError(f"missing {label} in {path}")


def starting_atoms(files: dict, problem: dict, displacement: float):
    from ase import Atoms
    from ase.constraints import FixAtoms

    conf = legacy_matrix(files["CuH2-init_PES_point.dat"], "conf")
    if conf.shape != (218, 5) or not np.array_equal(conf[:, 3], [1] * 216 + [0, 0]):
        raise ValueError("Cu-H2 fixture must contain 216 fixed Cu and two moving H atoms")
    if not np.array_equal(conf[:, 4], np.arange(218)):
        raise ValueError("Cu-H2 fixture atom identities differ")
    midpoint = legacy_matrix(files["displacedH2.dat"], "R_sp")
    direction = legacy_matrix(files["orients_CuH2.dat"], "orient_start")
    rotation = legacy_matrix(files["orients_CuH2.dat"], "orient_init")
    if any(value.shape != (1, 6) for value in (midpoint, direction, rotation)):
        raise ValueError("Cu-H2 displacements and orientations must have six coordinates")
    positions = conf[:, :3].copy()
    positions[-2:] = (midpoint + displacement * direction).reshape(2, 3)
    atoms = Atoms(numbers=[29] * 216 + [1, 1], positions=positions,
                  cell=problem["potential"]["cell"], pbc=problem["potential"]["pbc"])
    atoms.set_constraint(FixAtoms(indices=np.arange(216)))
    orientation = np.zeros((218, 3))
    norm = np.linalg.norm(rotation)
    if not np.isfinite(norm) or norm <= 1e-14:
        raise ValueError("Cu-H2 dimer orientation is zero or non-finite")
    orientation[-2:] = rotation.reshape(2, 3) / norm
    return atoms, orientation


def write_inputs(directory: Path, job: dict):
    import capnp
    from ase.io import write

    atoms, orientation = starting_atoms(job["files"], job["problem"], job["displacement"])
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "pos.con", atoms, format="eon")
    np.savetxt(directory / "direction.dat", orientation, fmt="%.17g")
    parameters = tomllib.loads(Path(job["files"]["params.toml"]).read_text())
    parameters["cellDimensions"] = atoms.cell.array.reshape(-1).tolist()
    parameters["problem"]["finalConvergenceForce"] = job["problem"]["convergence"]["force_norm"]
    parameters["problem"]["convergenceForceNorm"] = "linfNorm"
    parameters["problem"]["actdistFro"] = job["options"]["frozen_activation_distance"]
    serializer = load_module(job["sources"]["scripts/prepare_petmad_assets.py"],
                             "gpridge_parameter_serializer")
    schema = capnp.load(job["sources"]["schema/gprd_params.capnp"])
    message = serializer.populate_message_from_toml(parameters, schema)
    with (directory / "params.bin").open("wb") as stream:
        message.write(stream)
    return atoms


def command(binary: str, directory: Path, options: dict) -> list[str]:
    frozen = frozen_arguments(options)
    frozen = ["--actdist-fro" if value == "--gp-frozen-activation-distance" else value
              for value in frozen]
    return [
        binary, "--pos", str(directory / "pos.con"),
        "--orient", str(directory / "direction.dat"),
        "--config", str(directory / "params.bin"),
        "--potential", "cuh2", "--gp-nonperiodic", "false",
        "--dimer-project-rotations", "false", "--convergence-force-norm", "linf",
        "--gp-model-profile", options["model_profile"],
        "--hyperparameter-mode", options["hyperparameter_mode"],
        "--linalg-mode", options["arithmetic"],
        "--linalg-threads", str(options["threads"]),
        "--gp-hssvd", str(options["hssvd"]).lower(),
        "--gp-active-posterior", str(options["active_posterior"]).lower(),
        "--gp-prior-mean-statistic", options["prior_mean_statistic"],
        "--gp-no-output-norm", "true", "--gp-zbl-prior-mean", "false",
        "--gp-observation-archive", str(directory / "observations"),
    ] + frozen


def final_positions(text: str, initial: np.ndarray) -> np.ndarray:
    blocks = re.findall(
        r"Final midpoint coordinates \(\d+ atoms\):\s*\n(.*?)End of final midpoint coordinates",
        text, re.S,
    )
    if not blocks:
        raise ValueError("native dimer did not report final coordinates")
    positions = np.asarray([[float(x) for x in line.split()]
                            for line in blocks[-1].strip().splitlines()])
    if positions.shape != (2, 3) or not np.isfinite(positions).all():
        raise ValueError("native dimer must report the two moving H atoms")
    full = initial.copy()
    full[-2:] = positions
    return full


def verify_positions(atoms, positions: np.ndarray, calculator, force_stop: float,
                     progress: dict | None = None) -> dict:
    """Keep every moving mode and the fixed Cu coordinates in every force call."""
    if positions.shape != atoms.positions.shape or not np.array_equal(
            positions[:-2], atoms.positions[:-2]):
        raise ValueError("fixed Cu atoms moved")
    if not np.isfinite(positions).all():
        raise ValueError("non-finite final positions")
    point = atoms.copy()
    point.calc = calculator
    calls = 0
    if progress is None:
        progress = {}
    progress.update(completed=0, requested=0)

    def evaluate(values):
        nonlocal calls
        point.positions[:] = values
        point.calc.reset()
        progress["requested"] += 1
        energy = float(point.get_potential_energy())
        forces = np.asarray(point.get_forces(apply_constraint=False))
        calls += 1
        progress["completed"] = calls
        if not np.isfinite(energy) or forces.shape != positions.shape or not np.isfinite(forces).all():
            raise ValueError("non-finite or malformed independent Cu-H2 evaluation")
        return energy, forces

    energy, forces = evaluate(positions)
    force = float(np.max(np.abs(forces[-2:])))
    reasons = []
    index = None
    eigenvalues = []
    if force > force_stop:
        reasons.append("force_norm")
    else:
        step = hydrogen_hessian_step()
        hessian = np.empty((6, 6))
        for column in range(6):
            upper, lower = positions.copy(), positions.copy()
            row, axis = divmod(column, 3)
            upper[-2 + row, axis] += step
            lower[-2 + row, axis] -= step
            hessian[:, column] = -(evaluate(upper)[1][-2:].reshape(-1)
                                   - evaluate(lower)[1][-2:].reshape(-1)) / (2 * step)
        hessian = (hessian + hessian.T) / 2
        index = vibrational_index(hessian, positions[-2:], False)
        eigenvalues = np.linalg.eigvalsh(hessian).tolist()
        tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(hessian)))
        if index != 1:
            reasons.append("hessian_index")
        if any(abs(value) <= tolerance for value in eigenvalues):
            reasons.append("unresolved_mode")
    return {"energy": energy, "force_norm": force, "force_norm_kind": "linf",
            "hessian_index": index, "expected_index": 1, "verify_calls": calls,
            "verification_requests": progress["requested"], "verify_calls_complete": True,
            "hessian_eigenvalues": eigenvalues, "reasons": reasons}


def _execute(job: dict) -> dict:
    from ase.io import write

    directory = Path(job["directory"])
    atoms = write_inputs(directory, job)
    argv = command(job["runtime"]["binary"], directory, job["options"])
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = str(job["options"]["threads"])
    inputs = {name: digest(directory / name) for name in ("pos.con", "direction.dat", "params.bin")}
    (directory / "command.json").write_text(json.dumps({
        "argv": argv, "runtime": job["runtime"], "oracle": job["oracle"],
        "input_sha256": inputs, "source_input_sha256": job["problem"]["input"]["sha256"],
        "native_source_sha256": job["problem"]["native_sources"]["sha256"],
        "rpc_client": job["rpc_client"],
        "displacement": job["displacement"],
        "environment": {key: env[key] for key in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }, indent=2) + "\n")
    began = time.perf_counter()
    with (directory / "search.log").open("w") as stream:
        status = run_search(argv, directory, env, stream, job["options"]["timeout"])
    search_wall = time.perf_counter() - began
    text = (directory / "search.log").read_text(errors="replace")
    matches = re.findall(r"^Oracle calls \(dimer total\): (\d+)$", text, re.M)
    calls = int(matches[-1]) if matches else None
    reasons = []
    if status == 124:
        reasons.append("timeout")
    elif status not in (0, 2):
        reasons.append("crash")
    convergence = re.findall(r"Converged:\s*(yes|no)", text, re.I)
    if status != 0 or not convergence or convergence[-1].lower() != "yes":
        reasons.append("not_converged")
    if calls is None:
        reasons.append("missing_count")
    elif calls >= job["problem"]["convergence"]["max_force_calls"]:
        reasons.append("cap")
    checked = {}
    progress = {"completed": 0, "requested": 0}
    verification_start = time.perf_counter()
    try:
        positions = final_positions(text, atoms.positions)
        final = atoms.copy()
        final.positions[:] = positions
        write(directory / "saddle.con", final, format="eon")
        module = load_module(job["rpc_sources"]["CppCore/rgpot/rpc/ase_calculator.py"],
                             "gpridge_rgpot_calculator")
        with module.RgpotCalculator.spawn(
                server_bin=job["oracle"]["binary"], potential="CuH2",
                request_timeout=job["options"]["verification_timeout"]) as calc:
            checked = verify_positions(atoms, positions, calc,
                                       job["problem"]["convergence"]["force_norm"], progress)
        reasons.extend(checked.pop("reasons"))
    except Exception as error:
        reasons.append("invalid_output")
        checked.update(error=str(error), verify_calls=progress["completed"],
                       verification_requests=progress["requested"], verify_calls_complete=False)
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2", "start": job["name"],
        "counter": "client", "comparable": False,
        "force_calls": calls if calls is not None else 0, "count_available": calls is not None,
        "success": failure is None and not reasons, "failure": failure,
        "reasons": reasons, "returncode": status,
        "force_call_acceptance_limit": job["problem"]["convergence"]["max_force_calls"],
        "force_call_limit_enforced": False,
        "search_wall_seconds": search_wall,
        "verification_wall_seconds": time.perf_counter() - verification_start,
        "wall_seconds": time.perf_counter() - began, "work": str(directory), **checked,
    }
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run(root: Path, problem_name: str, template_name: str, *, output: Path,
        workers: int = 1, work: Path | None = None, limit: int = 0) -> dict:
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, "gpridge"))
    if (problem["id"] != "tsopt/cuh2" or template["problem"] != problem["id"]
            or problem["potential"]["kind"] != "cuh2"
            or problem["convergence"]["norm"] != "linf"):
        raise ValueError("native Cu-H2 dimer requires its fixed-substrate problem contract")
    frozen_arguments(template["gpridge"])
    deadline = template["gpridge"].get("verification_timeout")
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not np.isfinite(deadline) or deadline <= 0):
        raise ValueError("verification_timeout must be finite and positive")
    files = checked_files(root / problem["input"]["directory"], problem["input"]["sha256"])
    sources = checked_files(Path(os.environ["GPRIDGE_SOURCE"]), problem["native_sources"]["sha256"])
    rpc_sources, rpc_client = checked_rpc_client(
        Path(os.environ["GPRIDGE_RGPOT_SOURCE"]), problem["rpc_client"],
        sources["subprojects/rgpot/CppCore/rgpot/rpc/Potentials.capnp"])
    identity = runtime(template, "gprd")
    oracle = runtime(template, "potserv")
    displacements = problem["sample"]["displacements"]
    if limit > 0:
        displacements = displacements[:limit]
    work = (work or output.parent / output.stem).resolve()
    jobs = [{"name": f"displacement-{value:g}", "displacement": value,
             "directory": str(work / f"displacement-{value:g}"),
             "files": files, "sources": sources, "problem": problem,
             "rpc_sources": rpc_sources, "rpc_client": rpc_client,
             "runtime": identity, "oracle": oracle, "options": template["gpridge"]}
            for value in displacements]
    records = run_jobs(jobs, _execute, workers)
    settings = {"cap": problem["convergence"]["max_force_calls"]}
    return write_campaign(output, records, template, problem, settings,
                          {"runtime": identity, "oracle": oracle, "gpridge": template["gpridge"],
                           "rpc_client": rpc_client,
                           "input_sha256": problem["input"]["sha256"],
                           "native_source_sha256": problem["native_sources"]["sha256"]},
                          images=None, counter="client",
                          comment="# Native Cu-H2 search; independent frozen-substrate validation.")
