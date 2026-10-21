"""Minimize OptBench starts with vc-SQNM and ``HarnessCounter``.

One force call is one entry into the harness potential. An energy read
and a force read of the same geometry are one call. On a variable-cell
row the cell is part of that key, and ``dE/dA`` is produced by the same
entry. The analytic Hessian is not a force call.

SQNM returns a new coordinate array. The next step has to be that array.
The Python package also multiplies the lattice derivative by minus one,
so a variable-cell step is given ``-dE/dA``. The ASE wrapper builds the
same input from ``cell_stress``.

A 2014 success is under the cap and at the problem force stop. A v2
success also has Hessian index 0. The variable-cell loop stops only when
the atomic norm and the max-norm of ``dE/dA`` are both at that stop, so
the recorded point meets the atomic test or it has hit the cap.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import (
    lennard_jones,
    lennard_jones_hessian,
    morse_bulk,
    morse_bulk_hessian,
    read_con,
)
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance, write_starts
from optbench.vcsqnm_catalog import CODE_VERSION, COMMIT, DIST_VERSION, SOURCE_SHA256
from optbench.vcsqnm_pes import cell_stress, morse_cell, morse_cell_hessian, sqnm_lattice_derivative

EIGENVALUE_TOL = 1e-6
_METHODS = frozenset({"sqnm", "ase", "vc", "ase-vc"})
_VARIABLE = frozenset({"vc", "ase-vc"})
_ASE = frozenset({"ase", "ase-vc"})
_PARAMETERS = ("initial_step_size", "nhist_max", "alpha_min", "eps_subsp", "lattice_weight")
_PIN_OK = False
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


@contextlib.contextmanager
def _quiet_io():
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield stdout, stderr


def _desynced(stdout: io.StringIO, stderr: io.StringIO) -> str:
    text = stdout.getvalue() + "\n" + stderr.getvalue()
    if "not called with positions" in text:
        return "SQNM was not called with positions that were expected"
    return ""


def assert_pin() -> None:
    """The installed sources are tag v1.3, whose distribution version is 1.2."""
    global _PIN_OK
    if _PIN_OK:
        return
    import importlib.metadata

    import sqnm

    version = importlib.metadata.version("sqnm")
    if version != DIST_VERSION:
        raise RuntimeError(f"sqnm {version} is not distribution {DIST_VERSION} from tag v1.3")
    root = Path(list(sqnm.__path__)[0])
    for name, digest in SOURCE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"installed sqnm is missing {name}")
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != digest:
            raise RuntimeError(f"installed {name} sha256 {found} is not the v1.3 pin")
    _PIN_OK = True


def _kind(problem_id: str) -> str:
    if problem_id in ("minimization/lj38", "minimization/lj38-noise"):
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no vc-SQNM potential")


def _parameters(section: dict) -> dict:
    missing = [key for key in _PARAMETERS if key not in section]
    if missing:
        raise ValueError(f"sqnm section needs {', '.join(missing)}")
    method = str(section["method"])
    if method not in _METHODS:
        raise ValueError(f"unknown sqnm method {method}")
    return {
        "method": method,
        "initial_step_size": float(section["initial_step_size"]),
        "nhist_max": int(section["nhist_max"]),
        "alpha_min": float(section["alpha_min"]),
        "eps_subsp": float(section["eps_subsp"]),
        "lattice_weight": float(section["lattice_weight"]),
    }


def _columns(flat: np.ndarray) -> np.ndarray:
    """Atom-major coordinates as a (3, nat) array."""
    count = int(flat.size)
    if count % 3 != 0:
        raise ValueError("coordinates must be a multiple of 3")
    return np.ascontiguousarray(np.asarray(flat, dtype=np.float64).reshape(count // 3, 3).T)


def _flatten(cols: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(cols, dtype=np.float64)).T.reshape(-1).copy()


def _cell_ok(cell: np.ndarray) -> bool:
    if not np.isfinite(cell).all():
        return False
    det = float(np.linalg.det(cell))
    return math.isfinite(det) and abs(det) >= 1e-18


class _Surface:
    """One harness entry per geometry. The cell joins the key when it moves."""

    def __init__(self, kind: str, box, cap: int, variable: bool):
        self.kind = kind
        self.box = None if box is None else np.asarray(box, dtype=np.float64)
        self.variable = variable
        self.counter = HarnessCounter(cache=True, cap=cap)
        self.last_deriv: np.ndarray | None = None

    def evaluate(self, coords, cell=None):
        import numpy as np

        flat = np.asarray(coords, dtype=np.float64).reshape(-1)
        matrix = None if cell is None else np.asarray(cell, dtype=np.float64)
        parts = [float(value) for value in flat]
        if self.variable:
            if matrix is None:
                raise ValueError("variable-cell SQNM needs a cell")
            parts.extend(float(value) for value in matrix.reshape(-1))
        key = tuple(parts)
        fresh, cached = self.counter.charge(key)
        if not fresh and cached is not None:
            energy, grad_t, deriv_t = cached
            self.last_deriv = None if deriv_t is None else np.asarray(deriv_t, dtype=np.float64).reshape(3, 3)
            return float(energy), np.asarray(grad_t, dtype=np.float64)
        if self.kind == "lj":
            energy, grad = lennard_jones(flat)
            deriv = None
        elif self.variable:
            energy, grad, deriv = morse_cell(flat, matrix)
            deriv = np.asarray(deriv, dtype=np.float64)
        else:
            energy, grad = morse_bulk(flat, self.box)
            deriv = None
        self.last_deriv = None if deriv is None else deriv
        deriv_t = None if deriv is None else tuple(float(value) for value in np.asarray(deriv).reshape(-1))
        self.counter.remember(key, (float(energy), tuple(float(value) for value in grad), deriv_t))
        return float(energy), np.asarray(grad, dtype=np.float64)


def _index_at(kind: str, point: list[float], box, cell, force_stop: float, variable: bool):
    import numpy as np

    if kind == "lj":
        matrix = lennard_jones_hessian(point)
        values = vibrational_eigenvalues(matrix, point, rotations=True)
        tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    elif variable:
        matrix = morse_cell_hessian(point, cell)
        values = np.linalg.eigvalsh(matrix).tolist()
        tolerance = EIGENVALUE_TOL
    else:
        matrix = morse_bulk_hessian(point, box)
        values = np.linalg.eigvalsh(matrix).tolist()
        tolerance = EIGENVALUE_TOL
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _optimizer(method: str, nat: int, parameters: dict, cell: np.ndarray | None):
    if method == "sqnm":
        from sqnm.free_or_fixed_cell_sqnm import free_sqnm

        return free_sqnm(
            nat,
            parameters["initial_step_size"],
            parameters["nhist_max"],
            parameters["alpha_min"],
            parameters["eps_subsp"],
        )
    if method == "vc":
        from sqnm.periodic_sqnm import periodic_sqnm

        return periodic_sqnm(
            nat,
            cell,
            parameters["initial_step_size"],
            parameters["nhist_max"],
            parameters["lattice_weight"],
            parameters["alpha_min"],
            parameters["eps_subsp"],
        )
    raise ValueError(method)


def _ase_optimizer(atoms, method: str, parameters: dict, cap: int):
    from sqnm.vcsqnm_for_ase import aseOptimizer

    return aseOptimizer(
        atoms,
        vc_relax=method == "ase-vc",
        force_tol=0.0,
        maximalSteps=cap + 1,
        initial_step_size=parameters["initial_step_size"],
        nhist_max=parameters["nhist_max"],
        lattice_weigth=parameters["lattice_weight"],
        alpha_min=parameters["alpha_min"],
        eps_subsp=parameters["eps_subsp"],
    )


def _atoms(flat: np.ndarray, cell: np.ndarray | None):
    from ase import Atoms

    rows = flat.reshape(-1, 3)
    if cell is None:
        return Atoms(symbols=["H"] * len(rows), positions=rows, pbc=False)
    return Atoms(symbols=["H"] * len(rows), positions=rows, cell=np.asarray(cell, dtype=np.float64).T, pbc=True)


class _Calculator:
    """Fill energy, forces, and stress in one calculate so ASE's cache is one call."""

    def __new__(cls, evaluate, variable: bool):
        import numpy as np
        from ase.calculators.calculator import Calculator, all_changes

        class _Harness(Calculator):
            implemented_properties = ["energy", "forces", "stress"] if variable else ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
                Calculator.calculate(self, atoms, properties, system_changes)
                coords = np.asarray(self.atoms.positions, dtype=np.float64).reshape(-1)
                matrix = None
                if variable:
                    matrix = np.asarray(self.atoms.cell.array, dtype=np.float64).T.copy()
                energy, grad = evaluate(coords, matrix)
                self.results["energy"] = float(energy)
                self.results["forces"] = -np.asarray(grad, dtype=np.float64).reshape(-1, 3)
                if variable:
                    deriv = evaluate.last_deriv
                    self.results["stress"] = cell_stress(deriv, matrix)

        return _Harness()


def drive(coords, method: str, parameters: dict, evaluate, stop, calls, cap: int, cell=None) -> dict:
    """Step until ``stop`` or the harness cap.

    ``evaluate(flat, cell)`` returns energy and the gradient the stepper
    sees. ``stop()`` reads that evaluation. ``calls()`` is the harness tally.
    """
    import numpy as np

    flat = np.asarray(coords, dtype=np.float64).reshape(-1).copy()
    nat = flat.size // 3
    matrix = None if cell is None else np.asarray(cell, dtype=np.float64).copy()
    variable = method in _VARIABLE
    error = ""
    stopped = False
    diverged = False
    if variable and (matrix is None or matrix.shape != (3, 3)):
        raise ValueError("variable-cell SQNM needs the starting cell")
    try:
        if method in _ASE:
            atoms = _atoms(flat, matrix)
            atoms.calc = _Calculator(evaluate, variable)
            with _quiet_io() as (stdout, stderr):
                opt = _ase_optimizer(atoms, method, parameters, cap)
            error = _desynced(stdout, stderr)
            if error:
                flat = np.asarray(atoms.positions, dtype=np.float64).reshape(-1).copy()
            else:
                while not error:
                    if stop():
                        stopped = True
                        break
                    if calls() >= cap:
                        break
                    with _quiet_io() as (stdout, stderr):
                        opt.step(atoms)
                    error = _desynced(stdout, stderr)
                    flat = np.asarray(atoms.positions, dtype=np.float64).reshape(-1).copy()
                    if variable:
                        matrix = np.asarray(atoms.cell.array, dtype=np.float64).T.copy()
                    if error:
                        break
                    if not np.isfinite(flat).all() or (variable and not _cell_ok(matrix)):
                        diverged = True
                        break
                    if calls() >= cap:
                        break
                    atoms.get_forces()
        else:
            with _quiet_io() as (stdout, stderr):
                opt = _optimizer(method, nat, parameters, matrix)
            error = _desynced(stdout, stderr)
            while not error:
                if calls() >= cap:
                    break
                if variable and not _cell_ok(matrix):
                    diverged = True
                    break
                evaluate(flat, matrix)
                if stop():
                    stopped = True
                    break
                if calls() >= cap:
                    break
                cols = _columns(flat)
                forces = _columns(-evaluate.grad)
                with _quiet_io() as (stdout, stderr):
                    if variable:
                        deriv = sqnm_lattice_derivative(evaluate.last_deriv)
                        updated, matrix = opt.optimizer_step(
                            cols, matrix, float(evaluate.energy), forces, deriv
                        )
                        matrix = np.asarray(matrix, dtype=np.float64).copy()
                    else:
                        updated = opt.optimizer_step(cols, float(evaluate.energy), forces)
                error = _desynced(stdout, stderr)
                flat = _flatten(updated)
                if not np.isfinite(flat).all() or (variable and matrix is not None and not _cell_ok(matrix)):
                    diverged = True
                    break
    except ForceCallCap as exc:
        error = f"ForceCallCap: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "point": [float(value) for value in flat],
        "cell": None if matrix is None else [[float(value) for value in row] for row in matrix],
        "stopped": stopped and not error and not diverged,
        "error": error,
        "diverged": diverged,
    }


class _Seen:
    """Last evaluation, for the stop test and the ASE calculator."""

    def __init__(self, surface: _Surface):
        self._surface = surface
        self.energy = None
        self.grad = None
        self.last_deriv = None
        self.norm = None
        self.lattice = None

    def __call__(self, flat, cell=None):
        energy, grad = self._surface.evaluate(flat, cell)
        self.energy = float(energy)
        self.grad = np.asarray(grad, dtype=np.float64).reshape(-1)
        self.last_deriv = self._surface.last_deriv
        self.norm = float(np.linalg.norm(self.grad))
        self.lattice = None if self.last_deriv is None else float(np.max(np.abs(self.last_deriv)))
        return self.energy, self.grad


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    import numpy as np

    assert_pin()
    calls = 0
    verify_calls = 0
    error = ""
    energy = None
    force_norm = None
    point = None
    cell = None
    surface = None
    stopped = False
    diverged = False
    gtol = float(job["gtol"])
    cap = int(job["cap"])
    method = str(job["method"])
    variable = method in _VARIABLE
    try:
        if job["kind"] == "lj" and variable:
            raise ValueError("Lennard-Jones 38 has no cell coordinate")
        flat = np.asarray(job["coords"], dtype=np.float64).reshape(-1).copy()
        box = None if job.get("box") is None else np.asarray(job["box"], dtype=np.float64)
        matrix = None if not variable else np.diag(box)
        surface = _Surface(job["kind"], box, cap, variable)
        seen = _Seen(surface)

        def stop() -> bool:
            if seen.norm is None or not math.isfinite(seen.norm):
                return False
            if not variable:
                return seen.norm <= gtol
            return (
                seen.lattice is not None
                and math.isfinite(seen.lattice)
                and seen.norm <= gtol
                and seen.lattice <= gtol
            )

        result = drive(
            flat,
            method,
            job["parameters"],
            seen,
            stop,
            lambda: surface.counter.calls,
            cap,
            matrix,
        )
        point = result["point"]
        cell = result["cell"]
        stopped = bool(result["stopped"])
        error = str(result["error"])
        diverged = bool(result["diverged"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if surface is not None:
        calls = int(surface.counter.calls)
        if (
            not diverged
            and point is not None
            and all(math.isfinite(value) for value in point)
            and not (variable and cell is None)
        ):
            try:
                with surface.counter.verification():
                    probe = None if cell is None else np.asarray(cell, dtype=np.float64)
                    energy, gradient = surface.evaluate(point, probe)
                force_norm = float(np.linalg.norm(gradient))
                verify_calls = int(surface.counter.verify_calls)
            except Exception as exc:
                if not error:
                    error = f"{type(exc).__name__}: {exc}"
    hessian_index = None
    min_eigenvalues: list[float] = []
    if point is not None and all(math.isfinite(value) for value in point) and (not error or "ForceCallCap" in error):
        try:
            hessian_index, min_eigenvalues = _index_at(
                job["kind"], point, job.get("box"), cell, gtol, variable
            )
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    finite = (
        point is not None
        and all(math.isfinite(value) for value in point)
        and energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    cap_hit = "ForceCallCap" in error or "force-call cap" in error
    at_force = finite and stopped and force_norm is not None and force_norm <= gtol and calls < cap and not error
    reasons: list[str] = []
    if error and not cap_hit:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif calls >= cap or cap_hit:
        reasons.append("cap")
    elif not at_force:
        reasons.append("force_norm")
    elif hessian_index != 0:
        reasons.append("hessian_index")
    failure = classify_failure(reasons)
    lattice_max = None
    if variable and surface is not None and surface.last_deriv is not None and finite:
        lattice_max = float(np.max(np.abs(surface.last_deriv)))
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": verify_calls,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": force_norm,
        "hessian_index": hessian_index,
        "expected_index": 0,
        "converged": at_force,
        "hessian_calls": 0,
        "min_eigenvalues": min_eigenvalues,
        "method": method,
    }
    if lattice_max is not None:
        record["lattice_deriv_max"] = lattice_max
    if error:
        record["error"] = error
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter harness. One force call is one energy and gradient entry.",
        "# A repeated coordinate is cached. The refused call at the cap is not counted.",
        "# The analytic Hessian is not a force call.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
        "# energy_min and energy_max are the final potential values on the recorded points.",
    ]
    if str(template.get("problem")) == "minimization/lj38":
        lines.append(
            "# LJ38 index removes the six rigid modes. A curvature is negative only below "
            "-max(force_norm, 1e-8*||H||_F, 1e-6)."
        )
    if str(template.get("sqnm", {}).get("method")) in _VARIABLE:
        lines.append(
            "# The cell is a coordinate. The loop also holds max |dE/dA| to the problem force norm. "
            "Python periodic_sqnm is given -dE/dA."
        )
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
) -> dict:
    _quiet_threads()
    assert_pin()
    if template.get("engine") != "vcsqnm":
        raise ValueError(f"{template['id']} is not a vc-SQNM template")
    if str(template.get("commit") or "") != COMMIT:
        raise RuntimeError(f"{template['id']} commit is not {COMMIT}")
    if str(template.get("code_version") or "") != CODE_VERSION:
        raise RuntimeError(f"{template['id']} code_version is not tag {CODE_VERSION}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    optimizer = (template.get("ini") or {}).get("Optimizer") or {}
    if "converged_force" in optimizer and abs(float(optimizer["converged_force"]) - gtol) > 1e-12:
        raise ValueError("template converged_force is not the problem force norm")
    if "max_iterations" in optimizer and int(optimizer["max_iterations"]) < cap:
        raise ValueError("template max_iterations is below the problem force-call cap")
    parameters = _parameters(template.get("sqnm") or {})
    if parameters["method"] in _VARIABLE and problem["id"] != "minimization/morse-bulk":
        raise ValueError(f"{template['id']} moves the cell, and {problem['id']} has no cell coordinate")
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    if problem["id"] == "minimization/morse-bulk" and geometries is None and not limit and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    kind = _kind(problem["id"])
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if kind == "lj" else box.tolist(),
                "kind": kind,
                "method": parameters["method"],
                "parameters": {key: parameters[key] for key in _PARAMETERS},
                "gtol": gtol,
                "cap": cap,
            }
        )
    if workers <= 1:
        records = [_minimize_job(job) for job in jobs]
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_minimize_job, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                if done == total or done % 50 == 0:
                    print(f"{template['id']} {done}/{total}", flush=True)
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["converged"]]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in records
        ],
        cap,
    )
    passed = {"gtol": gtol, "maxiter": cap, **parameters}
    provenance = collect_provenance(
        code=str(template.get("code")),
        code_version=CODE_VERSION,
        commit=COMMIT,
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. Analytic Hessians are not force calls. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm", parameters["method"]),
                "code_version": CODE_VERSION,
                "commit": COMMIT,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": passed,
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "force_calls": summary.get("force_calls"),
                    "force_calls_min": summary.get("force_calls_min"),
                    "force_calls_max": summary.get("force_calls_max"),
                    "nfailed": summary.get("nfailed"),
                    "median": summary.get("median"),
                },
                "starts": records,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, COMMIT, CODE_VERSION), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    return summary
