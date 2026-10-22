"""Minimum-mode and P-RFO campaigns through the rgsaddle C ABI.

The cost is ``RgpotCounter.calculate`` on a ``HarnessCounter``. Library
evaluation counters are recorded beside that number and are not the
score. A tsopt success is
a point under the force cap, at the Euclidean force stop, whose
Cartesian Hessian has index 1. The reference geometry passed to
``judge_stationary`` is the converged point, so the energy and Kabsch
tests do not add a second stationary-point target; a non-saddle fails
on ``hessian_index``.

Nonlinear CG is not a min-mode translation here. ``rgsaddle`` refuses
any stepper that line-searches the value next to a projected force
(``check_force_driven``). The C ABI then maps method 1 to L-BFGS and
every other accepted code to FIRE.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .connection import ProcessTable, per_atom_max, periodic_lengths, tolerances
from .connection import connected as endpoints_connected
from .counter import HarnessCounter
from .failures import classify_failure
from .potserver import RgpotCounter
from .score import half_even, score_v2
from .stationary import judge_stationary, negative_index, numerical_hessian
from .surfaces import load_con, potential_from_problem, read_mode
from .v2 import collect_provenance, write_starts

ABI_MAJOR = 1
ABI_MINOR = 16
# Endpoint minimizations sit outside the climb cap. One min-mode step can
# finish after the climb counter has crossed max_iterations, so the
# per-search PAR-2 cap leaves room for that step.
_SADDLE_ENDPOINT_BUDGET = 2000
_SADDLE_CLIMB_OVERSHOOT = 64
OK = 0
METHOD_FIRE = 0
METHOD_LBFGS = 1
KIND_DIMER = 0
KIND_LANCZOS = 1
GATE_L2 = 0
EXPAND = {"lanczos": 0, "gd": 1, "jd0": 2, "jd0-alt": 3, "mjd0": 4, "mjd0-alt": 5}
HESS = {"bfgs": 0, "ts-bfgs": 1}
RESTRICTED = {"trust-region": 0, "max-internal": 1, "restricted-atomic": 2}


def _library_path() -> Path:
    env = os.environ.get("RGSADDLE_LIB")
    if not env:
        raise RuntimeError("set RGSADDLE_LIB to the librgsaddle.so built from the pinned commit")
    path = Path(env)
    if not path.is_file():
        raise RuntimeError(f"RGSADDLE_LIB is not a file: {path}")
    return path


def _assert_source_commit(expected: str) -> None:
    src = os.environ.get("RGSADDLE_SRC")
    if not src or not expected:
        return
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    if head != expected:
        raise RuntimeError(f"rgsaddle HEAD {head} does not match the template commit {expected}")


_LIB = None


def _load():
    import ctypes

    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(str(_library_path()))

    class MinModeConfig(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_uint32),
            ("minor", ctypes.c_uint32),
            ("flags", ctypes.c_uint64),
            ("kind", ctypes.c_int32),
            ("method", ctypes.c_int32),
            ("dr", ctypes.c_double),
            ("rotation_tol", ctypes.c_double),
            ("max_rotations", ctypes.c_int64),
            ("krylov_dim", ctypes.c_int64),
            ("force_tol", ctypes.c_double),
            ("max_move", ctypes.c_double),
            ("rotation_angle_tol", ctypes.c_double),
            ("difference", ctypes.c_int32),
            ("reserved", ctypes.c_int32),
        ]

    class SellaConfig(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_uint32),
            ("minor", ctypes.c_uint32),
            ("flags", ctypes.c_uint64),
            ("delta", ctypes.c_double),
            ("force_tol", ctypes.c_double),
            ("force_gate", ctypes.c_int32),
            ("order", ctypes.c_int64),
        ]

    class ConstraintsConfig(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_uint32),
            ("minor", ctypes.c_uint32),
            ("flags", ctypes.c_uint64),
        ]

    class Request(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_uint32),
            ("minor", ctypes.c_uint32),
            ("flags", ctypes.c_uint64),
            ("n_images", ctypes.c_int64),
            ("n_atoms", ctypes.c_int64),
            ("positions", ctypes.POINTER(ctypes.c_double)),
            ("energies", ctypes.POINTER(ctypes.c_double)),
            ("gradients", ctypes.POINTER(ctypes.c_double)),
            ("image", ctypes.c_int64),
        ]

    class Report(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_uint32),
            ("minor", ctypes.c_uint32),
            ("flags", ctypes.c_uint64),
            ("status", ctypes.c_int32),
            ("evaluations", ctypes.c_int32),
            ("max_force", ctypes.c_double),
            ("ci_index", ctypes.c_int64),
            ("iteration", ctypes.c_int64),
            ("curvature", ctypes.c_double),
            ("rotations", ctypes.c_int64),
        ]

    surface_fn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Request))
    lib.Request = Request
    lib.Report = Report
    lib.MinModeConfig = MinModeConfig
    lib.SellaConfig = SellaConfig
    lib.ConstraintsConfig = ConstraintsConfig
    lib.surface_fn = surface_fn
    lib.rgsaddle_last_error.restype = ctypes.c_char_p
    for name, restype, argtypes in (
        ("rgsaddle_minmode_create", ctypes.c_void_p, [ctypes.POINTER(MinModeConfig), ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]),
        ("rgsaddle_minmode_step", ctypes.c_int, [ctypes.c_void_p, surface_fn, ctypes.c_void_p, ctypes.POINTER(Report)]),
        ("rgsaddle_minmode_estimate", ctypes.c_int, [ctypes.c_void_p, surface_fn, ctypes.c_void_p, ctypes.POINTER(Report)]),
        ("rgsaddle_minmode_position", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]),
        ("rgsaddle_minmode_mode", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]),
        ("rgsaddle_minmode_set_force_gate", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int32]),
        ("rgsaddle_minmode_free", None, [ctypes.c_void_p]),
        ("rgsaddle_sella_saddle_create", ctypes.c_void_p, [ctypes.POINTER(SellaConfig), ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]),
        ("rgsaddle_sella_saddle_create_on", ctypes.c_void_p, [ctypes.POINTER(SellaConfig), ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_void_p]),
        ("rgsaddle_sella_saddle_step", ctypes.c_int, [ctypes.c_void_p, surface_fn, ctypes.c_void_p, ctypes.POINTER(Report)]),
        ("rgsaddle_sella_saddle_position", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]),
        ("rgsaddle_sella_saddle_set_hess_update", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int32]),
        ("rgsaddle_sella_saddle_set_expand", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int32]),
        ("rgsaddle_sella_saddle_set_restricted", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int32]),
        ("rgsaddle_sella_saddle_seed_mode", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_double]),
        ("rgsaddle_sella_saddle_free", None, [ctypes.c_void_p]),
        ("rgsaddle_constraints_create", ctypes.c_void_p, [ctypes.POINTER(ConstraintsConfig), ctypes.c_int64]),
        ("rgsaddle_constraints_free", None, [ctypes.c_void_p]),
    ):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    _LIB = lib
    return lib


def _section(template: dict, name: str) -> dict:
    return ((template.get("ini") or {}).get(name) or {})


def _stored(root: Path, archive_path: str) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(archive_path)


def _extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.is_dir():
        return archive
    if not tarfile.is_tarfile(archive):
        return archive
    marker = dest / ".extracted"
    if marker.is_file():
        return dest
    # Several asv processes unpack the same archive. One holds the lock.
    dest.mkdir(parents=True, exist_ok=True)
    lock_path = dest / ".extract.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not marker.is_file():
            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text("ok\n", encoding="utf-8")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return dest


def _norm(values) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _unit(values: np.ndarray) -> np.ndarray:
    scale = float(np.linalg.norm(values))
    if not math.isfinite(scale) or scale < 1e-14:
        out = np.zeros_like(values)
        out[0] = 1.0
        return out
    return values / scale


class HarnessSurface:
    """In-process surface. Each library request is one ``RgpotCounter.calculate``."""

    def __init__(self, evaluate):
        def rgpot_eval(positions, atomic_numbers=None, box=None):
            energy, gradient = evaluate(positions)
            return energy, gradient

        self.rgpot = RgpotCounter(rgpot_eval, HarnessCounter(cache=True))

    @property
    def calls(self) -> int:
        return self.rgpot.counter.calls

    @property
    def verify_calls(self) -> int:
        return self.rgpot.counter.verify_calls

    def energy_forces(self, coords):
        return self.rgpot.calculate(coords)

    def verification(self):
        return self.rgpot.counter.verification()


def _outcome(reasons: list[str]) -> tuple[bool, str | None]:
    failure = classify_failure(reasons)
    return failure is None, failure


def _callback(lib, potential, dof: int):
    errors: list[str] = []

    def surface(_user, req) -> int:
        try:
            body = req.contents
            rows = int(body.n_images)
            width = int(body.n_atoms) * 3
            if rows < 1 or width != dof:
                errors.append(f"shape rows={rows} width={width} dof={dof}")
                return -6
            positions = np.ctypeslib.as_array(body.positions, shape=(rows * width,))
            energies = np.ctypeslib.as_array(body.energies, shape=(rows,))
            gradients = np.ctypeslib.as_array(body.gradients, shape=(rows * width,))
            for row in range(rows):
                sl = slice(row * width, (row + 1) * width)
                energy, gradient = potential.energy_forces(np.array(positions[sl], dtype=np.float64, copy=True))
                energies[row] = float(energy)
                gradients[sl] = np.asarray(gradient, dtype=np.float64)
            return 0
        except Exception as exc:
            errors.append(repr(exc))
            return -6

    wrapped = lib.surface_fn(surface)
    wrapped.errors = errors
    return wrapped


def _position(lib, read, session, dof: int) -> np.ndarray:
    import ctypes

    out = np.empty(dof, dtype=np.float64)
    rc = read(session, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc != OK:
        raise RuntimeError(f"rgsaddle position read returned {rc}")
    return out


def _mode(lib, session, dof: int) -> np.ndarray:
    import ctypes

    out = np.empty(dof, dtype=np.float64)
    rc = lib.rgsaddle_minmode_mode(session, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc != OK:
        raise RuntimeError(f"rgsaddle mode read returned {rc}")
    return out


def _blank_report(lib):
    report = lib.Report()
    report.major = ABI_MAJOR
    report.minor = ABI_MINOR
    return report


def _judge_saddle(point: np.ndarray, energy: float, gradient: np.ndarray, *, force_tol: float, cap: int, calls: int, converged: bool) -> dict:
    counted = HarnessSurface(lambda _coords: (energy, gradient))
    # The Hessian is taken on the search potential by the caller. This
    # helper only packages a point that was already differentiated.
    del counted
    coords = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    return {
        "energy": float(energy),
        "force_norm": _norm(gradient),
        "converged": bool(converged) and _norm(gradient) <= force_tol and calls < cap,
        "coords": coords,
    }


def _rigid_basis(point: np.ndarray) -> np.ndarray:
    """Orthonormal translations and rotations of a free cluster.

    One atom has no separate rigid subspace: its three coordinates are
    the whole Hessian. Two atoms have translations and one rotation.
    """
    pos = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    count = len(pos)
    if count < 2:
        return np.zeros((point.size, 0), dtype=np.float64)
    columns = []
    for axis in range(3):
        column = np.zeros_like(pos)
        column[:, axis] = 1.0
        columns.append(column.reshape(-1))
    if count >= 3:
        centered = pos - pos.mean(axis=0)
        for axis in range(3):
            unit = np.zeros(3, dtype=np.float64)
            unit[axis] = 1.0
            columns.append(np.cross(centered, unit).reshape(-1))
    basis = np.column_stack(columns)
    _q, r = np.linalg.qr(basis)
    keep = np.abs(np.diag(r)) > 1e-8
    return _q[:, keep]


def _vibrational_eigenvalues(point: np.ndarray, eigenvalues: list[float], vectors) -> tuple[list[float], int]:
    """Drop numerical rigid modes before the index count.

    A free cluster's Cartesian Hessian has six null modes. Central
    differences leave some of them a few times 1e-6 below zero, and
    ``negative_index`` would count those as extra imaginary
    frequencies. A mode is removed only when it lies in the rigid
    subspace (overlap at least 0.9) and its eigenvalue is smaller than
    1e-3 in magnitude. The saddle threshold stays 1e-6. A fixed
    substrate has no separate rigid subspace in the free coordinates,
    so the caller skips this.
    """
    basis = _rigid_basis(point)
    if basis.shape[1] == 0:
        return list(eigenvalues), 0
    matrix = np.asarray(vectors, dtype=np.float64)
    adjusted = []
    removed = 0
    for index, value in enumerate(eigenvalues):
        column = matrix[:, index]
        overlap = float(np.linalg.norm(basis.T @ column))
        if overlap >= 0.9 and abs(float(value)) < 1e-3:
            adjusted.append(0.0)
            removed += 1
        else:
            adjusted.append(float(value))
    return adjusted, removed


def _hessian_spectrum(potential: HarnessSurface, point: np.ndarray) -> tuple[list[float], np.ndarray]:
    """Central-difference Hessian and its ascending eigenpairs.

    The free-atom Hessian of the Pt island is 525 by 525. A Python Jacobi
    sweep on that matrix does not finish with the search, so the spectrum
    is the symmetric LAPACK solve. Eigenvalues match the Jacobi sort.
    """
    flat = [float(value) for value in np.asarray(point, dtype=np.float64)]
    with potential.verification():
        hessian = numerical_hessian(lambda coords: potential.energy_forces(coords)[1], flat, step=1e-5)
    matrix = np.asarray(hessian, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, vectors = np.linalg.eigh(matrix)
    return [float(value) for value in eigenvalues], vectors


def _saddle_record(
    name: str,
    potential: HarnessSurface,
    point: np.ndarray,
    *,
    cap: int,
    force_tol: float,
    library_converged: bool,
    rotations: int,
    library_evals: int,
    steps: int,
    project_rigid: bool,
) -> dict:
    energy, gradient = potential.energy_forces(point)
    raw_eigenvalues, vectors = _hessian_spectrum(potential, point)
    cartesian_index = negative_index(raw_eigenvalues, 1e-6)
    if project_rigid:
        eigenvalues, rigid_removed = _vibrational_eigenvalues(point, raw_eigenvalues, vectors)
    else:
        eigenvalues, rigid_removed = list(raw_eigenvalues), 0
    index = negative_index(eigenvalues, 1e-6)
    coords = np.asarray(point, dtype=np.float64).reshape(-1, 3).tolist()
    judged = judge_stationary(
        kind="saddle",
        eigenvalues=eigenvalues,
        energy=float(energy),
        coords=coords,
        reference_energy=float(energy),
        reference_coords=coords,
        energy_tol=0.0,
        rmsd_tol=0.0,
    )
    reasons = [reason for reason in judged["reasons"] if reason != "rmsd" and reason != "energy"]
    force_norm = _norm(gradient)
    if potential.calls >= cap:
        reasons.append("cap")
    # The harness Euclidean norm is the stop. Library status is recorded
    # on ``converged`` and does not by itself reject a point already
    # under that norm with Hessian index 1.
    if not math.isfinite(force_norm) or force_norm > force_tol + 1e-10:
        reasons.append("force_norm")
    # judge_stationary already appends hessian_index when the index is wrong.
    reasons = list(dict.fromkeys(reasons))
    success, failure = _outcome(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(potential.calls),
        "verify_calls": int(potential.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": float(energy),
        "force_norm": force_norm,
        "hessian_index": int(index),
        "cartesian_index": int(cartesian_index),
        "rigid_modes_removed": int(rigid_removed),
        "expected_index": 1,
        "lowest_eigenvalue": float(eigenvalues[0]) if eigenvalues else None,
        "lowest_eigenvalues": [float(value) for value in eigenvalues[:8]],
        "rmsd": float(judged["rmsd"]),
        "reference": "converged-point",
        "converged": bool(library_converged) and force_norm <= force_tol,
        "rotations": int(rotations),
        "library_evaluations": int(library_evals),
        "steps": int(steps),
        "x": [float(value) for value in np.asarray(point, dtype=np.float64)],
    }


def _free_view(positions: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    free = np.flatnonzero(~fixed)
    if len(free) == 0:
        raise ValueError("every atom is fixed")
    return free, positions[free].reshape(-1).copy(), positions.copy()


def _bind(raw, positions: np.ndarray, free: np.ndarray, cell: np.ndarray):
    base = np.array(positions, dtype=np.float64, copy=True)

    def evaluate(flat):
        pos = base.copy()
        pos[free] = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        energy, gradient = raw(pos, cell)
        return float(energy), np.ascontiguousarray(gradient[free].reshape(-1), dtype=np.float64)

    return evaluate


def _seed(x: np.ndarray, potential: HarnessSurface, how: str, reactant: np.ndarray | None, file_mode: np.ndarray | None) -> np.ndarray:
    if how == "reactant" and reactant is not None:
        return _unit(x - reactant)
    if how == "file" and file_mode is not None:
        return _unit(file_mode)
    _energy, gradient = potential.energy_forces(x)
    return _unit(np.asarray(gradient, dtype=np.float64))


def _minmode_config(lib, kind: str, method: str, saddle: dict, optimizer: dict, force_tol: float):
    cfg = lib.MinModeConfig()
    cfg.major = ABI_MAJOR
    cfg.minor = ABI_MINOR
    cfg.flags = 0
    cfg.kind = KIND_LANCZOS if kind == "lanczos" else KIND_DIMER
    cfg.method = METHOD_LBFGS if method == "lbfgs" else METHOD_FIRE
    cfg.dr = float(saddle.get("dr", 1e-3))
    cfg.rotation_tol = float(saddle.get("rotation_tol", 1e-4))
    cfg.max_rotations = int(saddle.get("max_rotations", 20))
    cfg.krylov_dim = int(saddle.get("krylov_dim", 12))
    cfg.force_tol = force_tol
    cfg.max_move = float(optimizer.get("max_move", 0.2))
    cfg.rotation_angle_tol = float(saddle.get("rotation_angle_tol", 0.0))
    cfg.difference = 1 if str(saddle.get("difference", "forward")) == "central" else 0
    cfg.reserved = 0
    return cfg


def _potential_for(spec: dict, positions: np.ndarray, free: np.ndarray, cell: np.ndarray) -> HarnessSurface:
    if spec.get("evaluate") is not None:
        return HarnessSurface(spec["evaluate"])
    raw = potential_from_problem({"potential": spec["potential"]})
    return HarnessSurface(_bind(raw, positions, free, cell))


def _finite_point(point: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(point)))


def _run_minmode(spec: dict) -> dict:
    os.environ["RAYON_NUM_THREADS"] = "1"
    lib = _load()
    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    free, x, _base = _free_view(positions, fixed)
    potential = _potential_for(spec, positions, free, cell)
    dof = int(x.size)
    n_atoms = dof // 3
    saddle = spec["saddle"]
    optimizer = spec["optimizer"]
    force_tol = float(spec["force_tol"])
    cap = int(spec["cap"])
    reactant = None if spec.get("reactant") is None else np.asarray(spec["reactant"], dtype=np.float64)
    file_mode = None if spec.get("file_mode") is None else np.asarray(spec["file_mode"], dtype=np.float64)
    mode = _seed(x, potential, spec["mode_seed"], reactant, file_mode)
    cfg = _minmode_config(lib, spec["kind"], spec["method"], saddle, optimizer, force_tol)
    import ctypes

    session = lib.rgsaddle_minmode_create(
        ctypes.byref(cfg),
        n_atoms,
        np.ascontiguousarray(x).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        np.ascontiguousarray(mode).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not session:
        text = lib.rgsaddle_last_error()
        detail = text.decode() if text else "null session"
        raise RuntimeError(f"rgsaddle_minmode_create failed: {detail}")
    callback = _callback(lib, potential, dof)
    surface_errors: list[str] = []
    try:
        gate_rc = lib.rgsaddle_minmode_set_force_gate(session, GATE_L2)
        if gate_rc != OK:
            raise RuntimeError(f"rgsaddle_minmode_set_force_gate returned {gate_rc}")
        rotations = 0
        library_evals = 0
        steps = 0
        converged = False
        solver_rc = OK
        final_mode = None
        if spec["task"] == "mode":
            reference = _unit(np.asarray(spec["reference_mode"], dtype=np.float64))
            overlap = 0.0
            previous = None
            report = None
            while potential.calls < cap:
                before = potential.calls
                report = _blank_report(lib)
                rc = lib.rgsaddle_minmode_estimate(session, callback, None, ctypes.byref(report))
                steps += 1
                if rc != OK:
                    solver_rc = rc
                    break
                rotations += int(report.rotations)
                library_evals += int(report.evaluations)
                current = _mode(lib, session, dof)
                overlap = float(abs(np.dot(_unit(current), reference)))
                if overlap >= float(spec["overlap"]):
                    converged = True
                    break
                if previous is not None and float(np.linalg.norm(current - previous)) < 1e-8:
                    break
                # A repeated geometry is a cache hit. The harness count
                # stays put and the estimate is not moving.
                if potential.calls == before:
                    break
                previous = current
            point = _position(lib, lib.rgsaddle_minmode_position, session, dof)
            energy, gradient = potential.energy_forces(point)
            reasons = []
            if not (converged and potential.calls < cap):
                reasons.append("mode_overlap")
            if potential.calls >= cap:
                reasons.append("cap")
            if solver_rc != OK:
                reasons.append("solver")
            success, failure = _outcome(reasons)
            record = {
                "schema": "optbench.start/v2",
                "start": spec["name"],
                "force_calls": int(potential.calls),
                "verify_calls": 0,
                "counter": "harness",
                "comparable": True,
                "success": success,
                "failure": failure,
                "reasons": reasons,
                "energy": float(energy),
                "force_norm": _norm(gradient),
                "hessian_index": None,
                "expected_index": 1,
                "overlap": overlap,
                "converged": converged,
                "rotations": rotations,
                "library_evaluations": library_evals,
                "steps": steps,
                "curvature": float(report.curvature) if report is not None else None,
            }
            if _finite_point(point):
                raw_eigenvalues, vectors = _hessian_spectrum(potential, point)
                record["cartesian_index"] = negative_index(raw_eigenvalues, 1e-6)
                if not bool(np.any(fixed)):
                    eigenvalues, removed = _vibrational_eigenvalues(point, raw_eigenvalues, vectors)
                else:
                    eigenvalues, removed = list(raw_eigenvalues), 0
                record["hessian_index"] = negative_index(eigenvalues, 1e-6)
                record["rigid_modes_removed"] = removed
                record["lowest_eigenvalue"] = float(eigenvalues[0]) if eigenvalues else None
                record["lowest_eigenvalues"] = [float(value) for value in eigenvalues[:8]]
                record["verify_calls"] = int(potential.verify_calls)
            # Overlap is the mode-finder score. The Hessian index of the
            # unmoved guess is recorded and does not decide success.
            return record
        report = None
        while potential.calls < cap:
            before = potential.calls
            report = _blank_report(lib)
            rc = lib.rgsaddle_minmode_step(session, callback, None, ctypes.byref(report))
            steps += 1
            if rc != OK:
                solver_rc = rc
                break
            rotations += int(report.rotations)
            library_evals += int(report.evaluations)
            if int(report.status) == 1:
                converged = True
                break
            # FIRE can sit on one geometry. Those repeats are cache hits,
            # so the cap never arrives. Stop and judge the point we have.
            if potential.calls == before:
                break
        point = _position(lib, lib.rgsaddle_minmode_position, session, dof)
        if spec.get("defer_hessian"):
            final_mode = _mode(lib, session, dof)
        surface_errors = list(callback.errors)
    finally:
        lib.rgsaddle_minmode_free(session)
        del callback
    if solver_rc != OK and not _finite_point(point):
        reasons = ["solver"]
        success, failure = _outcome(reasons)
        return {
            "schema": "optbench.start/v2",
            "start": spec["name"],
            "force_calls": int(potential.calls),
            "verify_calls": int(potential.verify_calls),
            "counter": "harness",
            "comparable": True,
            "success": success,
            "failure": failure,
            "reasons": reasons,
            "hessian_index": None,
            "expected_index": 1,
            "converged": False,
            "rotations": rotations,
            "library_evaluations": library_evals,
            "steps": steps,
            "solver_rc": int(solver_rc),
            "surface_errors": surface_errors,
        }
    if spec.get("defer_hessian"):
        energy, gradient = potential.energy_forces(point)
        return {
            "name": spec["name"],
            "x": point,
            "mode": final_mode,
            "free": free,
            "calls": int(potential.calls),
            "verify_calls": int(potential.verify_calls),
            "energy": float(energy),
            "force_norm": _norm(gradient),
            "converged": converged,
            "rotations": rotations,
            "library_evaluations": library_evals,
            "steps": steps,
            "curvature": float(report.curvature) if report is not None else None,
            "solver_rc": int(solver_rc),
        }
    return _saddle_record(
        spec["name"],
        potential,
        point,
        cap=cap,
        force_tol=force_tol,
        library_converged=converged,
        rotations=rotations,
        library_evals=library_evals,
        steps=steps,
        project_rigid=not bool(np.any(fixed)),
    )


def _run_prfo(spec: dict) -> dict:
    os.environ["RAYON_NUM_THREADS"] = "1"
    lib = _load()
    import ctypes

    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    free, x, _base = _free_view(positions, fixed)
    potential = _potential_for(spec, positions, free, cell)
    dof = int(x.size)
    n_atoms = dof // 3
    force_tol = float(spec["force_tol"])
    cap = int(spec["cap"])
    sella = spec["sella"]
    cfg = lib.SellaConfig()
    cfg.major = ABI_MAJOR
    cfg.minor = ABI_MINOR
    cfg.flags = 0
    cfg.delta = float(sella.get("delta", 0.0))
    cfg.force_tol = force_tol
    cfg.force_gate = GATE_L2
    cfg.order = int(sella.get("order", 1))
    masses = np.ones(n_atoms, dtype=np.float64)
    x_buf = np.ascontiguousarray(x)
    constraints = None
    if bool(np.any(fixed)):
        blank = lib.ConstraintsConfig()
        blank.major = ABI_MAJOR
        blank.minor = ABI_MINOR
        blank.flags = 0
        constraints = lib.rgsaddle_constraints_create(ctypes.byref(blank), n_atoms)
        if not constraints:
            raise RuntimeError("rgsaddle_constraints_create failed")
        session = lib.rgsaddle_sella_saddle_create_on(
            ctypes.byref(cfg),
            n_atoms,
            x_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            constraints,
        )
    else:
        session = lib.rgsaddle_sella_saddle_create(
            ctypes.byref(cfg),
            n_atoms,
            x_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    if not session:
        text = lib.rgsaddle_last_error()
        detail = text.decode() if text else "null session"
        if constraints:
            lib.rgsaddle_constraints_free(constraints)
        raise RuntimeError(f"rgsaddle_sella_saddle_create failed: {detail}")
    lib.rgsaddle_sella_saddle_set_hess_update(session, HESS[str(sella.get("hess_update", "bfgs"))])
    lib.rgsaddle_sella_saddle_set_expand(session, EXPAND[str(sella.get("expand", "jd0"))])
    lib.rgsaddle_sella_saddle_set_restricted(session, RESTRICTED[str(sella.get("restricted", "trust-region"))])
    reactant = None if spec.get("reactant") is None else np.asarray(spec["reactant"], dtype=np.float64)
    if reactant is not None and str(sella.get("mode_seed", "reactant")) != "none":
        seed = np.ascontiguousarray(_unit(x - reactant))
        seed_rc = lib.rgsaddle_sella_saddle_seed_mode(
            session,
            seed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(-1.0),
        )
        if seed_rc != OK:
            lib.rgsaddle_sella_saddle_free(session)
            if constraints:
                lib.rgsaddle_constraints_free(constraints)
            raise RuntimeError(f"rgsaddle_sella_saddle_seed_mode returned {seed_rc}")
    callback = _callback(lib, potential, dof)
    converged = False
    steps = 0
    library_evals = 0
    try:
        while potential.calls < cap:
            before = potential.calls
            report = _blank_report(lib)
            rc = lib.rgsaddle_sella_saddle_step(session, callback, None, ctypes.byref(report))
            steps += 1
            if rc != OK:
                break
            library_evals += int(report.evaluations)
            if int(report.status) == 1:
                converged = True
                break
            if potential.calls == before:
                break
        point = _position(lib, lib.rgsaddle_sella_saddle_position, session, dof)
    finally:
        lib.rgsaddle_sella_saddle_free(session)
        if constraints:
            lib.rgsaddle_constraints_free(constraints)
        del callback
    return _saddle_record(
        spec["name"],
        potential,
        point,
        cap=cap,
        force_tol=force_tol,
        library_converged=converged,
        rotations=0,
        library_evals=library_evals,
        steps=steps,
        project_rigid=not bool(np.any(fixed)),
    )


def _dispatch(spec: dict) -> dict:
    if spec["session"] == "prfo":
        return _run_prfo(spec)
    return _run_minmode(spec)


def _geometries(root: Path, problem: dict, work: Path) -> list[dict]:
    archive_path = (problem.get("input") or {}).get("archive_path")
    if not archive_path:
        paths = (problem.get("input") or {}).get("geometries") or []
        if not paths:
            raise ValueError(f"{problem['id']} has no geometries")
        return [{"name": Path(paths[0]).name, "path": _stored(root, paths[0]), "mode": None}]
    extracted = _extract(_stored(root, archive_path), work / "geometries")
    ident = problem["id"]
    rows = []
    if ident == "tsopt/LJ38":
        folder = extracted / "coords-con"
        for index in range(200):
            path = folder / f"{index:04d}.con"
            rows.append({"name": f"{index:04d}", "path": path, "mode": None})
    elif ident == "tsopt/pt-island":
        folder = extracted / "pt-island-con"
        reactant = folder / "reactant.con"
        for index in range(49):
            rows.append({"name": f"initial_{index}", "path": folder / f"initial_{index}.con", "mode": None, "reactant": reactant})
    elif ident == "lowest-eigenvector/lj38":
        folder = extracted / "coords-con"
        for index in range(200):
            rows.append(
                {
                    "name": f"{index:04d}",
                    "path": folder / f"{index:04d}.con",
                    "mode": folder / f"{index:04d}_mode",
                    "initial_mode": folder / "initial_mode",
                }
            )
    else:
        raise ValueError(f"no geometry layout for {ident}")
    missing = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(str(missing[0]))
    return rows


def _spec_from_geometry(row: dict, problem: dict, template: dict, cap: int, force_tol: float) -> dict:
    positions, fixed, cell = load_con(Path(row["path"]))
    saddle = _section(template, "SaddleSearch")
    optimizer = _section(template, "Optimizer")
    main = _section(template, "Main")
    sella = _section(template, "Sella")
    reactant = None
    if row.get("reactant"):
        other, other_fixed, _cell = load_con(Path(row["reactant"]))
        reactant = other[~other_fixed].reshape(-1)
    file_mode = None
    reference = None
    if row.get("mode"):
        full = read_mode(Path(row["mode"]).read_text(encoding="utf-8", errors="replace"))
        reference = full.reshape(-1, 3)[~fixed].reshape(-1)
    if str(saddle.get("mode_seed", "gradient")) == "file" and row.get("initial_mode"):
        full = read_mode(Path(row["initial_mode"]).read_text(encoding="utf-8", errors="replace"))
        file_mode = full.reshape(-1, 3)[~fixed].reshape(-1)
    task = "mode" if problem["id"] == "lowest-eigenvector/lj38" else "saddle"
    mode_seed = str(saddle.get("mode_seed", "gradient"))
    # Pt tsopt guesses in the Sella entry start along x - reactant.
    if reactant is not None and mode_seed == "gradient":
        mode_seed = "reactant"
    return {
        "name": row["name"],
        "positions": positions,
        "fixed": fixed,
        "cell": cell,
        "potential": problem.get("potential") or {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0},
        "session": str(main.get("session", "minmode")),
        "kind": str(saddle.get("min_mode_method", "dimer")),
        "method": str(optimizer.get("opt_method", "lbfgs")),
        "mode_seed": mode_seed,
        "task": task,
        "saddle": saddle,
        "optimizer": optimizer,
        "sella": sella,
        "cap": cap,
        "force_tol": force_tol,
        "reactant": reactant,
        "file_mode": file_mode,
        "reference_mode": reference,
        "overlap": float((problem.get("sample") or {}).get("overlap", 0.99)),
    }


def _summarize(records: list[dict], template: dict, cap: int, problem_id: str) -> tuple[str, dict]:
    ok = [row for row in records if row["success"]]
    counts = [int(row["force_calls"]) for row in ok]
    lines = []
    if counts:
        lines.append(f"force_calls {half_even(sum(counts) / len(counts))}")
        lines.append(f"force_calls_min {min(counts)}")
        lines.append(f"force_calls_median {statistics.median(counts)}")
        lines.append(f"force_calls_max {max(counts)}")
    lines.append(f"nfailed {len(records) - len(ok)}")
    metrics = score_v2(records, cap)
    lines.append(f"par2_mean {metrics['par2_mean']:.10f}")
    lines.append(f"successes {metrics['successes']}")
    lines.append(f"rate_low {metrics['rate_low']:.10f}")
    lines.append(f"rate_high {metrics['rate_high']:.10f}")
    lines.append(f"par2_median {metrics['median']:.10f}")
    lines.append(f"n_cap {metrics['n_cap']}")
    lines.append(f"n_wrong_stationary {metrics['n_wrong_stationary']}")
    lines.append(f"n_divergence {metrics['n_divergence']}")
    lines.append(f"n_crash {metrics['n_crash']}")
    lines.append("counter harness")
    lines.append(f"code {template.get('code', 'rgsaddle')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem_id}")
    rotations = sum(int(row.get("rotations") or 0) for row in records)
    library = sum(int(row.get("library_evaluations") or 0) for row in records)
    harness = sum(int(row["force_calls"]) for row in records)
    lines.append(f"rotation_steps {rotations}")
    lines.append(f"library_evaluations {library}")
    lines.append(f"harness_calls {harness}")
    return "\n".join(lines) + "\n", metrics


def run_template(root: Path, problem: dict, template: dict, *, output: Path, workers: int = 1, work: Path | None = None) -> dict:
    """Run one rgsaddle template on every geometry of ``problem``."""
    from .campaign import _check_stop

    _assert_source_commit(str(template.get("commit", "")))
    cap = _check_stop(problem, template)
    optimizer = _section(template, "Optimizer")
    convergence = problem.get("convergence") or {}
    force_tol = float(optimizer.get("converged_force", convergence.get("force_norm", 1e-3)))
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    if problem["id"] == "saddle-search/pt-island":
        return _run_saddle_search(root, problem, template, output=output, cap=cap, force_tol=force_tol, work=scratch)
    rows = _geometries(root, problem, scratch)
    expected = int((problem.get("sample") or {}).get("starts") or len(rows))
    if len(rows) != expected:
        raise RuntimeError(f"{problem['id']} has {len(rows)} geometries, the problem states {expected}")
    specs = [_spec_from_geometry(row, problem, template, cap, force_tol) for row in rows]
    if workers <= 1:
        records = [_dispatch(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_dispatch, specs))
    text, metrics = _summarize(records, template, cap, problem["id"])
    provenance = collect_provenance(
        code=str(template.get("code", "rgsaddle")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={
            "template": template["id"],
            "problem": problem["id"],
            "cap": cap,
            "force_tol": force_tol,
            "session": _section(template, "Main").get("session", "minmode"),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    public = []
    for record in records:
        copied = {key: value for key, value in record.items() if key != "x"}
        copied["code"] = provenance["code"]
        copied["code_version"] = provenance["code_version"]
        copied["commit"] = provenance["commit"]
        copied["tuning"] = template.get("tuning", "")
        copied["template"] = template["id"]
        public.append(copied)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    return {
        "force_calls": half_even(statistics.fmean([row["force_calls"] for row in records if row["success"]])) if any(row["success"] for row in records) else None,
        "nfailed": metrics["nfailed"],
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
        "records": records,
    }


def _cartesian_rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
    """RMSD in the substrate frame. Fixed atoms pin the orientation."""
    delta = np.asarray(moved, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _lbfgs_minimize(potential: HarnessSurface, x0: np.ndarray, force_tol: float, budget: int) -> tuple[np.ndarray, bool]:
    """Minimize with L-BFGS. Every trial is a harness call on ``potential``."""
    x = np.array(x0, dtype=np.float64, copy=True)
    started = potential.calls
    energy, grad = potential.energy_forces(x)
    grad = np.asarray(grad, dtype=np.float64)
    memory_s: list[np.ndarray] = []
    memory_y: list[np.ndarray] = []
    memory_rho: list[float] = []
    converged = _norm(grad) <= force_tol
    while not converged and potential.calls - started < budget:
        q = grad.copy()
        alphas: list[float] = []
        for step_s, step_y, step_rho in zip(reversed(memory_s), reversed(memory_y), reversed(memory_rho)):
            alpha = step_rho * float(np.dot(step_s, q))
            alphas.append(alpha)
            q = q - alpha * step_y
        if memory_y:
            gamma = float(np.dot(memory_s[-1], memory_y[-1]) / np.dot(memory_y[-1], memory_y[-1]))
        else:
            gamma = 1.0
        direction = gamma * q
        for step_s, step_y, step_rho, alpha in zip(memory_s, memory_y, memory_rho, reversed(alphas)):
            beta = step_rho * float(np.dot(step_y, direction))
            direction = direction + (alpha - beta) * step_s
        direction = -direction
        slope = float(np.dot(grad, direction))
        if slope >= 0.0:
            direction = -grad
            slope = -float(np.dot(grad, grad))
        step = 1.0
        trial = x
        trial_energy = energy
        trial_grad = grad
        accepted = False
        for _attempt in range(10):
            if potential.calls - started >= budget:
                break
            delta = step * direction
            length = float(np.linalg.norm(delta))
            if length > 0.2:
                delta = delta * (0.2 / length)
                step = 0.2 / max(float(np.linalg.norm(direction)), 1e-30)
            candidate = x + delta
            trial_energy, trial_grad_raw = potential.energy_forces(candidate)
            trial_grad = np.asarray(trial_grad_raw, dtype=np.float64)
            if trial_energy <= energy + 1e-4 * step * slope:
                trial = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        change = trial - x
        rise = trial_grad - grad
        curvature = float(np.dot(rise, change))
        if curvature > 1e-12:
            memory_s.append(change)
            memory_y.append(rise)
            memory_rho.append(1.0 / curvature)
            if len(memory_s) > 10:
                memory_s.pop(0)
                memory_y.pop(0)
                memory_rho.pop(0)
        x = trial
        energy = float(trial_energy)
        grad = trial_grad
        converged = _norm(grad) <= force_tol
    return x, converged


def _rng_state(rng) -> dict:
    return json.loads(json.dumps(rng.bit_generator.state))


def _rng_restore(payload: dict):
    rng = np.random.default_rng()
    rng.bit_generator.state = payload
    return rng


def _checkpoint_paths(output: Path) -> tuple[Path, Path]:
    return output.with_name(output.stem + ".header.json"), output.with_name(output.stem + ".partial.jsonl")


def _load_partial(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _append_partial(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
        handle.flush()


def _run_saddle_search(root: Path, problem: dict, template: dict, *, output: Path, cap: int, force_tol: float, work: Path) -> dict:
    """Displace the Pt island until the goal count of connected index-1 saddles.

    Each search has its own harness count: the min-mode calls plus the
    two endpoint minimizations. The repetition total also includes the
    initial minimization. Hessian calls stay in ``verify_calls``.
    The link and the novelty test are ``optbench.connection``: per-atom
    minimum-image distance on the free atoms, as eOn compares structures.
    Only a connected saddle enters the process table.
    ``work`` is unused; the searches do not extract an archive.
    """
    del work
    if str(_section(template, "Main").get("session", "minmode")) == "prfo":
        raise RuntimeError("saddle-search displacements use the min-mode session")
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = load_con(_stored(root, con_path))
    search = _section(template, "Search")
    goal = int((problem.get("goal") or {}).get("connected_saddles", search.get("connected", 27)))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    repetitions = int(search.get("repetitions", 1))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    magnitude = float(search.get("displace_magnitude", 0.1))
    distance_difference, energy_difference = tolerances(problem)
    lengths = periodic_lengths(cell, (problem.get("potential") or {}).get("pbc"))
    max_searches = int(search.get("max_searches", 20000))
    free = np.flatnonzero(~fixed)
    raw = potential_from_problem(problem)
    comments = []
    all_records = []
    failed_reps = 0
    for rep in range(1, repetitions + 1):
        header_path, partial_path = _checkpoint_paths(output)
        header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
        saved = _load_partial(partial_path) if header and header.get("template") == template["id"] and int(header.get("rep", 0)) == rep else []
        if header and header.get("template") == template["id"] and int(header.get("rep", 0)) == rep:
            potential = HarnessSurface(_bind(raw, positions, free, cell))
            minimum = np.asarray(header["minimum"], dtype=np.float64)
            min_energy = float(header["min_energy"])
            minimize_calls = int(header["minimize_calls"])
            min_ok = bool(header["min_ok"])
            resumed_outer = minimize_calls + sum(int(row.get("endpoint_calls") or 0) for row in saved)
            search_calls = sum(int(row.get("climb_calls") or 0) for row in saved)
            found = ProcessTable(lengths, distance_difference, energy_difference)
            for row in saved:
                if row.get("saddle_x") is not None:
                    found.add(row["saddle_x"], float(row["barrier"]))
            connected = sum(1 for row in saved if row.get("success"))
            searches = len(saved)
            rng = _rng_restore(saved[-1]["rng_state"]) if saved else np.random.default_rng(20261015 + rep)
            for row in saved:
                kept = {key: value for key, value in row.items() if key not in {"rng_state", "saddle_x", "climb_calls"}}
                all_records.append(kept)
            print(
                f"{template['id']} rep {rep} resume searches {searches} connected {connected}",
                flush=True,
            )
        else:
            potential = HarnessSurface(_bind(raw, positions, free, cell))
            minimum, min_ok = _lbfgs_minimize(potential, positions[free].reshape(-1).copy(), force_tol, 5000)
            min_energy, _min_grad = potential.energy_forces(minimum)
            minimize_calls = int(potential.calls)
            resumed_outer = 0
            rng = np.random.default_rng(20261015 + rep)
            found = ProcessTable(lengths, distance_difference, energy_difference)
            connected = 0
            searches = 0
            search_calls = 0
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(
                json.dumps(
                    {
                        "template": template["id"],
                        "rep": rep,
                        "minimum": minimum.tolist(),
                        "min_energy": min_energy,
                        "minimize_calls": minimize_calls,
                        "min_ok": bool(min_ok),
                    }
                ),
                encoding="utf-8",
            )
            partial_path.unlink(missing_ok=True)
        while connected < goal and searches < max_searches:
            searches += 1
            atom = int(listed[int(rng.integers(0, len(listed)))])
            slots = np.flatnonzero(free == atom)
            if len(slots) != 1:
                raise RuntimeError(f"displace atom {atom} is not one free atom")
            slot = int(slots[0])
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            x0 = minimum.copy()
            x0[3 * slot : 3 * slot + 3] += magnitude * direction
            kick = np.zeros_like(x0)
            kick[3 * slot : 3 * slot + 3] = direction
            moved = positions.copy()
            moved[free] = x0.reshape(-1, 3)
            spec = {
                "name": f"rep{rep}-search{searches}",
                "positions": moved,
                "fixed": fixed,
                "cell": cell,
                "potential": problem["potential"],
                "session": "minmode",
                "kind": str(_section(template, "SaddleSearch").get("min_mode_method", "dimer")),
                "method": str(_section(template, "Optimizer").get("opt_method", "lbfgs")),
                "mode_seed": "file",
                "task": "saddle",
                "saddle": _section(template, "SaddleSearch"),
                "optimizer": _section(template, "Optimizer"),
                "sella": _section(template, "Sella"),
                "cap": cap,
                "force_tol": force_tol,
                "reactant": None,
                "file_mode": kick,
                "reference_mode": None,
                "overlap": 0.99,
                "defer_hessian": True,
            }
            before_outer = potential.calls
            outcome = _dispatch(spec)
            climb = int(outcome.get("calls", outcome.get("force_calls", 0)))
            record = {
                "schema": "optbench.start/v2",
                "start": spec["name"],
                "counter": "harness",
                "comparable": True,
                "success": False,
                "reasons": ["force_norm"],
                "converged": bool(outcome.get("converged", False)),
                "rotations": int(outcome.get("rotations") or 0),
                "library_evaluations": int(outcome.get("library_evaluations") or 0),
                "curvature": outcome.get("curvature"),
                "hessian_index": None,
            }
            curvature = outcome.get("curvature")
            negative = curvature is not None and math.isfinite(float(curvature)) and float(curvature) < 0.0
            saved_point = None
            if outcome.get("converged") and negative and _finite_point(np.asarray(outcome["x"])):
                raw_eigenvalues, vectors = _hessian_spectrum(potential, outcome["x"])
                # Free atoms of the island are pinned by the substrate.
                # Their Hessian is not a free-cluster null space.
                eigenvalues = list(raw_eigenvalues)
                index = negative_index(eigenvalues, 1e-6)
                record["hessian_index"] = int(index)
                record["cartesian_index"] = int(index)
                record["rigid_modes_removed"] = 0
                record["lowest_eigenvalue"] = float(eigenvalues[0])
                record["verify_calls"] = int(potential.verify_calls)
                if index == 1 and float(outcome["force_norm"]) <= force_tol + 1e-10:
                    saddle_x = np.asarray(outcome["x"], dtype=np.float64)
                    climb_mode = outcome.get("mode")
                    axis = _unit(np.asarray(climb_mode if climb_mode is not None else kick, dtype=np.float64))
                    plus, _plus_ok = _lbfgs_minimize(
                        potential, saddle_x + 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET
                    )
                    minus, _minus_ok = _lbfgs_minimize(
                        potential, saddle_x - 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET
                    )
                    links = endpoints_connected(minimum, plus, minus, lengths, distance_difference)
                    barrier = float(outcome["energy"] - min_energy)
                    novel = links and not found.repeat(saddle_x, barrier)
                    record["barrier"] = barrier
                    record["per_atom_max_plus"] = per_atom_max(plus, minimum, lengths)
                    record["per_atom_max_minus"] = per_atom_max(minus, minimum, lengths)
                    record["reasons"] = []
                    if novel:
                        found.add(saddle_x, barrier)
                        saved_point = saddle_x
                    if links and novel and barrier < barrier_max:
                        connected += 1
                        record["success"] = True
                    else:
                        if not links:
                            record["reasons"].append("not_connected")
                        elif not novel:
                            record["reasons"].append("duplicate")
                        if barrier >= barrier_max:
                            record["reasons"].append("barrier")
                else:
                    record["reasons"] = ["hessian_index"] if index != 1 else ["force_norm"]
            else:
                point = outcome.get("x")
                finite = point is not None and _finite_point(np.asarray(point, dtype=np.float64))
                if "calls" not in outcome:
                    record["reasons"] = [str(item) for item in (outcome.get("reasons") or ["solver"])]
                elif not finite:
                    record["reasons"] = ["divergence"]
                elif climb >= int(cap):
                    record["reasons"] = ["cap", "force_norm"]
            endpoint_calls = int(potential.calls - before_outer)
            record["force_calls"] = climb + endpoint_calls
            record["endpoint_calls"] = endpoint_calls
            success, failure = _outcome(record["reasons"])
            record["success"] = success
            record["failure"] = failure
            search_calls += climb
            all_records.append(record)
            stored = dict(record)
            stored["rng_state"] = _rng_state(rng)
            stored["climb_calls"] = climb
            if saved_point is not None:
                stored["saddle_x"] = saved_point.tolist()
            _append_partial(partial_path, stored)
            if searches == 1 or searches % 25 == 0 or connected >= goal:
                print(
                    f"{template['id']} rep {rep} searches {searches} connected {connected} "
                    f"calls {record['force_calls']} failure {failure}",
                    flush=True,
                )
        total_calls = int(resumed_outer + potential.calls + search_calls)
        if connected < goal:
            failed_reps += 1
        fraction = connected / searches if searches else 0.0
        comments.append(
            f"# rep {rep} force_calls {total_calls} unique_saddles {len(found)} "
            f"fraction_connected {fraction} total_searches {searches} "
            f"connected {connected} minimize_calls {minimize_calls} "
            f"minimum_converged {int(min_ok)} seed {20261015 + rep}"
        )
    totals = []
    for line in comments:
        parts = line.split()
        totals.append(int(parts[parts.index("force_calls") + 1]))
    text = "\n".join(comments) + "\n"
    if totals:
        text += f"force_calls {statistics.fmean(totals)}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
    search_cap = int(cap) + _SADDLE_CLIMB_OVERSHOOT + 2 * _SADDLE_ENDPOINT_BUDGET
    over = [
        record["start"]
        for record in all_records
        if record.get("success") and int(record["force_calls"]) >= search_cap
    ]
    if over:
        raise RuntimeError(f"{template['id']} success {over[0]} spent {search_cap} or more force calls")
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in all_records
        ],
        search_cap,
    ) if all_records else None
    text += (
        f"code {template.get('code', 'rgsaddle')}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(all_records)}\n"
    )
    if metrics is not None:
        text += (
            f"par2_mean {metrics['par2_mean']}\n"
            f"v2_nfailed {metrics['nfailed']}\n"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), all_records)
    if metrics is not None:
        provenance = collect_provenance(
            code=str(template.get("code", "rgsaddle")),
            code_version=str(template.get("code_version", "")),
            commit=str(template.get("commit", "")),
            parameters={"template": template["id"], "problem": problem["id"], "cap": search_cap},
        )
        document = {
            "schema": "optbench.result/v2",
            "status": "archive",
            "problem": problem["id"],
            "scoring": "par2-bootstrap",
            "par_factor": 2,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 20261015,
            "note": (
                "One row is one displacement search. The cap is the climb budget, "
                "one step of overshoot, and the two endpoint minimizations. "
                "The initial minimization is not in that count."
            ),
            "runs": [
                {
                    "entry": output.stem,
                    "code": template.get("code", "rgsaddle"),
                    "algorithm": template.get("algorithm", ""),
                    "code_version": template.get("code_version", ""),
                    "commit": template.get("commit", ""),
                    "tuning": template.get("tuning", ""),
                    "present_source": True,
                    "cap": search_cap,
                    "provenance": provenance,
                    "metrics": metrics,
                    "starts": all_records,
                }
            ],
        }
        output.with_suffix(".v2.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    header_path, partial_path = _checkpoint_paths(output)
    partial_path.unlink(missing_ok=True)
    header_path.unlink(missing_ok=True)
    return {
        "force_calls": statistics.fmean(totals) if totals else None,
        "nfailed": failed_reps,
        "starts": len(all_records),
        "template": template["id"],
        "problem": problem["id"],
        "records": all_records,
    }
