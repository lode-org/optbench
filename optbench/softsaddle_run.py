"""SoftSaddle on the harness counter.

The Lanczos and Davidson solvers are the 2017 libraries. Each shared
object exports one C entry, ``softsaddle_minimum_mode``, and the spec
potential supplies the gradient. The saddle campaign is the archived
adaptive-hypersphere LBFGS minimum-mode walk: memory 60, step 0.5,
eight eigen-iterations, and a batch of 25. A fresh library is loaded
for every search, which is what ``MyOwn_v10_morse`` does with
``unloadlibrary``. The lowest-mode campaign repeats that eigen-solve
until the overlap stop or the call cap.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .connection import ProcessTable, per_atom_max, periodic_lengths, tolerances
from .connection import connected as endpoints_connected
from .failures import classify_failure
from .rgsaddle_minmode import (
    HarnessSurface,
    _SADDLE_CLIMB_OVERSHOOT,
    _SADDLE_ENDPOINT_BUDGET,
    _bind,
    _finite_point,
    _geometries,
    _hessian_spectrum,
    _lbfgs_minimize,
    _norm,
    _outcome,
    _unit,
    _vibrational_eigenvalues,
)
from .score import half_even, score_v2
from .stationary import negative_index
from .surfaces import load_con, potential_from_problem, read_mode
from .v2 import collect_provenance, write_starts

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_GRAD = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_void_p,
)

INITIAL_RANDOM = 0
INITIAL_GRADIENT = 1
INITIAL_USER = 2
INITIAL_PREVIOUS = 3

CLIMB_BUDGET = 1000
HYPER_SOURCE = "eb78f6d9cde1e9aae308e3f8ecec777fdb655e3432e356c9da339b950bd3267a"
_LIBS: dict[str, ctypes.CDLL] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def library_path(kind: str) -> Path | None:
    """Shared object for ``lanczos`` or ``davidson``, or None when it was not built."""
    env = "SOFTSADDLE_DAVIDSON" if kind == "davidson" else "SOFTSADDLE_LANCZOS"
    named = os.environ.get(env)
    if named:
        path = Path(named)
        return path if path.is_file() else None
    directory = Path(os.environ.get("SOFTSADDLE_LIB_DIR", str(_repo_root() / "softsaddle" / "lib")))
    path = directory / f"libsoftsaddle_{kind}.so"
    return path if path.is_file() else None


def libraries_ready() -> bool:
    return library_path("lanczos") is not None and library_path("davidson") is not None


def _load(kind: str) -> ctypes.CDLL:
    cached = _LIBS.get(kind)
    if cached is not None:
        return cached
    path = library_path(kind)
    if path is None:
        raise RuntimeError(
            f"softsaddle {kind} library is not built; run softsaddle/driver/build.sh"
        )
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.softsaddle_reset.argtypes = []
    lib.softsaddle_seed.argtypes = [ctypes.c_uint]
    lib.softsaddle_set_error.argtypes = [ctypes.c_int]
    lib.softsaddle_kk.restype = ctypes.c_int
    lib.softsaddle_minimum_mode.restype = ctypes.c_int
    lib.softsaddle_minimum_mode.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        _GRAD,
        ctypes.c_void_p,
    ]
    _LIBS[kind] = lib
    return lib


def _section(template: dict, name: str) -> dict:
    return ((template.get("ini") or {}).get(name) or {})


@contextmanager
def _quiet(path: Path):
    """Send C++ stdout and stderr to ``path`` for one search."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(handle, 1)
        os.dup2(handle, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(handle)


@contextmanager
def _at(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        with _quiet(path / "stdio.log"):
            yield
    finally:
        os.chdir(previous)


def _minimum_mode(
    lib: ctypes.CDLL,
    potential: HarnessSurface,
    x: np.ndarray,
    *,
    initial: int,
    iteration_limit: int,
    finite: float,
    tolerance: float,
    seed_vector: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, int, int, float]:
    """One solver call. The gradient at ``x`` is the vector returned last."""
    point = np.ascontiguousarray(x, dtype=np.float64)
    n = int(point.size)
    eigenvalue = ctypes.c_double(float("inf"))
    vector = np.zeros(n, dtype=np.float64)
    if seed_vector is not None and seed_vector.size == n:
        vector[:] = seed_vector
    gradient = np.zeros(n, dtype=np.float64)
    iterations = ctypes.c_int()
    convergence = ctypes.c_double()
    status = ctypes.c_int()
    holder = {"energy": None, "error": None}

    def invoke(x_ptr, g_ptr, length, _user):
        try:
            coords = np.ctypeslib.as_array(x_ptr, shape=(length,)).copy()
            _energy, grad = potential.energy_forces(coords)
            out = np.ctypeslib.as_array(g_ptr, shape=(length,))
            out[:] = np.asarray(grad, dtype=np.float64)
            if length == n and float(np.max(np.abs(coords - point))) == 0.0:
                holder["energy"] = float(_energy)
        except Exception as exc:
            holder["error"] = repr(exc)
            np.ctypeslib.as_array(g_ptr, shape=(length,))[:] = 0.0
            lib.softsaddle_set_error(1)

    callback = _GRAD(invoke)
    rc = lib.softsaddle_minimum_mode(
        point.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        ctypes.byref(eigenvalue),
        vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(initial),
        int(iteration_limit),
        float(finite),
        float(tolerance),
        ctypes.byref(iterations),
        ctypes.byref(convergence),
        ctypes.byref(status),
        callback,
        None,
    )
    del callback
    if holder["error"] or rc:
        raise RuntimeError(holder["error"] or "softsaddle gradient callback failed")
    if holder["energy"] is None:
        energy, _gradient = potential.energy_forces(point)
        holder["energy"] = float(energy)
    return (
        float(eigenvalue.value),
        vector,
        gradient,
        int(iterations.value),
        int(status.value),
        float(holder["energy"]),
    )


def _effective(gradient: np.ndarray, eigenvalue: float, mode: np.ndarray) -> np.ndarray:
    """Minimum-mode gradient from ``get_func_and_grad_v8_morse``.

    The positive-curvature branch is the projection onto the mode.
    The negative branch reflects the gradient through that mode.
    ``use_same_formula`` is false, so the positive branch stays on
    while the search has not left the starting basin.
    """
    direction = _unit(mode)
    product = float(np.dot(gradient, direction))
    if eigenvalue >= 0.0:
        return -product * direction
    return gradient - 2.0 * product * direction


def _max_motion(vector: np.ndarray, limit: float) -> np.ndarray:
    length = _norm(vector)
    if length > limit > 0.0:
        return vector * (limit / length)
    return vector


def _sign(value: float) -> float:
    """MATLAB ``sign``. Zero stays zero; ``math.copysign`` does not."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _lambda_zero(xpos: np.ndarray, xneg: np.ndarray, epos: float, eneg: float) -> np.ndarray | None:
    """Linear crossing from ``x_lamda_zero.m``. The caller passes the MATLAB argument order."""
    delta = np.asarray(xneg, dtype=np.float64) - np.asarray(xpos, dtype=np.float64)
    distance = _norm(delta)
    span = float(epos) - float(eneg)
    if distance < 1e-15 or span == 0.0 or not math.isfinite(span):
        return None
    return np.asarray(xpos, dtype=np.float64) + (float(epos) * distance / span) * (delta / distance)


def _analyze(mode, vector, step, accumulated, count: int, tolerance: float):
    """Reuse the mode while the running step stays inside ``tolerance``."""
    if accumulated is None:
        total = np.array(step, dtype=np.float64, copy=True)
    else:
        total = accumulated + step
    if _norm(total) > tolerance:
        return None, None, np.zeros_like(step), 0
    return mode, vector, total, count


class _LBFGS:
    """Two-loop memory from ``LBFGS_morse_v6`` ``updateLBFGS`` and ``getStep``."""

    def __init__(self, dimension: int, memory: int):
        self.memory = int(memory)
        self.s = np.zeros((dimension, self.memory), dtype=np.float64)
        self.y = np.zeros((dimension, self.memory), dtype=np.float64)
        self.rho = np.zeros(self.memory, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray, x_prev: np.ndarray, g: np.ndarray, g_prev: np.ndarray) -> None:
        if self.count == self.memory:
            self.s[:, :-1] = self.s[:, 1:]
            self.y[:, :-1] = self.y[:, 1:]
            self.rho[:-1] = self.rho[1:]
            self.count = self.memory - 1
        step = x - x_prev
        change = g - g_prev
        self.count += 1
        slot = self.count - 1
        self.s[:, slot] = step
        self.y[:, slot] = change
        denom = float(np.dot(change, step))
        self.rho[slot] = 1.0 / denom if denom != 0.0 else 0.0

    def step(
        self,
        x: np.ndarray,
        g: np.ndarray,
        *,
        iteration: int,
        finite: float,
        max_step: float,
        potential: HarnessSurface,
        eigenvalue: float | None,
        mode: np.ndarray | None,
        bandera: bool,
    ) -> tuple[np.ndarray, bool]:
        forced = bandera
        if iteration > 0:
            if forced or self.count < 1:
                curvature = -0.01
                forced = False
            else:
                slot = self.count - 1
                denom = float(np.dot(self.s[:, slot], self.y[:, slot]))
                curvature = float(np.dot(self.y[:, slot], self.y[:, slot]) / denom) if denom != 0.0 else float("nan")
            if not math.isfinite(curvature) or curvature < 0.0:
                self.count = 0
                return _max_motion(-g * 1000.0, max_step), False
        if iteration == 0:
            if _norm(g) < 1e-15:
                kick = np.zeros_like(g)
                kick[0] = 1.0
                self.count = 0
                return _max_motion(kick, max_step), False
            if forced:
                curvature = -0.01
            else:
                length = _norm(g)
                direction = -g / length if length > 0.0 else -g
                trial = x + finite * direction
                _energy, trial_gradient = potential.energy_forces(trial)
                if eigenvalue is None or mode is None:
                    trial_effective = trial_gradient
                else:
                    trial_effective = _effective(trial_gradient, eigenvalue, mode)
                curvature = float(np.dot(trial_effective - g, direction) / finite) if finite != 0.0 else float("nan")
            if not math.isfinite(curvature) or curvature < 0.0:
                self.count = 0
                return _max_motion(-g * 1000.0, max_step), False
        scale = 1.0 / curvature
        q = np.array(g, dtype=np.float64, copy=True)
        alpha = np.zeros(self.count, dtype=np.float64)
        for index in range(self.count - 1, -1, -1):
            alpha[index] = self.rho[index] * float(np.dot(self.s[:, index], q))
            q = q - alpha[index] * self.y[:, index]
        direction_h = scale * q
        for index in range(self.count):
            beta = self.rho[index] * float(np.dot(self.y[:, index], direction_h))
            direction_h = direction_h + (alpha[index] - beta) * self.s[:, index]
        proposed = -direction_h
        length = _norm(proposed)
        gradient_length = _norm(g)
        if length >= max_step or length < 1e-15 or gradient_length < 1e-15:
            self.count = 0
            return _max_motion(scale * -g, max_step), False
        descent = float(np.dot(proposed / length, -g / gradient_length))
        if descent < 0.0:
            self.count = 0
            return _max_motion(scale * -g, max_step), False
        return _max_motion(proposed, max_step), False


def _settings(template: dict) -> dict:
    optimizer = _section(template, "Optimizer")
    saddle = _section(template, "SaddleSearch")
    search = _section(template, "Search")
    main = _section(template, "Main")
    factor = float(saddle.get("save_lanczos_factor", 50.0))
    max_step = float(optimizer.get("max_step", 0.5))
    ls_err = float(optimizer.get("ls_err", 1e-2))
    return {
        "solver": str(main.get("solver", "lanczos")),
        "seed_mode": str(saddle.get("mode_seed", "previous")),
        "iteration_limit": int(saddle.get("lanc_iter", 8)),
        "tolerance": float(saddle.get("lanc_abs_tol", 0.01)),
        "finite": float(saddle.get("lanc_finite", optimizer.get("finite_diff", 1e-4))),
        "memory": int(optimizer.get("memory_size", 60)),
        "max_step": max_step,
        "cg_iter_max": int(optimizer.get("max_iterations", CLIMB_BUDGET)),
        "cg_err": float(optimizer.get("cg_err", 1e-3)),
        "ls_err": ls_err,
        "reuse_tolerance": ls_err + (max_step - ls_err) * factor / 100.0,
        "save_lanczos": bool(saddle.get("save_lanczos_call", True)),
        "dE_max": float(optimizer.get("dE_max", 480.0)),
        "batch": int(search.get("batch", 25)),
        "hyper_rad0": float(search.get("hyper_rad0", 0.1)),
        "displace_atoms": int(search.get("displace_atoms", 7)),
        "max_searches": int(search.get("max_searches", 5000)),
        "repetitions": int(search.get("repetitions", 1)),
        "mix_radius": bool(search.get("mix_prev_hyper_rad", True)),
        "keep_efficiency": float(search.get("keep_efficiency", 0.2)),
    }


def _mode_at(lib, potential, x, settings, eigenvalue, mode) -> tuple[float, np.ndarray, np.ndarray, float]:
    if eigenvalue is None or mode is None:
        initial = INITIAL_GRADIENT if settings["seed_mode"] == "gradient" else INITIAL_PREVIOUS
        seed_vector = None
        if initial == INITIAL_GRADIENT:
            _energy, gradient = potential.energy_forces(x)
            if _norm(gradient) < 1e-14:
                initial = INITIAL_USER
                seed_vector = np.zeros(x.size, dtype=np.float64)
                seed_vector[0] = 1.0
        eigenvalue, mode, gradient, _iterations, _status, energy = _minimum_mode(
            lib,
            potential,
            x,
            initial=initial,
            iteration_limit=settings["iteration_limit"],
            finite=settings["finite"],
            tolerance=settings["tolerance"],
            seed_vector=seed_vector,
        )
    else:
        energy, gradient = potential.energy_forces(x)
    if not np.all(np.isfinite(gradient)) or not math.isfinite(float(energy)) or not math.isfinite(eigenvalue):
        raise RuntimeError("softsaddle mode returned a non-finite value")
    return float(eigenvalue), np.asarray(mode, dtype=np.float64), np.asarray(gradient, dtype=np.float64), float(energy)


def climb(
    lib,
    potential: HarnessSurface,
    x0: np.ndarray,
    *,
    settings: dict,
    force_tol: float,
    minimum_energy: float,
    climb_budget: int,
) -> dict:
    """One LBFGS minimum-mode climb. Endpoint minimizations are separate."""
    x = np.array(x0, dtype=np.float64, copy=True)
    dimension = int(x.size)
    memory = _LBFGS(dimension, settings["memory"])
    eigenvalue = None
    mode = None
    effective = np.ones(dimension, dtype=np.float64) * 1000.0
    true_gradient = effective.copy()
    energy = float(minimum_energy)
    previous_x = x.copy()
    previous_g = effective.copy()
    previous_eigenvalue = None
    step = np.zeros(dimension, dtype=np.float64)
    accumulated = None
    reuse_count = 0
    bandera = False
    crossed = None
    iteration = 0
    converged = False
    true_norm = float("inf")
    while (
        iteration < int(settings["cg_iter_max"])
        and potential.calls < climb_budget
        and energy - minimum_energy < float(settings["dE_max"])
    ):
        if iteration == 0 or not settings["save_lanczos"]:
            eigenvalue, mode = None, None
        else:
            reuse_count += 1
            eigenvalue, mode, accumulated, reuse_count = _analyze(
                eigenvalue, mode, step, accumulated, reuse_count, settings["reuse_tolerance"]
            )
        try:
            eigenvalue, mode, true_gradient, energy = _mode_at(lib, potential, x, settings, eigenvalue, mode)
        except RuntimeError as exc:
            return {
                "x": x,
                "energy": energy,
                "eigenvalue": eigenvalue,
                "mode": mode,
                "force_norm": true_norm,
                "converged": False,
                "iterations": iteration,
                "x_lambda": crossed,
                "error": str(exc),
            }
        effective = _effective(true_gradient, eigenvalue, mode)
        true_norm = _norm(true_gradient)
        if eigenvalue < 0.0 and true_norm <= force_tol and np.all(np.isfinite(x)):
            converged = True
            break
        if iteration > 0:
            memory.update(x, previous_x, effective, previous_g)
        step, bandera = memory.step(
            x,
            effective,
            iteration=iteration,
            finite=settings["finite"],
            max_step=settings["max_step"],
            potential=potential,
            eigenvalue=eigenvalue,
            mode=mode,
            bandera=bandera,
        )
        if bandera:
            step = _max_motion(step, settings["max_step"])
            bandera = False
        previous_x = x.copy()
        previous_g = effective.copy()
        x = x + step
        if (
            iteration > 0
            and previous_eigenvalue is not None
            and eigenvalue is not None
            and crossed is None
            and _sign(float(previous_eigenvalue)) + _sign(float(eigenvalue)) == 0.0
        ):
            crossed = _lambda_zero(previous_x, x, previous_eigenvalue, eigenvalue)
        previous_eigenvalue = eigenvalue
        iteration += 1
        if eigenvalue is not None and eigenvalue >= 0.0 and float(np.max(np.abs(effective))) <= settings["cg_err"]:
            length = _norm(effective)
            if length > 0.0:
                effective = 1000.0 * effective / length
            bandera = True
    if converged and settings["save_lanczos"]:
        try:
            eigenvalue, mode, true_gradient, energy = _mode_at(lib, potential, x, settings, None, None)
            effective = _effective(true_gradient, eigenvalue, mode)
            true_norm = _norm(true_gradient)
            converged = bool(eigenvalue < 0.0 and true_norm <= force_tol)
        except RuntimeError as exc:
            converged = False
            return {
                "x": x,
                "energy": energy,
                "eigenvalue": eigenvalue,
                "mode": mode,
                "force_norm": true_norm,
                "converged": False,
                "iterations": iteration,
                "x_lambda": crossed,
                "error": str(exc),
            }
    return {
        "x": x,
        "energy": float(energy),
        "eigenvalue": None if eigenvalue is None else float(eigenvalue),
        "mode": None if mode is None else np.asarray(mode, dtype=np.float64),
        "force_norm": float(true_norm) if math.isfinite(true_norm) else float("inf"),
        "converged": bool(converged and _finite_point(x)),
        "iterations": int(iteration),
        "x_lambda": None if crossed is None else np.asarray(crossed, dtype=np.float64),
        "error": None,
    }


def _public(name: str, calls: int, success: bool, failure: str | None, reasons: list[str], **extra) -> dict:
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(extra.pop("verify_calls", 0)),
        "counter": "harness",
        "comparable": True,
        "success": bool(success),
        "failure": failure,
        "reasons": reasons,
    }
    record.update(extra)
    return record


def run_mode_point(spec: dict) -> dict:
    """Lowest mode of one geometry. ``spec`` is picklable."""
    os.environ["OMP_NUM_THREADS"] = "1"
    kind = str(spec["solver"])
    lib = _load(kind)
    evaluate = spec.get("evaluate")
    if evaluate is None:
        positions = np.asarray(spec["positions"], dtype=np.float64)
        fixed = np.asarray(spec["fixed"], dtype=bool)
        cell = np.asarray(spec["cell"], dtype=np.float64)
        free = np.flatnonzero(~fixed)
        potential = HarnessSurface(_bind(potential_from_problem({"potential": spec["potential"]}), positions, free, cell))
        x = positions[free].reshape(-1).copy()
        reference = _unit(np.asarray(spec["reference"], dtype=np.float64))
        fixed_any = bool(np.any(fixed))
    else:
        potential = HarnessSurface(evaluate)
        x = np.asarray(spec["x"], dtype=np.float64).copy()
        reference = _unit(np.asarray(spec["reference"], dtype=np.float64))
        fixed_any = False
    cap = int(spec["cap"])
    overlap_stop = float(spec["overlap"])
    work = Path(spec["work"])
    settings = spec["settings"]
    with _at(work):
        lib.softsaddle_seed(int(spec["seed"]) & 0xFFFFFFFF)
        lib.softsaddle_reset()
        initial = INITIAL_GRADIENT if settings["seed_mode"] == "gradient" else INITIAL_PREVIOUS
        seed_vector = None
        overlap = 0.0
        best_overlap = -1.0
        previous = None
        eigenvalue = float("inf")
        vector = np.zeros(x.size, dtype=np.float64)
        steps = 0
        error = None
        try:
            if initial == INITIAL_GRADIENT:
                _energy, gradient = potential.energy_forces(x)
                if _norm(gradient) < 1e-14:
                    initial = INITIAL_USER
                    seed_vector = np.zeros(x.size, dtype=np.float64)
                    seed_vector[0] = 1.0
            while potential.calls < cap:
                before = potential.calls
                eigenvalue, vector, _gradient, _iterations, _status, _energy = _minimum_mode(
                    lib,
                    potential,
                    x,
                    initial=initial,
                    iteration_limit=settings["iteration_limit"],
                    finite=settings["finite"],
                    tolerance=settings["tolerance"],
                    seed_vector=seed_vector,
                )
                steps += 1
                overlap = float(abs(np.dot(_unit(vector), reference)))
                if overlap >= overlap_stop:
                    break
                if previous is not None and _norm(vector - previous) < 1e-8:
                    break
                # A repeated estimate at one geometry that does not raise the
                # overlap is the same stall. Davidson rebuilds its Hessian
                # when the point has not moved, and the vector jitters.
                if overlap <= best_overlap + 1e-4:
                    break
                best_overlap = overlap
                if potential.calls == before:
                    break
                previous = vector.copy()
                initial = INITIAL_PREVIOUS
                seed_vector = None
        except RuntimeError as exc:
            error = str(exc)
        energy, gradient = potential.energy_forces(x)
        reasons: list[str] = []
        if error:
            reasons.append("crash")
        if not (overlap >= overlap_stop and potential.calls < cap and error is None):
            reasons.append("mode_overlap")
        if potential.calls >= cap:
            reasons.append("cap")
        success, failure = _outcome(reasons)
        record = _public(
            spec["name"],
            potential.calls,
            success,
            failure,
            reasons,
            energy=float(energy),
            force_norm=_norm(gradient),
            overlap=overlap,
            curvature=float(eigenvalue) if math.isfinite(eigenvalue) else None,
            converged=bool(overlap >= overlap_stop and potential.calls < cap and error is None),
            steps=steps,
            solver=kind,
            error=error,
        )
        if error is None and _finite_point(x):
            raw, vectors = _hessian_spectrum(potential, x)
            if fixed_any:
                eigenvalues, removed = list(raw), 0
            else:
                eigenvalues, removed = _vibrational_eigenvalues(x, raw, vectors)
            record["cartesian_index"] = negative_index(raw, 1e-6)
            record["hessian_index"] = negative_index(eigenvalues, 1e-6)
            record["rigid_modes_removed"] = int(removed)
            record["lowest_eigenvalue"] = float(eigenvalues[0]) if eigenvalues else None
            record["lowest_eigenvalues"] = [float(value) for value in eigenvalues[:8]]
            record["verify_calls"] = int(potential.verify_calls)
        return record


def _judge_saddle(payload: dict, climbed: dict, potential: HarnessSurface, calls_before_endpoints: int) -> dict:
    force_tol = float(payload["force_tol"])
    minimum = np.asarray(payload["minimum"], dtype=np.float64)
    min_energy = float(payload["min_energy"])
    distance_difference = float(payload["distance_difference"])
    lengths = np.asarray(payload["lengths"], dtype=np.float64)
    barrier_max = float(payload["barrier_max"])
    search_cap = int(payload["search_cap"])
    reasons = ["force_norm"]
    record = {
        "converged": bool(climbed["converged"]),
        "curvature": climbed["eigenvalue"],
        "iterations": climbed["iterations"],
        "hessian_index": None,
        "barrier": None,
        "per_atom_max_plus": None,
        "per_atom_max_minus": None,
        "lambda_distance": None,
        "saddle_distance": None,
        "saddle_x": None,
        "novel_candidate": False,
        "error": climbed["error"],
    }
    point = np.asarray(climbed["x"], dtype=np.float64)
    finite = _finite_point(point)
    negative = (
        climbed["eigenvalue"] is not None
        and math.isfinite(float(climbed["eigenvalue"]))
        and float(climbed["eigenvalue"]) < 0.0
    )
    if climbed["error"]:
        reasons = ["crash"]
    elif climbed["converged"] and negative and finite:
        raw, _vectors = _hessian_spectrum(potential, point)
        index = negative_index(raw, 1e-6)
        record["hessian_index"] = int(index)
        record["lowest_eigenvalue"] = float(raw[0]) if raw else None
        record["verify_calls"] = int(potential.verify_calls)
        if index == 1 and float(climbed["force_norm"]) <= force_tol + 1e-10 and potential.calls < search_cap:
            mode = climbed["mode"] if climbed["mode"] is not None else np.zeros_like(point)
            axis = _unit(np.asarray(mode, dtype=np.float64))
            plus, _plus_ok = _lbfgs_minimize(potential, point + 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET)
            minus, _minus_ok = _lbfgs_minimize(potential, point - 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET)
            links = endpoints_connected(minimum, plus, minus, lengths, distance_difference)
            barrier = float(climbed["energy"] - min_energy)
            record["barrier"] = barrier
            record["per_atom_max_plus"] = per_atom_max(plus, minimum, lengths)
            record["per_atom_max_minus"] = per_atom_max(minus, minimum, lengths)
            record["saddle_distance"] = _norm(point - minimum)
            record["saddle_x"] = point.copy()
            record["novel_candidate"] = bool(links)
            if climbed["x_lambda"] is not None:
                distance = _norm(np.asarray(climbed["x_lambda"], dtype=np.float64) - minimum)
                if distance > 0.0:
                    record["lambda_distance"] = distance
            reasons = []
            if not links:
                reasons.append("not_connected")
            if barrier >= barrier_max:
                reasons.append("barrier")
        else:
            reasons = ["hessian_index"] if index != 1 else ["force_norm"]
    elif not finite:
        reasons = ["divergence"]
    elif calls_before_endpoints >= int(payload["climb_budget"]) or potential.calls >= search_cap:
        reasons = ["cap", "force_norm"]
    if potential.calls >= search_cap and "cap" not in reasons and reasons:
        reasons.append("cap")
    success, failure = _outcome(reasons)
    if success and potential.calls >= search_cap:
        success, failure = False, "cap"
        reasons = ["cap"]
    record["reasons"] = reasons
    record["success"] = success
    record["failure"] = failure
    record["force_calls"] = int(potential.calls)
    record["force_norm"] = float(climbed["force_norm"]) if math.isfinite(float(climbed["force_norm"])) else None
    record["energy"] = float(climbed["energy"]) if climbed["energy"] is not None else None
    return record


def _saddle_worker(payload: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    positions = np.asarray(payload["positions"], dtype=np.float64)
    fixed = np.asarray(payload["fixed"], dtype=bool)
    cell = np.asarray(payload["cell"], dtype=np.float64)
    free = np.flatnonzero(~fixed)
    potential = HarnessSurface(_bind(potential_from_problem({"potential": payload["potential"]}), positions, free, cell))
    lib = _load(str(payload["settings"]["solver"]))
    work = Path(payload["work"])
    with _at(work):
        lib.softsaddle_seed(int(payload["seed"]) & 0xFFFFFFFF)
        lib.softsaddle_reset()
        climbed = climb(
            lib,
            potential,
            np.asarray(payload["x0"], dtype=np.float64),
            settings=payload["settings"],
            force_tol=float(payload["force_tol"]),
            minimum_energy=float(payload["min_energy"]),
            climb_budget=int(payload["climb_budget"]),
        )
        judged = _judge_saddle(payload, climbed, potential, int(potential.calls))
    judged["verify_calls"] = int(potential.verify_calls)
    judged["name"] = payload["name"]
    judged["index"] = int(payload["index"])
    if judged.get("saddle_x") is not None:
        judged["saddle_x"] = np.asarray(judged["saddle_x"], dtype=np.float64).tolist()
    return judged


def _apply_novelty(judged: dict, found: ProcessTable) -> dict:
    reasons = list(judged["reasons"])
    success = bool(judged["success"])
    point = judged.get("saddle_x")
    if judged.get("novel_candidate") and point is not None:
        saddle = np.asarray(point, dtype=np.float64)
        barrier = float(judged["barrier"])
        if not found.repeat(saddle, barrier):
            found.add(saddle, barrier)
            judged["accepted_novel"] = True
        else:
            if "duplicate" not in reasons:
                reasons.append("duplicate")
            success = False
        if success and reasons:
            success = False
    failure = None if success and not reasons else classify_failure(reasons or ["divergence"])
    if success:
        failure = None
        reasons = []
    judged = dict(judged)
    judged["reasons"] = reasons
    judged["success"] = bool(success and failure is None)
    judged["failure"] = failure
    return judged


def _radius_update(radius: float, batch: list[dict], lam: list[float], sad: list[float], settings: dict) -> float:
    connected = [row for row in batch if row.get("success")]
    efficiency = len(connected) / float(settings["batch"])
    if not connected and not sad:
        return radius
    for row in connected:
        if row.get("lambda_distance"):
            lam.append(float(row["lambda_distance"]))
        if row.get("saddle_distance"):
            sad.append(float(row["saddle_distance"]))
    if not lam:
        return radius
    proposed = float(np.mean(lam))
    if settings["mix_radius"]:
        proposed = 0.5 * (proposed + radius)
    if efficiency >= float(settings["keep_efficiency"]):
        return radius
    if not math.isfinite(proposed) or proposed <= 0.0:
        return radius
    return float(proposed)


def _replay_radius(records: list[dict], settings: dict) -> tuple[float, list[float], list[float]]:
    radius = float(settings["hyper_rad0"])
    lam: list[float] = []
    sad: list[float] = []
    batch = int(settings["batch"])
    complete = len(records) - (len(records) % batch)
    for start in range(0, complete, batch):
        radius = _radius_update(radius, records[start : start + batch], lam, sad, settings)
    return radius, lam, sad


def hypersphere() -> tuple[np.ndarray, np.ndarray]:
    path = _repo_root() / "softsaddle" / "data" / "hypersphere_5000_rep1.npz"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    packed = np.load(path)
    return np.asarray(packed["V"], dtype=np.float64), np.asarray(packed["order"], dtype=np.int64)


def _displace(minimum: np.ndarray, direction: np.ndarray, radius: float, atoms: int) -> np.ndarray:
    width = 3 * int(atoms)
    x0 = np.array(minimum, dtype=np.float64, copy=True)
    x0[:width] = minimum[:width] + float(radius) * np.asarray(direction, dtype=np.float64)
    return x0


def _checkpoint_paths(output: Path) -> tuple[Path, Path]:
    return output.with_name(output.stem + ".header.json"), output.with_name(output.stem + ".partial.jsonl")


def _load_partial(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_partial(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _jsonable(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _run_saddle(root: Path, problem: dict, template: dict, *, output: Path, force_tol: float, work: Path, workers: int) -> dict:
    settings = _settings(template)
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    from .rgsaddle_minmode import _stored

    positions, fixed, cell = load_con(_stored(root, con_path))
    lengths = periodic_lengths(cell, (problem.get("potential") or {}).get("pbc"))
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    distance_difference, energy_difference = tolerances(problem)
    climb_budget = int(settings["cg_iter_max"])
    search_cap = climb_budget + _SADDLE_CLIMB_OVERSHOOT + 2 * _SADDLE_ENDPOINT_BUDGET
    vectors, order = hypersphere()
    free = np.flatnonzero(~fixed)
    raw = potential_from_problem(problem)
    header_path, partial_path = _checkpoint_paths(output)
    header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
    saved = _load_partial(partial_path) if header and header.get("template") == template["id"] else []
    if header and header.get("template") == template["id"]:
        minimum = np.asarray(header["minimum"], dtype=np.float64)
        min_energy = float(header["min_energy"])
        minimize_calls = int(header["minimize_calls"])
        min_ok = bool(header["min_ok"])
    else:
        potential = HarnessSurface(_bind(raw, positions, free, cell))
        minimum, min_ok = _lbfgs_minimize(potential, positions[free].reshape(-1).copy(), force_tol, 5000)
        min_energy, _gradient = potential.energy_forces(minimum)
        minimize_calls = int(potential.calls)
        saved = []
        partial_path.unlink(missing_ok=True)
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(
            json.dumps(
                {
                    "template": template["id"],
                    "minimum": minimum.tolist(),
                    "min_energy": min_energy,
                    "minimize_calls": minimize_calls,
                    "min_ok": bool(min_ok),
                }
            ),
            encoding="utf-8",
        )
    radius, _lam, _sad = _replay_radius(saved, settings)
    found = ProcessTable(lengths, distance_difference, energy_difference)
    connected = 0
    for row in saved:
        if row.get("saddle_stored") is not None:
            found.add(row["saddle_stored"], float(row["barrier"]))
        if row.get("success"):
            connected += 1
    print(
        f"{template['id']} resume searches {len(saved)} connected {connected} radius {radius:.4f}",
        flush=True,
    )
    batch = int(settings["batch"])
    max_searches = min(int(settings["max_searches"]), int(order.size))

    def _batch(start: int, stop: int, radius_now: float) -> list[dict]:
        payloads = []
        for index in range(start, stop):
            direction = vectors[int(order[index])]
            x0 = _displace(minimum, direction, radius_now, settings["displace_atoms"])
            payloads.append(
                {
                    "name": f"rep1-search{index + 1}",
                    "index": index,
                    "positions": positions,
                    "fixed": fixed,
                    "cell": cell,
                    "potential": problem["potential"],
                    "x0": x0,
                    "minimum": minimum,
                    "min_energy": min_energy,
                    "settings": settings,
                    "force_tol": force_tol,
                    "distance_difference": distance_difference,
                    "lengths": lengths,
                    "barrier_max": barrier_max,
                    "search_cap": search_cap,
                    "climb_budget": climb_budget,
                    "seed": 20261015 + index + 1,
                    "work": str(work / f"search-{index + 1:05d}"),
                }
            )
        return payloads

    pool = None
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        while connected < goal and len(saved) < max_searches:
            start = len(saved)
            stop = min(start + batch, max_searches)
            payloads = _batch(start, stop, radius)
            if pool is None or len(payloads) == 1:
                judged_rows = [_saddle_worker(item) for item in payloads]
            else:
                judged_rows = list(pool.map(_saddle_worker, payloads))
            judged_rows.sort(key=lambda row: int(row["index"]))
            for judged in judged_rows:
                applied = _apply_novelty(judged, found)
                if applied["success"]:
                    connected += 1
                stored_point = None
                if applied.get("accepted_novel") and applied.get("saddle_x") is not None:
                    stored_point = list(applied["saddle_x"])
                public = _public(
                    applied["name"],
                    applied["force_calls"],
                    applied["success"],
                    applied["failure"],
                    list(applied["reasons"]),
                    verify_calls=int(applied.get("verify_calls") or 0),
                    energy=applied.get("energy"),
                    force_norm=applied.get("force_norm"),
                    hessian_index=applied.get("hessian_index"),
                    barrier=applied.get("barrier"),
                    per_atom_max_plus=applied.get("per_atom_max_plus"),
                    per_atom_max_minus=applied.get("per_atom_max_minus"),
                    curvature=applied.get("curvature"),
                    converged=bool(applied.get("converged")),
                    lambda_distance=applied.get("lambda_distance"),
                    saddle_distance=applied.get("saddle_distance"),
                    iterations=applied.get("iterations"),
                    solver=settings["solver"],
                    radius=radius,
                    error=applied.get("error"),
                )
                partial = dict(public)
                partial["saddle_stored"] = stored_point
                saved.append(partial)
                _append_partial(partial_path, partial)
            if len(saved) % batch == 0 or len(saved) >= max_searches or connected >= goal:
                radius, _lam, _sad = _replay_radius(saved, settings)
            print(
                f"{template['id']} searches {len(saved)} connected {connected} radius {radius:.4f} "
                f"last {saved[-1]['force_calls']} {saved[-1]['failure']}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
    records = []
    for row in saved:
        public = {key: value for key, value in row.items() if key != "saddle_stored"}
        records.append(public)
    fraction = connected / len(records) if records else 0.0
    unique = len(found)
    comments = (
        f"# rep 1 force_calls {sum(int(row['force_calls']) for row in records)} "
        f"unique_saddles {unique} fraction_connected {fraction} total_searches {len(records)} "
        f"connected {connected} minimize_calls {minimize_calls} minimum_converged {int(bool(min_ok))} "
        f"seed 20261016 hypersphere {HYPER_SOURCE}"
    )
    reached = connected >= goal
    finished = reached or len(records) >= max_searches
    text = comments + "\n"
    if records:
        ok = [int(row["force_calls"]) for row in records if row["success"]]
        if ok:
            text += f"force_calls {half_even(sum(ok) / len(ok))}\n"
            text += f"force_calls_min {min(ok)}\n"
            text += f"force_calls_median {float(np.median(ok))}\n"
            text += f"force_calls_max {max(ok)}\n"
    metrics = score_v2(
        [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records],
        search_cap,
    ) if records else None
    text += (
        f"code {template.get('code', 'softsaddle')}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"repetitions 1\n"
        f"nfailed {0 if reached else 1}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(records)}\n"
    )
    if not finished:
        text += "partial true\n"
    if metrics is not None:
        text += (
            f"par2_mean {metrics['par2_mean']}\n"
            f"successes {metrics['successes']}\n"
            f"rate_low {metrics['rate_low']}\n"
            f"rate_high {metrics['rate_high']}\n"
            f"par2_median {metrics['median']}\n"
            f"n_cap {metrics['n_cap']}\n"
            f"n_wrong_stationary {metrics['n_wrong_stationary']}\n"
            f"n_divergence {metrics['n_divergence']}\n"
            f"n_crash {metrics['n_crash']}\n"
            f"v2_nfailed {metrics['nfailed']}\n"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    if metrics is not None:
        provenance = collect_provenance(
            code=str(template.get("code", "softsaddle")),
            code_version=str(template.get("code_version", "")),
            commit=str(template.get("commit", "")),
            parameters={
                "template": template["id"],
                "problem": problem["id"],
                "cap": search_cap,
                "solver": settings["solver"],
                "hypersphere_sha256": HYPER_SOURCE,
            },
        )
        document = {
            "schema": "optbench.result/v2",
            "status": "measured" if finished else "partial",
            "problem": problem["id"],
            "scoring": "par2-bootstrap",
            "par_factor": 2,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 20261015,
            "note": (
                "One row is one hypersphere displacement. The cap is the climb budget, "
                "64 calls of overshoot, and the two endpoint minimizations. "
                "The initial minimization is not in that count."
            ),
            "runs": [
                {
                    "entry": output.stem,
                    "code": template.get("code", "softsaddle"),
                    "algorithm": template.get("algorithm", ""),
                    "code_version": template.get("code_version", ""),
                    "commit": template.get("commit", ""),
                    "tuning": template.get("tuning", ""),
                    "present_source": True,
                    "cap": search_cap,
                    "provenance": provenance,
                    "metrics": metrics,
                    "starts": records,
                }
            ],
        }
        output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if finished:
        partial_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
    return {
        "force_calls": metrics["success_only_mean"] if metrics else None,
        "nfailed": 0 if reached else 1,
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "connected": connected,
        "records": records,
    }


def mode_spec(root: Path, problem: dict, template: dict, row: dict, cap: int, work: Path, index: int) -> dict:
    """One lowest-mode start. The geometry index selects the library seed."""
    settings = _settings(template)
    positions, fixed, cell = load_con(Path(row["path"]))
    reference = None
    if row.get("mode"):
        full = read_mode(Path(row["mode"]).read_text(encoding="utf-8", errors="replace"))
        reference = full.reshape(-1, 3)[~fixed].reshape(-1)
    return {
        "name": row["name"],
        "positions": positions,
        "fixed": fixed,
        "cell": cell,
        "potential": problem.get("potential") or {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0},
        "reference": reference,
        "cap": cap,
        "overlap": float((problem.get("sample") or {}).get("overlap", 0.99)),
        "solver": settings["solver"],
        "settings": settings,
        "seed": 20261015 + index,
        "work": str(work / f"mode-{row['name']}"),
    }


def _run_modes(root: Path, problem: dict, template: dict, *, output: Path, cap: int, work: Path, workers: int) -> dict:
    rows = _geometries(root, problem, work)
    expected = int((problem.get("sample") or {}).get("starts") or len(rows))
    if len(rows) != expected:
        raise RuntimeError(f"{problem['id']} has {len(rows)} geometries, the problem states {expected}")
    specs = [mode_spec(root, problem, template, row, cap, work, index) for index, row in enumerate(rows)]
    if workers <= 1:
        records = [run_mode_point(spec) for spec in specs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            records = list(pool.map(run_mode_point, specs))
    ok = [int(row["force_calls"]) for row in records if row["success"]]
    lines = []
    if ok:
        lines.append(f"force_calls {half_even(sum(ok) / len(ok))}")
        lines.append(f"force_calls_min {min(ok)}")
        lines.append(f"force_calls_median {float(np.median(ok))}")
        lines.append(f"force_calls_max {max(ok)}")
    metrics = score_v2(
        [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records],
        cap,
    )
    lines.append(f"nfailed {metrics['nfailed']}")
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
    lines.append(f"code {template.get('code', 'softsaddle')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append(f"harness_calls {sum(int(row['force_calls']) for row in records)}")
    if len(records) < expected:
        lines.append("partial true")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = collect_provenance(
        code=str(template.get("code", "softsaddle")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={
            "template": template["id"],
            "problem": problem["id"],
            "cap": cap,
            "solver": str(_section(template, "Main").get("solver", "lanczos")),
        },
    )
    public = []
    for record in records:
        copied = _jsonable(record)
        copied["code"] = provenance["code"]
        copied["code_version"] = provenance["code_version"]
        copied["commit"] = provenance["commit"]
        copied["tuning"] = template.get("tuning", "")
        copied["template"] = template["id"]
        public.append(copied)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    return {
        "force_calls": half_even(sum(ok) / len(ok)) if ok else None,
        "nfailed": metrics["nfailed"],
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
        "records": records,
    }


def run_template(root: Path, problem: dict, template: dict, *, output: Path, workers: int = 1, work: Path | None = None) -> dict:
    """Run one SoftSaddle template."""
    from .campaign import _check_stop

    solver = str(_section(template, "Main").get("solver", "lanczos"))
    if library_path(solver) is None:
        raise RuntimeError(f"softsaddle {solver} library is not built")
    cap = _check_stop(problem, template)
    optimizer = _section(template, "Optimizer")
    convergence = problem.get("convergence") or {}
    force_tol = float(optimizer.get("converged_force", convergence.get("force_norm", 1e-3)))
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    if problem["id"] == "saddle-search/pt-island":
        return _run_saddle(root, problem, template, output=output, force_tol=force_tol, work=scratch, workers=workers)
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    return _run_modes(root, problem, template, output=output, cap=cap, work=scratch, workers=workers)
