"""CI-NEB campaigns through the rgsaddle C band and min-mode sessions.

rgsaddle 0.2.2 on the recorded main does not build ``Stencil.band``.
This module calls ``librgsaddle.so`` (``RGSADDLE_LIBRARY`` or the
release build under ``/tmp/rgsaddle``). A missing library raises when
a search starts; unit tests that only check geometry skip that path.

Cost is ``RgpotCounter.calculate`` on a ``HarnessCounter`` for the
free atoms. The initial-path IDPP and S-IDPP relaxations use the
surrogate and are not potential calls. Hessian finite differences go
through ``verification`` and are not search cost.

The band stores raw potential evaluations, not a per-image quotient.
Energy-weighted springs are the eOn constants from one initial
evaluation and then stay fixed, because the C session takes them at
create. OCI-NEB is a CI-NEB band followed by a min-mode refinement.
The min-mode L-BFGS memory is 20: ``rgsaddle_minmode_create`` passes
memory 0 and the C binding then uses 20.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .counter import HarnessCounter
from .failures import classify_failure
from .pes import (
    energy_weighted_springs,
    evaluate,
    idpp_objective,
    interpolate_free,
    load_pairs,
    pair_distances,
)
from .potserver import RgpotCounter
from .score import score_v2
from .stationary import negative_index, numerical_hessian
from .v2 import collect_provenance, write_starts

_ABI_MAJOR = 1
_ABI_MINOR = 16
_KEEP_TRANSLATION = 1 << 1
_OK = 0
_SURFACE_FAILED = -6
_NO_EVALUATION = -12
_FORCE_L2 = 0
_FORCE_LINF = 1
_SPRING_UNIFORM = 0
_SPRING_WEIGHTED = 1
_TANGENT_IMPROVED = 1
_PROJECTION_NEB = 1
_METHOD_FIRE = 0
_METHOD_LBFGS = 1
_MINMODE_DIMER = 0
_MINMODE_LANCZOS = 1
_DIFFERENCE_CENTRAL = 1
_STATUS_CONVERGED = 1
_EIGEN_TOL = 1e-6
_FD_STEP = 1e-5
_MINMODE_LBFGS_MEMORY = 20

_SURFACE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


class _Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class BandConfig(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("flags", ctypes.c_uint64),
        ("tangent", ctypes.c_int32),
        ("spring", ctypes.c_int32),
        ("projection", ctypes.c_int32),
        ("method", ctypes.c_int32),
        ("spring_k", ctypes.c_double),
        ("spring_ks", ctypes.POINTER(ctypes.c_double)),
        ("ci_trigger_factor", ctypes.c_double),
        ("ci_trigger_force", ctypes.c_double),
        ("cell", ctypes.POINTER(ctypes.c_double)),
        ("force_tol", ctypes.c_double),
        ("max_move", ctypes.c_double),
        ("memory", ctypes.c_int64),
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


class _Request(ctypes.Structure):
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


def library_path() -> Path | None:
    """The shared library this process would load, if the file exists."""
    env = os.environ.get("RGSADDLE_LIBRARY")
    if env:
        path = Path(env)
        return path if path.is_file() else None
    default = Path("/tmp/rgsaddle/target/release/librgsaddle.so")
    return default if default.is_file() else None


def _load_library():
    path = library_path()
    if path is None:
        raise FileNotFoundError("librgsaddle.so is not built and RGSADDLE_LIBRARY is unset")
    lib = ctypes.CDLL(str(path))
    lib.rgsaddle_band_create.argtypes = [
        ctypes.POINTER(BandConfig),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rgsaddle_band_create.restype = ctypes.c_void_p
    lib.rgsaddle_band_step.argtypes = [
        ctypes.c_void_p,
        _SURFACE,
        ctypes.c_void_p,
        ctypes.POINTER(Report),
    ]
    lib.rgsaddle_band_step.restype = ctypes.c_int
    lib.rgsaddle_band_set_force_gate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rgsaddle_band_set_force_gate.restype = ctypes.c_int
    lib.rgsaddle_band_positions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.rgsaddle_band_positions.restype = ctypes.c_int
    lib.rgsaddle_band_evaluation.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rgsaddle_band_evaluation.restype = ctypes.c_int
    lib.rgsaddle_band_free.argtypes = [ctypes.c_void_p]
    lib.rgsaddle_band_free.restype = None
    lib.rgsaddle_minmode_create.argtypes = [
        ctypes.POINTER(MinModeConfig),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rgsaddle_minmode_create.restype = ctypes.c_void_p
    lib.rgsaddle_minmode_step.argtypes = [
        ctypes.c_void_p,
        _SURFACE,
        ctypes.c_void_p,
        ctypes.POINTER(Report),
    ]
    lib.rgsaddle_minmode_step.restype = ctypes.c_int
    lib.rgsaddle_minmode_set_force_gate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rgsaddle_minmode_set_force_gate.restype = ctypes.c_int
    lib.rgsaddle_minmode_position.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.rgsaddle_minmode_position.restype = ctypes.c_int
    lib.rgsaddle_minmode_free.argtypes = [ctypes.c_void_p]
    lib.rgsaddle_minmode_free.restype = None
    return lib


def _threads_one() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"


def vibrational_index(
    hessian: np.ndarray,
    coords: np.ndarray,
    project_rigid: bool,
    rotations: int | None = None,
) -> int:
    """Negative eigenvalues of a Cartesian Hessian.

    A free pair potential is invariant to translation and rotation, so
    those six directions are removed before the count. A slab with
    fixed atoms, and a potential that is not translation invariant,
    keep every mode: projecting them would hide the real unstable
    direction. ``rotations`` overrides that choice: 0 keeps the three
    translations out and leaves rotations in, which is a periodic cell.
    """
    matrix = np.asarray(hessian, dtype=np.float64)
    if rotations is None:
        if project_rigid:
            matrix = _project_rigid(matrix, np.asarray(coords, dtype=np.float64))
    else:
        matrix = _project_rigid(matrix, np.asarray(coords, dtype=np.float64), int(rotations))
    values = np.linalg.eigvalsh(matrix)
    return negative_index(values, _EIGEN_TOL)


def _project_rigid(hessian: np.ndarray, coords: np.ndarray, rotations: int = 3) -> np.ndarray:
    n_atoms = coords.shape[0]
    if n_atoms < 2:
        return hessian
    centered = coords - coords.mean(axis=0)
    seeds = []
    for axis in range(3):
        translation = np.zeros(n_atoms * 3, dtype=np.float64)
        translation[axis::3] = 1.0
        seeds.append(translation)
        if rotations:
            rotation = np.cross(centered, np.eye(3)[axis])
            seeds.append(rotation.reshape(-1))
    basis = []
    for seed in seeds:
        mode = seed.copy()
        for kept in basis:
            mode -= np.dot(kept, mode) * kept
        norm = float(np.linalg.norm(mode))
        if norm > 1e-8:
            basis.append(mode / norm)
    if not basis:
        return hessian
    modes = np.column_stack(basis)
    projector = np.eye(hessian.shape[0]) - modes @ modes.T
    # The external block is shifted to +1 so a numerical zero cannot
    # be counted as a negative curvature.
    return projector @ hessian @ projector + modes @ modes.T


class _Surface:
    """One callback target. The C session calls it on the stepping thread."""

    def __init__(self, kind: str, rgpot: RgpotCounter | None, cap: int):
        self.kind = kind
        self.rgpot = rgpot
        self.cap = cap
        self.n_band = 0
        self.slots: list[int] = []
        self.n_req = 0
        self.template: np.ndarray | None = None
        self.free: np.ndarray | None = None
        self.box: np.ndarray | None = None
        self.box_flat: tuple[float, ...] | None = None
        self.d_reactant: np.ndarray | None = None
        self.d_product: np.ndarray | None = None
        self.hit_cap = False
        self.crashed = False
        self.crash_name = ""
        self.last_point: np.ndarray | None = None
        self.last_energy = float("nan")
        self.last_grad: np.ndarray | None = None

    def calls(self) -> int:
        if self.rgpot is None:
            return 0
        return int(self.rgpot.counter.calls)

    def __call__(self, _user, raw) -> int:
        try:
            return self._fill(_Request.from_address(raw))
        except Exception as exc:
            self.crashed = True
            self.crash_name = type(exc).__name__
            traceback.print_exc()
            return _SURFACE_FAILED

    def _fill(self, req: _Request) -> int:
        n_rows = int(req.n_images)
        dof = int(req.n_atoms) * 3
        positions = np.ctypeslib.as_array(req.positions, shape=(n_rows * dof,)).reshape(n_rows, dof).copy()
        energies = np.ctypeslib.as_array(req.energies, shape=(n_rows,))
        gradients = np.ctypeslib.as_array(req.gradients, shape=(n_rows * dof,)).reshape(n_rows, dof)
        whole = n_rows == self.n_band
        for row in range(n_rows):
            if self.rgpot is not None and self.calls() >= self.cap:
                self.hit_cap = True
                return _SURFACE_FAILED
            image = row if whole else row + 1
            energy, grad = self._row(positions[row], image)
            if not np.isfinite(energy) or not np.all(np.isfinite(grad)):
                return -7
            energies[row] = energy
            gradients[row] = grad
        return _OK

    def _row(self, flat: np.ndarray, image: int) -> tuple[float, np.ndarray]:
        if self.kind == "pes":
            assert self.rgpot is not None
            energy, grad = self.rgpot.calculate(flat, None, self.box_flat)
            vector = np.asarray(grad, dtype=np.float64)
            self.last_point = np.array(flat, dtype=np.float64, copy=True)
            self.last_energy = float(energy)
            self.last_grad = vector
            return float(energy), vector
        assert self.template is not None and self.free is not None
        assert self.d_reactant is not None and self.d_product is not None
        full = np.array(self.template, dtype=np.float64, copy=True)
        full[self.free] = flat.reshape((-1, 3))
        slot = self.slots[image]
        lam = slot / (self.n_req - 1)
        target = (1.0 - lam) * self.d_reactant + lam * self.d_product
        energy, grad = idpp_objective(full, target, self.box)
        return float(energy), np.asarray(grad, dtype=np.float64)[self.free].reshape(-1)


class _Session:
    def __init__(self, lib, handle, callback, n_images: int, dof: int, free_fn):
        self.lib = lib
        self.handle = handle
        self.callback = callback
        self.n_images = n_images
        self.dof = dof
        self._free = free_fn

    def step(self) -> tuple[int, Report]:
        report = Report()
        rc = int(self.lib.rgsaddle_band_step(self.handle, self.callback, None, ctypes.byref(report)))
        return rc, report

    def positions(self) -> np.ndarray:
        buf = np.empty(self.n_images * self.dof, dtype=np.float64)
        rc = int(self.lib.rgsaddle_band_positions(self.handle, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if rc != _OK:
            raise RuntimeError(f"rgsaddle_band_positions returned {rc}")
        return buf.reshape(self.n_images, self.dof)

    def evaluation(self) -> tuple[np.ndarray, np.ndarray] | None:
        energies = np.empty(self.n_images, dtype=np.float64)
        grads = np.empty(self.n_images * self.dof, dtype=np.float64)
        rc = int(
            self.lib.rgsaddle_band_evaluation(
                self.handle,
                energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                grads.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                None,
            )
        )
        if rc == _NO_EVALUATION:
            return None
        if rc != _OK:
            raise RuntimeError(f"rgsaddle_band_evaluation returned {rc}")
        return energies, grads.reshape(self.n_images, self.dof)

    def close(self) -> None:
        if self.handle:
            self._free(self.handle)
            self.handle = None


def _cell(box: np.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    cell = np.zeros(9, dtype=np.float64)
    cell[0] = float(box[0])
    cell[4] = float(box[1])
    cell[8] = float(box[2])
    return cell


def _open_band(
    lib,
    positions: np.ndarray,
    surface: _Surface,
    *,
    spring_k: float,
    spring_ks: np.ndarray | None,
    climbing: bool,
    ci_trigger_factor: float,
    ci_trigger_force: float,
    force_tol: float,
    force_gate: int,
    max_move: float,
    method: str,
    memory: int,
    keep_translation: bool,
    box: np.ndarray | None,
) -> _Session:
    n_images, dof = positions.shape
    if dof % 3 != 0:
        raise ValueError("band coordinates are not a multiple of three")
    surface.n_band = n_images
    held = np.ascontiguousarray(positions, dtype=np.float64)
    cfg = BandConfig()
    cfg.major = _ABI_MAJOR
    cfg.minor = _ABI_MINOR
    cfg.flags = _KEEP_TRANSLATION if keep_translation else 0
    cfg.tangent = _TANGENT_IMPROVED
    cfg.projection = _PROJECTION_NEB
    cfg.method = _METHOD_LBFGS if method == "lbfgs" else _METHOD_FIRE
    cfg.spring_k = float(spring_k)
    ks = None
    if spring_ks is None:
        cfg.spring = _SPRING_UNIFORM
        cfg.spring_ks = None
    else:
        ks = np.ascontiguousarray(spring_ks, dtype=np.float64)
        if ks.shape != (n_images - 1,):
            raise ValueError("weighted springs need one constant per segment")
        cfg.spring = _SPRING_WEIGHTED
        cfg.spring_ks = ks.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cfg.ci_trigger_factor = float(ci_trigger_factor) if climbing else 0.0
    cfg.ci_trigger_force = float(ci_trigger_force) if climbing else 0.0
    cfg.force_tol = float(force_tol)
    cfg.max_move = float(max_move)
    cfg.memory = int(memory)
    cell = _cell(box)
    if cell is None:
        cfg.cell = None
    else:
        cfg.cell = cell.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    callback = _SURFACE(surface)
    handle = lib.rgsaddle_band_create(
        ctypes.byref(cfg),
        n_images,
        dof // 3,
        held.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not handle:
        raise RuntimeError("rgsaddle_band_create returned null")
    rc = int(lib.rgsaddle_band_set_force_gate(handle, force_gate))
    if rc != _OK:
        lib.rgsaddle_band_free(handle)
        raise RuntimeError(f"rgsaddle_band_set_force_gate returned {rc}")
    return _Session(lib, handle, callback, n_images, dof, lib.rgsaddle_band_free)


def _relax(
    lib,
    positions: np.ndarray,
    surface: _Surface,
    *,
    max_steps: int,
    force_tol: float,
    force_gate: int,
    spring_k: float,
    spring_ks: np.ndarray | None,
    max_move: float,
    method: str,
    memory: int,
    climbing: bool,
    ci_trigger_factor: float,
    ci_trigger_force: float,
    keep_translation: bool,
    box: np.ndarray | None,
    stop_on_candidate: bool,
    candidate_tol: float,
) -> tuple[np.ndarray, Report | None, bool, np.ndarray | None, np.ndarray | None]:
    """Step until the gate, the candidate force, the cap, or ``max_steps``.

    The third result is true when the stop was the candidate's own
    Euclidean force, before the whole-band gate. The last two results
    are the session's cached energies and raw gradients at the accepted
    band. Reading them does not call the potential again.
    """
    session = _open_band(
        lib,
        positions,
        surface,
        spring_k=spring_k,
        spring_ks=spring_ks,
        climbing=climbing,
        ci_trigger_factor=ci_trigger_factor,
        ci_trigger_force=ci_trigger_force,
        force_tol=force_tol,
        force_gate=force_gate,
        max_move=max_move,
        method=method,
        memory=memory,
        keep_translation=keep_translation,
        box=box,
    )
    report = None
    early = False
    energies = None
    grads = None
    try:
        for _ in range(max_steps):
            if surface.rgpot is not None and surface.calls() >= surface.cap:
                surface.hit_cap = True
                break
            rc, report = session.step()
            if rc != _OK:
                break
            evaluation = session.evaluation()
            if evaluation is not None:
                energies, grads = evaluation
            if int(report.status) == _STATUS_CONVERGED:
                break
            if (
                stop_on_candidate
                and grads is not None
                and int(report.ci_index) >= 0
                and float(np.linalg.norm(grads[int(report.ci_index)])) <= candidate_tol
            ):
                early = True
                break
        return session.positions(), report, early, energies, grads
    finally:
        session.close()


def _surrogate_path(
    lib,
    linear: np.ndarray,
    *,
    path: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    spring_k: float,
    force_tol: float,
    max_steps: int,
    stage_steps: int,
    max_move: float,
    keep_translation: bool,
) -> tuple[np.ndarray, int]:
    """IDPP or S-IDPP initial path. Returns the path and the surrogate steps."""
    n_images = linear.shape[0]
    slots = list(range(n_images))
    surface = _idpp_surface(reactant, product, free, box, slots, n_images)
    steps = 0
    if path == "idpp":
        relaxed, report, _early, _energies, _grads = _relax(
            lib,
            linear,
            surface,
            max_steps=max_steps,
            force_tol=force_tol,
            force_gate=_FORCE_LINF,
            spring_k=spring_k,
            spring_ks=None,
            max_move=max_move,
            method="fire",
            memory=0,
            climbing=False,
            ci_trigger_factor=0.0,
            ci_trigger_force=0.0,
            keep_translation=keep_translation,
            box=box,
            stop_on_candidate=False,
            candidate_tol=force_tol,
        )
        steps += int(report.iteration) if report is not None else 0
        return relaxed, steps
    if path != "sidpp":
        raise ValueError(f"unknown path {path}")
    if n_images <= 3:
        # One interior image has no second chain to grow.
        return _surrogate_path(
            lib,
            linear,
            path="idpp",
            reactant=reactant,
            product=product,
            free=free,
            box=box,
            spring_k=spring_k,
            force_tol=force_tol,
            max_steps=max_steps,
            stage_steps=stage_steps,
            max_move=max_move,
            keep_translation=keep_translation,
        )
    frames = {0: linear[0].copy(), n_images - 1: linear[-1].copy()}
    left = [1]
    right = [n_images - 2]
    frames[1] = linear[1].copy()
    frames[n_images - 2] = linear[-2].copy()
    d_id = _path_length([frames[0], frames[n_images - 1]]) / (n_images - 1)

    def grow(add_left: bool) -> None:
        nonlocal d_id
        order = _chain_order(left, right, n_images)
        d_id = _path_length([frames[index] for index in order]) / (n_images - 1)
        if add_left:
            frontier, other, slot = left[-1], right[-1], left[-1] + 1
            left.append(slot)
        else:
            frontier, other, slot = right[-1], left[-1], right[-1] - 1
            right.append(slot)
        gap = frames[other] - frames[frontier]
        dist = float(np.linalg.norm(gap))
        step = min(d_id, 0.5 * dist) if dist > 1e-12 else 0.0
        frames[slot] = frames[frontier] + (step / dist) * gap if dist > 1e-12 else frames[frontier].copy()

    def relax_chain() -> None:
        nonlocal steps
        order = _chain_order(left, right, n_images)
        current = np.vstack([frames[index] for index in order])
        chain = _idpp_surface(reactant, product, free, box, order, n_images)
        ks = _sidpp_springs(frames, left, right, n_images, spring_k, d_id)
        relaxed, report, _early, _energies, _grads = _relax(
            lib,
            current,
            chain,
            max_steps=stage_steps,
            force_tol=force_tol,
            force_gate=_FORCE_LINF,
            spring_k=spring_k,
            spring_ks=ks,
            max_move=max_move,
            method="fire",
            memory=0,
            climbing=False,
            ci_trigger_factor=0.0,
            ci_trigger_force=0.0,
            keep_translation=keep_translation,
            box=box,
            stop_on_candidate=False,
            candidate_tol=force_tol,
        )
        steps += int(report.iteration) if report is not None else 0
        for row, index in enumerate(order):
            frames[index] = relaxed[row]

    relax_chain()
    add_left = True
    while len(left) + len(right) < n_images - 2:
        grow(add_left)
        add_left = not add_left
        relax_chain()
    order = _chain_order(left, right, n_images)
    full = np.vstack([frames[index] for index in order])
    final, report, _early, _energies, _grads = _relax(
        lib,
        full,
        surface,
        max_steps=max_steps,
        force_tol=force_tol,
        force_gate=_FORCE_LINF,
        spring_k=spring_k,
        spring_ks=None,
        max_move=max_move,
        method="fire",
        memory=0,
        climbing=False,
        ci_trigger_factor=0.0,
        ci_trigger_force=0.0,
        keep_translation=keep_translation,
        box=box,
        stop_on_candidate=False,
        candidate_tol=force_tol,
    )
    steps += int(report.iteration) if report is not None else 0
    return final, steps


def _idpp_surface(reactant, product, free, box, slots: list[int], n_req: int) -> _Surface:
    surface = _Surface("idpp", None, cap=0)
    surface.slots = list(slots)
    surface.n_req = n_req
    surface.template = np.array(reactant, dtype=np.float64, copy=True)
    surface.free = np.asarray(free, dtype=bool)
    surface.box = None if box is None else np.asarray(box, dtype=np.float64)
    surface.d_reactant = pair_distances(reactant, box)
    surface.d_product = pair_distances(product, box)
    return surface


def _chain_order(left: list[int], right: list[int], n_images: int) -> list[int]:
    return [0, *left, *reversed(right), n_images - 1]


def _path_length(frames: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(frames[i + 1] - frames[i]) for i in range(len(frames) - 1)))


def _sidpp_springs(frames, left, right, n_images, spring_k: float, d_id: float) -> np.ndarray:
    """Base ``k`` on the short chains and ``k * d_id / gap`` across the centre.

    Schmerwitz, Ásgeirsson, and Jónsson, JCTC 20, 155 (2024), eq. 10,
    applied to whichever segment currently joins the two chains.
    """
    order = _chain_order(left, right, n_images)
    centre = {left[-1], right[-1]}
    constants = []
    for left_index, right_index in zip(order, order[1:], strict=False):
        if {left_index, right_index} == centre:
            gap = float(np.linalg.norm(frames[right_index] - frames[left_index]))
            constants.append(spring_k * d_id / max(gap, 1e-8))
        else:
            constants.append(spring_k)
    return np.asarray(constants, dtype=np.float64)


def _refine_minmode(
    lib,
    rgpot: RgpotCounter,
    surface: _Surface,
    point: np.ndarray,
    mode: np.ndarray,
    *,
    kind: str,
    method: str,
    force_tol: float,
    max_move: float,
    cap: int,
) -> tuple[np.ndarray, bool]:
    dof = point.shape[0]
    n_atoms = dof // 3
    cfg = MinModeConfig()
    cfg.major = _ABI_MAJOR
    cfg.minor = _ABI_MINOR
    cfg.flags = 0
    cfg.kind = _MINMODE_LANCZOS if kind == "lanczos" else _MINMODE_DIMER
    cfg.method = _METHOD_LBFGS if method == "lbfgs" else _METHOD_FIRE
    cfg.dr = 1e-3
    cfg.rotation_tol = 1e-4
    cfg.max_rotations = 10
    cfg.krylov_dim = max(1, min(12, dof - 1))
    cfg.force_tol = float(force_tol)
    cfg.max_move = float(max_move)
    cfg.rotation_angle_tol = 0.0
    cfg.difference = _DIFFERENCE_CENTRAL
    cfg.reserved = 0
    surface.n_band = 1
    surface.kind = "pes"
    surface.rgpot = rgpot
    held = np.ascontiguousarray(point, dtype=np.float64)
    direction = np.ascontiguousarray(mode, dtype=np.float64)
    callback = _SURFACE(surface)
    handle = lib.rgsaddle_minmode_create(
        ctypes.byref(cfg),
        n_atoms,
        held.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        direction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not handle:
        raise RuntimeError("rgsaddle_minmode_create returned null")
    try:
        rc = int(lib.rgsaddle_minmode_set_force_gate(handle, _FORCE_L2))
        if rc != _OK:
            raise RuntimeError(f"rgsaddle_minmode_set_force_gate returned {rc}")
        converged = False
        while rgpot.counter.calls < cap:
            report = Report()
            rc = int(lib.rgsaddle_minmode_step(handle, callback, None, ctypes.byref(report)))
            if rc != _OK:
                break
            if int(report.status) == _STATUS_CONVERGED:
                converged = True
                break
        out = np.empty(dof, dtype=np.float64)
        rc = int(lib.rgsaddle_minmode_position(handle, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if rc != _OK:
            raise RuntimeError(f"rgsaddle_minmode_position returned {rc}")
        return out, converged
    finally:
        lib.rgsaddle_minmode_free(handle)


def _mode_between(positions: np.ndarray, index: int) -> np.ndarray:
    tangent = positions[index + 1] - positions[index - 1]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-14:
        tangent = np.zeros(positions.shape[1], dtype=np.float64)
        tangent[0] = 1.0
        return tangent
    return tangent / norm


def _candidate(positions: np.ndarray, energies: np.ndarray, report: Report | None) -> tuple[int, str]:
    if report is not None and int(report.ci_index) >= 0:
        return int(report.ci_index), "climbing"
    interior = energies[1:-1]
    return 1 + int(np.argmax(interior)), "highest"


def _crash_record(name: str, calls: int, verify: int, crash: str) -> dict:
    """A search that raised. The class is crash and the calls are those already charged."""
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify),
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "crash": crash,
        "energy": float("nan"),
        "force_norm": float("nan"),
        "hessian_index": None,
        "converged": False,
        "candidate": "crash",
        "ci_index": None,
        "refined": False,
        "surrogate_steps": 0,
        "geodesic_seconds": 0.0,
    }


def run_band_search(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    evaluate_full,
    n_interior: int,
    force_norm: float,
    cap: int,
    max_move: float,
    climbing: bool,
    spring: str,
    path: str,
    refine: str,
    method: str,
    memory: int,
    spring_k: float,
    k_min: float,
    k_max: float,
    ci_trigger_factor: float,
    ci_trigger_force: float,
    idpp_spring_k: float,
    idpp_force_tol: float,
    idpp_max_steps: int,
    sidpp_stage_steps: int,
    project_rigid: bool,
    keep_translation: bool,
    potential: str = "",
) -> dict:
    """One reactant/product pair. The record uses the harness counter."""
    state: dict = {"counter": None}
    try:
        return _run_band_search(
            name=name,
            reactant=reactant,
            product=product,
            free=free,
            box=box,
            evaluate_full=evaluate_full,
            n_interior=n_interior,
            force_norm=force_norm,
            cap=cap,
            max_move=max_move,
            climbing=climbing,
            spring=spring,
            path=path,
            refine=refine,
            method=method,
            memory=memory,
            spring_k=spring_k,
            k_min=k_min,
            k_max=k_max,
            ci_trigger_factor=ci_trigger_factor,
            ci_trigger_force=ci_trigger_force,
            idpp_spring_k=idpp_spring_k,
            idpp_force_tol=idpp_force_tol,
            idpp_max_steps=idpp_max_steps,
            sidpp_stage_steps=sidpp_stage_steps,
            project_rigid=project_rigid,
            keep_translation=keep_translation,
            potential=potential,
            state=state,
        )
    except Exception as exc:
        counter = state["counter"]
        calls = 0 if counter is None else int(counter.calls)
        verify = 0 if counter is None else int(counter.verify_calls)
        return _crash_record(name, calls, verify, type(exc).__name__)


def _run_band_search(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    evaluate_full,
    n_interior: int,
    force_norm: float,
    cap: int,
    max_move: float,
    climbing: bool,
    spring: str,
    path: str,
    refine: str,
    method: str,
    memory: int,
    spring_k: float,
    k_min: float,
    k_max: float,
    ci_trigger_factor: float,
    ci_trigger_force: float,
    idpp_spring_k: float,
    idpp_force_tol: float,
    idpp_max_steps: int,
    sidpp_stage_steps: int,
    project_rigid: bool,
    keep_translation: bool,
    potential: str,
    state: dict,
) -> dict:
    lib = _load_library()
    linear = interpolate_free(reactant, product, free, n_interior, box)
    surrogate_steps = 0
    geodesic_seconds = 0.0
    if path == "linear":
        band_path = linear
    elif path == "geodesic":
        from .geodesic_path import geodesic_free, symbols_for

        info: dict = {}
        band_path = geodesic_free(
            reactant,
            product,
            free,
            n_interior,
            box,
            symbols_for(potential, reactant.shape[0]),
            info=info,
        )
        geodesic_seconds = float(info.get("seconds") or 0.0)
    else:
        band_path, surrogate_steps = _surrogate_path(
            lib,
            linear,
            path=path,
            reactant=reactant,
            product=product,
            free=free,
            box=box,
            spring_k=idpp_spring_k,
            force_tol=idpp_force_tol,
            max_steps=idpp_max_steps,
            stage_steps=sidpp_stage_steps,
            max_move=max_move,
            keep_translation=keep_translation,
        )
    template = np.array(reactant, dtype=np.float64, copy=True)
    mask = np.asarray(free, dtype=bool)

    def evaluate_free(flat):
        full = np.array(template, copy=True)
        full[mask] = np.asarray(flat, dtype=np.float64).reshape((-1, 3))
        energy, grad = evaluate_full(full)
        return float(energy), np.asarray(grad, dtype=np.float64)[mask].reshape(-1)

    counter = HarnessCounter(cache=True)
    state["counter"] = counter
    box_flat = None if box is None else tuple(float(value) for value in np.asarray(box, dtype=np.float64).reshape(-1))

    def evaluate_rgpot(coords, _atomic_numbers, _box):
        energy, grad = evaluate_free(coords)
        gradient = tuple(float(item) for item in np.asarray(grad, dtype=np.float64).reshape(-1))
        return float(energy), gradient

    rgpot = RgpotCounter(evaluate_rgpot, counter)
    spring_ks = None
    if spring == "weighted":
        energies = [rgpot.calculate(row, None, box_flat)[0] for row in band_path]
        spring_ks = energy_weighted_springs(np.asarray(energies, dtype=np.float64), k_min, k_max)
    elif spring != "uniform":
        raise ValueError(f"unknown spring {spring}")
    surface = _Surface("pes", rgpot, cap)
    surface.box_flat = box_flat
    positions, report, early, band_energies, band_grads = _relax(
        lib,
        band_path,
        surface,
        max_steps=cap,
        force_tol=force_norm,
        force_gate=_FORCE_L2,
        spring_k=spring_k,
        spring_ks=spring_ks,
        max_move=max_move,
        method=method,
        memory=memory,
        climbing=climbing,
        ci_trigger_factor=ci_trigger_factor,
        ci_trigger_force=ci_trigger_force,
        keep_translation=keep_translation,
        box=box,
        stop_on_candidate=climbing,
        candidate_tol=force_norm,
    )
    if band_energies is None or band_grads is None:
        # No accepted band evaluation. A later potential call would be a
        # new search cost, so a start that already spent the cap stops here.
        index, candidate = 1, "highest"
        point = positions[index]
        if counter.calls >= cap or surface.hit_cap:
            energy = float("nan")
            grad = np.full(point.shape, np.nan)
        else:
            energy, grad = rgpot.calculate(point, None, box_flat)
            grad = np.asarray(grad, dtype=np.float64)
    else:
        index, candidate = _candidate(positions, band_energies, report)
        point = positions[index]
        energy = float(band_energies[index])
        grad = np.asarray(band_grads[index], dtype=np.float64)
    refined = False
    minmode_converged = False
    if refine and counter.calls < cap:
        mode = _mode_between(positions, index)
        point, minmode_converged = _refine_minmode(
            lib,
            rgpot,
            surface,
            point,
            mode,
            kind=refine,
            method=method,
            force_tol=force_norm,
            max_move=max_move,
            cap=cap,
        )
        if surface.last_point is not None and np.array_equal(point, surface.last_point):
            energy = surface.last_energy
            grad = surface.last_grad if surface.last_grad is not None else grad
        elif counter.calls >= cap or surface.hit_cap:
            energy = float("nan")
            grad = np.full(point.shape, np.nan)
        else:
            energy, grad = rgpot.calculate(point, None, box_flat)
            grad = np.asarray(grad, dtype=np.float64)
        candidate = "min-mode"
        refined = True
    force = float(np.linalg.norm(grad))
    reasons: list[str] = []
    if surface.crashed:
        reasons.append("crash")
    if counter.calls >= cap or surface.hit_cap:
        reasons.append("cap")
    if not np.isfinite(force) or force > force_norm:
        reasons.append("force_norm")
    hessian_index = None
    if "crash" not in reasons and "cap" not in reasons and "force_norm" not in reasons:
        with counter.verification():
            hessian = numerical_hessian(
                lambda coords: rgpot.calculate(coords, None, box_flat)[1],
                np.asarray(point, dtype=np.float64).tolist(),
                step=_FD_STEP,
            )
        hessian_index = vibrational_index(
            np.asarray(hessian, dtype=np.float64),
            np.asarray(point, dtype=np.float64).reshape((-1, 3)),
            project_rigid,
        )
        if hessian_index != 1:
            reasons.append("hessian_index")
    converged = (report is not None and int(report.status) == _STATUS_CONVERGED) or early or minmode_converged
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(counter.calls),
        "verify_calls": int(counter.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": float(energy),
        "force_norm": force,
        "hessian_index": hessian_index,
        "converged": bool(converged),
        "candidate": candidate,
        "ci_index": None if report is None else int(report.ci_index),
        "refined": refined,
        "surrogate_steps": int(surrogate_steps),
        "geodesic_seconds": float(geodesic_seconds),
    }


def _registry(root: Path, code_id: str = "rgsaddle") -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == code_id:
            return row
    raise ValueError(f"codes/registry.toml has no {code_id} entry")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        code = template.get("code") or "rgsaddle"
        raise ValueError(f"{template['id']} commit does not match the {code} registry")
    band = template["band"]
    problem_band = problem["band"]
    convergence = problem["convergence"]
    return {
        "potential": problem_band["potential"],
        "n_interior": int(problem_band["interior_images"]),
        "max_move": float(problem_band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": problem["geometry"]["kind"],
        "climbing": bool(band["climbing"]),
        "spring": str(band["spring"]),
        "path": str(band["path"]),
        "refine": str(band["refine"]),
        "method": str(band["method"]),
        "memory": int(band["memory"]),
        "spring_k": float(band["spring_k"]),
        "k_min": float(band["k_min"]),
        "k_max": float(band["k_max"]),
        "ci_trigger_factor": float(band["ci_trigger_factor"]),
        "ci_trigger_force": float(band["ci_trigger_force"]),
        "idpp_spring_k": float(band["idpp_spring_k"]),
        "idpp_force_tol": float(band["idpp_force_tol"]),
        "idpp_max_steps": int(band["idpp_max_steps"]),
        "sidpp_stage_steps": int(band["sidpp_stage_steps"]),
    }


def _job(pair, settings: dict) -> dict:
    return {
        "name": pair.name,
        "reactant": pair.reactant,
        "product": pair.product,
        "free": pair.free,
        "box": pair.box,
        "settings": settings,
    }


def _execute(job: dict) -> dict:
    import time

    began = time.perf_counter()
    record = _execute_search(job)
    record["wall_seconds"] = time.perf_counter() - began
    return record


def _execute_search(job: dict) -> dict:
    _threads_one()
    settings = job["settings"]
    free = np.asarray(job["free"], dtype=bool)
    n_free = int(np.count_nonzero(free))
    kind = settings["potential"]
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)

    def evaluate_full(coords):
        return evaluate(kind, coords, box)

    return run_band_search(
        name=job["name"],
        reactant=np.asarray(job["reactant"], dtype=np.float64),
        product=np.asarray(job["product"], dtype=np.float64),
        free=free,
        box=box,
        evaluate_full=evaluate_full,
        n_interior=settings["n_interior"],
        force_norm=settings["force_norm"],
        cap=settings["cap"],
        max_move=settings["max_move"],
        climbing=settings["climbing"],
        spring=settings["spring"],
        path=settings["path"],
        refine=settings["refine"],
        method=settings["method"],
        memory=settings["memory"],
        spring_k=settings["spring_k"],
        k_min=settings["k_min"],
        k_max=settings["k_max"],
        ci_trigger_factor=settings["ci_trigger_factor"],
        ci_trigger_force=settings["ci_trigger_force"],
        idpp_spring_k=settings["idpp_spring_k"],
        idpp_force_tol=settings["idpp_force_tol"],
        idpp_max_steps=settings["idpp_max_steps"],
        sidpp_stage_steps=settings["sidpp_stage_steps"],
        project_rigid=bool(np.all(free) and n_free >= 2 and kind == "lj"),
        keep_translation=bool(n_free < 2 or not np.all(free)),
        potential=kind,
    )


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness RgpotCounter.calculate",
        "# force_calls count raw potential evaluations and are not divided by the image count",
        "# the mean is round-half-even over successes; cap, force, or Hessian index other than 1 is a failure",
    ]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        images = int(settings["n_interior"])
        if images > 0:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
            lines.append("call_unit total")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row['hessian_index'] == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append(f"images {settings['n_interior']}")
    note = str(template.get("note") or "").strip()
    if note:
        lines.append(f"note {note}")
    return "\n".join(lines) + "\n"


def _success_mean(records: list[dict]) -> dict:
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary = {
        "successes": len(successes),
        "nfailed": len(records) - len(successes),
        "n": len(records),
        "transition_states": sum(1 for row in records if row["hessian_index"] == 1),
    }
    if successes:
        summary["force_calls"] = int(round(statistics.fmean(successes)))
        summary["force_calls_min"] = min(successes)
        summary["force_calls_max"] = max(successes)
        summary["force_calls_median"] = statistics.median(successes)
    return summary


_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    """Pair objects for one problem, loaded once in this process."""
    key = (str(Path(root).resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        loaded = load_pairs(
            geometry_archive(root, problem),
            settings["geometry"],
            settings["pairs"],
        )
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def search_named(
    root: Path,
    problem_name: str,
    template_name: str,
    start: str,
    seed: str = "0",
) -> dict:
    """One archived pair.

    ``seed`` is recorded on the start. The band does not draw random
    numbers, so the value does not change the search.
    """
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root, str(template.get("code") or "rgsaddle"))["commit"]
    settings = _settings(template, problem, commit)
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    _threads_one()
    record = _execute(_job(pairs[start], settings))
    record["seed"] = str(seed)
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def write_outputs(
    output: Path,
    records: list[dict],
    template: dict,
    settings: dict,
    problem: dict,
) -> dict:
    """Write ``benchmark.dat``, the per-start file, and the v2 score."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(records, template, settings), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    parameters = dict(settings)
    parameters["minmode_lbfgs_memory"] = _MINMODE_LBFGS_MEMORY
    parameters["counter"] = "harness"
    parameters["seed"] = "0"
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score_v2(
            [
                {
                    "force_calls": row["force_calls"],
                    "success": row["success"],
                    "failure": row.get("failure"),
                    "reasons": row.get("reasons") or [],
                }
                for row in records
            ],
            settings["cap"],
        ),
        "success_mean": _success_mean(records),
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success_only = document["success_mean"]
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": success_only.get("force_calls"),
        "nfailed": success_only["nfailed"],
        "starts": success_only["n"],
        "successes": success_only["successes"],
        "output": str(output),
    }


def publish_sidecars(root: Path, directory: Path, output_dir: Path) -> list[dict]:
    """Reduce a complete asv sidecar tree into one dat per problem and template.

    A published mean is the full start set. A missing pair is an error.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    paths = sorted(Path(directory).rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no asv sidecar records in {directory}")
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault((row["problem"], row["template"]), []).append(row)
    summaries = []
    for problem_name, template_name in sorted(grouped):
        records = grouped[(problem_name, template_name)]
        problem = resolve_problem(root, problem_name)
        template = load_template(root, template_name)
        commit = _registry(root, str(template.get("code") or "rgsaddle"))["commit"]
        settings = _settings(template, problem, commit)
        order = list(cached_pairs(root, problem, settings))
        found = {row["start"] for row in records}
        if found != set(order) or len(records) != len(order):
            raise ValueError(
                f"{template_name} on {problem_name} has {len(records)} records, the problem has {len(order)} pairs"
            )
        rank = {name: index for index, name in enumerate(order)}
        records.sort(key=lambda row: rank[row["start"]])
        slug = f"{problem['id'].replace('/', '-')}-{template['id'].replace('/', '-')}"
        summaries.append(write_outputs(output_dir / f"{slug}.dat", records, template, settings, problem))
    return summaries


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
) -> dict:
    """Run every pair in one problem and write the dat, starts, and v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] == "rgsaddle-band":
        pass
    elif template["engine"] == "geodesic" and template.get("backend") == "rgsaddle":
        pass
    else:
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root, str(template.get("code") or "rgsaddle"))["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_job(pair, settings) for pair in pairs]
    _threads_one()
    if workers <= 1:
        records = [_execute(job) for job in jobs]
    else:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(_execute, jobs))
    order = {job["name"]: index for index, job in enumerate(jobs)}
    records.sort(key=lambda row: order[row["start"]])
    for record in records:
        record["seed"] = "0"
        record["problem"] = problem["id"]
        record["template"] = template["id"]
    return write_outputs(output, records, template, settings, problem)
