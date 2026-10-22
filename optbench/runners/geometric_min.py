"""Minimize OptBench starts with geomeTRIC's custom engine.

The geometry file is the problem distance. It is stored as the molecule
frame geomeTRIC calls Angstrom. ``calc_new`` receives Bohr, converts back,
and returns Hartree and Hartree/Bohr. Lennard-Jones epsilon is 1, so that
energy number is already the Hartree value. Morse energies are eV and use
the library ``ev2au`` factor. The gradient is multiplied by ``bohr2ang``,
and by ``ev2au`` for Morse.

The search stops when the problem force norm is at the problem stop.
geomeTRIC's GAU thresholds are set far below that so they do not stop the
run, and ``tmin`` stays at 1e-4, the floor the library uses when the
displacement threshold is not tighter than a milliangstrom. Trust radius,
maximum trust, and the BFGS update stay at the minimization defaults. The
Hessian is never built by finite difference.

Cartesian passes an empty bond list. TRIC is
``DelocalizedInternalCoordinates(connect=False, addcart=False)`` on the
two nearest neighbors. The covalent-radius graph is not used: the builder's
1 Angstrom minimum distance bonds a compressed Lennard-Jones start into a
primitive set that does not finish, and a Morse first shell asks for a
Wilson matrix of about a tebibyte. Lennard-Jones atoms are labeled H so
Schlegel's guess stays positive near sigma. Morse atoms are Pt. The distance
coordinates are the library's Cartesian differences, so the neighbor
list uses those distances too. A minimum-image neighbor can be a long
Cartesian vector, and that coordinate does not describe the periodic
contact.

Primitive membership uses a hashed bucket with the same equality test.
Distance, angle, and dihedral values and first derivatives use the same
formulas in batches. The other primitive types stay on the library methods.

A coordinate rebuild calls ``Molecule.build_topology`` with bond discovery
left on. That replaces the stored neighbor list with covalent radii. The
runner turns discovery off when the bonds were read, so the rebuild keeps
the two nearest neighbors. Two failed rebuilds still continue in Cartesian
coordinates.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potserver import RgpotCounter
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_BATCH_TYPES = frozenset({"Distance", "Angle", "Dihedral"})
_TMIN = 1e-4
_NEIGHBORS = 2


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _init_worker() -> None:
    _quiet_threads()


def install_fast_add() -> None:
    """Hash primitive membership. Equality is unchanged."""
    from geometric.internal import PrimitiveInternalCoordinates

    if getattr(PrimitiveInternalCoordinates, "_optbench_fast_add", False):
        return
    original_delete = PrimitiveInternalCoordinates.delete

    def bucket(dof):
        atoms = []
        for name in ("a", "b", "c", "d"):
            if not hasattr(dof, name):
                continue
            value = getattr(dof, name)
            if isinstance(value, (list, tuple)):
                atoms.extend(int(item) for item in value)
            else:
                try:
                    atoms.append(int(value))
                except (TypeError, ValueError):
                    atoms.append(repr(value))
        return (type(dof), frozenset(atoms))

    def add(self, dof):
        groups = getattr(self, "_buckets", None)
        if groups is None:
            groups = {}
            self._buckets = groups
        group = groups.setdefault(bucket(dof), [])
        if dof not in group:
            group.append(dof)
            self.Internals.append(dof)

    def delete(self, dof):
        original_delete(self, dof)
        groups = getattr(self, "_buckets", None)
        if not groups:
            return
        key = bucket(dof)
        group = groups.get(key)
        if group:
            groups[key] = [item for item in group if item != dof]

    PrimitiveInternalCoordinates.add = add
    PrimitiveInternalCoordinates.delete = delete
    PrimitiveInternalCoordinates._optbench_fast_add = True


def install_keep_read_bonds() -> None:
    """Keep a bond list that was read when a coordinate rebuild asks for bonds.

    ``Optimizer.checkCoordinateSystem`` copies the molecule and calls
    ``build_topology`` with discovery left on. Discovery replaces
    ``Data['bonds']`` even when ``read_bonds`` is set. The TRIC runs store
    the two nearest Cartesian neighbors in that list.
    """
    from geometric.molecule import Molecule

    current = Molecule.build_topology
    if getattr(current, "_optbench_keep_read_bonds", False):
        return

    def build_topology(self, force_bonds=True, bond_order=0.0, metal_bo_factor=0.5, **kwargs):
        if self.top_settings.get("read_bonds"):
            force_bonds = False
        return current(
            self,
            force_bonds=force_bonds,
            bond_order=bond_order,
            metal_bo_factor=metal_bo_factor,
            **kwargs,
        )

    build_topology._optbench_keep_read_bonds = True
    Molecule.build_topology = build_topology


def _cross(left, right):
    import numpy as np

    return np.cross(left, right)


def _distance_batch(xyz, pairs):
    import numpy as np

    left = xyz[pairs[:, 0]]
    right = xyz[pairs[:, 1]]
    difference = left - right
    length = np.linalg.norm(difference, axis=1)
    unit = difference / length[:, None]
    deriv = np.zeros((len(pairs), xyz.shape[0], 3))
    rows = np.arange(len(pairs))
    deriv[rows, pairs[:, 0]] = unit
    deriv[rows, pairs[:, 1]] = -unit
    return length, deriv


def _angle_batch(xyz, triples):
    import numpy as np

    first = xyz[triples[:, 0]]
    center = xyz[triples[:, 1]]
    last = xyz[triples[:, 2]]
    u_prime = first - center
    v_prime = last - center
    u_norm = np.linalg.norm(u_prime, axis=1)
    v_norm = np.linalg.norm(v_prime, axis=1)
    u = u_prime / u_norm[:, None]
    v = v_prime / v_norm[:, None]
    cosine = np.einsum("ij,ij->i", u, v)
    high = cosine >= 1.0
    if np.any(np.abs(cosine[high]) - 1.0 > 1e-6):
        raise RuntimeError("Encountered invalid value in angle")
    values = np.arccos(np.clip(cosine, -1.0, 1.0))
    values[cosine <= -1.0] = np.pi
    values[high] = 0.0
    w_prime = _cross(u, v)
    w_norm = np.linalg.norm(w_prime, axis=1)
    parallel = (np.linalg.norm(u + v, axis=1) < 1e-10) | (np.linalg.norm(u - v, axis=1) < 1e-10)
    w = np.zeros_like(u)
    usable = ~parallel & (w_norm > 0.0)
    w[usable] = w_prime[usable] / w_norm[usable, None]
    if np.any(parallel):
        vector1 = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)
        vector2 = np.array([-1.0, 1.0, 1.0]) / np.sqrt(3.0)
        picked = u[parallel]
        alternate = (np.linalg.norm(picked + vector1, axis=1) < 1e-10) | (
            np.linalg.norm(picked - vector2, axis=1) < 1e-10
        )
        special = _cross(picked, np.where(alternate[:, None], vector2, vector1))
        w[parallel] = special / np.linalg.norm(special, axis=1)[:, None]
    term1 = _cross(u, w) / u_norm[:, None]
    term2 = _cross(w, v) / v_norm[:, None]
    deriv = np.zeros((len(triples), xyz.shape[0], 3))
    rows = np.arange(len(triples))
    deriv[rows, triples[:, 0]] = term1
    deriv[rows, triples[:, 2]] = term2
    deriv[rows, triples[:, 1]] = -(term1 + term2)
    return values, deriv


def _wrap_periodic(diff):
    import numpy as np

    plus = diff + 2.0 * np.pi
    minus = diff - 2.0 * np.pi
    diff = np.where(np.abs(diff) > np.abs(plus), plus, diff)
    return np.where(np.abs(diff) > np.abs(minus), minus, diff)


def _dihedral_batch(xyz, quads):
    import numpy as np

    atom_a = xyz[quads[:, 0]]
    atom_b = xyz[quads[:, 1]]
    atom_c = xyz[quads[:, 2]]
    atom_d = xyz[quads[:, 3]]
    vec1 = atom_b - atom_a
    vec2 = atom_c - atom_b
    vec3 = atom_d - atom_c
    cross1 = _cross(vec2, vec3)
    cross2 = _cross(vec1, vec2)
    arg1 = np.einsum("ij,ij->i", vec1, cross1) * np.linalg.norm(vec2, axis=1)
    arg2 = np.einsum("ij,ij->i", cross1, cross2)
    values = np.arctan2(arg1, arg2)
    u_prime = atom_a - atom_b
    w_prime = atom_c - atom_b
    v_prime = atom_d - atom_c
    u_norm = np.linalg.norm(u_prime, axis=1)
    w_norm = np.linalg.norm(w_prime, axis=1)
    v_norm = np.linalg.norm(v_prime, axis=1)
    u = u_prime / u_norm[:, None]
    w = w_prime / w_norm[:, None]
    v = v_prime / v_norm[:, None]
    uw = np.einsum("ij,ij->i", u, w)
    vw = np.einsum("ij,ij->i", v, w)
    su2 = 1.0 - uw**2
    sv2 = 1.0 - vw**2
    ok_u = su2 >= 1e-6
    ok_v = sv2 >= 1e-6
    uxw = _cross(u, w)
    vxw = _cross(v, w)
    term1 = np.zeros_like(u)
    term2 = np.zeros_like(u)
    term3 = np.zeros_like(u)
    term4 = np.zeros_like(u)
    term1[ok_u] = uxw[ok_u] / (u_norm[ok_u, None] * su2[ok_u, None])
    term3[ok_u] = uxw[ok_u] * uw[ok_u, None] / (w_norm[ok_u, None] * su2[ok_u, None])
    term2[ok_v] = vxw[ok_v] / (v_norm[ok_v, None] * sv2[ok_v, None])
    term4[ok_v] = vxw[ok_v] * vw[ok_v, None] / (w_norm[ok_v, None] * sv2[ok_v, None])
    deriv = np.zeros((len(quads), xyz.shape[0], 3))
    rows = np.arange(len(quads))
    deriv[rows, quads[:, 0]] = term1
    deriv[rows, quads[:, 3]] = -term2
    deriv[rows, quads[:, 1]] = -term1 + term3 - term4
    deriv[rows, quads[:, 2]] = term2 - term3 + term4
    return values, deriv


def _groups(ic) -> dict[str, tuple]:
    import numpy as np

    found = {name: [] for name in _BATCH_TYPES}
    for index, internal in enumerate(ic.Internals):
        name = type(internal).__name__
        if name in found:
            found[name].append(index)
    packed = {}
    if found["Distance"]:
        packed["Distance"] = (
            np.asarray(found["Distance"], dtype=np.int64),
            np.asarray([(ic.Internals[i].a, ic.Internals[i].b) for i in found["Distance"]], dtype=np.int64),
        )
    if found["Angle"]:
        packed["Angle"] = (
            np.asarray(found["Angle"], dtype=np.int64),
            np.asarray(
                [(ic.Internals[i].a, ic.Internals[i].b, ic.Internals[i].c) for i in found["Angle"]],
                dtype=np.int64,
            ),
        )
    if found["Dihedral"]:
        packed["Dihedral"] = (
            np.asarray(found["Dihedral"], dtype=np.int64),
            np.asarray(
                [
                    (ic.Internals[i].a, ic.Internals[i].b, ic.Internals[i].c, ic.Internals[i].d)
                    for i in found["Dihedral"]
                ],
                dtype=np.int64,
            ),
        )
    return packed


def _batchable(ic) -> bool:
    return any(type(internal).__name__ in _BATCH_TYPES for internal in ic.Internals)


def install_batch_primitives() -> None:
    """Evaluate distance, angle, and dihedral blocks with the library formulas."""
    from geometric.internal import PrimitiveInternalCoordinates

    if getattr(PrimitiveInternalCoordinates, "_optbench_batch", False):
        return
    original_calculate = PrimitiveInternalCoordinates.calculate
    original_derivatives = PrimitiveInternalCoordinates.derivatives
    original_calc_diff = PrimitiveInternalCoordinates.calcDiff

    def calculate(self, xyz):
        if not _batchable(self):
            return original_calculate(self, xyz)
        import numpy as np

        flat = np.asarray(xyz, dtype=np.float64)
        shaped = flat.reshape(-1, 3)
        grouped = _groups(self)
        pre = {}
        if "Distance" in grouped:
            slots, pairs = grouped["Distance"]
            values, _deriv = _distance_batch(shaped, pairs)
            for slot, value in zip(slots, values):
                pre[int(slot)] = float(value)
        if "Angle" in grouped:
            slots, triples = grouped["Angle"]
            values, _deriv = _angle_batch(shaped, triples)
            for slot, value in zip(slots, values):
                pre[int(slot)] = float(value)
        if "Dihedral" in grouped:
            slots, quads = grouped["Dihedral"]
            values, _deriv = _dihedral_batch(shaped, quads)
            for slot, value in zip(slots, values):
                pre[int(slot)] = float(value)
        out = np.empty(len(self.Internals))
        for index, internal in enumerate(self.Internals):
            if index in pre:
                out[index] = pre[index]
            else:
                out[index] = internal.value(flat)
        return out

    def derivatives(self, xyz):
        if not _batchable(self):
            return original_derivatives(self, xyz)
        import numpy as np

        flat = np.asarray(xyz, dtype=np.float64)
        shaped = flat.reshape(-1, 3)
        self.calculate(flat)
        grouped = _groups(self)
        pre = {}
        if "Distance" in grouped:
            slots, pairs = grouped["Distance"]
            _values, deriv = _distance_batch(shaped, pairs)
            for row, slot in enumerate(slots):
                pre[int(slot)] = deriv[row]
        if "Angle" in grouped:
            slots, triples = grouped["Angle"]
            _values, deriv = _angle_batch(shaped, triples)
            for row, slot in enumerate(slots):
                pre[int(slot)] = deriv[row]
        if "Dihedral" in grouped:
            slots, quads = grouped["Dihedral"]
            _values, deriv = _dihedral_batch(shaped, quads)
            for row, slot in enumerate(slots):
                pre[int(slot)] = deriv[row]
        out = np.empty((len(self.Internals), shaped.shape[0], 3))
        for index, internal in enumerate(self.Internals):
            if index in pre:
                out[index] = pre[index]
            else:
                out[index] = internal.derivative(flat)
        return out

    def calc_diff(self, xyz1, xyz2):
        if not _batchable(self):
            return original_calc_diff(self, xyz1, xyz2)
        import numpy as np

        left = np.asarray(xyz1, dtype=np.float64).reshape(-1, 3)
        right = np.asarray(xyz2, dtype=np.float64).reshape(-1, 3)
        grouped = _groups(self)
        pre = {}
        if "Distance" in grouped:
            slots, pairs = grouped["Distance"]
            first, _deriv = _distance_batch(left, pairs)
            second, _deriv = _distance_batch(right, pairs)
            for slot, value in zip(slots, first - second):
                pre[int(slot)] = float(value)
        if "Angle" in grouped:
            slots, triples = grouped["Angle"]
            first, _deriv = _angle_batch(left, triples)
            second, _deriv = _angle_batch(right, triples)
            for slot, value in zip(slots, first - second):
                pre[int(slot)] = float(value)
        if "Dihedral" in grouped:
            slots, quads = grouped["Dihedral"]
            first, _deriv = _dihedral_batch(left, quads)
            second, _deriv = _dihedral_batch(right, quads)
            for slot, value in zip(slots, _wrap_periodic(first - second)):
                pre[int(slot)] = float(value)
        out = np.empty(len(self.Internals))
        for index, internal in enumerate(self.Internals):
            if index in pre:
                out[index] = pre[index]
            else:
                out[index] = internal.calcDiff(xyz1, xyz2)
        return out

    PrimitiveInternalCoordinates.calculate = calculate
    PrimitiveInternalCoordinates.derivatives = derivatives
    PrimitiveInternalCoordinates.calcDiff = calc_diff
    PrimitiveInternalCoordinates._optbench_batch = True


def knearest(coords, k: int, box) -> list[tuple[int, int]]:
    import numpy as np

    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    count = len(pos)
    if k <= 0 or count <= 1:
        return []
    width = min(int(k), count - 1)
    lengths = None if box is None else np.asarray(box, dtype=np.float64).reshape(-1)
    bonds = set()
    for index in range(count):
        diff = pos - pos[index]
        if lengths is not None and lengths.size >= 3:
            span = lengths[:3]
            diff -= span * np.round(diff / span)
        dist = np.einsum("ij,ij->i", diff, diff)
        dist[index] = np.inf
        for other in np.argpartition(dist, width)[:width]:
            left, right = (index, int(other)) if index < int(other) else (int(other), index)
            bonds.add((left, right))
    return sorted(bonds)


def _silence() -> None:
    logging.getLogger("geometric").setLevel(logging.ERROR)
    logging.getLogger("geometric.nifty").setLevel(logging.ERROR)


class _HarnessEngine:
    """Built inside ``optimize`` so geometric is imported in the worker."""


def _make_molecule(coords, element: str, bonds: list[tuple[int, int]]):
    import numpy as np
    from geometric.molecule import Molecule

    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    molecule = Molecule()
    molecule.elem = [element] * len(pos)
    molecule.xyzs = [pos.copy()]
    molecule.Data["bonds"] = list(bonds)
    molecule.top_settings["read_bonds"] = True
    molecule.build_topology(force_bonds=False)
    return molecule


def _coordinates(molecule, coordsys: str):
    from geometric.optimize import CartesianCoordinates, DelocalizedInternalCoordinates

    if coordsys == "cart":
        return CartesianCoordinates(molecule)
    if coordsys == "tric":
        return DelocalizedInternalCoordinates(molecule, build=True, connect=False, addcart=False)
    raise ValueError(f"unknown geomeTRIC coordinate system {coordsys}")


def _report_step(job, steps, engine, opt) -> None:
    if os.environ.get("OPTBENCH_GEOMETRIC_PROGRESS") != "1":
        return
    import numpy as np

    gradient = engine.problem_gradient
    if gradient is None:
        return
    norm = float(np.linalg.norm(gradient))
    print(
        f"{job.get('start')} step={steps} norm={norm:.6g} trust={float(opt.trust):.4g} ic={type(opt.IC).__name__}",
        flush=True,
    )


def optimize(job: dict, evaluate, stop) -> dict:
    """Run one geomeTRIC minimization.

    ``evaluate`` returns problem-unit energy and gradient. ``stop`` reads
    the latest problem-unit gradient and reports whether the search is done.
    """
    import numpy as np
    from geometric.engine import Engine
    from geometric.nifty import ang2bohr, bohr2ang, ev2au
    from geometric.optimize import OPT_STATE, Optimizer
    from geometric.params import OptParams

    _silence()
    install_fast_add()
    install_batch_primitives()
    install_keep_read_bonds()
    kind = str(job["kind"])
    coordsys = str(job["coordsys"])
    element = str(job["element"])
    neighbors = int(job.get("neighbors") or 0)
    cap = int(job["cap"])
    box = job.get("box")
    bonds = []
    if coordsys == "tric":
        bonds = knearest(job["coords"], neighbors, None)
    molecule = _make_molecule(job["coords"], element, bonds)
    internals = _coordinates(molecule, coordsys)
    energy_scale = float(ev2au) if kind == "morse" else 1.0

    class HarnessEngine(Engine):
        def __init__(self):
            super().__init__(molecule)
            self.problem_gradient = None

        def calc(self, coords, dirname, read_data=False, copydir=None):
            result = super().calc(coords, dirname, read_data=read_data, copydir=copydir)
            grad = np.asarray(result["gradient"], dtype=np.float64).reshape(-1)
            self.problem_gradient = grad / (bohr2ang * energy_scale)
            return result

        def calc_new(self, coords, dirname):
            flat = np.asarray(coords, dtype=np.float64).reshape(-1) * bohr2ang
            energy, gradient = evaluate(flat)
            grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
            return {
                "energy": float(energy) * energy_scale,
                "gradient": grad * bohr2ang * energy_scale,
            }

    engine = HarnessEngine()
    params = OptParams(
        maxiter=cap,
        convergence_energy=1e-30,
        convergence_grms=1e-30,
        convergence_gmax=1e-30,
        convergence_drms=1e-30,
        convergence_dmax=1e-30,
        tmin=_TMIN,
        xyzout=None,
        qdata=False,
        hessian="never",
    )
    dirname = tempfile.mkdtemp(prefix="geometric-optbench-")
    opt = None
    stopped = False
    error = ""
    steps = 0
    rebuilds = 0
    try:
        opt = Optimizer(
            molecule.xyzs[0].flatten() * ang2bohr,
            molecule,
            internals,
            engine,
            dirname,
            params,
            print_info=False,
        )
        opt.calcEnergyForce()
        opt.prepareFirstStep()
        if engine.problem_gradient is not None and stop(engine.problem_gradient):
            stopped = True
        _report_step(job, steps, engine, opt)
        while not stopped and opt.state not in (OPT_STATE.CONVERGED, OPT_STATE.FAILED):
            opt.step()
            if opt.state == OPT_STATE.SKIP_EVALUATION:
                rebuilds += 1
                if rebuilds > 100:
                    error = "coordinate rebuild did not settle"
                    break
                continue
            if opt.state != OPT_STATE.NEEDS_EVALUATION:
                break
            opt.calcEnergyForce()
            steps += 1
            if steps % 25 == 0:
                _report_step(job, steps, engine, opt)
            if engine.problem_gradient is not None and stop(engine.problem_gradient):
                stopped = True
                break
            opt.evaluateStep()
    except ForceCallCap as exc:
        error = f"ForceCallCap: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(dirname, ignore_errors=True)
    point = None
    coordinate_system = ""
    if opt is not None and getattr(opt, "X", None) is not None:
        point = (np.asarray(opt.X, dtype=np.float64).reshape(-1) * bohr2ang).tolist()
        coordinate_system = type(opt.IC).__name__
    return {
        "point": point,
        "stopped": stopped,
        "error": error,
        "steps": steps,
        "rebuilds": rebuilds,
        "coordinate_system": coordinate_system,
        "bonds": len(bonds),
    }


def _kind(problem_id: str) -> str:
    if problem_id in {"minimization/lj38", "minimization/lj38-noise"}:
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no geomeTRIC potential")


def _section(template: dict) -> dict:
    section = template.get("geometric") or template.get("solver") or {}
    coordsys = str(section.get("coordsys") or "")
    if coordsys not in {"cart", "tric"}:
        raise ValueError(f"{template['id']} needs geometric.coordsys cart or tric")
    element = str(section.get("element") or "")
    if not element:
        raise ValueError(f"{template['id']} needs geometric.element")
    neighbors = int(section.get("neighbors") or (_NEIGHBORS if coordsys == "tric" else 0))
    return {"coordsys": coordsys, "element": element, "neighbors": neighbors}


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    import numpy as np

    from optbench.potentials import lennard_jones, morse_bulk
    from optbench.runners.rgmin_min import _index_at

    started = time.perf_counter()
    calls = 0
    verify_calls = 0
    error = ""
    energy = None
    force_norm = None
    point = None
    potential = None
    report = None
    gtol = float(job["gtol"])
    cap = int(job["cap"])
    kind = str(job["kind"])
    try:
        box = None if job.get("box") is None else np.asarray(job["box"], dtype=np.float64)
        counter = HarnessCounter(cache=True, cap=cap)

        def raw(positions, _numbers=None, _box=None):
            if kind == "lj":
                return lennard_jones(positions)
            return morse_bulk(positions, box)

        potential = RgpotCounter(raw, counter)

        def evaluate(flat):
            value, gradient = potential.calculate(np.asarray(flat, dtype=np.float64).reshape(-1))
            return float(value), np.asarray(gradient, dtype=np.float64)

        def stop(gradient):
            return float(np.linalg.norm(np.asarray(gradient, dtype=np.float64))) <= gtol

        report = optimize(job, evaluate, stop)
        point = report["point"]
        error = str(report.get("error") or "")
    except ForceCallCap as exc:
        error = f"ForceCallCap: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if potential is not None and point is not None:
        calls = int(potential.counter.calls)
        try:
            with potential.counter.verification():
                energy, gradient = potential.calculate(point)
            force_norm = float(np.linalg.norm(gradient))
            verify_calls = int(potential.counter.verify_calls)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    elif potential is not None:
        calls = int(potential.counter.calls)
        verify_calls = int(potential.counter.verify_calls)
    hessian_index = None
    min_eigenvalues: list[float] = []
    if point is not None and (not error or "ForceCallCap" in error or "force-call cap" in error):
        try:
            hessian_index, min_eigenvalues = _index_at(
                kind,
                point,
                None if kind == "lj" else job.get("box"),
                gtol,
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
    stopped = bool(report and report.get("stopped"))
    at_force = finite and stopped and force_norm is not None and force_norm <= gtol and calls < cap and not error
    reasons: list[str] = []
    cap_hit = "ForceCallCap" in error or "force-call cap" in error
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
        "seconds": time.perf_counter() - started,
        "geometric_steps": None if report is None else int(report.get("steps") or 0),
        "coordinate_system": "" if report is None else str(report.get("coordinate_system") or ""),
        "rebuilds": None if report is None else int(report.get("rebuilds") or 0),
    }
    if error:
        record["error"] = error
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, ended: dict[str, int]) -> str:
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
        "problem": template.get("problem"),
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "partial": "true" if summary.get("partial") else "",
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter harness. One force call is one RgpotCounter.calculate entry.",
        "# geomeTRIC finite-difference Hessians are not used.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
    ]
    if str(template.get("problem")) == "minimization/lj38":
        lines.append(
            "# LJ38 index removes the six rigid modes. A curvature is negative only below "
            "-max(force_norm, 1e-8*||H||_F, 1e-6)."
        )
    if ended:
        parts = ", ".join(f"{name} {count}" for name, count in sorted(ended.items()))
        lines.append("# final coordinate system: " + parts)
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
    import geometric

    version = str(geometric.__version__)
    revision = str(getattr(geometric, "__git_revision__", "") or "")
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"geometric {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if commit and revision and revision != commit:
        raise RuntimeError(f"geometric revision {revision} does not match template {commit}")
    if not commit:
        raise RuntimeError(f"{template['id']} has no commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    section = _section(template)
    kind = _kind(problem["id"])
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if kind == "lj" or box is None else box.tolist(),
                "kind": kind,
                "coordsys": section["coordsys"],
                "neighbors": section["neighbors"],
                "element": section["element"],
                "gtol": gtol,
                "cap": cap,
            }
        )
    done_rows: dict[int, dict] = {}
    partial_path = output.with_suffix(".partial.jsonl")
    if partial_path.is_file():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_rows[int(row["index"])] = row
    pending = [job for job in jobs if job["index"] not in done_rows]
    if pending and workers > 1:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
            futures = {pool.submit(_minimize_job, job): job["index"] for job in pending}
            total = len(jobs)
            finished = len(done_rows)
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with partial_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    record = future.result()
                    done_rows[int(record["index"])] = record
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    finished += 1
                    if finished == total or finished % 50 == 0:
                        print(f"{template['id']} {finished}/{total}", flush=True)
    else:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("a", encoding="utf-8") as handle:
            for job in pending:
                record = _minimize_job(job)
                done_rows[int(record["index"])] = record
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
    records = [done_rows[job["index"]] for job in jobs]
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
    expected = (problem.get("sample") or {}).get("starts")
    summary["partial"] = expected is not None and len(records) < int(expected)
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in records
        ],
        cap,
    )
    ended: dict[str, int] = {}
    for record in records:
        name = str(record.get("coordinate_system") or "unknown")
        ended[name] = ended.get(name, 0) + 1
    passed = {"coordsys": section["coordsys"], "element": section["element"], "neighbors": section["neighbors"], "tmin": _TMIN}
    provenance = collect_provenance(
        code=str(template.get("code") or "geomeTRIC"),
        code_version=version,
        commit=commit,
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
        "note": "Harness counter. geomeTRIC does not build a finite-difference Hessian. Energy and Kabsch RMSD were not applied.",
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm", section["coordsys"]),
                "code_version": version,
                "commit": commit,
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
    output.write_text(_dat(summary, template, metrics, commit, version, ended), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if partial_path.is_file() and not summary["partial"]:
        partial_path.unlink()
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["coordinate_systems"] = ended
    return summary
