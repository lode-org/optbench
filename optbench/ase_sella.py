"""ASE and Sella campaigns with one force-call counter.

A force call is one calculator evaluation. ASE optimizers, Sella, NEB
images, and the dimer all increment that counter. An evaluation that
returns both energy and forces counts once.
"""

from __future__ import annotations

import json
import os
import statistics
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from .campaign import load_template, resolve_problem
from .counter import HarnessCounter, attach_ase_counter
from .score import score_v2, summarize_all, summarize_capped, summarize_divided
from .v2 import host_label

CONTRIBUTOR = "Rohit Goswami"

ASE_DEFAULTS = {
    "BFGS": {"maxstep": 0.2, "alpha": 70.0},
    "LBFGS": {
        "maxstep": 0.2,
        "memory": 100,
        "damping": 1.0,
        "alpha": 70.0,
        "use_line_search": False,
    },
    "FIRE": {
        "dt": 0.1,
        "maxstep": 0.2,
        "dtmax": 1.0,
        "Nmin": 5,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.1,
        "fa": 0.99,
        "a": 0.1,
        "downhill_check": False,
    },
    "BFGSLineSearch": {
        "maxstep": 0.2,
        "c1": 0.23,
        "c2": 0.46,
        "alpha": 10.0,
        "stpmax": 50.0,
    },
    "LBFGSLineSearch": {
        "maxstep": 0.2,
        "memory": 100,
        "damping": 1.0,
        "alpha": 70.0,
        "c1": 0.23,
        "c2": 0.46,
    },
    "GPMin": {"update_prior_strategy": "maximum", "update_hyperparams": False},
    "PreconLBFGS": {
        "maxstep": 0.04,
        "memory": 100,
        "damping": 1.0,
        "alpha": 70.0,
        "precon": "auto",
        "use_armijo": True,
        "c1": 0.23,
        "c2": 0.46,
        "variable_cell": False,
    },
    "PreconFIRE": {
        "dt": 0.1,
        "maxmove": 0.2,
        "dtmax": 1.0,
        "Nmin": 5,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.1,
        "fa": 0.99,
        "a": 0.1,
        "theta": 0.1,
        "precon": "none",
        "use_armijo": True,
        "variable_cell": False,
    },
    "PreconODE12r": {"alpha": 1.0, "precon": "Exp", "rtol": 0.01, "verbose": 0},
}

SELLA_MINIMUM = {
    "order": 0,
    "eta": 1e-4,
    "gamma": 0.1,
    "delta0": 0.1,
    "sigma_inc": 1.15,
    "sigma_dec": 0.90,
    "rho_inc": 1.035,
    "rho_dec": 100.0,
    "method": "qn",
    "eig": False,
    "internal": False,
    "threepoint": False,
}
SELLA_SADDLE = {
    "order": 1,
    "eta": 1e-4,
    "gamma": 0.1,
    "delta0": 0.1,
    "sigma_inc": 1.15,
    "sigma_dec": 0.65,
    "rho_inc": 1.035,
    "rho_dec": 5.0,
    "method": "prfo",
    "eig": True,
    "internal": False,
    "threepoint": False,
}

DIMER_DEFAULTS = {
    "eigenmode_method": "dimer",
    "f_rot_min": 0.1,
    "f_rot_max": 1.0,
    "max_num_rot": 1,
    "trial_angle": 0.7853981633974483,
    "trial_trans_step": 0.001,
    "maximum_translation": 0.1,
    "cg_translation": True,
    "use_central_forces": True,
    "dimer_separation": 0.0001,
    "initial_eigenmode_method": "gauss",
    "extrapolate_forces": False,
    "displacement_method": "gauss",
    "gauss_std": 0.1,
    "order": 1,
}

_SCALAR_ATTRS = (
    "maxstep",
    "alpha",
    "memory",
    "damping",
    "dt",
    "dtmax",
    "Nmin",
    "finc",
    "fdec",
    "astart",
    "fa",
    "a",
    "use_line_search",
    "use_armijo",
    "c1",
    "c2",
    "stpmax",
    "downhill_check",
    "theta",
    "rtol",
    "maxmove",
    "update_hyperparams",
    "update_prior_strategy",
    "verbose",
)


class ForceCap(RuntimeError):
    """The harness force-call cap was reached."""


class CallCounter:
    """One shared ``HarnessCounter`` for every calculator in a search.

    Index 0 is the call count and index 1 is the cap, which is what the
    drivers already read. NEB images attach this same counter.
    """

    def __init__(self, cap: int):
        self.harness = HarnessCounter(cache=False)
        self.state = {
            "calls": 0,
            "verify_calls": 0,
            "verifying": False,
            "cap": int(cap),
        }

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return int(self.harness.calls)
        if index == 1:
            return int(self.state["cap"])
        raise IndexError(index)


def _cap_error(exc: BaseException) -> bool:
    return isinstance(exc, ForceCap) or (
        isinstance(exc, RuntimeError) and "force-call cap" in str(exc)
    )


def _project_rigid(hessian, positions, n_rot: int):
    """Remove translation, and rotation when ``n_rot`` is 3.

    A free cluster has six rigid modes. A periodic cell keeps translations
    and loses rotations. Modes that are linearly dependent are dropped.
    """
    import numpy as np

    count = len(positions)
    modes = []
    for axis in range(3):
        vector = np.zeros((count, 3))
        vector[:, axis] = 1.0
        modes.append(vector.ravel())
    if n_rot:
        centered = np.asarray(positions, dtype=float) - np.mean(positions, axis=0)
        for axis in range(3):
            unit = np.zeros(3)
            unit[axis] = 1.0
            modes.append(np.cross(centered, unit).ravel())
    basis = np.stack(modes, axis=1)
    orthogonal, triangular = np.linalg.qr(basis)
    keep = np.abs(np.diag(triangular)) > 1e-8
    if not np.any(keep):
        return hessian
    axes = orthogonal[:, keep]
    projector = np.eye(hessian.shape[0]) - axes @ axes.T
    return projector @ hessian @ projector


def vibrational_index(atoms, step: float = 1e-4, tol: float = 1e-6) -> int:
    """Hessian index on the free atoms.

    The difference is central. Calls made here must already be marked as
    verification on the attached counter. Fixed atoms are left out of the
    matrix. Rigid modes are projected only when every atom is free, because
    a fixed substrate removes those zeros itself.
    """
    import numpy as np

    positions = np.array(atoms.get_positions(), dtype=float, copy=True)
    free = np.asarray(_free_mask(atoms), dtype=bool)
    free_index = np.flatnonzero(free)
    if len(free_index) == 0:
        return 0
    dimension = 3 * len(free_index)
    hessian = np.zeros((dimension, dimension))
    try:
        for local, atom in enumerate(free_index):
            for axis in range(3):
                column = 3 * local + axis
                for sign in (1.0, -1.0):
                    trial = positions.copy()
                    trial[atom, axis] += sign * step
                    atoms.set_positions(trial)
                    forces = np.asarray(atoms.get_forces(), dtype=float)[free].ravel()
                    hessian[:, column] += -sign * forces / (2.0 * step)
    finally:
        atoms.set_positions(positions)
    atoms.get_forces()
    hessian = 0.5 * (hessian + hessian.T)
    if bool(free.all()):
        rotations = 0 if np.any(atoms.pbc) else 3
        hessian = _project_rigid(hessian, positions, rotations)
    values = np.linalg.eigvalsh(hessian)
    return int(np.sum(values < -float(tol)))


def _verify_index(atoms, expected: int) -> int:
    """Count a Hessian as verification, not as search cost."""
    state = getattr(getattr(atoms, "calc", None), "optbench_counter", None)
    previous = False
    if state is not None:
        previous = bool(state.get("verifying", False))
        state["verifying"] = True
    try:
        return vibrational_index(atoms)
    finally:
        if state is not None:
            state["verifying"] = previous


def _geodesic_ase(template: dict) -> bool:
    return template.get("engine") == "geodesic" and template.get("backend") == "ase"


def entry_id(template: dict) -> str:
    prefix = "ase" if template["engine"] == "ase" else "sella"
    rest = template["id"].split("/", 1)[1]
    return f"{prefix}-{template['code_version']}-{rest}"


def raw_cap(template: dict) -> int:
    """Harness-call cap. A band stops on the total across every image."""
    return int(template["stop"]["max_force_calls"])


def _plain(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return value
    return str(value)


def format_metric(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def _lookup_stored(root: Path, archive_path: str) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(archive_path)


def materialize(root: Path, template: dict, problem: dict, cache: Path) -> Path:
    archive_path = (template.get("input") or {}).get("archive_path")
    if not archive_path:
        archive_path = (problem.get("input") or {}).get("archive_path")
    if not archive_path:
        raise FileNotFoundError(f"{template['id']} has no geometry archive")
    stored = _lookup_stored(root, archive_path)
    if not tarfile.is_tarfile(stored):
        return stored
    dest = cache / stored.parent.name
    marker = dest / ".complete"
    if not marker.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(stored) as packed:
            packed.extractall(dest, filter="data")
        marker.write_text("ok\n", encoding="utf-8")
    return dest


def _prefer_con(paths: list[Path]) -> Path:
    for path in paths:
        if path.suffix == ".con":
            return path
    return paths[0]


def discover(material: Path, template: dict) -> list[tuple[str, list[Path], int | None]]:
    """Return (start name, geometry paths, seed) in a stable order."""
    task = template["task"]
    if task in {"dimer", "sella-search"}:
        path = material if material.is_file() else _prefer_con(sorted(material.rglob("*.con")))
        block = template.get("dimer") if task == "dimer" else template.get("sella_search")
        searches = int((block or {}).get("searches", 1))
        return [(f"seed-{seed:04d}", [path], seed) for seed in range(searches)]
    if task in {"neb", "sella-neb"}:
        return _discover_neb(material, template["neb"])
    select = (template.get("input") or {}).get("select", "con")
    if material.is_file():
        return [(material.name, [material], None)]
    files = [path for path in material.rglob("*") if path.is_file() and path.suffix == ".con"]
    if select == "cluster":
        files = [path for path in files if path.name.startswith("cluster_")]
    elif select == "initial":
        files = [path for path in files if path.name.startswith("initial_")]
    else:
        files = [path for path in files if path.name != "reactant.con"]
    return [(path.name, [path], None) for path in sorted(files, key=_nat_key)]


def _discover_neb(material: Path, neb: dict) -> list[tuple[str, list[Path], int | None]]:
    layout = neb["layout"]
    if layout == "minima":
        rows = []
        directories = sorted(
            (path for path in material.rglob("*") if path.is_dir() and path.name.startswith("min")),
            key=_nat_key,
        )
        for directory in directories:
            left = directory / "a.con"
            right = directory / "b.con"
            if left.is_file() and right.is_file():
                rows.append((directory.name, [left, right], None))
        return rows
    grouped: dict[tuple[str, int | None], list[Path]] = {}
    import re

    pattern = re.compile(r"^(start|end|product)_(\d+)\.(con|xyz)$")
    for path in material.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "reactant.con":
            grouped.setdefault(("reactant", None), []).append(path)
            continue
        match = pattern.match(path.name)
        if match:
            grouped.setdefault((match.group(1), int(match.group(2))), []).append(path)
    if layout == "reactant-products":
        reactant = _prefer_con(grouped[("reactant", None)])
        products = sorted(
            (
                (index, _prefer_con(paths))
                for (role, index), paths in grouped.items()
                if role == "product" and index is not None
            ),
            key=lambda item: item[0],
        )
        return [(f"product_{index}", [reactant, path], None) for index, path in products]
    starts = {
        index: _prefer_con(paths)
        for (role, index), paths in grouped.items()
        if role == "start" and index is not None
    }
    ends = {
        index: _prefer_con(paths)
        for (role, index), paths in grouped.items()
        if role == "end" and index is not None
    }
    indices = sorted(set(starts) & set(ends))
    count = neb.get("count")
    if count is not None:
        indices = [index for index in indices if index < int(count)]
    return [(f"{index:04d}", [starts[index], ends[index]], None) for index in indices]


def _nat_key(path: Path) -> list[int | str]:
    import re

    pieces = re.split(r"(\d+)", path.name)
    return [int(piece) if piece.isdigit() else piece for piece in pieces]


def _read_positions(path: Path) -> list[list[float]]:
    atoms = _read_atoms(path)
    return atoms.positions.tolist()


def _read_xyz(path: Path):
    """Read an xyz file whose symbols need not be chemical elements."""
    from ase import Atoms
    from ase.data import atomic_numbers

    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].split()[0])
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append(parts)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"{path} has {len(rows)} coordinates, expected {count}")
    symbols = []
    positions = []
    for parts in rows:
        symbol = parts[0]
        if symbol not in atomic_numbers:
            symbol = "H"
        symbols.append(symbol)
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return Atoms(symbols=symbols, positions=positions)


def _read_atoms(path: Path):
    from ase.io import read

    if path.suffix == ".xyz":
        try:
            return read(path, format="xyz")
        except Exception:
            return _read_xyz(path)
    try:
        return read(path, format="eon")
    except Exception:
        return read(str(path))


def _apply_pbc(atoms, potential: dict):
    pbc = potential.get("pbc")
    if pbc is not None:
        atoms.set_pbc(list(pbc))
    return atoms


class _Morse:
    """Pt Morse used by the eOn morse_pt and smooth-linear archives."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, D: float, alpha: float, r0: float, rc: float, form: str):
        from ase.calculators.calculator import Calculator

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self):
                super().__init__()
                self.D = D
                self.alpha = alpha
                self.r0 = r0
                self.rc = rc
                self.form = form

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
                from ase.calculators.calculator import all_changes

                changes = all_changes if system_changes is None else system_changes
                super().calculate(atoms, properties, changes)
                energy, forces = _morse_arrays(
                    atoms.positions,
                    atoms.cell.array,
                    atoms.pbc,
                    self.D,
                    self.alpha,
                    self.r0,
                    self.rc,
                    self.form,
                )
                self.results = {"energy": energy, "forces": forces}

        self.calc = _Calc()


def _morse_arrays(positions, cell, pbc, D, alpha, r0, rc, form):
    import numpy as np

    delta = positions[:, None, :] - positions[None, :, :]
    if bool(np.any(pbc)):
        frac = delta @ np.linalg.inv(cell)
        shift = np.round(frac)
        for axis, periodic in enumerate(pbc):
            if not periodic:
                shift[..., axis] = 0.0
        delta = (frac - shift) @ cell
    dist = np.linalg.norm(delta, axis=-1)
    mask = (dist > 1e-12) & (dist < rc)
    x = np.exp(-alpha * (dist - r0))
    energy_pair = D * (x * x - 2.0 * x)
    slope = 2.0 * alpha * D * x * (1.0 - x)
    x_cut = np.exp(-alpha * (rc - r0))
    energy_cut = D * (x_cut * x_cut - 2.0 * x_cut)
    slope_cut = 2.0 * alpha * D * x_cut * (1.0 - x_cut)
    if form == "shift":
        energy_pair = energy_pair - energy_cut
    elif form == "smooth":
        linear = -slope_cut
        energy_pair = energy_pair + linear * dist - energy_cut - linear * rc
        slope = slope + linear
    else:
        raise ValueError(form)
    energy_pair = np.where(mask, energy_pair, 0.0)
    slope = np.where(mask, slope, 0.0)
    energy = float(0.5 * energy_pair.sum())
    hat = np.zeros_like(delta)
    np.divide(delta, dist[..., None], out=hat, where=mask[..., None])
    forces = -np.sum(slope[..., None] * hat, axis=1)
    return energy, forces


def _inner_calculator(potential: dict):
    if potential["kind"] == "lennard-jones":
        from ase.calculators.lj import LennardJones

        return LennardJones(
            epsilon=float(potential["epsilon"]),
            sigma=float(potential["sigma"]),
            rc=float(potential["rc"]),
        )
    if potential["kind"] == "morse":
        return _Morse(
            float(potential["D"]),
            float(potential["alpha"]),
            float(potential["r0"]),
            float(potential["rc"]),
            str(potential["form"]),
        ).calc
    raise ValueError(potential["kind"])


def _attach(atoms, potential: dict, counter: CallCounter):
    _apply_pbc(atoms, potential)
    calculator = _inner_calculator(potential)
    attach_ase_counter(
        calculator,
        cap=counter[1],
        state=counter.state,
        counter=counter.harness,
    )
    atoms.calc = calculator
    return atoms


def _cached_l2(calc, atoms=None) -> float | None:
    """Euclidean norm of the forces the optimizer is allowed to move.

    Fixed atoms keep a reaction force in the raw calculator result. The stop
    uses the constrained forces, which is what ``atoms.get_forces`` returns
    from the cached evaluation.
    """
    import numpy as np

    if atoms is not None and getattr(atoms, "calc", None) is not None:
        results = getattr(atoms.calc, "results", None) or {}
        if "forces" in results:
            return float(np.linalg.norm(np.asarray(atoms.get_forces(), dtype=float)))
    results = getattr(calc, "results", None) or {}
    forces = results.get("forces")
    if forces is None:
        inner = getattr(calc, "inner", None)
        if inner is not None:
            forces = getattr(inner, "results", {}).get("forces")
    if forces is None:
        return None
    return float(np.linalg.norm(np.asarray(forces, dtype=float)))


def _lbfgs_line_search():
    try:
        from ase.optimize import LBFGSLineSearch
    except ImportError:
        from ase.optimize.lbfgs import LBFGSLineSearch
    return LBFGSLineSearch


def _build_optimizer(spec: dict, target):
    from ase.optimize import BFGS, BFGSLineSearch, FIRE, GPMin, LBFGS
    from ase.optimize.precon import Exp, PreconFIRE, PreconLBFGS, PreconODE12r

    classes = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "FIRE": FIRE,
        "BFGSLineSearch": BFGSLineSearch,
        "LBFGSLineSearch": _lbfgs_line_search(),
        "GPMin": GPMin,
        "PreconLBFGS": PreconLBFGS,
        "PreconFIRE": PreconFIRE,
        "PreconODE12r": PreconODE12r,
    }
    kwargs = dict(spec.get("parameters") or {})
    precon = kwargs.pop("precon", None)
    if isinstance(precon, dict):
        kwargs["precon"] = Exp(A=float(precon.get("A", 3.0)))
    elif precon is not None:
        kwargs["precon"] = precon
    if spec["name"] == "PreconFIRE" and "maxstep" in kwargs:
        kwargs["maxmove"] = kwargs.pop("maxstep")
    cls = classes[spec["name"]]
    try:
        return cls(target, logfile=None, **kwargs)
    except TypeError:
        return cls(target, **kwargs)


def _describe_precon(precon) -> str:
    if precon is None:
        return "none"
    if isinstance(precon, str):
        return precon
    text = type(precon).__name__
    amplitude = getattr(precon, "A", None)
    if isinstance(amplitude, (int, float)):
        text += f" A={float(amplitude)}"
    return text


def _requested_parameters(spec: dict) -> dict:
    requested = dict(spec.get("parameters") or {})
    if spec.get("name") == "PreconFIRE" and "maxstep" in requested:
        requested["maxmove"] = requested.pop("maxstep")
    return requested


def _snapshot_optimizer(opt, spec: dict) -> dict:
    name = spec["name"]
    params = {key: _plain(value) for key, value in ASE_DEFAULTS.get(name, {}).items()}
    requested = _requested_parameters(spec)
    precon = requested.pop("precon", None)
    for key, value in requested.items():
        params[key] = _plain(value)
    if precon is not None:
        params["precon"] = _describe_precon(precon) if not isinstance(precon, dict) else (
            f"{precon.get('name', 'Exp')} A={float(precon.get('A', 3.0))}"
        )
    for key in _SCALAR_ATTRS:
        if hasattr(opt, key):
            value = getattr(opt, key)
            if isinstance(value, (int, float, str, bool)):
                params[key] = _plain(value)
    if hasattr(opt, "precon"):
        params["precon"] = _describe_precon(opt.precon)
    return params


def _snapshot_sella(opt, order: int, requested: dict) -> dict:
    params = dict(SELLA_MINIMUM if order == 0 else SELLA_SADDLE)
    params.update(requested)
    params["order"] = int(getattr(opt, "ord", order))
    for key in ("eta", "sigma_inc", "sigma_dec", "rho_inc", "rho_dec", "method", "eig"):
        if hasattr(opt, key):
            params[key] = _plain(getattr(opt, key))
    if hasattr(opt, "delta"):
        params["delta0"] = float(opt.delta)
    diag = getattr(opt, "diagkwargs", {}) or {}
    if "gamma" in diag:
        params["gamma"] = float(diag["gamma"])
    if "threepoint" in diag:
        params["threepoint"] = bool(diag["threepoint"])
    params["internal"] = bool(getattr(opt, "user_internal", False))
    return params


def _drive_l2(
    opt,
    calc,
    bucket: CallCounter,
    tol: float,
    max_steps: int,
    atoms=None,
    norm_of=None,
) -> tuple[bool, float | None]:
    from ase.optimize.optimize import Optimizer

    measure = norm_of or (lambda: _cached_l2(calc, atoms))
    if type(opt).step is Optimizer.step:
        return _drive_l2_scipy(opt, calc, bucket, tol, max_steps, atoms, measure)
    norm = None
    try:
        for _converged in opt.irun(fmax=0.0, steps=max_steps):
            norm = measure()
            if bucket[0] >= bucket[1]:
                raise ForceCap()
            if norm is not None and norm <= tol:
                return False, norm
    except Exception as exc:
        if not _cap_error(exc):
            raise
        return True, measure()
    norm = measure()
    failed = norm is None or norm > tol or bucket[0] >= bucket[1]
    return failed, norm


def _drive_l2_scipy(
    opt,
    calc,
    bucket: CallCounter,
    tol: float,
    max_steps: int,
    atoms=None,
    measure=None,
) -> tuple[bool, float | None]:
    measure = measure or (lambda: _cached_l2(calc, atoms))

    def gradient_converged(_gradient) -> bool:
        if bucket[0] >= bucket[1]:
            raise ForceCap()
        norm = measure()
        return norm is not None and norm <= tol

    opt.gradient_converged = gradient_converged
    try:
        opt.run(fmax=tol, steps=max_steps)
    except Exception as exc:
        if not _cap_error(exc):
            raise
        return True, measure()
    norm = measure()
    return norm is None or norm > tol or bucket[0] >= bucket[1], norm


def _fmax(atoms) -> float | None:
    import numpy as np

    if getattr(atoms, "calc", None) is None:
        return None
    results = getattr(atoms.calc, "results", None) or {}
    if "forces" not in results:
        return None
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.size == 0:
        return None
    return float(np.sqrt((forces * forces).sum(axis=1)).max())


def _cap_row(row: dict, atoms, message: str = "force-call cap") -> dict:
    row["failed"] = True
    row["error"] = message
    if atoms is not None and row.get("force_norm") is None:
        row["force_norm"] = _cached_l2(getattr(atoms, "calc", None), atoms)
    return row


def _run_minimize(payload: dict, bucket: CallCounter) -> dict:
    atoms = _attach(_read_atoms(Path(payload["paths"][0])), payload["potential"], bucket)
    opt = _build_optimizer(payload["optimizer"], atoms)
    row = {"failed": True, "force_norm": None, "expected_index": 0}
    if payload.get("snapshot"):
        row["parameters"] = _snapshot_optimizer(opt, payload["optimizer"])
    try:
        measure = (lambda: _fmax(atoms)) if payload.get("norm") == "fmax" else None
        failed, norm = _drive_l2(
            opt,
            atoms.calc,
            bucket,
            payload["tol"],
            payload["max_steps"],
            atoms,
            norm_of=measure,
        )
    except Exception as exc:
        if _cap_error(exc):
            return _cap_row(row, atoms)
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return row
    row["failed"] = failed
    row["force_norm"] = norm
    row["energy"] = _safe_energy(atoms)
    if not failed and bucket[0] < bucket[1]:
        _attach_index(row, atoms, 0)
    return row


def _reactant_mode(atoms, reference) -> object:
    """Unit displacement from the reactant, in the free-atom coordinates."""
    import numpy as np
    from ase.constraints import FixAtoms

    if reference is None:
        return None
    reference = np.asarray(reference, dtype=float)
    if reference.shape != atoms.positions.shape:
        return None
    displacement = atoms.positions - reference
    free = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            free[np.asarray(constraint.index, dtype=int)] = False
    vector = displacement[free].ravel()
    scale = float(np.linalg.norm(vector))
    if scale == 0.0:
        return None
    return vector / scale


def _safe_energy(atoms) -> float | None:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


def _attach_index(row: dict, atoms, expected: int) -> None:
    row["expected_index"] = int(expected)
    try:
        row["hessian_index"] = int(_verify_index(atoms, expected))
    except Exception as exc:
        row["error"] = f"Hessian {type(exc).__name__}: {exc}"[:500]
        row["failed"] = True
        return
    if row["hessian_index"] != expected:
        row["failed"] = True


def _run_sella(payload: dict, bucket: CallCounter) -> dict:
    from sella import Sella

    atoms = _attach(_read_atoms(Path(payload["paths"][0])), payload["potential"], bucket)
    requested = dict(payload["optimizer"].get("parameters") or {})
    order = int(requested.pop("order", payload.get("sella", {}).get("order", 1)))
    v0 = _reactant_mode(atoms, payload.get("reactant_positions"))
    kwargs = {key: value for key, value in requested.items() if value is not None}
    opt = Sella(atoms, logfile=None, order=order, v0=v0, **kwargs)
    row = {"failed": True, "force_norm": None, "expected_index": order}
    if payload.get("snapshot"):
        snapped = _snapshot_sella(opt, order, requested)
        if payload.get("sella", {}).get("v0") not in (None, "none"):
            snapped["v0"] = payload["sella"]["v0"]
        row["parameters"] = snapped
    try:
        failed, norm = _drive_l2(opt, atoms.calc, bucket, payload["tol"], payload["max_steps"], atoms)
    except Exception as exc:
        if _cap_error(exc):
            return _cap_row(row, atoms)
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return row
    row["failed"] = failed
    row["force_norm"] = norm
    row["energy"] = _safe_energy(atoms)
    library = getattr(opt, "pes", None)
    if library is not None and hasattr(library, "neval"):
        row["library_neval"] = int(library.neval)
    if not failed and bucket[0] < bucket[1]:
        _attach_index(row, atoms, order)
    return row


def _make_band(payload: dict, bucket: CallCounter):
    from ase.mep import NEB

    neb_spec = payload["neb"]
    moving = int(neb_spec["moving_images"])
    left = _apply_pbc(_read_atoms(Path(payload["paths"][0])), payload["potential"])
    right = _apply_pbc(_read_atoms(Path(payload["paths"][1])), payload["potential"])
    images = [left]
    for _ in range(moving):
        images.append(left.copy())
    images.append(right)
    climb_after = neb_spec.get("climb_after")
    climb = bool(neb_spec["climb"]) if climb_after is None else False
    band = NEB(
        images,
        k=float(neb_spec["spring"]),
        climb=climb,
        method=str(neb_spec.get("method", "improvedtangent")),
    )
    method = str(neb_spec.get("interpolate", "linear"))
    if method == "geodesic":
        _geodesic_images(images, payload["potential"], seed=int(payload.get("seed") or 0))
    else:
        band.interpolate(method=method)
    for image in images:
        _attach(image, payload["potential"], bucket)
    return band, images


def _geodesic_images(images, potential: dict, seed: int) -> None:
    """Replace a linear or IDPP guess with the geodesic initial path.

    The smoother does not call the calculator. Endpoints stay on the
    geometries the problem supplied.
    """
    import numpy as np

    from .geodesic_path import geodesic_positions, symbols_for

    left = images[0]
    right = images[-1]
    free = np.asarray(_free_mask(left), dtype=bool)
    box = None
    if any(bool(flag) for flag in left.pbc):
        box = np.asarray(left.cell.lengths(), dtype=np.float64)
    frames = geodesic_positions(
        np.asarray(left.positions, dtype=np.float64),
        np.asarray(right.positions, dtype=np.float64),
        symbols_for(str(potential["kind"]), len(left)),
        len(images) - 2,
        box,
        free=free,
        seed=seed,
    )
    for image, frame in zip(images, frames, strict=True):
        image.set_positions(frame)


def _neb_parameters(opt, payload: dict, neb_spec: dict) -> dict:
    params = _snapshot_optimizer(opt, payload["optimizer"])
    params["spring"] = float(neb_spec["spring"])
    params["climb"] = bool(neb_spec["climb"])
    params["moving_images"] = int(neb_spec["moving_images"])
    params["method"] = str(neb_spec.get("method", "improvedtangent"))
    if neb_spec.get("interpolate"):
        params["interpolate"] = str(neb_spec["interpolate"])
    if neb_spec.get("climb_after") is not None:
        params["climb_after"] = float(neb_spec["climb_after"])
        params["climb"] = False
    return params


def _highest_moving(images):
    best = images[1]
    best_energy = float("-inf")
    for image in images[1:-1]:
        energy = float(image.get_potential_energy())
        if energy > best_energy:
            best = image
            best_energy = energy
    return best


def _run_neb(payload: dict, bucket: CallCounter, band_limit: int | None = None) -> dict:
    neb_spec = payload["neb"]
    band, images = _make_band(payload, bucket)
    opt = _build_optimizer(payload["optimizer"], band)
    climb_after = neb_spec.get("climb_after")
    limit = bucket[1] if band_limit is None else int(band_limit)
    converged = False
    try:
        for converged in opt.irun(fmax=payload["tol"], steps=payload["max_steps"]):
            if bucket[0] >= limit:
                converged = False
                raise ForceCap()
            if climb_after is not None and not band.climb:
                residuals = [float(value) for value in getattr(band, "residuals", []) or []]
                if residuals and max(residuals) <= float(climb_after):
                    band.climb = True
    except Exception as exc:
        if not _cap_error(exc):
            raise
        converged = False
    residuals = [float(value) for value in getattr(band, "residuals", []) or []]
    failed = (not converged) or bucket[0] >= bucket[1]
    row = {"failed": failed, "force_norm": max(residuals) if residuals else None, "images": images}
    if payload.get("snapshot"):
        row["parameters"] = _neb_parameters(opt, payload, neb_spec)
    return row


def _free_mask(atoms) -> list[bool]:
    """True where the dimer may displace an atom. Fixed atoms stay put."""
    import numpy as np
    from ase.constraints import FixAtoms

    free = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            free[np.asarray(constraint.index, dtype=int)] = False
    return [bool(value) for value in free]


def _displace_free(atoms, seed: int, std: float) -> None:
    import numpy as np

    mask = np.asarray(_free_mask(atoms), dtype=bool)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(std), size=atoms.positions.shape)
    noise[~mask] = 0.0
    atoms.set_positions(atoms.positions + noise)


def _run_sella_neb(payload: dict, bucket: CallCounter) -> dict:
    """Relax the band, then refine the highest image with Sella.

    The band stops at ``reserve`` of the cap so the saddle step has calls
    left. ``v0`` stays unset: on a free cluster Sella drops the rigid modes
    and a full Cartesian mode has the wrong length.
    """
    from sella import Sella

    neb_spec = payload["neb"]
    reserve = float((payload.get("sella") or {}).get("reserve", 0.2))
    band_limit = max(1, int(bucket[1] * (1.0 - reserve)))
    band_payload = dict(payload)
    band_payload["optimizer"] = {
        "name": str(neb_spec.get("band_optimizer", "FIRE")),
        "parameters": dict(neb_spec.get("band_parameters") or {}),
    }
    try:
        band_row = _run_neb(band_payload, bucket, band_limit=band_limit)
    except Exception as exc:
        if _cap_error(exc):
            return _cap_row({"failed": True, "force_norm": None}, None)
        return {"failed": True, "force_norm": None, "error": f"{type(exc).__name__}: {exc}"[:500]}
    images = band_row.pop("images", None)
    if bucket[0] >= bucket[1] or not images:
        band_row["failed"] = True
        return band_row
    chosen = _highest_moving(images)
    requested = dict(payload["optimizer"].get("parameters") or {})
    order = int(requested.pop("order", 1))
    kwargs = {key: value for key, value in requested.items() if value is not None}
    try:
        opt = Sella(chosen, logfile=None, order=order, v0=None, **kwargs)
        failed, _norm = _drive_l2(
            opt,
            chosen.calc,
            bucket,
            payload["tol"],
            payload["max_steps"],
            chosen,
            norm_of=lambda: _fmax(chosen),
        )
    except Exception as exc:
        if _cap_error(exc):
            band_row["failed"] = True
            band_row["error"] = "force-call cap"
            band_row["force_norm"] = _fmax(chosen)
            return band_row
        band_row["failed"] = True
        band_row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return band_row
    force_norm = _fmax(chosen)
    failed = failed or force_norm is None or force_norm > payload["tol"] or bucket[0] >= bucket[1]
    band_row["failed"] = failed
    band_row["force_norm"] = force_norm
    band_row["energy"] = _safe_energy(chosen)
    if payload.get("snapshot"):
        params = dict(band_row.get("parameters") or {})
        params.update(_snapshot_sella(opt, order, requested))
        params["v0"] = "none"
        params["reserve"] = reserve
        params["band_optimizer"] = band_payload["optimizer"]["name"]
        band_row["parameters"] = params
    return band_row


def _run_sella_search(payload: dict, bucket: CallCounter) -> dict:
    """One Sella order-1 search from a Gaussian kick of the free atoms."""
    spec = payload.get("sella_search") or {}
    atoms = _read_atoms(Path(payload["paths"][0]))
    _apply_pbc(atoms, payload["potential"])
    _displace_free(atoms, int(payload["seed"]), float(spec.get("gauss_std", 0.1)))
    _attach(atoms, payload["potential"], bucket)
    search = dict(payload)
    search["paths"] = payload["paths"]
    # The geometry is already displaced and attached. Run Sella on it.
    from sella import Sella

    requested = dict(payload["optimizer"].get("parameters") or {})
    order = int(requested.pop("order", 1))
    kwargs = {key: value for key, value in requested.items() if value is not None}
    row = {"failed": True, "force_norm": None, "expected_index": order}
    try:
        opt = Sella(atoms, logfile=None, order=order, v0=None, **kwargs)
        if payload.get("snapshot"):
            snapped = _snapshot_sella(opt, order, requested)
            snapped["v0"] = "none"
            snapped["gauss_std"] = float(spec.get("gauss_std", 0.1))
            row["parameters"] = snapped
        failed, norm = _drive_l2(opt, atoms.calc, bucket, payload["tol"], payload["max_steps"], atoms)
    except Exception as exc:
        if _cap_error(exc):
            return _cap_row(row, atoms)
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return row
    row["failed"] = failed
    row["force_norm"] = norm
    row["energy"] = _safe_energy(atoms)
    if not failed and bucket[0] < bucket[1]:
        _attach_index(row, atoms, order)
    return row


def _run_dimer(payload: dict, bucket: CallCounter) -> dict:
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

    atoms = _attach(_read_atoms(Path(payload["paths"][0])), payload["potential"], bucket)
    mask = _free_mask(atoms)
    control = DimerControl(
        logfile=os.devnull,
        mask=mask,
        **dict(payload["optimizer"].get("parameters") or {}),
    )
    mode_atoms = MinModeAtoms(atoms, control, random_seed=int(payload["seed"]))
    opt = MinModeTranslate(mode_atoms, logfile=None)
    norm = None
    curvature = None
    try:
        # ASE stops on the largest force component. The problem stop is the
        # Euclidean norm, so the fmax gate stays open and this loop applies it.
        for _ase_converged in opt.irun(fmax=0.0, steps=payload["max_steps"]):
            if bucket[0] >= bucket[1]:
                raise ForceCap()
            norm = _cached_l2(atoms.calc, atoms)
            curvature = float(mode_atoms.get_curvature())
            if norm is not None and norm <= payload["tol"] and curvature < 0.0:
                break
    except Exception as exc:
        if not _cap_error(exc):
            raise
    if curvature is None:
        curvature = float(mode_atoms.get_curvature())
    if norm is None:
        norm = _cached_l2(atoms.calc, atoms)
    saddle = curvature < 0.0 and norm is not None and norm <= payload["tol"]
    failed = (not saddle) or bucket[0] >= bucket[1]
    row = {
        "failed": failed,
        "force_norm": norm,
        "curvature": curvature,
        "saddle": saddle,
        "expected_index": 1,
    }
    if saddle and bucket[0] < bucket[1]:
        _attach_index(row, atoms, 1)
        if row.get("hessian_index") != 1:
            row["saddle"] = False
            row["failed"] = True
    if payload.get("snapshot"):
        params = dict(DIMER_DEFAULTS)
        params.update(payload["optimizer"].get("parameters") or {})
        params["mask"] = "free" if not all(mask) else "all"
        row["parameters"] = params
    return row


def _reason_list(row: dict, cap: int, tol: float) -> list[str]:
    reasons: list[str] = []
    if int(row.get("force_calls") or 0) >= cap or row.get("error") == "force-call cap":
        reasons.append("cap")
    error = row.get("error")
    if error and error != "force-call cap":
        reasons.append("error")
    norm = row.get("force_norm")
    if norm is None or float(norm) > float(tol):
        reasons.append("force_norm")
    if row.get("expected_index") is not None and "hessian_index" in row:
        if int(row["hessian_index"]) != int(row["expected_index"]):
            reasons.append("hessian_index")
    elif row.get("expected_index") is not None and "hessian_index" not in row and "error" not in reasons:
        if "cap" not in reasons and "force_norm" not in reasons:
            reasons.append("hessian_index")
    return reasons


def _run_start(payload: dict) -> dict:
    import logging
    import time
    import warnings

    warnings.filterwarnings("ignore", message="The system is likely too small")
    began = time.perf_counter()
    logging.getLogger("sella").setLevel(logging.ERROR)
    bucket = CallCounter(int(payload["raw_cap"]))
    row = {
        "schema": "optbench.start/v2",
        "start": payload["start"],
        "seed": int(payload.get("seed") or 0),
        "force_calls": 0,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failed": True,
        "force_norm": None,
        "reference": "none",
    }
    try:
        task = payload["task"]
        if task == "minimize":
            extra = _run_minimize(payload, bucket)
        elif task == "sella":
            extra = _run_sella(payload, bucket)
        elif task == "neb":
            extra = _run_neb(payload, bucket)
        elif task == "sella-neb":
            extra = _run_sella_neb(payload, bucket)
        elif task == "sella-search":
            extra = _run_sella_search(payload, bucket)
        elif task == "dimer":
            extra = _run_dimer(payload, bucket)
        else:
            raise ValueError(task)
        extra.pop("images", None)
        row.update(extra)
    except Exception as exc:
        if _cap_error(exc):
            row["failed"] = True
            row["error"] = "force-call cap"
        else:
            row["failed"] = True
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    row["force_calls"] = int(bucket[0])
    row["verify_calls"] = int(bucket.state["verify_calls"])
    reasons = _reason_list(row, bucket[1], float(payload["tol"]))
    # NEB success is the band residual, not a Hessian index.
    if payload["task"] in {"neb", "sella-neb"}:
        reasons = [reason for reason in reasons if reason != "hessian_index"]
    row["reasons"] = reasons
    row["success"] = not reasons
    row["failed"] = not row["success"]
    row["converged"] = "force_norm" not in reasons and "cap" not in reasons and "error" not in reasons
    row["wall_seconds"] = time.perf_counter() - began
    return row


def _load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["start"])] = row
    return rows


def _v2_records(rows: list[dict], template: dict) -> tuple[list[dict], int]:
    """Costs that enter PAR-2.

    The cost is the harness total, including a band that used to be
    reported per image. Failures are marked unsuccessful so the penalty
    is ``2 * cap`` rather than the raw count.
    """
    cap = int(template["stop"]["max_force_calls"])
    records = []
    for row in rows:
        calls = float(row["force_calls"])
        success = bool(row.get("success")) and calls < cap
        records.append({"force_calls": calls, "success": success})
    return records, cap


def _summarize(rows: list[dict], template: dict) -> dict:
    cap = int(template["stop"]["max_force_calls"])
    reducer = template["reducer"]
    if reducer == "capped":
        counts = [
            cap if row["failed"] or int(row["force_calls"]) >= cap else int(row["force_calls"])
            for row in rows
        ]
        summary = summarize_capped(counts, cap)
        ok = [count for count in counts if count < cap]
        summary["force_calls_median"] = float(statistics.median(ok)) if ok else None
        return summary
    if reducer == "all":
        return summarize_all(
            [int(row["force_calls"]) for row in rows],
            sum(1 for row in rows if row["failed"]),
        )
    if reducer == "per_image":
        return summarize_divided(
            [int(row["force_calls"]) for row in rows],
            [bool(row["failed"]) for row in rows],
            images=int(template["neb"]["moving_images"]),
            cap=cap,
        )
    raise ValueError(reducer)


def _baseline_parameters(template: dict) -> dict:
    task = template["task"]
    requested = dict((template.get("optimizer") or {}).get("parameters") or {})
    if task == "sella":
        order = int(requested.get("order", template.get("sella", {}).get("order", 1)))
        params = dict(SELLA_MINIMUM if order == 0 else SELLA_SADDLE)
        params.update(requested)
        v0 = (template.get("sella") or {}).get("v0")
        if v0 not in (None, "none"):
            params["v0"] = v0
        return params
    if task == "dimer":
        params = dict(DIMER_DEFAULTS)
        params.update(requested)
        return params
    name = template["optimizer"]["name"]
    params = dict(ASE_DEFAULTS.get(name, {}))
    if name == "PreconFIRE" and "maxstep" in requested:
        requested["maxmove"] = requested.pop("maxstep")
    precon = requested.pop("precon", None)
    params.update(requested)
    if isinstance(precon, dict):
        params["precon"] = f"{precon.get('name', 'Exp')} A={float(precon.get('A', 3.0))}"
    if task in {"neb", "sella-neb"}:
        neb = template["neb"]
        if task == "sella-neb":
            order = int(requested.get("order", (template.get("sella") or {}).get("order", 1)))
            params = dict(SELLA_SADDLE if order else SELLA_MINIMUM)
            params.update(requested)
            params["band_optimizer"] = str(neb.get("band_optimizer", "FIRE"))
            params["band_parameters"] = dict(neb.get("band_parameters") or {})
            params["reserve"] = float((template.get("sella") or {}).get("reserve", 0.2))
            params["v0"] = "none"
        params["spring"] = float(neb["spring"])
        params["climb"] = bool(neb["climb"])
        params["moving_images"] = int(neb["moving_images"])
        params["method"] = str(neb.get("method", "improvedtangent"))
        if neb.get("interpolate"):
            params["interpolate"] = str(neb["interpolate"])
        if neb.get("climb_after") is not None:
            params["climb_after"] = float(neb["climb_after"])
    if task == "sella-search":
        order = int(requested.get("order", 1))
        params = dict(SELLA_SADDLE if order else SELLA_MINIMUM)
        params.update(requested)
        params["v0"] = "none"
        params["gauss_std"] = float((template.get("sella_search") or {}).get("gauss_std", 0.1))
    return params


def _write_dat(
    path: Path,
    template: dict,
    summary: dict,
    parameters: dict,
    *,
    starts: int,
    partial: bool,
    host: str,
    scored: dict | None = None,
) -> None:
    method = {
        "profile": template["profile"],
        "parameter_source": template.get("parameter_source", ""),
        "counter": "harness attach_ase_counter",
        "reducer": template["reducer"],
        "parameters": parameters,
        "potential": template.get("potential") or {},
        "stop": template.get("stop") or {},
    }
    note = template.get("note", "")
    if template["task"] in {"minimize", "sella", "sella-search", "dimer"}:
        note = (
            note
            + " v2 success is the harness cap, the stop norm, and the Hessian index"
            + " (0 minimum, 1 saddle) after rigid modes are projected out when no atom is fixed."
            + " Energy and Kabsch RMSD are not applied: these starts have no reference geometry."
        ).strip()
    elif template["task"] in {"neb", "sella-neb"}:
        note = (
            note
            + " v2 success is the harness cap and the template fmax stop."
            + " The band has no reference saddle, so the Hessian and Kabsch RMSD are not applied."
        ).strip()
    fields = [
        ("force_calls", summary.get("force_calls")),
        ("force_calls_min", summary.get("force_calls_min")),
        ("force_calls_median", summary.get("force_calls_median")),
        ("force_calls_max", summary.get("force_calls_max")),
        ("nfailed", summary.get("nfailed")),
        ("force_calls_raw", summary.get("force_calls_raw")),
        ("force_calls_per_image", summary.get("force_calls_per_image")),
        ("force_calls_per_image_min", summary.get("force_calls_per_image_min")),
        ("force_calls_per_image_max", summary.get("force_calls_per_image_max")),
        ("force_calls_per_image_median", summary.get("force_calls_per_image_median")),
        ("call_unit", summary.get("call_unit")),
        ("starts", starts),
        ("code", template["code"]),
        ("code_version", template["code_version"]),
        ("commit", template.get("commit", "")),
        ("algorithm", template["algorithm"]),
        ("profile", template["profile"]),
        ("template", template["id"]),
        ("problem", template["problem"]),
        ("counter", "harness"),
        ("reducer", template["reducer"]),
        ("parameter_source", template.get("parameter_source", "")),
        ("method", json.dumps(_jsonable(method), sort_keys=True, separators=(",", ":"))),
        ("note", note),
        ("host", host),
        ("date", date.today().isoformat()),
        ("contributor", CONTRIBUTOR),
    ]
    if scored:
        fields.extend(
            [
                ("scoring", scored.get("scoring")),
                ("par2_mean", scored.get("par2_mean")),
                ("par2_median", scored.get("median")),
                ("par2_median_low", scored.get("median_low")),
                ("par2_median_high", scored.get("median_high")),
                ("par2_iqr", scored.get("iqr")),
                ("success_rate", scored.get("success_rate")),
                ("rate_low", scored.get("rate_low")),
                ("rate_high", scored.get("rate_high")),
                ("par_applied", scored.get("par_applied")),
                ("par_factor", scored.get("par_factor")),
                ("bootstrap_resamples", scored.get("bootstrap_resamples")),
                ("bootstrap_seed", scored.get("bootstrap_seed")),
            ]
        )
    if partial:
        fields.append(("partial", "true"))
    if template.get("ranked") is False:
        fields.append(("ranked", "false"))
    if template["problem"] == "double-ended-saddle-search/lj38_2ts_separation":
        kept = starts - int(summary.get("nfailed") or 0)
        fields.append(("transition_states", kept))
    lines = []
    for key, value in fields:
        text = format_metric(value)
        if text is None or text == "":
            continue
        lines.append(f"{key} {text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_template(
    root: Path,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
    cache: Path | None = None,
) -> dict:
    """Run one ASE or Sella template and write benchmark.dat plus per-start rows."""
    host = host_label()
    template = load_template(root, template_name)
    if template["engine"] not in {"ase", "sella"} and not _geodesic_ase(template):
        raise ValueError(f"{template_name} is not an ASE or Sella template")
    problem = resolve_problem(root, template["problem"])
    cache = cache or Path("/tmp/optbench-geometry")
    material = materialize(root, template, problem, cache)
    jobs = discover(material, template)
    if limit > 0:
        jobs = jobs[:limit]
    if not jobs:
        raise FileNotFoundError(f"{template['id']} found no geometries")
    reactant_positions = None
    if (template.get("sella") or {}).get("v0") == "reactant":
        reactant = next(path for path in material.rglob("reactant.con") if path.is_file())
        reactant_positions = _read_positions(reactant)
    starts_path = output.with_name(output.stem + ".starts.jsonl")
    done = _load_rows(starts_path) if resume else {}
    if not resume and starts_path.exists():
        starts_path.unlink()
        done = {}
    cap = raw_cap(template)
    tol = float(template["stop"].get("force_norm", template["stop"].get("gradient_norm")))
    payloads = []
    for name, paths, seed in jobs:
        if name in done:
            continue
        payloads.append(
            {
                "start": name,
                "task": template["task"],
                "paths": [str(path) for path in paths],
                "potential": template["potential"],
                "optimizer": template["optimizer"],
                "neb": template.get("neb"),
                "sella": template.get("sella"),
                "sella_search": template.get("sella_search"),
                "reactant_positions": reactant_positions,
                "seed": 0 if seed is None else seed,
                "tol": tol,
                "norm": str(template["stop"].get("norm") or "l2"),
                "max_steps": int(template["stop"]["max_steps"]),
                "raw_cap": cap,
                "snapshot": False,
            }
        )
    if payloads:
        payloads[0]["snapshot"] = True
    fresh: list[dict] = []
    if payloads:
        starts_path.parent.mkdir(parents=True, exist_ok=True)
        with starts_path.open("a", encoding="utf-8") as handle:
            if workers <= 1:
                iterator = (_run_start(payload) for payload in payloads)
                total = len(payloads)
                for index, row in enumerate(iterator, 1):
                    fresh.append(row)
                    handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                    handle.flush()
                    if index == 1 or index == total or index % 25 == 0:
                        print(f"{template['id']} {index}/{total}", flush=True)
            else:
                import multiprocessing as mp

                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    futures = [pool.submit(_run_start, payload) for payload in payloads]
                    total = len(futures)
                    for index, future in enumerate(as_completed(futures), 1):
                        row = future.result()
                        fresh.append(row)
                        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        handle.flush()
                        if index == 1 or index == total or index % 25 == 0:
                            print(f"{template['id']} {index}/{total}", flush=True)
    by_start = dict(done)
    for row in fresh:
        by_start[row["start"]] = row
    ordered = [by_start[name] for name, _paths, _seed in jobs]
    starts_path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    errors = [row for row in ordered if row.get("error")]
    for row in errors[:5]:
        print(f"error {template['id']} {row['start']}: {row['error']}", flush=True)
    summary = _summarize(ordered, template)
    records, score_cap = _v2_records(ordered, template)
    scored = score_v2(records, score_cap)
    summary["par2_mean"] = scored.get("par2_mean")
    summary["scoring"] = scored.get("scoring")
    parameters = _baseline_parameters(template)
    for row in ordered:
        if row.get("parameters"):
            parameters = row["parameters"]
            break
    _write_dat(
        output,
        template,
        summary,
        parameters,
        starts=len(ordered),
        partial=limit > 0,
        host=host,
        scored=scored,
    )
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = template["problem"]
    summary["entry"] = entry_id(template)
    summary["output"] = str(output)
    return summary


def run_point(root: Path, template_name: str, start: str, seed: int, *, cache: Path | None = None) -> dict:
    """Run one start and return its v2 record, including wall time.

    ``start`` is the geometry name, or the geometry file name for a search
    that varies ``seed``. The record's ``start`` stays the discover name,
    so a dimer seed is still ``seed-0000``.
    """
    import time

    template = load_template(root, template_name)
    problem = resolve_problem(root, template["problem"])
    cache = cache or Path("/tmp/optbench-geometry")
    material = materialize(root, template, problem, cache)
    jobs = discover(material, template)
    chosen = None
    for name, paths, job_seed in jobs:
        job_seed_i = 0 if job_seed is None else int(job_seed)
        if template["task"] in {"dimer", "sella-search"}:
            if job_seed_i == int(seed):
                chosen = (name, paths, job_seed_i)
                break
        elif name == start and job_seed_i == int(seed):
            chosen = (name, paths, job_seed_i)
            break
    if chosen is None:
        raise KeyError(f"{template_name} has no start {start!r} seed {seed}")
    name, paths, job_seed_i = chosen
    reactant_positions = None
    if (template.get("sella") or {}).get("v0") == "reactant":
        reactant = next(path for path in material.rglob("reactant.con") if path.is_file())
        reactant_positions = _read_positions(reactant)
    tol = float(template["stop"].get("force_norm", template["stop"].get("gradient_norm")))
    payload = {
        "start": name,
        "task": template["task"],
        "paths": [str(path) for path in paths],
        "potential": template["potential"],
        "optimizer": template["optimizer"],
        "neb": template.get("neb"),
        "sella": template.get("sella"),
        "sella_search": template.get("sella_search"),
        "reactant_positions": reactant_positions,
        "seed": job_seed_i,
        "tol": tol,
        "norm": str(template["stop"].get("norm") or "l2"),
        "max_steps": int(template["stop"]["max_steps"]),
        "raw_cap": raw_cap(template),
        "snapshot": True,
    }
    started = time.perf_counter()
    row = _run_start(payload)
    row["wall_seconds"] = time.perf_counter() - started
    row["template"] = template["id"]
    row["problem"] = template["problem"]
    return row


def write_rows(root: Path, template_name: str, rows: list[dict], output: Path) -> dict:
    """Write one campaign dat and its starts file from finished v2 rows."""
    template = load_template(root, template_name)
    for row in rows:
        row["failed"] = not bool(row.get("success"))
        row.setdefault("schema", "optbench.start/v2")
        row.setdefault("counter", "harness")
        row.setdefault("comparable", True)
    starts_path = output.with_name(output.stem + ".starts.jsonl")
    starts_path.parent.mkdir(parents=True, exist_ok=True)
    starts_path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = _summarize(rows, template)
    records, score_cap = _v2_records(rows, template)
    scored = score_v2(records, score_cap)
    summary["par2_mean"] = scored.get("par2_mean")
    parameters = _baseline_parameters(template)
    for row in rows:
        if row.get("parameters"):
            parameters = row["parameters"]
            break
    _write_dat(
        output,
        template,
        summary,
        parameters,
        starts=len(rows),
        partial=False,
        host=host_label(),
        scored=scored,
    )
    summary["starts"] = len(rows)
    summary["template"] = template["id"]
    summary["problem"] = template["problem"]
    summary["entry"] = entry_id(template)
    summary["output"] = str(output)
    return summary


def run_many(
    root: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    limit: int = 0,
    match: str = "",
    ranked_only: bool = True,
    resume: bool = False,
) -> list[dict]:
    """Run every ranked ASE and Sella template, light problems first."""
    from .campaign import list_templates

    found = []
    for template in list_templates(root):
        if template.get("engine") not in {"ase", "sella"}:
            continue
        if ranked_only and not template.get("ranked", True):
            continue
        if match and match not in template["id"]:
            continue
        found.append(template)

    def weight(template: dict) -> tuple:
        ident = template["id"]
        score = 0
        if any(token in ident for token in ("gpmin", "precon", "de-pt", "neb-pt", "pt-island")):
            score += 2
        if "lj38" in ident:
            score += 1
        return (score, ident)

    summaries = []
    cache = Path("/tmp/optbench-geometry")
    for template in sorted(found, key=weight):
        output = output_dir / f"{entry_id(template)}.dat"
        print(f"run {template['id']} -> {output.name}", flush=True)
        summaries.append(
            run_template(
                root,
                template["id"],
                output=output,
                workers=workers,
                limit=limit,
                resume=resume,
                cache=cache,
            )
        )
    return summaries


def write_problem_record(root: Path, campaign: Path, dest: Path, problem: str, status: str) -> None:
    """Write one optbench.result/v1 file from campaign dat files for ``problem``."""
    from .dat import parse_lenient

    runs = []
    for path in sorted(campaign.glob("*.dat")):
        fields, _duplicates = parse_lenient(path.read_text(encoding="utf-8"))
        if fields.get("problem") != problem or fields.get("partial") == "true":
            continue
        metrics = {}
        for key in (
            "force_calls",
            "force_calls_min",
            "force_calls_max",
            "force_calls_median",
            "force_calls_raw",
            "nfailed",
        ):
            if key not in fields:
                continue
            text = fields[key]
            metrics[key] = float(text) if "." in text else int(text)
        method = json.loads(fields["method"]) if fields.get("method") else {}
        runs.append(
            {
                "entry": path.stem,
                "code": fields.get("code", ""),
                "algorithm": fields.get("algorithm", ""),
                "code_version": fields.get("code_version", ""),
                "commit": fields.get("commit", ""),
                "present_source": True,
                "starts": int(fields.get("starts", "0")),
                "nfailed": int(fields.get("nfailed", "0")),
                "method": method,
                "metrics": metrics,
                "starts_file": str(path.with_name(path.stem + ".starts.jsonl").relative_to(root)),
                "notes": fields.get("note", ""),
            }
        )
    record = {
        "schema": "optbench.result/v1",
        "status": status,
        "problem": problem,
        "submitter": CONTRIBUTOR,
        "date": date.today().isoformat(),
        "host": host_label(),
        "runs": runs,
    }
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
