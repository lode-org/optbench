"""TSASE solid-state NEB, loaded without the package initializer.

``tsase/__init__.py`` imports the Fortran calculators and calls
``numpy.seterr(all='raise')``. Those extensions are built with
``numpy.distutils``, which this interpreter does not provide. The harness
loads ``neb/ssneb.py`` and ``neb/fire_ssneb.py`` by file path and counts
ASE ``calculate`` entries on one shared calculator. ``SSDimer.forceCalls``
is not the score.

``ss=False`` leaves the cell stress out. The cell is diagonal. A free
cluster is shifted into a large orthorhombic cell so the fractional-coordinate
interpolation does not wrap. Fixed atoms keep a zero potential force, and
each FIRE step puts them back on the reactant coordinate. The class
multiplies the spring constant by the image count; that factor is its own.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from .common import (
    cached_pairs,
    judge_mode,
    judge_saddle,
    pair_seed,
    prepared,
    run_jobs,
    write_campaign,
)
from .geometries import load_starts, point_settings

ENGINE = "tsase"
CODE = "tsase"


def source_root() -> Path:
    env = os.environ.get("TSASE_SRC")
    if env:
        return Path(env)
    return Path("/tmp/src/tsase")


def available() -> str | None:
    neb = source_root() / "tsase" / "neb" / "ssneb.py"
    fire = source_root() / "tsase" / "neb" / "fire_ssneb.py"
    if not neb.is_file() or not fire.is_file():
        return (
            f"TSASE ssneb.py is not under {source_root()}. Set TSASE_SRC to the checkout. "
            "The Fortran calculators are not required for this runner."
        )
    return None


def _load_classes():
    root = source_root() / "tsase"
    if "tsase" not in sys.modules:
        package = types.ModuleType("tsase")
        package.__path__ = [str(root)]
        package.__file__ = str(root / "__init__.py")
        sys.modules["tsase"] = package
    neb_name = "tsase.neb"
    if neb_name not in sys.modules:
        neb = types.ModuleType(neb_name)
        neb.__path__ = [str(root / "neb")]
        neb.__file__ = str(root / "neb" / "__init__.py")
        sys.modules[neb_name] = neb
        sys.modules["tsase"].neb = neb

    def load(name: str, filename: str):
        full = f"tsase.neb.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, root / "neb" / filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {full}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        setattr(sys.modules[neb_name], name, module)
        return module

    load("util", "util.py")
    load("minimizer_ssneb", "minimizer_ssneb.py")
    ssneb = load("ssneb", "ssneb.py")
    fire = load("fire_ssneb", "fire_ssneb.py")
    return ssneb.ssneb, fire.fire_ssneb


def _frame(reactant: np.ndarray, product: np.ndarray, box, kind: str):
    """Coordinates and a lower-triangular cell. Lennard-Jones is shifted into the cell."""
    if kind == "lj" or box is None:
        stacked = np.vstack([reactant, product])
        low = stacked.min(axis=0)
        high = stacked.max(axis=0)
        pad = 20.0
        length = max(80.0, float(np.max(high - low) + 2.0 * pad))
        shift = pad - low
        cell = np.eye(3, dtype=np.float64) * length
        return reactant + shift, product + shift, cell, None
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)
    cell = np.diag(lengths[:3].astype(np.float64))
    return reactant, product, cell, lengths[:3]


def _section(template: dict) -> dict:
    section = dict(template.get("tsase") or {})
    section.setdefault("method", "ci")
    section.setdefault("spring_k", 5.0)
    section.setdefault("ss", False)
    return section


def _job(pair, settings: dict, template: dict, seed: str) -> dict:
    section = _section(template)
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64).tolist(),
        "product": np.asarray(pair.product, dtype=np.float64).tolist(),
        "free": np.asarray(pair.free, dtype=bool).tolist(),
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64).reshape(-1).tolist(),
        "kind": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
        "n_interior": settings["n_interior"],
        "max_move": settings["max_move"],
        "method": str(section["method"]),
        "spring_k": float(section["spring_k"]),
        "seed": str(seed),
    }


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    from optbench.counter import HarnessCounter, attach_ase_counter
    from optbench.pes import evaluate

    ssneb_cls, fire_cls = _load_classes()
    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = np.asarray(job["product"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    raw_box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    left, right, cell, eval_box = _frame(reactant, product, raw_box, kind)
    counter = HarnessCounter(cache=False, cap=cap)
    error = ""
    point = None
    energy = None
    gradient = None
    band = None
    previous = Path.cwd()

    class _Surface(Calculator):
        implemented_properties = ("energy", "forces")
        nolabel = True

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
            if system_changes is None:
                system_changes = ["positions"]
            Calculator.calculate(self, atoms, properties, system_changes)
            positions = np.asarray(atoms.get_positions(), dtype=np.float64)
            value, grad = evaluate(kind, positions, eval_box)
            forces = -np.asarray(grad, dtype=np.float64)
            forces[~free] = 0.0
            self.results["energy"] = float(value)
            self.results["forces"] = forces

    try:
        with tempfile.TemporaryDirectory(prefix="tsase-") as folder:
            os.chdir(folder)
            log = Path(folder) / "stdout.txt"
            calculator = attach_ase_counter(_Surface(), cap=cap, counter=counter)
            ends = []
            for coords in (left, right):
                atoms = Atoms(positions=np.asarray(coords, dtype=np.float64), cell=cell, pbc=True)
                atoms.calc = calculator
                ends.append(atoms)
            with log.open("w", encoding="utf-8") as handle, redirect_stdout(handle):
                band = ssneb_cls(
                    ends[0],
                    ends[1],
                    numImages=int(job["n_interior"]) + 2,
                    k=float(job["spring_k"]),
                    method=job["method"],
                    ss=False,
                    parallel=False,
                )
                opt = fire_cls(band, maxmove=float(job["max_move"]))
                original = opt.step

                def step():
                    original()
                    for image in band.path[1:-1]:
                        coords = image.get_positions()
                        coords[~free] = left[~free]
                        image.set_positions(coords)

                opt.step = step
                opt.minimize(forceConverged=float(job["force_norm"]), maxIterations=cap)
            index = int(getattr(band, "Umaxi", 1))
            index = min(max(index, 1), len(band.path) - 2)
            chosen_atoms = band.path[index]
            chosen = np.asarray(chosen_atoms.get_positions(), dtype=np.float64)
            chosen[~free] = left[~free]
            chosen_atoms.set_positions(chosen)
            point = chosen
            forces = np.asarray(chosen_atoms.get_forces(), dtype=np.float64)
            energy = float(chosen_atoms.get_potential_energy())
            gradient = -forces
    except SystemExit as exc:
        error = f"SystemExit: {exc}"
    except Exception as exc:
        message = str(exc)
        if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
            error = f"force-call cap: {message}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        if point is None and band is not None:
            try:
                index = int(getattr(band, "Umaxi", 1))
                index = min(max(index, 1), len(band.path) - 2)
                point = np.asarray(band.path[index].get_positions(), dtype=np.float64)
            except Exception:
                point = None
    finally:
        os.chdir(previous)
    prefix: list[str] = []
    if error:
        if "force-call cap" in error:
            prefix.append("cap")
        else:
            prefix.append("crash")
    record = judge_saddle(
        job["name"],
        point,
        None if energy is None else float(energy),
        None if gradient is None else np.asarray(gradient, dtype=np.float64),
        counter=counter,
        cap=cap,
        force_norm=float(job["force_norm"]),
        kind=kind,
        free=free,
        box=eval_box,
        prefix=prefix,
    )
    if error:
        record["error"] = error
    record["seed"] = job["seed"]
    record["neb_method"] = job["method"]
    return record


def _load_dimer():
    _load_classes()
    root = source_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib

    importlib.import_module("tsase.io")
    module = importlib.import_module("tsase.dimer.ssdimer")
    return module.SSDimer_atoms


def _load_minimizer(method: str):
    import ase.io.trajectory as trajectory

    if not hasattr(trajectory, "PickleTrajectory"):

        class PickleTrajectory:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        trajectory.PickleTrajectory = PickleTrajectory
    _load_classes()
    root = source_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib

    if method == "mdmin":
        return importlib.import_module("tsase.optimize.mdmin").MDMin
    if method == "lbfgs":
        return importlib.import_module("tsase.optimize.sdlbfgs").SDLBFGS
    raise ValueError(f"TSASE has no minimizer {method}")


def _surface(kind: str, free: np.ndarray, box):
    from ase.calculators.calculator import Calculator

    from optbench.pes import evaluate

    class _Surface(Calculator):
        implemented_properties = ("energy", "forces")
        nolabel = True

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
            if system_changes is None:
                system_changes = ["positions"]
            Calculator.calculate(self, atoms, properties, system_changes)
            positions = np.asarray(atoms.get_positions(), dtype=np.float64)
            value, grad = evaluate(kind, positions, box)
            forces = -np.asarray(grad, dtype=np.float64)
            forces[~free] = 0.0
            self.results["energy"] = float(value)
            self.results["forces"] = forces

    return _Surface


def _execute_dimer(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ase import Atoms

    from optbench.counter import HarnessCounter, attach_ase_counter

    dimer_cls = _load_dimer()
    positions = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    counter = HarnessCounter(cache=False, cap=cap)
    task = str(job["task"])
    error = ""
    overlap = 0.0
    energy = None
    gradient = None
    point = positions.copy()
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="tsase-dimer-") as folder:
            os.chdir(folder)
            log = Path(folder) / "stdout.txt"
            atoms = Atoms(positions=positions, cell=_cell(positions, box), pbc=box is not None)
            atoms.calc = attach_ase_counter(_surface(kind, free, box)(), cap=cap, counter=counter)
            rng = np.random.default_rng(int(job["zseed"]))
            mode = rng.normal(size=positions.shape)
            mode[~free] = 0.0
            scale = float(np.linalg.norm(mode))
            mode = mode / scale
            with log.open("w", encoding="utf-8") as handle, redirect_stdout(handle):
                dimer = dimer_cls(
                    R0=atoms,
                    mode=mode,
                    maxStep=float(job["max_step"]),
                    dR=float(job["dr"]),
                    phi_tol=float(job["phi_tol"]),
                    rotationMax=int(job["rotation_max"]),
                    ss=False,
                    noZeroModes=bool(np.all(free)),
                    rotationOpt="cg",
                )
                stalled = 0
                previous_overlap = None
                while counter.calls < cap:
                    before = counter.calls
                    if task == "mode":
                        dimer.get_forces()
                        current = np.asarray(dimer.N, dtype=np.float64)
                        piece = current[: positions.shape[0]][free].reshape(-1)
                        scale = float(np.linalg.norm(piece))
                        target = np.asarray(job["reference"], dtype=np.float64)
                        if scale > 1e-14:
                            overlap = abs(float(np.dot(piece / scale, target / np.linalg.norm(target))))
                        if overlap >= float(job["overlap"]):
                            break
                        if previous_overlap is not None and abs(overlap - previous_overlap) < 1e-6:
                            stalled += 1
                        else:
                            stalled = 0
                        previous_overlap = overlap
                        if stalled >= 3:
                            break
                    else:
                        dimer.step()
                        coords = atoms.get_positions()
                        coords[~free] = positions[~free]
                        atoms.set_positions(coords)
                        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
                        if float(np.linalg.norm(forces[free])) <= float(job["force_norm"]) and float(getattr(dimer, "curvature", 1.0)) < 0.0:
                            break
                    if counter.calls == before:
                        break
            point = np.asarray(atoms.get_positions(), dtype=np.float64)
            point[~free] = positions[~free]
            atoms.set_positions(point)
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            energy = float(atoms.get_potential_energy())
            gradient = -forces
    except Exception as exc:
        message = str(exc)
        if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
            error = f"force-call cap: {message}"
        else:
            error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
    if task == "mode":
        record = judge_mode(job["name"], int(counter.calls), cap, overlap, float(job["overlap"]), error)
        record["seed"] = "0"
        return record
    prefix: list[str] = []
    if error:
        prefix.append("cap" if "force-call cap" in error else "crash")
    record = judge_saddle(
        job["name"],
        point,
        None if energy is None else float(energy),
        None if gradient is None else np.asarray(gradient, dtype=np.float64),
        counter=counter,
        cap=cap,
        force_norm=float(job["force_norm"]),
        kind=kind,
        free=free,
        box=box,
        prefix=prefix,
    )
    if error:
        record["error"] = error
    record["seed"] = "0"
    return record


def _cell(positions: np.ndarray, box) -> np.ndarray:
    if box is None:
        span = float(np.max(np.abs(positions))) if positions.size else 1.0
        return np.eye(3) * max(80.0, 4.0 * span + 10.0)
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)
    return np.diag(lengths[:3])


def _dimer_job(start: dict, settings: dict, template: dict) -> dict:
    section = dict(template.get("tsase") or {})
    return {
        "name": start["name"],
        "positions": start["positions"],
        "free": [not bool(flag) for flag in start["fixed"]],
        "box": start["box"],
        "reference": start.get("reference"),
        "kind": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
        "overlap": settings["overlap"],
        "task": settings["task"],
        "max_step": float(section.get("max_step", 0.2)),
        "dr": float(section.get("dr", 0.001)),
        "phi_tol": float(section.get("phi_tol", 5.0)),
        "rotation_max": int(section.get("rotation_max", 4)),
        "zseed": pair_seed(f"0:{start['name']}", 20261015),
    }


def _execute_minimize(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ase import Atoms

    from optbench.counter import HarnessCounter, attach_ase_counter

    method = str(job["method"])
    opt_cls = _load_minimizer(method)
    positions = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    counter = HarnessCounter(cache=True, cap=cap)
    error = ""
    energy = None
    gradient = None
    point = positions.copy()
    try:
        atoms = Atoms(positions=positions, cell=_cell(positions, box), pbc=box is not None)
        atoms.calc = attach_ase_counter(_surface(kind, free, box)(), cap=cap, counter=counter)
        kwargs = {"logfile": None, "trajectory": None}
        if method == "lbfgs":
            kwargs["maxstep"] = float(job["max_step"])
        else:
            kwargs["maxstep"] = float(job["max_step"])
        opt = opt_cls(atoms, **kwargs)
        stalled = 0
        while counter.calls < cap:
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            if float(np.linalg.norm(forces[free])) <= float(job["force_norm"]):
                break
            before_pos = np.asarray(atoms.get_positions(), dtype=np.float64)
            opt.step(forces)
            coords = atoms.get_positions()
            coords[~free] = positions[~free]
            atoms.set_positions(coords)
            moved = float(np.linalg.norm(np.asarray(atoms.get_positions()) - before_pos))
            stalled = stalled + 1 if moved < 1e-12 else 0
            if stalled >= 3:
                break
        point = np.asarray(atoms.get_positions(), dtype=np.float64)
        point[~free] = positions[~free]
        atoms.set_positions(point)
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        energy = float(atoms.get_potential_energy())
        gradient = -forces
    except Exception as exc:
        message = str(exc)
        if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
            error = f"force-call cap: {message}"
        else:
            error = f"{type(exc).__name__}: {exc}"
    prefix: list[str] = []
    if error:
        prefix.append("cap" if "force-call cap" in error else "crash")
    # A periodic pair potential is invariant to translation. Rotations stay
    # in the count: the cell breaks them.
    rotations = 0 if box is not None and bool(np.all(free)) else None
    record = judge_saddle(
        job["name"],
        point,
        None if energy is None else float(energy),
        None if gradient is None else np.asarray(gradient, dtype=np.float64),
        counter=counter,
        cap=cap,
        force_norm=float(job["force_norm"]),
        kind=kind,
        free=free,
        box=box,
        prefix=prefix,
        expected_index=0,
        rotations=rotations,
    )
    if error:
        record["error"] = error
    record["seed"] = "0"
    record["minimizer"] = method
    return record


def search_named(root: Path, problem_name: str, template_name: str, start: str, seed: str = "0") -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    problem, template, settings = prepared(root, problem_name, template_name, CODE)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(pairs[start], settings, template, seed))
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def _run_points(root, problem, template, settings, output: Path, workers: int, executor) -> dict:
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    if settings["task"] == "minimize":
        section = dict(template.get("tsase") or {})
        jobs = []
        for start in starts:
            jobs.append(
                {
                    "name": start["name"],
                    "positions": start["positions"],
                    "free": [not bool(flag) for flag in start["fixed"]],
                    "box": start["box"],
                    "kind": settings["potential"],
                    "cap": settings["cap"],
                    "force_norm": settings["force_norm"],
                    "method": str(section.get("method")),
                    "max_step": float(section.get("max_step", 0.2)),
                }
            )
        comment = (
            "# counter harness. One force call is one ASE calculate.\n"
            "# The stop is the Euclidean force norm and Hessian index 0.\n"
            "# TSASE's own step counter is not this count."
        )
    else:
        jobs = [_dimer_job(start, settings, template) for start in starts]
        comment = (
            "# counter harness. One force call is one ASE calculate inside the solid-state dimer.\n"
            "# TSASE forceCalls is not this count.\n"
            "# A mode success is the problem overlap. A saddle success is Hessian index 1."
        )
    records = run_jobs(jobs, executor, workers)
    parameters = {
        "potential": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
        "task": settings["task"],
        "counter": "harness",
        "seed": "0",
        **dict(template.get("tsase") or {}),
    }
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    method = str((template.get("tsase") or {}).get("method", "ci"))
    # quenchDimer is the 2014 AKMC client. This module runs ssneb, the
    # solid-state dimer, and the two minimizers. Refusing here avoids
    # treating that client as a band.
    if method not in {"ci", "normal", "dimer", "lbfgs", "mdmin"}:
        raise ValueError(f"{template_name} method {method} is not a search this module runs")
    reason = available()
    if reason:
        raise RuntimeError(reason)
    if method == "dimer":
        settings = point_settings(problem)
        if settings["cap"] is None:
            raise ValueError(f"{problem['id']} has no force-call cap")
        settings["task"] = "mode" if problem["suite"] == "lowest-eigenvector" else "saddle"
        return _run_points(root, problem, template, settings, output, workers, _execute_dimer)
    if method in {"lbfgs", "mdmin"}:
        settings = point_settings(problem)
        if settings["cap"] is None:
            raise ValueError(f"{problem['id']} has no force-call cap")
        settings["task"] = "minimize"
        return _run_points(root, problem, template, settings, output, workers, _execute_minimize)
    problem, template, settings = prepared(root, problem_name, template_name, CODE)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_job(pair, settings, template, "0") for pair in pairs]
    records = run_jobs(jobs, _execute, workers)
    for record in records:
        record["problem"] = problem["id"]
        record["template"] = template["id"]
    section = _section(template)
    parameters = {
        **settings,
        "method": section["method"],
        "spring_k": section["spring_k"],
        "spring_k_times_images": "ssneb stores k * numImages",
        "ss": False,
        "minimizer": "fire_ssneb",
        "counter": "harness",
        "seed": "0",
    }
    comment = (
        "# counter harness. One force call is one ASE calculate on the shared calculator.\n"
        "# TSASE forceCalls is not this count.\n"
        "# ssneb multiplies the spring constant by the image count.\n"
        "# A success is under the cap, at the potential force stop, and Hessian index 1."
    )
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=settings["n_interior"],
        comment=comment,
    )
