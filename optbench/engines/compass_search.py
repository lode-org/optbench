"""COMPASS double-ended search through an ASE calculator.

The calculator is the score. COMPASS does not count its own calls.
The library treats any periodic cell as a variable cell. These problems
keep the given cell, so ``is_varshape`` stays false. Fixed atoms are
written back on every ``set_positions``. The chain is the one COMPASS
builds. It does not use the problem's interior image count.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from .common import cached_pairs, judge_saddle, prepared, run_jobs, write_campaign

ENGINE = "compass"
CODE = "compass"


def source_root() -> Path:
    env = os.environ.get("COMPASS_SRC")
    if env:
        return Path(env)
    return Path("/tmp/src/compass")


def available() -> str | None:
    script = source_root() / "compass.py"
    if not script.is_file():
        return (
            f"COMPASS compass.py is not under {source_root()}. "
            "The pin is 1.0.0, commit 22da7bd. The ASE calculator is the harness score."
        )
    return None


def _import_compass():
    """Load COMPASS. ASE 3.29 moved the cell filters out of ``ase.constraints``."""
    import ase.constraints as constraints
    from ase.filters import ExpCellFilter, UnitCellFilter

    if not hasattr(constraints, "ExpCellFilter"):
        constraints.ExpCellFilter = ExpCellFilter
    if not hasattr(constraints, "UnitCellFilter"):
        constraints.UnitCellFilter = UnitCellFilter
    root = str(source_root())
    if root not in sys.path:
        sys.path.append(root)
    from compass import Compass

    return Compass


def _cell(positions: np.ndarray, box) -> tuple[np.ndarray, bool]:
    if box is None:
        span = float(np.max(np.abs(positions))) if positions.size else 1.0
        return np.eye(3) * max(80.0, 4.0 * span + 10.0), False
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)
    return np.diag(lengths[:3].astype(np.float64)), True


def _lock(atoms, reference: np.ndarray, free: np.ndarray) -> None:
    original = atoms.set_positions

    def set_positions(positions, **kwargs):
        coords = np.array(positions, dtype=np.float64, copy=True)
        coords[~free] = reference[~free]
        return original(coords, **kwargs)

    atoms.set_positions = set_positions


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


def _job(pair, settings: dict) -> dict:
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64).tolist(),
        "product": np.asarray(pair.product, dtype=np.float64).tolist(),
        "free": np.asarray(pair.free, dtype=bool).tolist(),
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64).reshape(-1).tolist(),
        "kind": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
    }


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ase import Atoms

    from optbench.counter import HarnessCounter, attach_ase_counter

    compass_cls = _import_compass()
    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = np.asarray(job["product"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    cell, periodic = _cell(np.vstack([reactant, product]), box)
    counter = HarnessCounter(cache=False, cap=cap)
    error = ""
    point = None
    energy = None
    gradient = None
    images: list = []
    previous = Path.cwd()

    class FixedCell(compass_cls):
        def is_varshape(self):
            return False

    try:
        with tempfile.TemporaryDirectory(prefix="compass-") as folder:
            os.chdir(folder)
            log = Path(folder) / "stdout.txt"
            surface = _surface(kind, free, box)
            images = []
            for coords in (reactant, product):
                atoms = Atoms(
                    numbers=np.ones(coords.shape[0], dtype=int),
                    positions=np.asarray(coords, dtype=np.float64),
                    cell=cell,
                    pbc=periodic,
                )
                atoms.calc = attach_ase_counter(surface(), cap=cap, counter=counter)
                _lock(atoms, reactant, free)
                images.append(atoms)
            with log.open("w", encoding="utf-8") as handle, redirect_stdout(handle):
                # The images stop `targetdistance*` apart. On LJ38 that separation
                # has to be the force stop before the returned geometry is stationary.
                separation = float(job["force_norm"])
                search = FixedCell(
                    images[0],
                    images[1],
                    forcetol=separation,
                    minimize_fmax=separation,
                    targetdistancesaddle=separation,
                    targetdistanceother=separation,
                    minimize_nsteps=cap,
                    nsteps=cap,
                    restart_level=0,
                    parallel=False,
                    basename="clog_",
                )
                nodes = search.calc_initial_nodes()
                barrier, _saddles = search.run(nodes)
            if barrier is not None:
                point = np.asarray(barrier.get_positions(), dtype=np.float64)
                point[~free] = reactant[~free]
    except Exception as exc:
        message = str(exc)
        if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
            error = f"force-call cap: {message}"
        else:
            error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
    if point is not None and images and counter.calls < cap and not error:
        images[0].set_positions(point)
        try:
            energy = float(images[0].get_potential_energy())
            gradient = -np.asarray(images[0].get_forces(), dtype=np.float64)
        except Exception as exc:
            message = str(exc)
            if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
                error = f"force-call cap: {message}"
            else:
                error = f"{type(exc).__name__}: {exc}"
    prefix: list[str] = []
    if error:
        prefix.append("cap" if "force-call cap" in error else "crash")
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
        rotations=rotations,
    )
    if error and not record["success"]:
        record["error"] = error
    record["seed"] = "0"
    return record


def plan(root: Path, problem_name: str, template_name: str, work: Path) -> dict:
    """The jobs ``run`` sends to ``run_jobs``, one per pair, in problem order."""
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    problem, template, settings = prepared(root, problem_name, template_name, CODE)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    if problem.get("suite") not in {"neb", "double-ended-saddle-search"}:
        raise RuntimeError(
            f"{problem['id']} is not a double-ended band. "
            "COMPASS searches between two endpoints."
        )
    pairs = list(cached_pairs(root, problem, settings).values())
    return {
        "problem": problem,
        "template": template,
        "settings": settings,
        "jobs": [_job(pair, settings) for pair in pairs],
        "execute": _execute,
    }


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    planned = plan(root, problem_name, template_name, output.parent / "work")
    problem, template, settings = planned["problem"], planned["template"], planned["settings"]
    records = run_jobs(planned["jobs"], planned["execute"], workers)
    comment = (
        "# counter harness. COMPASS calls the ASE calculator, and that calculator calls pes.evaluate.\n"
        "# forcetol, minimize_fmax, and both target distances are the problem force stop.\n"
        "# The cell is fixed. COMPASS would otherwise treat any periodic boundary as a variable cell.\n"
        "# A success is under the cap, at the force stop, and Hessian index 1."
    )
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        {
            "forcetol": settings["force_norm"],
            "minimize_fmax": settings["force_norm"],
            "targetdistancesaddle": settings["force_norm"],
            "targetdistanceother": settings["force_norm"],
            "fixed_cell": True,
            "counter": "harness",
            "potential": settings["potential"],
        },
        images=None,
        comment=comment,
    )
