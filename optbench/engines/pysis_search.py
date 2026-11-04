"""pysisyphus Cartesian optimizers and Lanczos, scored on the harness.

The optimizers keep their Bohr defaults, including the step limit. The
calculator converts Bohr coordinates back to the harness unit before
``pes.evaluate`` and returns ``dE/dx`` per Bohr. Fixed atoms are restored
in the harness unit. The root-mean-square stop is the harness Euclidean
force stop, multiplied by the bohr-to-angstrom factor and divided by
``sqrt`` of the Cartesian components. Only that root-mean-square is
checked, so the harness norm matches the problem. Lanczos stays in the
harness unit. Its displacement and gradient are projected off the six
rigid modes of a free cluster. The finite-difference step and the
eigenvalue threshold are the library defaults.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import zlib
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from .common import judge_mode, judge_saddle, write_campaign
from .geometries import load_starts, point_settings

ENGINE = "pysisyphus"
CODE = "pysisyphus"
_MINIMIZERS = ("lbfgs", "cg", "sd")
_READY = False


def site_packages() -> Path:
    env = os.environ.get("PYSIS_SITE")
    if env:
        return Path(env)
    return Path("/tmp/venv-pysis/lib/python3.12/site-packages")


def available() -> str | None:
    package = site_packages() / "pysisyphus"
    if not package.is_dir():
        return (
            "pysisyphus is not installed at "
            f"{site_packages()}. The pin is commit a4ce10dd. "
            "The installed tree reports version 0.1.dev1 because that clone has no tag. "
            "The harness calculator is the score."
        )
    return None


def _section(template: dict) -> dict:
    section = dict(template.get("pysisyphus") or {})
    method = str(section.get("method", "lbfgs"))
    known = _MINIMIZERS + ("lanczos",)
    if method not in known:
        raise ValueError(f"{template['id']} method {method} is not one of {', '.join(known)}")
    section["method"] = method
    return section


def _prepare() -> None:
    global _READY
    if _READY:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    site = str(site_packages())
    if site not in sys.path:
        sys.path.insert(0, site)
    import pysisyphus

    pysisyphus.logger.setLevel(logging.ERROR)
    for handler in list(pysisyphus.logger.handlers):
        if type(handler) is logging.StreamHandler:
            pysisyphus.logger.removeHandler(handler)
    _READY = True


def _optimizer(method: str):
    _prepare()
    if method == "lbfgs":
        from pysisyphus.optimizers.LBFGS import LBFGS

        return LBFGS
    if method == "cg":
        from pysisyphus.optimizers.ConjugateGradient import ConjugateGradient

        return ConjugateGradient
    if method == "sd":
        from pysisyphus.optimizers.SteepestDescent import SteepestDescent

        return SteepestDescent
    raise ValueError(method)


def _rigid_basis(coords: np.ndarray, rotations: int) -> np.ndarray:
    """Orthonormal rigid motions. An empty basis leaves the vector unchanged."""
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    n_atoms = points.shape[0]
    width = n_atoms * 3
    if n_atoms < 2 or rotations < 0:
        return np.zeros((width, 0), dtype=np.float64)
    centered = points - points.mean(axis=0)
    seeds: list[np.ndarray] = []
    for axis in range(3):
        translation = np.zeros(width, dtype=np.float64)
        translation[axis::3] = 1.0
        seeds.append(translation)
        if rotations:
            rotation = np.cross(centered, np.eye(3)[axis])
            seeds.append(rotation.reshape(-1))
    basis: list[np.ndarray] = []
    for seed in seeds:
        vector = np.array(seed, dtype=np.float64, copy=True)
        for previous in basis:
            vector -= previous * float(np.dot(previous, vector))
        scale = float(np.linalg.norm(vector))
        if scale > 1e-8:
            basis.append(vector / scale)
    if not basis:
        return np.zeros((width, 0), dtype=np.float64)
    return np.column_stack(basis)


def _project(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if basis.size == 0:
        return values
    return values - basis @ (basis.T @ values)


def _remember_eval(counter, kind: str, xyz: np.ndarray, box, key: tuple[float, ...]):
    from optbench.pes import evaluate

    value, grad = evaluate(kind, xyz, box)
    gradient = np.asarray(grad, dtype=np.float64)
    counter.remember(key, (float(value), gradient))
    return float(value), gradient


def _lookup(counter, kind: str, xyz: np.ndarray, box):
    from optbench.counter import ForceCallCap

    key = tuple(float(value) for value in np.asarray(xyz, dtype=np.float64).reshape(-1))
    fresh, cached = counter.charge(key)
    if fresh:
        energy, gradient = _remember_eval(counter, kind, xyz, box, key)
    else:
        energy, gradient = cached
        gradient = np.asarray(gradient, dtype=np.float64)
    return float(energy), gradient, ForceCallCap


def _execute(job: dict) -> dict:
    if job["method"] == "lanczos":
        return _execute_lanczos(job)
    return _execute_minimize(job)


def _execute_minimize(job: dict) -> dict:
    from optbench.counter import ForceCallCap, HarnessCounter

    positions = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    ftol = float(job["force_norm"])
    counter = HarnessCounter(cache=True, cap=cap)
    holder: dict = {}
    error = ""
    capped = False
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="pysis-min-") as folder:
            os.chdir(folder)
            _prepare()
            from pysisyphus.Geometry import Geometry
            from pysisyphus.calculators.Calculator import Calculator
            from pysisyphus.constants import ANG2BOHR, BOHR2ANG
            from pysisyphus.optimizers.exceptions import ZeroStepLength

            class HarnessCalc(Calculator):
                def __init__(self):
                    super().__init__(base_name="harness")

                def get_forces(self, atoms, coords, **kwargs):
                    del atoms, kwargs
                    bohr = np.array(coords, dtype=np.float64, copy=True).reshape(-1, 3)
                    xyz = bohr * BOHR2ANG
                    xyz[~free] = positions[~free]
                    raw = np.asarray(coords)
                    raw[...] = (xyz * ANG2BOHR).reshape(raw.shape)
                    try:
                        energy, gradient, _cap = _lookup(counter, kind, xyz, box)
                    except ForceCallCap as exc:
                        holder["capped"] = True
                        holder["error"] = f"force-call cap: {exc}"
                        raise
                    # dE/dx_bohr = dE/dx_harness * BOHR2ANG. pysisyphus steps on the force.
                    reported = -np.asarray(gradient, dtype=np.float64) * BOHR2ANG
                    reported[~free] = 0.0
                    holder["point"] = xyz
                    holder["energy"] = float(energy)
                    holder["gradient"] = np.asarray(gradient, dtype=np.float64)
                    return {"energy": float(energy), "forces": reported.reshape(-1)}

            geom = Geometry(["H"] * positions.shape[0], (positions * ANG2BOHR).reshape(-1))
            geom.set_calculator(HarnessCalc())
            width = max(int(positions.size), 1)
            optimizer = _optimizer(job["method"])
            opt = optimizer(
                geom,
                max_cycles=cap,
                rms_force=(ftol * BOHR2ANG) / np.sqrt(width),
                rms_force_only=True,
                dump=False,
                out_dir=folder,
                logging_level=logging.ERROR,
            )
            log = Path(folder) / "stdout.txt"
            with log.open("w", encoding="utf-8") as handle, redirect_stdout(handle):
                try:
                    for _cycle in opt.run_generator:
                        placed = np.array(geom.coords3d, dtype=np.float64, copy=True) * BOHR2ANG
                        placed[~free] = positions[~free]
                        geom.coords3d = placed * ANG2BOHR
                except ZeroStepLength:
                    holder["stopped"] = "zero-step"
                except ForceCallCap as exc:
                    capped = True
                    error = f"force-call cap: {exc}"
            if holder.get("capped"):
                capped = True
                error = str(holder.get("error") or error)
            opt_logger = logging.getLogger("pysis.opt")
            for handler in list(opt_logger.handlers):
                handler.close()
                opt_logger.removeHandler(handler)
            final = np.array(geom.coords3d, dtype=np.float64, copy=True) * BOHR2ANG
            final[~free] = positions[~free]
            stored = holder.get("point")
            if not capped and (stored is None or not np.allclose(final, stored)):
                try:
                    energy, gradient, _cap = _lookup(counter, kind, final, box)
                    holder["point"] = final
                    holder["energy"] = float(energy)
                    holder["gradient"] = gradient
                except ForceCallCap as exc:
                    capped = True
                    error = f"force-call cap: {exc}"
    except Exception as exc:
        if "force-call cap" in str(exc) or type(exc).__name__ == "ForceCallCap":
            capped = True
            error = f"force-call cap: {exc}"
        else:
            error = f"{type(exc).__name__}: {exc}"
            holder["crashed"] = True
    finally:
        opt_logger = logging.getLogger("pysis.opt")
        for handler in list(opt_logger.handlers):
            handler.close()
            opt_logger.removeHandler(handler)
        os.chdir(previous)
    point = holder.get("point")
    energy = holder.get("energy")
    gradient = holder.get("gradient")
    prefix: list[str] = []
    if capped or counter.calls >= cap:
        prefix.append("cap")
    elif holder.get("crashed") or (error and "zero-step" not in str(holder.get("stopped") or "")):
        if error:
            prefix.append("crash")
    rotations = 0 if box is not None and bool(np.all(free)) else None
    record = judge_saddle(
        job["name"],
        None if point is None else np.asarray(point, dtype=np.float64),
        None if energy is None else float(energy),
        None if gradient is None else np.asarray(gradient, dtype=np.float64),
        counter=counter,
        cap=cap,
        force_norm=ftol,
        kind=kind,
        free=free,
        box=box,
        prefix=prefix,
        expected_index=0,
        rotations=rotations,
    )
    if error and not record["success"]:
        record["error"] = error
    record["method"] = job["method"]
    return record


def _execute_lanczos(job: dict) -> dict:
    from optbench.counter import ForceCallCap, HarnessCounter

    positions = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    counter = HarnessCounter(cache=True, cap=cap)
    reference = np.asarray(job["reference"], dtype=np.float64).reshape(-1)
    free_coords = positions[free].reshape(-1)
    rotations = 3 if box is None and bool(np.all(free)) else 0
    basis = _rigid_basis(free_coords, rotations) if rotations else np.zeros((free_coords.size, 0))
    error = ""
    overlap = 0.0
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="pysis-lanczos-") as folder:
            os.chdir(folder)
            _prepare()
            from pysisyphus.modefollow.lanczos import lanczos

            def grad_getter(displacement: np.ndarray) -> np.ndarray:
                disp = _project(np.asarray(displacement, dtype=np.float64), basis)
                xyz = np.array(positions, dtype=np.float64, copy=True)
                xyz[free] = xyz[free] + disp.reshape(-1, 3)
                _energy, gradient, _cap = _lookup(counter, kind, xyz, box)
                return _project(np.asarray(gradient, dtype=np.float64)[free], basis)

            seed = zlib.crc32(str(job["name"]).encode("utf-8")) & 0xFFFFFFFF
            guess = np.random.default_rng(seed).normal(size=free_coords.size)
            guess = _project(guess, basis)
            scale = float(np.linalg.norm(guess))
            if scale < 1e-14:
                guess = np.zeros(free_coords.size, dtype=np.float64)
                guess[0] = 1.0
                guess = _project(guess, basis)
                scale = float(np.linalg.norm(guess))
            guess = guess / scale
            _value, vector, _converged = lanczos(free_coords.size, grad_getter, guess=guess)
            piece = _project(np.asarray(vector, dtype=np.float64), basis)
            scale = float(np.linalg.norm(piece))
            target_scale = float(np.linalg.norm(reference))
            if scale > 1e-14 and target_scale > 1e-14 and piece.size == reference.size:
                overlap = abs(float(np.dot(piece / scale, reference / target_scale)))
    except ForceCallCap as exc:
        error = f"force-call cap: {exc}"
    except Exception as exc:
        message = str(exc)
        if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
            error = f"force-call cap: {message}"
        else:
            error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
    record = judge_mode(job["name"], int(counter.calls), cap, overlap, float(job["overlap"]), error)
    record["method"] = "lanczos"
    return record


def _jobs(starts: list[dict], settings: dict, template: dict) -> list[dict]:
    section = _section(template)
    rows = []
    for index, start in enumerate(starts):
        row = {
            "name": start["name"],
            "index": int(index),
            "positions": start["positions"],
            "free": [not bool(flag) for flag in start["fixed"]],
            "box": start["box"],
            "kind": settings["potential"],
            "cap": int(settings["cap"]),
            "force_norm": float(settings["force_norm"]),
            "overlap": float(settings["overlap"]),
            "method": section["method"],
        }
        if start.get("reference") is not None:
            row["reference"] = start["reference"]
        rows.append(row)
    return rows


def _collect(jobs: list[dict], workers: int) -> list[dict]:
    if len(jobs) <= 1:
        records = [_execute(job) for job in jobs]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            records = list(pool.map(_execute, jobs))
    order = {job["name"]: index for index, job in enumerate(jobs)}
    records.sort(key=lambda row: order[row["start"]])
    return records


def plan(root: Path, problem_name: str, template_name: str, work: Path) -> dict:
    """The jobs ``run`` sends to ``_collect``, one per archived start, in archive order."""
    reason = available()
    if reason:
        raise RuntimeError(reason)
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
    section = _section(template)
    settings = point_settings(problem)
    if settings["cap"] is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    if section["method"] == "lanczos":
        if problem.get("suite") != "lowest-eigenvector":
            raise RuntimeError(f"{problem['id']} is not a lowest-eigenvector problem")
    elif section["method"] in _MINIMIZERS:
        if problem.get("suite") != "minimization":
            raise RuntimeError(
                f"{problem['id']} is not a minimization. "
                "The dimer and NEB are separate pysisyphus tasks."
            )
    else:
        raise RuntimeError(f"{template['id']} has no runner")
    scratch = Path(work) / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    return {
        "problem": problem,
        "template": template,
        "settings": settings,
        "jobs": _jobs(starts, settings, template),
        "execute": _execute,
    }


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    planned = plan(root, problem_name, template_name, output.parent / "work")
    problem, template, settings = planned["problem"], planned["template"], planned["settings"]
    section = _section(template)
    if section["method"] == "lanczos":
        comment = (
            "# counter harness. pysisyphus Lanczos calls pes.evaluate.\n"
            "# dx is 5e-3 and dl is 1e-2. A free cluster drops six rigid modes.\n"
            "# The guess is seeded from the start name. Success is absolute overlap."
        )
        parameters = {"method": "lanczos", "dx": 5e-3, "dl": 1e-2, "max_cycles": 25, "counter": "harness"}
    else:
        comment = (
            "# counter harness. The pysisyphus calculator calls pes.evaluate.\n"
            f"# method {section['method']}. Coordinates inside the optimizer are Bohr.\n"
            "# The RMS stop matches the harness Euclidean force stop.\n"
            "# A success is under the cap, at the force stop, and Hessian index 0."
        )
        parameters = {
            "method": section["method"],
            "rms_force_only": True,
            "coordinates": "bohr",
            "counter": "harness",
        }
    records = _collect(planned["jobs"], workers)
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
