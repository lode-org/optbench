"""LAMMPS minimizers through ``fix external pf/callback``.

The pair style is ``zero``. Every force the minimizer uses is written
by the callback from the harness potential, and that callback is the
score. A native ``pair_style lj/cut`` or ``morse`` evaluation is not a
harness call, so it is not used.

``min_style`` is one of ``cg``, ``sd``, ``hftn``, ``quickmin``, and
``fire``. NEB, PRD, TAD, and hyperdynamics are not this runner: the
Al and Pt problems need an EAM evaluator the harness does not have.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np

from .common import judge_saddle, run_jobs, write_campaign
from .geometries import load_starts, point_settings

ENGINE = "lammps"
CODE = "lammps"
_STYLES = ("cg", "sd", "hftn", "quickmin", "fire")


def library_path() -> Path:
    env = os.environ.get("LAMMPS_LIB")
    if env:
        return Path(env)
    return Path("/tmp/src/lammps/build/liblammps.so")


def python_dir() -> Path:
    env = os.environ.get("LAMMPS_PYTHON")
    if env:
        return Path(env)
    return Path("/tmp/src/lammps/python")


def available() -> str | None:
    library = library_path()
    if not library.is_file():
        return (
            f"liblammps.so is not at {library}. The pin is stable_30Sep2026. "
            "Build the shared library with PYTHON, EXTRA-PAIR, MANYBODY, REPLICA, and PHONON."
        )
    package = python_dir() / "lammps"
    if not package.is_dir():
        return f"the LAMMPS Python module is not at {package}."
    return None


def _link_library() -> None:
    package = python_dir() / "lammps"
    link = package / "liblammps.so"
    target = library_path().resolve()
    if link.is_symlink() or link.is_file():
        if link.resolve() == target:
            return
        link.unlink()
    link.symlink_to(target)


def _section(template: dict) -> dict:
    section = dict(template.get("lammps") or {})
    style = str(section.get("min_style", "cg"))
    if style not in _STYLES:
        raise ValueError(f"{template['id']} min_style {style} is not one of {', '.join(_STYLES)}")
    section["min_style"] = style
    return section


def _noise(template: dict) -> dict:
    section = template.get("noise")
    if not section:
        return {"sigma": 0.0, "kappa": 3.0, "window": 5, "seed": 0}
    return {
        "sigma": float(section["sigma"]),
        "kappa": float(section.get("kappa", 3.0)),
        "window": int(section.get("window", 5)),
        "seed": int(section["seed"]),
    }


def _job(start: dict, settings: dict, template: dict, index: int) -> dict:
    section = _section(template)
    noise = _noise(template)
    return {
        "name": start["name"],
        "index": int(index),
        "positions": start["positions"],
        "free": [not bool(flag) for flag in start["fixed"]],
        "box": start["box"],
        "kind": settings["potential"],
        "cap": int(settings["cap"]),
        "force_norm": float(settings["force_norm"]),
        "min_style": section["min_style"],
        "sigma": noise["sigma"],
        "kappa": noise["kappa"],
        "window": noise["window"],
        "noise_seed": noise["seed"] + int(index),
    }


def _place(positions: np.ndarray, box):
    """Coordinates LAMMPS will read, the harness box, and the boundary style.

    A missing box is a free cluster: non-periodic, with room to move.
    A periodic cell is wrapped into the box so ``read_data`` accepts it.
    The harness potential applies the minimum image on that same cell.
    """
    xyz = np.asarray(positions, dtype=np.float64).copy()
    if box is None:
        low = xyz.min(axis=0) - 30.0
        high = np.maximum(xyz.max(axis=0) + 30.0, low + 1.0)
        return xyz, None, "f f f", low, high
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3].copy()
    if np.any(lengths <= 0.0):
        raise ValueError("a periodic cell needs three positive lengths")
    xyz = xyz - lengths * np.floor(xyz / lengths)
    return xyz, lengths, "p p p", np.zeros(3), lengths


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import sys

    from optbench.counter import ForceCallCap, HarnessCounter
    from optbench.failures import classify_failure
    from optbench.noise import converged_under_noise, stop_threshold
    from optbench.pes import evaluate

    _link_library()
    module = str(python_dir())
    if module not in sys.path:
        sys.path.insert(0, module)
    from lammps import lammps

    raw = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    given = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    positions, box, boundary, low, high = _place(raw, given)
    nat = int(positions.shape[0])
    sigma = float(job.get("sigma") or 0.0)
    kappa = float(job.get("kappa") or 3.0)
    window = int(job.get("window") or 5)
    components = int(np.count_nonzero(free)) * 3
    clean_tol = float(job["force_norm"])
    ftol = stop_threshold(clean_tol, sigma, components, kappa) if components else clean_tol
    counter = HarnessCounter(cache=True, cap=cap)
    error = ""
    point = positions.copy()
    energy = None
    gradient = None
    energies: list[float] = []
    noisy_norm = None
    noise_stopped = False
    rng = random.Random(int(job["noise_seed"])) if sigma > 0.0 else None
    holder: dict = {"stop": False, "capped": False}

    def callback(_caller, _step, nlocal, tag, coords, forces):
        nonlocal noisy_norm, noise_stopped
        if holder["stop"]:
            forces[:] = 0.0
            return
        try:
            if int(nlocal) != nat:
                raise RuntimeError(f"LAMMPS exposed {nlocal} atoms, the geometry has {nat}")
            ids = np.asarray(tag, dtype=np.int64) - 1
            xyz = np.zeros((nat, 3), dtype=np.float64)
            xyz[ids] = np.asarray(coords, dtype=np.float64)
            xyz[~free] = positions[~free]
            key = tuple(float(value) for value in xyz.reshape(-1))
            fresh, cached = counter.charge(key)
            if fresh:
                value, grad = evaluate(kind, xyz, box)
                grad = np.asarray(grad, dtype=np.float64)
                seen = grad
                if sigma > 0.0:
                    draw = np.fromiter(
                        (rng.gauss(0.0, sigma) for _ in range(grad.size)),
                        dtype=np.float64,
                        count=grad.size,
                    ).reshape(grad.shape)
                    draw[~free] = 0.0
                    seen = grad + draw
                    energies.append(float(value))
                counter.remember(key, (float(value), grad, seen))
            else:
                value, grad, seen = cached
                grad = np.asarray(grad, dtype=np.float64)
                seen = np.asarray(seen, dtype=np.float64)
            reported = seen
            if sigma > 0.0:
                noisy_norm = float(np.linalg.norm(seen[free]))
                noise_stopped = converged_under_noise(
                    force_norm=noisy_norm,
                    energies=energies,
                    clean_tol=clean_tol,
                    sigma=sigma,
                    components=components,
                    energy_tol=max(clean_tol, kappa * sigma),
                    kappa=kappa,
                    window=window,
                )
                # LAMMPS stops when the force it sees is under ftol. Hold the
                # reported norm just above that floor until the energy window
                # is also met, so one lucky sample does not end the search.
                if noisy_norm <= ftol and not noise_stopped and noisy_norm > 0.0:
                    reported = np.array(seen, copy=True)
                    reported[free] *= (1.01 * ftol) / noisy_norm
            local = -np.asarray(reported, dtype=np.float64)[ids]
            local[~free[ids]] = 0.0
            forces[:] = local
            holder["lmp"].fix_external_set_energy_global("ext", float(value))
            holder["point"] = xyz
            holder["energy"] = float(value)
            holder["gradient"] = grad
            holder["noisy_norm"] = noisy_norm
            holder["noise_stopped"] = noise_stopped
        except ForceCallCap as exc:
            holder["stop"] = True
            holder["capped"] = True
            holder["error"] = f"force-call cap: {exc}"
            forces[:] = 0.0
        except Exception as exc:
            holder["stop"] = True
            message = f"{type(exc).__name__}: {exc}"
            holder["error"] = message
            if "exposed" in message or "Lost atoms" in message:
                holder["diverged"] = True
            forces[:] = 0.0

    try:
        with tempfile.TemporaryDirectory(prefix="lammps-") as folder:
            data = Path(folder) / "geometry.data"
            _write_data(data, positions, low, high)
            command = lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])
            holder["lmp"] = command
            lines = [
                "units lj",
                f"boundary {boundary}",
                "atom_style atomic",
                "atom_modify map array sort 0 0.0",
                f"read_data {data}",
                "mass * 1.0",
                "pair_style zero 1.0",
                "pair_coeff * *",
                "neighbor 2.0 bin",
                "neigh_modify every 1 delay 0 check no binsize 1000",
                "fix ext all external pf/callback 1 1",
                "fix_modify ext energy yes",
                f"min_style {job['min_style']}",
                "min_modify dmax 0.2 norm two",
            ]
            # FIRE adds the force to the velocity before the move, and that
            # kick is not limited by dmax. A short timestep keeps the first
            # kick inside dmax when the force is large. tmax 10 lets it grow.
            if job["min_style"] in {"fire", "quickmin"}:
                lines.append("timestep 0.002")
            frozen = [str(index) for index, flag in enumerate(free, start=1) if not flag]
            if frozen:
                lines.append("group frozen id " + " ".join(frozen))
                lines.append("fix hold frozen setforce 0.0 0.0 0.0")
            command.commands_string("\n".join(lines))
            command.set_fix_external_callback("ext", callback)
            # etol 0 does not stop on energy. ftol is the Euclidean norm.
            command.command(f"minimize 0.0 {ftol} {cap} {cap}")
            command.close()
    except Exception as exc:
        message = str(exc)
        if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
            holder["capped"] = True
            holder["error"] = f"force-call cap: {message}"
        elif "error" not in holder:
            holder["error"] = f"{type(exc).__name__}: {exc}"
            if "Lost atoms" in message or "Too many neighbor bins" in message:
                holder["diverged"] = True
    if "point" in holder:
        point = holder["point"]
        energy = holder["energy"]
        gradient = holder["gradient"]
        noisy_norm = holder.get("noisy_norm", noisy_norm)
        noise_stopped = bool(holder.get("noise_stopped", noise_stopped))
    error = str(holder.get("error") or "")
    prefix: list[str] = []
    if holder.get("capped"):
        prefix.append("cap")
    elif holder.get("diverged"):
        prefix.append("divergence")
    elif error:
        prefix.append("crash")
    # A noisy stop is the noise rule, not the clean force norm. The Hessian
    # is the clean potential, and it is checked once that stop is met.
    judge_tol = 1.0e9 if sigma > 0.0 and noise_stopped else clean_tol
    # A periodic pair potential is invariant to translation. Rotations are not
    # removed: the cell breaks them. A free cluster uses the usual six modes.
    rotations = 0 if box is not None and bool(np.all(free)) else None
    record = judge_saddle(
        job["name"],
        point,
        None if energy is None else float(energy),
        None if gradient is None else np.asarray(gradient, dtype=np.float64),
        counter=counter,
        cap=cap,
        force_norm=judge_tol,
        kind=kind,
        free=free,
        box=box,
        prefix=prefix,
        expected_index=0,
        rotations=rotations,
    )
    if sigma > 0.0:
        clean_norm = record["force_norm"]
        record["clean_force_norm"] = clean_norm
        record["force_norm"] = None if noisy_norm is None else float(noisy_norm)
        record["sigma"] = sigma
        record["noise_stopped"] = noise_stopped
        if not noise_stopped and "cap" not in record["reasons"] and "crash" not in record["reasons"]:
            record["reasons"].append("force_norm")
            record["failure"] = classify_failure(record["reasons"])
            record["success"] = record["failure"] is None
            record["converged"] = record["success"]
    if error:
        record["error"] = error
    record["min_style"] = job["min_style"]
    return record


def _write_data(path: Path, positions: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
    xyz = np.asarray(positions, dtype=np.float64)
    lines = [
        "optbench geometry",
        "",
        f"{len(xyz)} atoms",
        "1 atom types",
        "",
        f"{low[0]} {high[0]} xlo xhi",
        f"{low[1]} {high[1]} ylo yhi",
        f"{low[2]} {high[2]} zlo zhi",
        "",
        "Atoms # atomic",
        "",
    ]
    for index, coord in enumerate(xyz, start=1):
        lines.append(f"{index} 1 {coord[0]} {coord[1]} {coord[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan(root: Path, problem_name: str, template_name: str, work: Path) -> dict:
    """The jobs ``run`` sends to ``run_jobs``, one per archived start, in archive order."""
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
    if problem.get("suite") != "minimization":
        raise RuntimeError(
            f"{problem['id']} is not a minimization. LAMMPS NEB, PRD, TAD, and hyper "
            "are not scored: hTST and accelerated-md need an EAM potential the harness does not evaluate."
        )
    settings = point_settings(problem)
    if settings["cap"] is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    scratch = Path(work) / template["id"].replace("/", "-")
    jobs = [
        _job(start, settings, template, index)
        for index, start in enumerate(load_starts(root, problem, scratch))
    ]
    return {"problem": problem, "template": template, "settings": settings, "jobs": jobs, "execute": _execute}


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    planned = plan(root, problem_name, template_name, output.parent / "work")
    problem, template, settings = planned["problem"], planned["template"], planned["settings"]
    records = run_jobs(planned["jobs"], planned["execute"], workers)
    section = _section(template)
    noise = _noise(template)
    comment = (
        "# counter harness. fix external pf/callback writes the harness force. "
        "pair_style zero contributes nothing.\n"
        f"# min_style is {section['min_style']}. A success is under the cap, "
        "at the Euclidean force stop, and Hessian index 0."
    )
    if noise["sigma"] > 0.0:
        comment += (
            "\n# Gaussian force noise from [noise]. The stop is the noisy norm and "
            "a clean-energy window. The Hessian is clean."
        )
    parameters = {
        "min_style": section["min_style"],
        "pair_style": "zero",
        "fix": "external pf/callback",
        "counter": "harness",
        "sigma": noise["sigma"],
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
