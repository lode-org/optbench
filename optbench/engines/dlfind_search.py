"""DL-FIND through the libdlfind gradient callback.

The callback is the score. DL-FIND's own energy tally is not. Coordinates
stay in the harness unit. The gradient is ``dE/dx`` from ``pes.evaluate``.
``iopt`` 3 is L-BFGS, 0 is steepest descent, and 1 is conjugate gradient.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

from .common import judge_saddle, run_jobs, write_campaign
from .geometries import load_starts, point_settings

ENGINE = "dlfind"
CODE = "dlfind"
_IOPT = {"lbfgs": 3, "sd": 0, "cg": 1}


def site_packages() -> Path:
    env = os.environ.get("DLFIND_SITE")
    if env:
        return Path(env)
    return Path("/tmp/venv-dlfind/lib/python3.12/site-packages")


def available() -> str | None:
    package = site_packages() / "libdlfind"
    if not package.is_dir():
        return (
            "libdlfind is not installed at "
            f"{site_packages()}. The pin is 0.0.3, commit 4167998d. "
            "The gradient callback is the harness score."
        )
    return None


def _section(template: dict) -> dict:
    section = dict(template.get("dlfind") or {})
    method = str(section.get("method", "lbfgs"))
    if method not in _IOPT:
        raise ValueError(f"{template['id']} method {method} is not one of {', '.join(_IOPT)}")
    section["method"] = method
    section["iopt"] = _IOPT[method]
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
        "method": section["method"],
        "iopt": section["iopt"],
        "sigma": noise["sigma"],
        "kappa": noise["kappa"],
        "window": noise["window"],
        "noise_seed": noise["seed"] + int(index),
    }


def _import_dlfind():
    site = str(site_packages())
    if site not in sys.path:
        sys.path.append(site)
    from libdlfind import dl_find
    from libdlfind.callback import dlf_get_gradient_wrapper, dlf_put_coords_wrapper, make_dlf_get_params

    return dl_find, dlf_get_gradient_wrapper, dlf_put_coords_wrapper, make_dlf_get_params


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from optbench.counter import ForceCallCap, HarnessCounter
    from optbench.failures import classify_failure
    from optbench.noise import converged_under_noise, stop_threshold
    from optbench.pes import evaluate

    dl_find, dlf_get_gradient_wrapper, dlf_put_coords_wrapper, make_dlf_get_params = _import_dlfind()
    positions = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    sigma = float(job.get("sigma") or 0.0)
    kappa = float(job.get("kappa") or 3.0)
    window = int(job.get("window") or 5)
    components = int(np.count_nonzero(free)) * 3
    clean_tol = float(job["force_norm"])
    ftol = stop_threshold(clean_tol, sigma, components, kappa) if components else clean_tol
    # DL-FIND stops on the largest gradient component. That bound keeps the
    # Euclidean norm at or under the harness stop.
    component_tol = ftol / np.sqrt(components) if components else ftol
    counter = HarnessCounter(cache=True, cap=cap)
    rng = random.Random(int(job["noise_seed"])) if sigma > 0.0 else None
    energies: list[float] = []
    noisy_norm = None
    noise_stopped = False
    holder: dict = {"capped": False}
    point = positions.copy()
    energy = None
    gradient = None

    @dlf_get_gradient_wrapper
    def gradient_of(coords, _image, _kiter):
        nonlocal noisy_norm, noise_stopped
        if holder.get("capped"):
            last = holder.get("gradient")
            if last is None:
                return 0.0, np.zeros(coords.size, dtype=np.float64)
            return float(holder.get("energy") or 0.0), np.asarray(last, dtype=np.float64).reshape(-1)
        try:
            xyz = np.array(coords, dtype=np.float64, copy=True)
            xyz[~free] = positions[~free]
            coords[~free] = positions[~free]
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
            reported = np.array(seen, copy=True)
            reported[~free] = 0.0
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
                if noisy_norm <= ftol and not noise_stopped and noisy_norm > 0.0:
                    reported[free] *= (1.01 * ftol) / noisy_norm
            holder["point"] = xyz
            holder["energy"] = float(value)
            holder["gradient"] = grad
            holder["noisy_norm"] = noisy_norm
            holder["noise_stopped"] = noise_stopped
            return float(value), reported.reshape(-1)
        except ForceCallCap as exc:
            holder["capped"] = True
            holder["error"] = f"force-call cap: {exc}"
            last = holder.get("gradient")
            if last is None:
                return 0.0, np.zeros(int(np.asarray(coords).size), dtype=np.float64)
            return float(holder.get("energy") or 0.0), np.asarray(last, dtype=np.float64).reshape(-1)
        except Exception as exc:
            holder["crashed"] = True
            holder["error"] = f"{type(exc).__name__}: {exc}"
            return 0.0, np.zeros(int(np.asarray(coords).size), dtype=np.float64)

    @dlf_put_coords_wrapper
    def store(_switch, _energy, coords, _iam):
        holder["reported"] = np.array(coords, dtype=np.float64, copy=True)

    params = make_dlf_get_params(
        coords=positions,
        tolerance=float(component_tol),
        printl=0,
        printf=0,
        maxcycle=cap,
        maxene=cap,
        iopt=int(job["iopt"]),
        icoord=0,
        maxstep=0.2,
        massweight=0,
        dump=0,
        restart=0,
    )
    try:
        dl_find(
            nvarin=int(positions.size),
            dlf_get_gradient=gradient_of,
            dlf_get_params=params,
            dlf_put_coords=store,
        )
    except Exception as exc:
        message = str(exc)
        if "error" not in holder:
            holder["error"] = f"{type(exc).__name__}: {message}"
    if "point" in holder:
        point = holder["point"]
        energy = holder["energy"]
        gradient = holder["gradient"]
        noisy_norm = holder.get("noisy_norm", noisy_norm)
        noise_stopped = bool(holder.get("noise_stopped", noise_stopped))
    error = str(holder.get("error") or "")
    prefix: list[str] = []
    if holder.get("capped") or counter.calls >= cap:
        prefix.append("cap")
    elif holder.get("crashed") or error:
        prefix.append("crash")
    judge_tol = 1.0e9 if sigma > 0.0 and noise_stopped else clean_tol
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
        record["clean_force_norm"] = record["force_norm"]
        record["force_norm"] = None if noisy_norm is None else float(noisy_norm)
        record["sigma"] = sigma
        record["noise_stopped"] = noise_stopped
        if not noise_stopped and "cap" not in record["reasons"] and "crash" not in record["reasons"]:
            record["reasons"].append("force_norm")
            record["failure"] = classify_failure(record["reasons"])
            record["success"] = record["failure"] is None
            record["converged"] = record["success"]
    if error and not record["success"]:
        record["error"] = error
    record["iopt"] = int(job["iopt"])
    record["method"] = job["method"]
    return record


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
            f"{problem['id']} is not a minimization. This runner scores iopt "
            "0, 1, and 3. NEB and the dimer are separate DL-FIND tasks."
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
        "# counter harness. The DL-FIND gradient callback calls pes.evaluate.\n"
        f"# iopt {section['iopt']} is {section['method']}. A success is under the cap, "
        "at the Euclidean force stop, and Hessian index 0."
    )
    if noise["sigma"] > 0.0:
        comment += (
            "\n# Gaussian force noise from [noise]. The stop is the noisy norm and "
            "a clean-energy window. The Hessian is clean."
        )
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        {
            "iopt": section["iopt"],
            "method": section["method"],
            "maxstep": 0.2,
            "counter": "harness",
            "sigma": noise["sigma"],
        },
        images=None,
        comment=comment,
    )
