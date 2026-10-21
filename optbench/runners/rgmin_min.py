"""Minimize OptBench starts with rgmin and ``RgpotCounter``.

The callback is ``RgpotCounter.calculate`` on a ``HarnessCounter`` that
caches a repeated coordinate and raises ``ForceCallCap`` before a search
call once the budget is spent. That refused call is not counted. Analytic
Hessians used by Newton and dogleg are not force calls. They are counted
separately.

A 2014 success is a start whose harness count is under the cap and whose
force norm is at the problem stop. A v2 success also has Hessian index 0.
On LJ38 that index is the vibrational index: translations and rotations are
removed, and a curvature counts only below ``-curvature_tolerance`` of the
force stop. The problem files do not give ``energy_tol``, ``rmsd_tol``, or a
per-start reference geometry, so energy and Kabsch RMSD are not applied. The
primary failure class is ``optbench.failures.classify_failure``.
"""

from __future__ import annotations

import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import HarnessCounter
from optbench.failures import classify_failure
from optbench.noise import converged_under_noise, stop_threshold
from optbench.potentials import (
    lennard_jones,
    lennard_jones_hessian,
    morse_bulk,
    morse_bulk_hessian,
    read_con,
)
from optbench.potserver import RgpotCounter
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance, write_starts

EIGENVALUE_TOL = 1e-6
HESSIAN_METHODS = frozenset({"newton", "rfo", "dogleg"})
_PASSED = ("method",)


def _kind(problem_id: str) -> str:
    if problem_id in {"minimization/lj38", "minimization/lj38-noise"}:
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no rgmin potential")


def _noise(template: dict, problem: dict) -> dict | None:
    """Gaussian force noise from ``[noise]``. Absent on a clean template."""
    section = template.get("noise")
    if not section:
        return None
    if "sigma" not in section:
        raise ValueError(f"{template['id']} noise section needs sigma")
    stated = problem.get("noise") or {}
    sigma = float(section["sigma"])
    allowed = [float(value) for value in stated.get("sigma") or []]
    if allowed and sigma not in allowed:
        raise ValueError(f"{template['id']} sigma {sigma} is not in the problem list")
    kappa = float(section.get("kappa", stated.get("kappa", 3.0)))
    window = int(section.get("window", stated.get("window", 5)))
    seed = int(section.get("seed", stated.get("seed", 20261016)))
    return {"sigma": sigma, "kappa": kappa, "window": window, "seed": seed}


def _options(template: dict) -> tuple[str, dict]:
    section = template.get("rgmin") or {}
    if "method" not in section:
        raise ValueError(f"{template['id']} needs rgmin.method")
    method = str(section["method"])
    options = {key: value for key, value in section.items() if key not in _PASSED}
    return method, options


def _index_at(kind: str, point: list[float], box, force_stop: float) -> tuple[int, list[float]]:
    import numpy as np

    if kind == "lj":
        matrix = lennard_jones_hessian(point)
        # A free cluster. The six rigid modes are not vibrational curvature,
        # and at a 0.01 stop the rotations are not an exact kernel.
        values = vibrational_eigenvalues(matrix, point, rotations=True)
        tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    else:
        matrix = morse_bulk_hessian(point, box)
        values = np.linalg.eigvalsh(matrix).tolist()
        tolerance = EIGENVALUE_TOL
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _minimize_job(job: dict) -> dict:
    import numpy as np
    import rgmin

    calls = 0
    verify_calls = 0
    hessian_calls = 0
    error = ""
    report = None
    energy = None
    force_norm = None
    clean_force_norm = None
    point = None
    potential = None
    noise_stopped = False
    sigma = float(job.get("sigma") or 0.0)
    noisy_norm = None
    try:
        x = np.asarray(job["coords"], dtype=np.float64).reshape(-1).copy()
        box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
        kind = job["kind"]
        cap = int(job["cap"])

        def evaluate(positions, _numbers=None, _box=None):
            if kind == "lj":
                return lennard_jones(positions)
            return morse_bulk(positions, box)

        counter = HarnessCounter(cache=True, cap=cap)
        potential = RgpotCounter(evaluate, counter)
        kappa = float(job.get("kappa") or 3.0)
        window = int(job.get("window") or 5)
        energies: list[float] = []
        noisy_norm = None
        rng = None
        if sigma > 0.0:
            import random

            rng = random.Random(int(job["noise_seed"]))

        def fg(coords):
            nonlocal noisy_norm
            value, gradient = potential.calculate(coords)
            grad = np.asarray(gradient, dtype=np.float64)
            if sigma > 0.0:
                energies.append(float(value))
                grad = grad + np.fromiter((rng.gauss(0.0, sigma) for _ in range(grad.size)), dtype=np.float64)
                noisy_norm = float(np.linalg.norm(grad))
            return float(value), grad

        def hess(coords):
            nonlocal hessian_calls
            hessian_calls += 1
            if kind == "lj":
                return lennard_jones_hessian(coords)
            return morse_bulk_hessian(coords, box)

        method = job["method"]
        solver_gtol = float(job["gtol"])
        if sigma > 0.0:
            solver_gtol = stop_threshold(solver_gtol, sigma, int(x.size), kappa)
        config = rgmin.SolverConfig(
            method,
            gtol=solver_gtol,
            maxiter=int(job["maxiter"]),
            **job["options"],
        )
        solver = rgmin.Solver(x.size, config)
        energy_tol = max(float(job["gtol"]), kappa * sigma)
        for _ in range(int(job["maxiter"])):
            if counter.calls >= cap:
                break
            if method in HESSIAN_METHODS:
                report = solver.step_hess(fg, hess, x)
            else:
                report = solver.step(fg, x)
            if sigma > 0.0:
                if counter.calls >= cap:
                    break
                fg(x)
                if noisy_norm is not None and converged_under_noise(
                    force_norm=noisy_norm,
                    energies=energies,
                    clean_tol=float(job["gtol"]),
                    sigma=sigma,
                    components=int(x.size),
                    energy_tol=energy_tol,
                    kappa=kappa,
                    window=window,
                ):
                    noise_stopped = True
                    break
                if report.stop not in (rgmin.Stop.Budget, rgmin.Stop.Tolerance):
                    break
            elif report.stop != rgmin.Stop.Budget:
                break
        point = [float(value) for value in x]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if "x" in locals():
            point = [float(value) for value in x]
    if potential is not None:
        calls = int(potential.counter.calls)
        if point is not None:
            try:
                with potential.counter.verification():
                    energy, gradient = potential.calculate(point)
                clean_force_norm = float(np.linalg.norm(gradient))
                force_norm = noisy_norm if sigma > 0.0 and noisy_norm is not None else clean_force_norm
                verify_calls = int(potential.counter.verify_calls)
            except Exception as exc:
                if not error:
                    error = f"{type(exc).__name__}: {exc}"
    gtol = float(job["gtol"])
    cap = int(job["cap"])
    hessian_index = None
    min_eigenvalues: list[float] = []
    if point is not None and (not error or "force-call cap" in error):
        try:
            hessian_index, min_eigenvalues = _index_at(job["kind"], point, job["box"], gtol)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    stopped = report is not None and report.stop == rgmin.Stop.Tolerance
    finite = (
        point is not None
        and all(math.isfinite(value) for value in point)
        and energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    if sigma > 0.0:
        at_force = finite and noise_stopped and calls < cap and not error
    else:
        at_force = (
            finite
            and stopped
            and force_norm is not None
            and force_norm <= gtol
            and calls < cap
            and not error
        )
    reasons: list[str] = []
    if error and "force-call cap" not in error:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif calls >= cap or (error and "force-call cap" in error):
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
        "clean_force_norm": clean_force_norm,
        "sigma": sigma,
        "noise_stopped": noise_stopped,
        "hessian_index": hessian_index,
        "expected_index": 0,
        "converged": at_force,
        "hessian_calls": hessian_calls,
        "min_eigenvalues": min_eigenvalues,
        "rgmin_steps": None if report is None else int(report.steps),
        "rgmin_stop": None if report is None else str(report.stop),
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


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str) -> str:
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
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter harness. One force call is one RgpotCounter.calculate entry.",
        "# Newton and dogleg Hessians are analytic and are not force calls.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
        "# energy_min and energy_max are the final potential values on the recorded points.",
    ]
    if str(template.get("problem")) in {"minimization/lj38", "minimization/lj38-noise"}:
        lines.append(
            "# LJ38 index removes the six rigid modes. A curvature is negative only below "
            "-max(force_norm, 1e-8*||H||_F, 1e-6)."
        )
    if template.get("noise"):
        lines.append(
            "# Gaussian force noise from [noise]. sigma is not an rgmin.SolverConfig field. "
            "The stop is the noisy force norm at max(clean tolerance, kappa * sigma * sqrt(components)), "
            "and the clean energy over the last window evaluations spans at most max(clean tolerance, kappa * sigma). "
            "The Hessian is the clean force."
        )
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
) -> dict:
    import rgmin

    version = str(rgmin.__version__)
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"rgmin {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if not commit:
        raise RuntimeError(f"{template['id']} has no commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    method, options = _options(template)
    if "gtol" in options and abs(float(options["gtol"]) - gtol) > 1e-12:
        raise ValueError("template gtol is not the problem force norm")
    if "maxiter" in options and int(options["maxiter"]) < cap:
        raise ValueError("template maxiter is below the problem force-call cap")
    options.pop("gtol", None)
    options.pop("maxiter", None)
    if problem["id"] in {"minimization/lj38", "minimization/lj38-noise"}:
        step = options.get("maxstep")
        if step is None or float(step) <= 0.0:
            raise ValueError(
                f"{template['id']} needs rgmin.maxstep, the per-atom displacement cap. "
                "An uncapped istep*g trial leaves the Lennard-Jones cutoff."
            )
    return _run(
        problem,
        template,
        kind=_kind(problem["id"]),
        method=method,
        options=options,
        gtol=gtol,
        cap=cap,
        geometries=geometries,
        output=output,
        workers=workers,
        work=work,
        version=version,
        commit=commit,
        root=root,
        noise=_noise(template, problem),
    )


def _load_geometries(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path):
    import tarfile

    from optbench.campaign import geometry_archive, iter_geometries

    if geometries is None:
        archive = geometry_archive(root, problem)
        dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".extracted"
        recorded = archive.name
        if marker.is_file():
            previous = marker.read_text(encoding="utf-8").strip()
            if previous != recorded:
                raise RuntimeError(f"{dest} was extracted from {previous}, not {recorded}")
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text(recorded, encoding="utf-8")
        cons = sorted(dest.rglob("*.con"))
        if not cons:
            raise FileNotFoundError(f"{archive} has no .con files")
    else:
        cons = iter_geometries(geometries)
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected is not None and len(cons) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(cons)}")
    loaded = []
    for path in cons:
        pos, box = read_con(path.read_text(encoding="utf-8", errors="replace"))
        loaded.append((path.stem, pos.reshape(-1), box))
    return loaded


def _run(
    problem,
    template,
    *,
    kind,
    method,
    options,
    gtol,
    cap,
    geometries,
    output,
    workers,
    work,
    version,
    commit,
    root,
    noise: dict | None = None,
) -> dict:
    loaded = _load_geometries(root, problem, geometries, work, output)
    if problem["id"] == "minimization/morse-bulk" and geometries is None and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if kind == "lj" else box.tolist(),
                "kind": kind,
                "method": method,
                "options": options,
                "gtol": gtol,
                "maxiter": cap,
                "cap": cap,
                "sigma": 0.0 if noise is None else float(noise["sigma"]),
                "kappa": 3.0 if noise is None else float(noise["kappa"]),
                "window": 5 if noise is None else int(noise["window"]),
                "noise_seed": 0 if noise is None else int(noise["seed"]) + int(index),
            }
        )
    if workers <= 1:
        records = [_minimize_job(job) for job in jobs]
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_minimize_job, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                if done == total or done % 50 == 0:
                    print(f"{template['id']} {done}/{total}", flush=True)
    records.sort(key=lambda row: row["index"])
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
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in records
        ],
        cap,
    )
    passed = {
        "gtol": gtol,
        "maxiter": cap,
        "method": method,
        **options,
    }
    if noise is not None:
        passed["sigma"] = float(noise["sigma"])
        passed["kappa"] = float(noise["kappa"])
        passed["window"] = int(noise["window"])
        passed["noise_seed"] = int(noise["seed"])
        passed["energy_tol"] = max(gtol, float(noise["kappa"]) * float(noise["sigma"]))
    provenance = collect_provenance(
        code="rgmin",
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
        "note": (
            "Harness counter. Analytic Hessians are not force calls. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "rgmin",
                "algorithm": template.get("algorithm", method),
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
    output.write_text(_dat(summary, template, metrics, commit, version), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    return summary
