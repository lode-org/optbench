"""Minimize OptBench starts with GMIN MYMYLBFGS.

The callback evaluates the harness potential and charges a
``HarnessCounter`` that caches a repeated coordinate. ``ForceCallCap``
is raised before a search call once the budget is spent, and that
refused call is not counted. The Hessian used for the index is analytic
and is not a force call.

A 2014 success is a start whose harness count is under the cap and
whose force norm is at the problem stop. A v2 success also has Hessian
index 0. On LJ38 that index is the vibrational index. The problem files
do not give ``energy_tol``, ``rmsd_tol``, or a per-start reference
geometry, so energy and Kabsch RMSD are not applied. The primary failure
class is ``optbench.failures.classify_failure``.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import HarnessCounter
from optbench.failures import classify_failure
from optbench.gmin_lbfgs import DEFAULTS, VERSION, minimize, require_revision
from optbench.potentials import lennard_jones, morse_bulk
from optbench.potserver import RgpotCounter
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
_PASSED = ("m", "maxbfgs", "dguess", "maxerise", "maxefall")


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _kind(problem_id: str) -> str:
    if problem_id == "minimization/lj38":
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no GMIN potential")


def _options(template: dict) -> dict:
    section = template.get("gmin") or {}
    options = dict(DEFAULTS)
    for key in _PASSED:
        if key in section:
            options[key] = section[key]
    return options


def _index_at(kind: str, point: list[float], box, force_stop: float) -> tuple[int, list[float]]:
    import numpy as np

    from optbench.potentials import lennard_jones_hessian, morse_bulk_hessian
    from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues

    if kind == "lj":
        matrix = lennard_jones_hessian(point)
        values = vibrational_eigenvalues(matrix, point, rotations=True)
        tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    else:
        matrix = morse_bulk_hessian(point, box)
        values = np.linalg.eigvalsh(matrix).tolist()
        tolerance = 1e-6
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    import numpy as np

    calls = 0
    verify_calls = 0
    error = ""
    result = None
    energy = None
    force_norm = None
    point = None
    potential = None
    try:
        x = np.asarray(job["coords"], dtype=np.float64).reshape(-1).copy()
        box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
        kind = job["kind"]
        cap = int(job["cap"])
        gtol = float(job["gtol"])

        def evaluate_raw(positions, _numbers=None, _box=None):
            if kind == "lj":
                return lennard_jones(positions)
            return morse_bulk(positions, box)

        counter = HarnessCounter(cache=True, cap=cap)
        potential = RgpotCounter(evaluate_raw, counter)

        def evaluate(coords):
            value, gradient = potential.calculate(coords)
            return float(value), np.asarray(gradient, dtype=np.float64)

        options = job["options"]
        result = minimize(
            x,
            evaluate,
            gtol=gtol,
            cap=cap,
            m=int(options["m"]),
            maxbfgs=float(options["maxbfgs"]),
            dguess=float(options["dguess"]),
            maxerise=float(options["maxerise"]),
            maxefall=float(options["maxefall"]),
        )
        point = [float(value) for value in result.coords]
        if result.capped:
            error = "ForceCallCap: force-call cap"
        elif result.info == 3:
            error = "ValueError: GMIN rejected the coordinate length"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if potential is not None:
        calls = int(potential.counter.calls)
        if point is not None:
            try:
                with potential.counter.verification():
                    energy, gradient = potential.calculate(point)
                force_norm = float(np.linalg.norm(gradient))
                verify_calls = int(potential.counter.verify_calls)
            except Exception as exc:
                if not error:
                    error = f"{type(exc).__name__}: {exc}"
    gtol = float(job["gtol"])
    cap = int(job["cap"])
    hessian_index = None
    min_eigenvalues: list[float] = []
    if point is not None and (not error or "force-call cap" in error or "ForceCallCap" in error):
        try:
            hessian_index, min_eigenvalues = _index_at(job["kind"], point, job["box"], gtol)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    stopped = result is not None and result.mflag and result.info == 0 and not result.capped
    finite = (
        point is not None
        and all(math.isfinite(value) for value in point)
        and energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    at_force = (
        finite
        and stopped
        and force_norm is not None
        and force_norm <= gtol
        and calls < cap
        and not error
    )
    reasons: list[str] = []
    cap_hit = "force-call cap" in error or "ForceCallCap" in error
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
        "gmin_itdone": None if result is None else int(result.itdone),
        "gmin_mflag": None if result is None else bool(result.mflag),
        "gmin_info": None if result is None else int(result.info),
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


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, partial: bool) -> str:
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
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "partial": "true" if partial else None,
    }
    lines = [
        "# counter harness. One force call is one RgpotCounter.calculate entry.",
        "# The quench is GMIN MYMYLBFGS. The Hessian index is analytic and is not a force call.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
        "# energy_min and energy_max are the final potential values on the recorded points.",
    ]
    if str(template.get("problem")) == "minimization/lj38":
        lines.append(
            "# LJ38 index removes the six rigid modes. A curvature is negative only below "
            "-max(force_norm, 1e-8*||H||_F, 1e-6)."
        )
    else:
        lines.append("# Morse index uses eigenvalue tolerance 1e-6.")
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
    found = require_revision(str(template.get("commit") or ""))
    version = str(template.get("code_version") or VERSION)
    if not found.startswith(version):
        raise RuntimeError(f"GMIN revision {found} does not match template {version}")
    commit = str(template.get("commit") or "")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    options = _options(template)
    return _run(
        problem,
        template,
        kind=_kind(problem["id"]),
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
        limit=int(limit),
    )


def _run(
    problem,
    template,
    *,
    kind,
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
    limit,
) -> dict:
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if problem["id"] == "minimization/morse-bulk" and geometries is None and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    full = len(loaded)
    if limit:
        loaded = loaded[:limit]
    partial = len(loaded) < full
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if kind == "lj" else box.tolist(),
                "kind": kind,
                "options": options,
                "gtol": gtol,
                "cap": cap,
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
    passed = {"gtol": gtol, "maxiter": cap, **options}
    provenance = collect_provenance(
        code="GMIN",
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
            "Harness counter. The quench is GMIN MYMYLBFGS. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "GMIN",
                "algorithm": template.get("algorithm", "LBFGS"),
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
    output.write_text(_dat(summary, template, metrics, commit, version, partial), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary
