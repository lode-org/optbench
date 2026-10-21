"""Build a v2 start record and read the Pt-island repetition lines.

The 2014 ``benchmark.dat`` mean is a different record. A start is
comparable only when ``counter`` is ``harness``.
"""

from __future__ import annotations

import json
import math
import platform
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

from .counter import CountedPotential
from .failures import classify_failure
from .score import score_v2
from .stationary import jacobi_eigenvalues, judge_stationary, numerical_hessian

# Keys a v2 result must carry, taken from the run, its metrics, its
# provenance, and one start. Problem files list the same names.
V2_KEYS = (
    "schema",
    "counter",
    "force_calls",
    "success",
    "failure",
    "comparable",
    "code",
    "code_version",
    "commit",
    "tuning",
    "par2_mean",
    "median",
    "median_low",
    "median_high",
    "wilson_low",
    "wilson_high",
    "n",
    "nfailed",
    "n_cap",
    "n_wrong_stationary",
    "n_divergence",
    "n_crash",
)


def collect_provenance(
    *,
    code: str,
    code_version: str,
    commit: str,
    parameters: dict,
    machine: dict | str | None = None,
) -> dict:
    """Code, version, commit, parameters, and machine.

    A string ``machine`` replaces the node name and keeps the rest of
    this process. A mapping is stored as given.
    """
    if isinstance(machine, dict):
        recorded = dict(machine)
    else:
        recorded = {
            "node": machine or platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
    return {
        "code": code,
        "code_version": code_version,
        "commit": commit,
        "parameters": parameters,
        "machine": recorded,
    }


def embed_point(coords: Sequence[float]) -> list[list[float]]:
    """One particle in the plane becomes an x, y, z row."""
    values = [float(value) for value in coords]
    if len(values) == 2:
        return [[values[0], values[1], 0.0]]
    if len(values) == 3:
        return [values]
    raise ValueError("embed a 2-vector or a 3-vector")


def _finite(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _crash_record(name: str, potential: CountedPotential, error: str) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": potential.calls,
        "verify_calls": potential.verify_calls,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "error": error,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": None,
        "rmsd": None,
        "converged": False,
    }


def record_start(
    name: str,
    evaluate: Callable,
    optimize: Callable,
    *,
    cap: int,
    kind: str,
    reference_energy: float,
    reference_coords: Sequence[Sequence[float]],
    force_norm: float,
    energy_tol: float,
    rmsd_tol: float,
    eigenvalue_tol: float = 1e-6,
    fd_step: float = 1e-5,
) -> dict:
    """Run one start on a counted potential and judge the stationary point.

    ``optimize`` receives the counted potential and returns a mapping
    with ``x``, ``energy``, ``force_norm``, and ``converged``. Hessian
    evaluations go through ``verification`` and are not search calls.
    """
    potential = CountedPotential(evaluate)
    try:
        outcome = optimize(potential)
    except Exception as exc:
        return _crash_record(name, potential, exc.__class__.__name__)
    point = [float(value) for value in outcome["x"]]
    energy = float(outcome["energy"])
    observed_force = float(outcome["force_norm"])
    finite = _finite([energy, observed_force, *point])
    judged = {
        "hessian_index": None,
        "expected_index": None,
        "rmsd": None,
        "reasons": [],
    }
    if finite:
        with potential.verification():
            hessian = numerical_hessian(
                lambda coords: potential.energy_forces(coords)[1],
                point,
                step=fd_step,
            )
            eigenvalues, _vectors = jacobi_eigenvalues(hessian)
        coords = embed_point(point)
        judged = judge_stationary(
            kind=kind,
            eigenvalues=eigenvalues,
            energy=energy,
            coords=coords,
            reference_energy=reference_energy,
            reference_coords=reference_coords,
            energy_tol=energy_tol,
            rmsd_tol=rmsd_tol,
            eigenvalue_tol=eigenvalue_tol,
        )
    reasons = list(judged["reasons"])
    if not finite or outcome.get("diverged"):
        reasons.append("divergence")
    if potential.calls >= cap:
        reasons.append("cap")
    if (not outcome["converged"]) or (not finite) or observed_force > force_norm:
        reasons.append("force_norm")
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": potential.calls,
        "verify_calls": potential.verify_calls,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": energy if finite else None,
        "force_norm": observed_force if math.isfinite(observed_force) else None,
        "hessian_index": judged["hessian_index"],
        "expected_index": judged["expected_index"],
        "rmsd": judged["rmsd"],
        "converged": bool(outcome["converged"]) and finite,
    }


def parse_repetitions(text: str) -> list[dict]:
    """Per-search costs from ``# rep`` comment lines.

    A line looks like ``# rep 1 force_calls 2611052 ... total_searches 9936``.
    The v2 cost of that repetition is ``force_calls / total_searches``.
    """
    rows = []
    for line in text.splitlines():
        if not line.startswith("# rep "):
            continue
        parts = line.split()
        if "force_calls" not in parts or "total_searches" not in parts:
            continue
        force_calls = int(parts[parts.index("force_calls") + 1])
        searches = int(parts[parts.index("total_searches") + 1])
        if searches <= 0:
            raise ValueError(f"total_searches must be positive in {line!r}")
        rows.append(
            {
                "repetition": int(parts[2]),
                "force_calls": force_calls,
                "searches": searches,
                "per_search": force_calls / searches,
            }
        )
    return rows


def write_starts(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(body, encoding="utf-8")


def client_start_records(names: list[str], counts: list[int], cap: int) -> list[dict]:
    """Per-start rows for a subprocess client.

    The client counted these calls. They are not a harness score.
    """
    records = []
    for name, count in zip(names, counts, strict=True):
        records.append(
            {
                "schema": "optbench.start/v2",
                "start": name,
                "force_calls": count,
                "counter": "client",
                "comparable": False,
                "success": count < cap,
            }
        )
    return records


def tuning_from_template(root: Path, template_id: str) -> str:
    if not template_id:
        return ""
    path = root / "templates" / f"{template_id}.toml"
    if not path.is_file():
        return ""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    value = str(data.get("tuning", ""))
    if value in ("default", "tuned"):
        return value
    return ""


def tuning_label(run: dict) -> str:
    """``default`` or ``tuned`` when the record or the eOn memory says so.

    eOn 3.5.0 ships L-BFGS memory 20. Memory 100 is the archived setting.
    Anything else stays unlabelled.
    """
    named = str(run.get("tuning") or "")
    if named in ("default", "tuned"):
        return named
    method = run.get("method") or {}
    code = str(run.get("code") or "")
    if str(run.get("algorithm") or "") == "LBFGS" and code in ("eOn", "Eon") and "lbfgs_memory" in method:
        memory = int(method["lbfgs_memory"])
        if memory == 20:
            return "default"
        if memory == 100:
            return "tuned"
    return "unlabelled"


def summarise_run(starts: list[dict], cap: int | None) -> dict:
    """PAR-2 of these starts. A stored failure class is kept."""
    records = []
    for start in starts:
        record = {"force_calls": start["force_calls"], "success": start["success"]}
        if "failure" in start:
            record["failure"] = start["failure"]
        if start.get("reasons"):
            record["reasons"] = start["reasons"]
        records.append(record)
    return score_v2(records, cap)
