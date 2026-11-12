"""Basin hopping with pele on the harness Lennard-Jones potential.

The archived driver sets the quench tolerance to 1e-3, the temperature
to 1.0, ``RandomDisplacement(stepsize=0.41)``, and
``AdaptiveStepsize`` with acceptance ratio 0.5 and interval 100. The
adaptive factor stays the constructor value, 0.9. The quench is
``lbfgs_cpp``. Only ``tol`` is set. Memory 4, maxstep 0.1, maxErise
1e-4, H0 0.1, and 10000 iterations stay the constructor values.

The driver takes up to 100000 quenches and checks the Markov energy
after each hop, not before the first hop. The target is the harness
energy of the Wales LJ38 coordinates plus the problem tolerance. One
force call is one ``getEnergyGradient`` entry on a ``HarnessCounter``.
A repeated coordinate is cached. The template call ceiling is the PAR-2
cap. It sits above the archived maximum and is not a quench setting.
"""

from __future__ import annotations

import io
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones
from optbench.runners.airss_go import reference_energy
from optbench.runners.pele_min import PELE_COMMIT, _load_geometries
from optbench.score import score_v2, summarize_capped
from optbench.v2 import write_starts

# A converged energy this far below the Wales minimum is not this surface.
_BELOW_MINIMUM = 0.01


class _HarnessLJ:
    """Lennard-Jones behind one ``HarnessCounter``."""

    def __init__(self, cap: int):
        self.counter = HarnessCounter(cache=True, cap=cap)

    def getEnergyGradient(self, coords):
        import numpy as np

        array = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in array)
        fresh, cached = self.counter.charge(key)
        if not fresh and cached is not None:
            energy, gradient = cached
            return float(energy), np.asarray(gradient, dtype=np.float64)
        energy, gradient = lennard_jones(array)
        gradient = np.ascontiguousarray(gradient, dtype=np.float64).reshape(-1)
        self.counter.remember(key, (float(energy), tuple(float(value) for value in gradient)))
        return float(energy), gradient

    def getEnergy(self, coords):
        energy, _gradient = self.getEnergyGradient(coords)
        return energy


def _seed_of(name: str) -> int:
    tail = name.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    raise ValueError(f"{name} has no integer seed")


def _meets(energy: float | None, minimum: float, tolerance: float, converged: bool, calls: int, cap: int) -> bool:
    if energy is None or not converged or calls >= cap:
        return False
    if not math.isfinite(energy):
        return False
    if energy < minimum - _BELOW_MINIMUM:
        return False
    return energy <= minimum + tolerance


def _hop_job(job: dict) -> dict:
    import numpy as np
    from pele.basinhopping import BasinHopping
    from pele.optimize import lbfgs_cpp
    from pele.takestep import AdaptiveStepsize, RandomDisplacement

    cap = int(job["cap"])
    minimum = float(job["energy_minimum"])
    tolerance = float(job["energy_tolerance"])
    seed = int(job["seed"])
    nquenches = int(job["nquenches"])
    np.random.seed(seed)
    potential = _HarnessLJ(cap)
    error = ""
    steps = 0
    caught_cap = False
    energy = None

    def quench(coords):
        return lbfgs_cpp(coords, potential, tol=float(job["quench_tol"]))

    try:
        step = AdaptiveStepsize(
            RandomDisplacement(stepsize=float(job["stepsize"])),
            acc_ratio=float(job["acc_ratio"]),
            interval=int(job["interval"]),
        )
        hopper = BasinHopping(
            np.asarray(job["coords"], dtype=np.float64),
            potential,
            step,
            quench=quench,
            temperature=float(job["temperature"]),
            outstream=io.StringIO(),
        )
        # The constructor quenches once. The driver checks the Markov
        # energy only after each later hop.
        for _ in range(nquenches):
            if potential.counter.calls >= cap:
                caught_cap = True
                break
            hopper.run(1)
            steps += 1
            trial = float(hopper.markovE)
            energy = trial
            if trial < minimum - _BELOW_MINIMUM:
                break
            if math.isfinite(trial) and trial <= minimum + tolerance:
                break
    except Exception as exc:
        if isinstance(exc, ForceCallCap) or potential.counter.calls >= cap:
            caught_cap = True
        else:
            error = f"{type(exc).__name__}: {exc}"
    calls = min(int(potential.counter.calls), cap)
    converged = energy is not None and math.isfinite(energy)
    success = _meets(energy, minimum, tolerance, converged, calls, cap)
    reasons: list[str] = []
    if error:
        reasons.append("crash")
    elif energy is None or not math.isfinite(energy if energy is not None else math.nan):
        reasons.append("divergence" if not caught_cap and calls < cap else "cap")
    elif energy < minimum - _BELOW_MINIMUM:
        reasons.append("divergence")
    elif calls >= cap or caught_cap:
        reasons.append("cap")
    elif not success:
        reasons.append("cap")
    if success:
        reasons = []
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "seed": str(seed),
        "force_calls": calls,
        "verify_calls": int(potential.counter.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "bh_steps": steps,
        "problem": job["problem"],
        "template": job["template"],
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": "default",
        "error": error,
    }


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
    }
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one getEnergyGradient entry.",
        "# The potential is the harness Lennard-Jones potential. A repeated coordinate is cached.",
        "# A start succeeds when a converged quench is within energy_above_minimum of the Wales LJ38 energy.",
        "# The quench is lbfgs_cpp with the archived tolerance. Other quench arguments stay at the constructor.",
        "# The call ceiling is the PAR-2 cap. It is above the archived maximum and is not a quench setting.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _installed_version() -> str:
    from importlib.metadata import version

    return version("pele")


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
    version = _installed_version()
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"pele {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if commit != PELE_COMMIT:
        raise RuntimeError(f"{template['id']} commit {commit} is not the built pele revision {PELE_COMMIT}")
    section = template.get("pele") or {}
    expected = ("method", "temperature", "stepsize", "acc_ratio", "interval", "quench_tol", "nquenches")
    if str(section.get("method") or "") != "basinhopping":
        raise ValueError(f"{template['id']} is not basin hopping")
    unknown = [key for key in section if key not in expected]
    if unknown:
        raise ValueError(f"{template['id']} sets {unknown}; the archived driver does not")
    missing = [key for key in expected if key not in section]
    if missing:
        raise ValueError(f"{template['id']} is missing {missing}")
    tolerance = float((problem.get("convergence") or {})["energy_above_minimum"])
    minimum = reference_energy()
    cap = int(template["failure_cap"])
    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    jobs = []
    for index, (name, coords, _box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "seed": _seed_of(name),
                "cap": cap,
                "temperature": float(section["temperature"]),
                "stepsize": float(section["stepsize"]),
                "acc_ratio": float(section["acc_ratio"]),
                "interval": int(section["interval"]),
                "quench_tol": float(section["quench_tol"]),
                "nquenches": int(section["nquenches"]),
                "energy_minimum": minimum,
                "energy_tolerance": tolerance,
                "problem": problem["id"],
                "template": template["id"],
                "code": template.get("code", "pele"),
                "code_version": version,
                "commit": commit,
            }
        )
    if workers <= 1:
        records = [_hop_job(job) for job in jobs]
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_hop_job, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                print(f"{template['id']} {done}/{total}", flush=True)
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    expected = (problem.get("sample") or {}).get("starts")
    partial = bool(limit) or (expected is not None and len(records) < int(expected))
    legacy_counts = [cap if not record["success"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["success"]]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in records
        ],
        cap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, partial), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(
        json.dumps(
            {
                "schema": "optbench.result/v2",
                "problem": problem["id"],
                "template": template["id"],
                "energy_minimum": minimum,
                "energy_above_minimum": tolerance,
                "failure_cap": cap,
                "metrics": metrics,
                "starts": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary
