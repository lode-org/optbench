"""Minimize Lennard-Jones 38 under Gaussian force noise.

The harness potential is the clean Lennard-Jones energy and gradient.
Noise is added after that call is counted, and a repeated coordinate
returns the same noisy force. Verification reads the clean gradient and
increments ``verify_calls``. Newton and dogleg are refused when sigma is
positive: their analytic Hessian is the clean force, and the noisy force
is not that gradient.

The index removes the six rigid modes. A curvature is negative only
below ``-curvature_tolerance`` of the noise-aware stop.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.noise import (
    KAPPA,
    WINDOW,
    converged_under_noise,
    stop_threshold,
    window_energy_tolerance,
)
from optbench.noise_catalog import HESSIAN_METHODS, allows_method
from optbench.potentials import lennard_jones, lennard_jones_hessian
from optbench.potserver import RgpotCounter
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance, write_starts

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _exception_text(exc: BaseException) -> str:
    """One ``Type: message`` string.

    geomeTRIC already stores ``RuntimeError: Not bracketed``. Raising that
    text again must not add a second type prefix.
    """
    message = str(exc)
    prefix = f"{type(exc).__name__}: "
    if message.startswith(prefix):
        return message
    return prefix + message


class _Observed:
    """Clean potential plus the noisy force the stepper sees."""

    def __init__(self, potential: RgpotCounter, sigma: float, rng: random.Random):
        self.potential = potential
        self.sigma = float(sigma)
        self.rng = rng
        self.energies: list[float] = []
        self.last_norm: float | None = None
        self._key: tuple[float, ...] | None = None
        self._noisy: tuple[float, ...] | None = None

    def __call__(self, coords) -> tuple[float, tuple[float, ...]]:
        key = tuple(float(value) for value in coords)
        before = self.potential.counter.calls
        energy, gradient = self.potential.calculate(key)
        clean = tuple(float(value) for value in gradient)
        if self.potential.counter._verifying:
            return float(energy), clean
        if self.potential.counter.calls == before and self._key == key and self._noisy is not None:
            return float(energy), self._noisy
        noisy = clean
        if self.sigma != 0.0:
            noisy = tuple(value + self.rng.gauss(0.0, self.sigma) for value in clean)
        self._key = key
        self._noisy = noisy
        self.energies.append(float(energy))
        self.last_norm = math.sqrt(sum(value * value for value in noisy))
        return float(energy), noisy

    def met(self, clean_tol: float, energy_tol: float, components: int) -> bool:
        if self.last_norm is None:
            return False
        return converged_under_noise(
            force_norm=self.last_norm,
            energies=self.energies,
            clean_tol=clean_tol,
            sigma=self.sigma,
            components=components,
            energy_tol=energy_tol,
            kappa=KAPPA,
            window=WINDOW,
        )


def _index(point: list[float], force_stop: float) -> tuple[int, list[float]]:
    import numpy as np

    matrix = lennard_jones_hessian(point)
    values = vibrational_eigenvalues(matrix, point, rotations=True)
    tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    import numpy as np

    started = time.perf_counter()
    calls = 0
    verify_calls = 0
    hessian_calls = 0
    error = ""
    report = None
    energy = None
    force_norm = None
    point = None
    potential = None
    stopped = False
    sigma = float(job["sigma"])
    method = str(job["method"])
    family = str(job["family"])
    clean_tol = float(job["clean_tol"])
    energy_tol = float(job["energy_tol"])
    cap = int(job["cap"])
    if family == "rgmin" and not allows_method(method, sigma):
        raise ValueError(
            f"{method} needs the exact Hessian of the clean force, "
            "and the noisy force is not that gradient"
        )
    try:
        x0 = [float(value) for value in job["coords"]]
        components = len(x0)
        counter = HarnessCounter(cache=True, cap=cap)
        potential = RgpotCounter(lambda positions, _numbers=None, _box=None: lennard_jones(positions), counter)
        observed = _Observed(potential, sigma, random.Random(int(job["seed"]) + int(job["index"])))
        if family == "rgmin":
            point, report, hessian_calls, stopped = _drive_rgmin(
                job, x0, observed, potential, cap, sigma, clean_tol, energy_tol, components
            )
        elif family == "ase":
            point, stopped = _drive_ase(job, x0, observed, potential, cap, clean_tol, energy_tol, components)
        elif family == "sella":
            point, stopped = _drive_sella(job, x0, observed, potential, cap, clean_tol, energy_tol, components)
        elif family == "geometric":
            point, stopped = _drive_geometric(job, x0, observed, potential, cap, clean_tol, energy_tol, components)
        elif family == "gmin":
            point, stopped = _drive_gmin(job, x0, observed, potential, cap, clean_tol, energy_tol, components)
        elif family == "vcsqnm":
            point, stopped, drive_error = _drive_vcsqnm(
                job, x0, observed, potential, cap, clean_tol, energy_tol, components
            )
            if drive_error:
                error = drive_error
        elif family == "pele":
            point, stopped, drive_error = _drive_pele(
                job, x0, observed, cap, clean_tol, energy_tol, components
            )
            if drive_error:
                error = drive_error
        else:
            raise ValueError(f"unknown solver family {family}")
    except ForceCallCap as exc:
        error = _exception_text(exc)
    except Exception as exc:
        error = _exception_text(exc)
    if point is None and potential is not None and getattr(potential, "last_point", None) is not None:
        point = potential.last_point
    if potential is not None and point is not None:
        calls = int(potential.counter.calls)
        try:
            with potential.counter.verification():
                energy, gradient = potential.calculate(point)
            force_norm = float(np.linalg.norm(gradient))
            verify_calls = int(potential.counter.verify_calls)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    elif potential is not None:
        calls = int(potential.counter.calls)
        verify_calls = int(potential.counter.verify_calls)
    threshold = stop_threshold(clean_tol, sigma, len(job["coords"]), KAPPA)
    hessian_index = None
    min_eigenvalues: list[float] = []
    if point is not None and (not error or "force-call cap" in error or "ForceCallCap" in error):
        try:
            hessian_index, min_eigenvalues = _index(point, threshold)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    finite = (
        point is not None
        and all(math.isfinite(value) for value in point)
        and energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    at_force = finite and stopped and force_norm is not None and force_norm <= threshold and calls < cap and not error
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
        "hessian_calls": hessian_calls,
        "min_eigenvalues": min_eigenvalues,
        "sigma": sigma,
        "seed": int(job["seed"]) + int(job["index"]),
        "seconds": time.perf_counter() - started,
        "rgmin_stop": None if report is None else str(getattr(report, "stop", "")),
    }
    meta = getattr(potential, "geometric_meta", None) if potential is not None else None
    if meta:
        record["coordinate_system"] = meta["coordinate_system"]
        record["geometric_steps"] = meta["geometric_steps"]
        record["rebuilds"] = meta["rebuilds"]
    if error:
        record["error"] = error
    return record


def _drive_rgmin(job, x0, observed, potential, cap, sigma, clean_tol, energy_tol, components):
    import numpy as np
    import rgmin

    x = np.asarray(x0, dtype=np.float64).copy()
    method = str(job["method"])
    options = dict(job["options"])
    gtol = clean_tol if sigma == 0.0 else 0.0
    config = rgmin.SolverConfig(method, gtol=gtol, maxiter=cap, **options)
    solver = rgmin.Solver(x.size, config)
    hessian_calls = 0
    report = None
    stopped = False

    def fg(coords):
        energy, gradient = observed(coords)
        return float(energy), np.asarray(gradient, dtype=np.float64)

    def hess(coords):
        nonlocal hessian_calls
        hessian_calls += 1
        return lennard_jones_hessian(coords)

    restarts = 0
    for _ in range(cap + 1):
        if potential.counter.calls >= cap:
            break
        before = potential.counter.calls
        if method in HESSIAN_METHODS:
            report = solver.step_hess(fg, hess, x)
        else:
            report = solver.step(fg, x)
        if sigma == 0.0:
            if report.stop != rgmin.Stop.Budget:
                break
            continue
        if observed.met(clean_tol, energy_tol, components):
            stopped = True
            break
        # Budget is the running state. Any other stop, or a step that
        # spends no call, rebuilds the solver at this point. A second
        # stall ends the search.
        if report.stop == rgmin.Stop.Budget and potential.counter.calls != before:
            continue
        restarts += 1
        if restarts > 2 and potential.counter.calls == before:
            break
        solver = rgmin.Solver(x.size, config)
    if sigma == 0.0:
        stopped = report is not None and report.stop == rgmin.Stop.Tolerance
    point = [float(value) for value in x]
    return point, report, hessian_calls, stopped


def _atoms(coords):
    from ase import Atoms

    rows = [coords[offset : offset + 3] for offset in range(0, len(coords), 3)]
    return Atoms(symbols=["H"] * len(rows), positions=rows, pbc=False)


def _calculator(atoms, observed, potential):
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    class _HarnessLJ(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            flat = np.asarray(self.atoms.positions, dtype=np.float64).reshape(-1)
            energy, gradient = observed(flat)
            self.results["energy"] = float(energy)
            self.results["forces"] = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)

    atoms.calc = _HarnessLJ()
    return atoms


def _step_optimizer(opt, observed, potential, cap, clean_tol, energy_tol, components) -> bool:
    while potential.counter.calls < cap:
        before = potential.counter.calls
        try:
            opt.run(fmax=0.0, steps=1)
        except ForceCallCap:
            break
        except RuntimeError as exc:
            if "force-call cap" not in str(exc):
                raise
            break
        if observed.met(clean_tol, energy_tol, components):
            return True
        if potential.counter.calls == before:
            break
    return observed.met(clean_tol, energy_tol, components)


def _drive_gmin(job, x0, observed, potential, cap, clean_tol, energy_tol, components):
    import numpy as np

    from optbench.gmin_lbfgs import DEFAULTS, minimize

    x = np.asarray(x0, dtype=np.float64).copy()
    sigma = float(job["sigma"])
    options = job["options"]
    threshold = stop_threshold(clean_tol, sigma, components, KAPPA)

    def evaluate(coords):
        energy, gradient = observed(coords)
        return float(energy), np.asarray(gradient, dtype=np.float64)

    def ready():
        return observed.met(clean_tol, energy_tol, components)

    result = minimize(
        x,
        evaluate,
        gtol=threshold,
        cap=cap,
        m=int(options.get("m", DEFAULTS["m"])),
        maxbfgs=float(options.get("maxbfgs", DEFAULTS["maxbfgs"])),
        dguess=float(options.get("dguess", DEFAULTS["dguess"])),
        maxerise=float(options.get("maxerise", DEFAULTS["maxerise"])),
        maxefall=float(options.get("maxefall", DEFAULTS["maxefall"])),
        ready=ready,
    )
    if result.capped:
        raise ForceCallCap("force-call cap")
    stopped = bool(result.mflag and result.info == 0 and observed.met(clean_tol, energy_tol, components))
    return [float(value) for value in result.coords], stopped


def _drive_ase(job, x0, observed, potential, cap, clean_tol, energy_tol, components):
    from optbench.ase_sella import _build_optimizer

    atoms = _calculator(_atoms(x0), observed, potential)
    parameters = {}
    if job["options"].get("maxstep") is not None:
        parameters["maxstep"] = float(job["options"]["maxstep"])
    opt = _build_optimizer({"name": job["optimizer"], "parameters": parameters}, atoms)
    stopped = _step_optimizer(opt, observed, potential, cap, clean_tol, energy_tol, components)
    point = [float(value) for row in atoms.positions for value in row]
    return point, stopped


def _drive_geometric(job, x0, observed, potential, cap, clean_tol, energy_tol, components):
    import numpy as np

    from optbench.runners.geometric_min import optimize

    def evaluate(flat):
        energy, gradient = observed(np.asarray(flat, dtype=np.float64).reshape(-1))
        return float(energy), np.asarray(gradient, dtype=np.float64)

    def stop(_gradient):
        return observed.met(clean_tol, energy_tol, components)

    report = optimize(
        {
            "coords": x0,
            "box": None,
            "kind": "lj",
            "coordsys": str(job["options"].get("coordsys") or job.get("method") or "cart"),
            "neighbors": int(job["options"].get("neighbors") or 2),
            "element": str(job["options"].get("element") or "H"),
            "cap": cap,
            "start": job.get("start"),
        },
        evaluate,
        stop,
    )
    potential.last_point = report["point"]
    potential.geometric_meta = {
        "coordinate_system": str(report.get("coordinate_system") or ""),
        "geometric_steps": int(report.get("steps") or 0),
        "rebuilds": int(report.get("rebuilds") or 0),
    }
    if report.get("error"):
        text = str(report["error"])
        if "ForceCallCap" in text or "force-call cap" in text:
            raise ForceCallCap(text)
        raise RuntimeError(text)
    return report["point"], bool(report["stopped"])


def _drive_vcsqnm(job, x0, observed, potential, cap, clean_tol, energy_tol, components):
    import numpy as np

    from optbench.runners.vcsqnm_min import drive

    method = str(job["method"])
    if method not in {"sqnm", "ase"}:
        raise ValueError(f"{method} is not a fixed-cell SQNM method")

    class _Noisy:
        def __init__(self):
            self.energy = None
            self.grad = None
            self.last_deriv = None

        def __call__(self, flat, cell=None):
            energy, grad = observed(flat)
            self.energy = float(energy)
            self.grad = np.asarray(grad, dtype=np.float64).reshape(-1)
            return self.energy, self.grad

    seen = _Noisy()
    result = drive(
        x0,
        method,
        dict(job["options"]),
        seen,
        lambda: observed.met(clean_tol, energy_tol, components),
        lambda: potential.counter.calls,
        cap,
        None,
    )
    return result["point"], bool(result["stopped"]), str(result["error"])


class _PeleObserved:
    """pele potential over the noisy harness force.

    Counting stays in ``observed``. ``last`` is the last evaluated point.
    """

    def __init__(self, observed):
        self.observed = observed
        self.counter = observed.potential.counter
        self.last: list[float] | None = None

    def getEnergyGradient(self, coords):
        import numpy as np

        flat = np.asarray(coords, dtype=np.float64).reshape(-1)
        energy, gradient = self.observed(flat)
        self.last = [float(value) for value in flat]
        return float(energy), np.asarray(gradient, dtype=np.float64)

    def getEnergy(self, coords):
        energy, _gradient = self.getEnergyGradient(coords)
        return energy


def _drive_pele(job, x0, observed, cap, clean_tol, energy_tol, components):
    from optbench.runners.pele_min import drive_minimizer, merged_options

    method = str(job["method"])
    options = merged_options(method, dict(job["options"]))

    def stop(*, energy, gradient, tol, coords=None) -> bool:
        return observed.met(clean_tol, energy_tol, components)

    return drive_minimizer(
        method,
        x0,
        _PeleObserved(observed),
        stop=stop,
        cap=int(cap),
        options=options,
    )


def _drive_sella(job, x0, observed, potential, cap, clean_tol, energy_tol, components):
    from sella import Sella

    atoms = _calculator(_atoms(x0), observed, potential)
    opt = Sella(
        atoms,
        order=int(job["options"].get("order", 0)),
        logfile=None,
        max_cpu_threads=1,
    )
    stopped = _step_optimizer(opt, observed, potential, cap, clean_tol, energy_tol, components)
    point = [float(value) for row in atoms.positions for value in row]
    return point, stopped


def _dat(summary: dict, template: dict, metrics: dict, sigma: float, ended: dict[str, int] | None = None) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "problem": template.get("problem"),
        "sigma": sigma,
        "par2_mean": metrics.get("par2_mean"),
        "rate_low": metrics.get("rate_low"),
        "rate_high": metrics.get("rate_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "cap": summary.get("cap"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter harness. One force call is one clean energy and gradient.",
        "# Noise is added after the call. A repeated coordinate keeps that sample.",
        "# The index uses the clean Hessian. Six rigid modes are removed.",
        "# A curvature is negative only below -max(force_norm, 1e-8*||H||_F, 1e-6).",
        "# Energy and Kabsch RMSD are not applied: the problem file has no per-start reference.",
    ]
    if ended:
        parts = ", ".join(f"{name} {count}" for name, count in sorted(ended.items()))
        lines.append("# final coordinate system: " + parts)
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
    if template.get("engine") != "noise":
        raise ValueError(f"{template['id']} is not a noise template")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    if problem["id"] != "minimization/lj38-noise":
        raise ValueError(f"{template['id']} does not run on {problem['id']}")
    solver = template.get("solver") or {}
    family = str(solver.get("family") or "")
    noise = template.get("noise") or {}
    sigma = float(noise["sigma"])
    method = str(solver.get("method") or solver.get("name") or family)
    if family == "rgmin" and not allows_method(str(solver.get("method")), sigma):
        raise ValueError(
            f"{template['id']} asks for {solver.get('method')} at sigma {sigma}. "
            "That method needs the exact Hessian of the clean force."
        )
    if family == "rgmin":
        import rgmin

        version = str(rgmin.__version__)
        if template.get("code_version") and version != str(template["code_version"]):
            raise RuntimeError(f"rgmin {version} does not match template {template['code_version']}")
    if family == "geometric":
        import geometric

        version = str(geometric.__version__)
        revision = str(getattr(geometric, "__git_revision__", "") or "")
        if template.get("code_version") and version != str(template["code_version"]):
            raise RuntimeError(f"geometric {version} does not match template {template['code_version']}")
        if template.get("commit") and revision and revision != str(template["commit"]):
            raise RuntimeError(f"geometric revision {revision} does not match template {template['commit']}")
    if family == "gmin":
        from optbench.gmin_lbfgs import require_revision

        found = require_revision(str(template.get("commit") or ""))
        version = str(template.get("code_version") or "")
        if version and not found.startswith(version):
            raise RuntimeError(f"GMIN revision {found} does not match template {version}")
    if family == "vcsqnm":
        from optbench.runners.vcsqnm_min import assert_pin
        from optbench.vcsqnm_catalog import CODE_VERSION, COMMIT

        assert_pin()
        if str(template.get("commit") or "") != COMMIT:
            raise RuntimeError(f"{template['id']} commit is not {COMMIT}")
        if str(template.get("code_version") or "") != CODE_VERSION:
            raise RuntimeError(f"{template['id']} code_version is not tag {CODE_VERSION}")
        if str(solver.get("method")) not in {"sqnm", "ase"}:
            raise ValueError(f"{template['id']} moves a cell this potential does not have")
    if family == "pele":
        from importlib.metadata import version

        found = version("pele")
        if template.get("code_version") and found != str(template["code_version"]):
            raise RuntimeError(f"pele {found} does not match template {template['code_version']}")
    convergence = problem.get("convergence") or {}
    clean_tol = float(convergence["force_norm"])
    if "energy_tol" not in convergence:
        raise ValueError(f"{problem['id']} has no energy_tol for the noise window")
    energy_tol = window_energy_tolerance(float(convergence["energy_tol"]), sigma, float((problem.get("noise") or {}).get("kappa", KAPPA)))
    cap = int(convergence["max_force_calls"])
    seed = int(noise.get("seed", (problem.get("noise") or {})["seed"]))
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    options = {
        key: value
        for key, value in solver.items()
        if key not in {"family", "method", "name"}
    }
    jobs = []
    for index, (name, coords, _box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": [float(value) for value in coords],
                "family": family,
                "method": str(solver.get("method") or ""),
                "optimizer": str(solver.get("name") or ""),
                "options": options,
                "sigma": sigma,
                "seed": seed,
                "clean_tol": clean_tol,
                "energy_tol": energy_tol,
                "cap": cap,
            }
        )
    done_rows: dict[int, dict] = {}
    partial_path = output.with_suffix(".partial.jsonl")
    if partial_path.is_file():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_rows[int(row["index"])] = row
    pending = [job for job in jobs if job["index"] not in done_rows]
    if pending:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("a", encoding="utf-8") as handle:
            if workers <= 1:
                for job in pending:
                    record = _minimize_job(job)
                    done_rows[int(record["index"])] = record
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
            else:
                import multiprocessing as mp

                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    futures = {pool.submit(_minimize_job, job): job["index"] for job in pending}
                    total = len(jobs)
                    finished = len(done_rows)
                    for future in as_completed(futures):
                        record = future.result()
                        done_rows[int(record["index"])] = record
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        handle.flush()
                        finished += 1
                        if finished == total or finished % 50 == 0:
                            print(f"{template['id']} {finished}/{total}", flush=True)
    records = [done_rows[job["index"]] for job in jobs]
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["converged"]]
    summary["median"] = statistics.median(successes) if successes else None
    summary["cap"] = cap
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
    passed = {"sigma": sigma, "family": family, "method": method, "cap": cap, "energy_tol": energy_tol, **options}
    provenance = collect_provenance(
        code=str(template.get("code")),
        code_version=str(template.get("code_version")),
        commit=str(template.get("commit")),
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. Noise is applied to the force after the clean call. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm", method),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
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
    ended: dict[str, int] = {}
    for record in records:
        name = str(record.get("coordinate_system") or "")
        if name:
            ended[name] = ended.get(name, 0) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, sigma, ended), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if partial_path.is_file() and len(records) == len(jobs):
        partial_path.unlink()
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["seconds"] = sum(float(record.get("seconds") or 0.0) for record in records)
    return summary
