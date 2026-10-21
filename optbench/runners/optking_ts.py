"""OptKing transition-state search on the harness potential.

The step types on revision 767e2c0c are ``RS_I_RFO`` and ``P_RFO``.
``RS_PRFO`` is a comment on the RFO base class and is not a step type.

One force call is one ``HarnessCounter`` entry. The initial Hessian is a
central difference of that gradient on the free coordinates, and each
displaced evaluation is a force call. Later steps use the Bofill update
and cost one gradient. A repeated geometry is one call. The Hessian
index is an analytic matrix and is not a search call.

A 2014 success is under the cap and at the problem force stop. A v2
success also has Hessian index 1. LJ38 removes rigid modes. The Pt
island leaves fixed atoms out of the matrix and does not project rigid
modes. Curvature counts only below ``-curvature_tolerance`` of the
force stop.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

PIN = "767e2c0c13ff2b851ac82101ccd5aab9a558a4f1"
VERSION = "0.5.1.dev2+g767e2c0c1"
STEP_TYPES = frozenset({"RS_I_RFO", "P_RFO"})
FD_STEP = 1.0e-5
STALL_STEPS = 12
STALL_DISPLACEMENT = 1.0e-8


def installed_version() -> str:
    """Distribution version. Importing OptKing also opens a log file."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("optking")
    except PackageNotFoundError as exc:
        raise RuntimeError("optking is not installed") from exc


def parse_con(text: str) -> dict:
    """Positions, frozen flags, element symbol, and orthorhombic lengths.

    Column four of a coordinate row is the frozen flag. ``1`` is frozen.
    ``read_con`` refuses those rows, and the Pt island uses them.
    """
    lines = text.splitlines()
    box = None
    for index, line in enumerate(lines[:-1]):
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            lengths = [float(part) for part in parts]
            angles = [float(part) for part in lines[index + 1].split()]
        except ValueError:
            continue
        if len(angles) == 3:
            box = lengths
            break
    if box is None:
        raise ValueError("con file has no box")
    start = next(index for index, line in enumerate(lines) if line.startswith("Coordinates of Component"))
    symbol = "X"
    for line in reversed(lines[:start]):
        token = line.strip()
        if token.isalpha():
            symbol = token
            break
    positions: list[list[float]] = []
    fixed: list[bool] = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            if positions:
                break
            continue
        try:
            xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            if positions:
                break
            continue
        flag = 0
        if len(parts) >= 4:
            try:
                flag = int(float(parts[3]))
            except ValueError:
                flag = 0
        positions.append(xyz)
        fixed.append(flag != 0)
    if not positions:
        raise ValueError("con file has no coordinates")
    return {"positions": positions, "fixed": fixed, "symbol": symbol, "box": box}


def counted_call(counter: HarnessCounter, evaluate, coords):
    """One harness entry, unless the coordinates repeat the cached geometry."""
    import numpy as np

    flat = np.asarray(coords, dtype=np.float64).reshape(-1)
    key = tuple(float(value) for value in flat)
    fresh, cached = counter.charge(key)
    if not fresh and cached is not None:
        energy, gradient = cached
        return float(energy), np.asarray(gradient, dtype=np.float64)
    energy, gradient = evaluate(flat)
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    stored = (float(energy), tuple(float(value) for value in gradient))
    counter.remember(key, stored)
    return stored[0], np.asarray(stored[1], dtype=np.float64)


def finite_difference_hessian(evaluate, coords, step: float):
    """Central difference of a Cartesian gradient. ``evaluate`` is counted."""
    import numpy as np

    point = np.asarray(coords, dtype=np.float64).reshape(-1).copy()
    dimension = int(point.size)
    hessian = np.zeros((dimension, dimension), dtype=np.float64)
    step = float(step)
    for index in range(dimension):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        _plus_energy, plus_gradient = evaluate(plus)
        _minus_energy, minus_gradient = evaluate(minus)
        hessian[:, index] = (plus_gradient - minus_gradient) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def morse_smooth_hessian(positions, cell_lengths, pbc, D: float, alpha: float, r0: float, rc: float):
    """Cartesian Hessian of the smooth-linear Morse used for the Pt island.

    The linear term matches energy and slope at the cutoff, so it does not
    change the second derivative inside the cutoff. Axes marked periodic
    use the orthorhombic minimum image. A nonperiodic axis is not wrapped.
    """
    import numpy as np

    from optbench.potentials import _hessian_from_pairs

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lengths = np.asarray(cell_lengths, dtype=np.float64).reshape(3)
    periodic = np.asarray(pbc, dtype=bool).reshape(3)
    count = int(pos.shape[0])
    iu, ju = np.triu_indices(count, k=1)
    diff = pos[iu] - pos[ju]
    if bool(periodic.any()):
        wrapped = lengths * np.round(diff / lengths)
        wrapped[:, ~periodic] = 0.0
        diff = diff - wrapped
    r2 = np.einsum("ij,ij->i", diff, diff)
    mask = (r2 > 0.0) & (r2 < rc * rc)
    iu, ju, diff, r2 = iu[mask], ju[mask], diff[mask], r2[mask]
    if iu.size == 0:
        return np.zeros((3 * count, 3 * count), dtype=np.float64)
    distance = np.sqrt(r2)
    x = np.exp(-alpha * (distance - r0))
    x_cut = math.exp(-alpha * (rc - r0))
    slope_cut = 2.0 * alpha * D * x_cut * (1.0 - x_cut)
    first = 2.0 * alpha * D * x * (1.0 - x) - slope_cut
    second = 2.0 * alpha * alpha * D * x * (2.0 * x - 1.0)
    return _hessian_from_pairs(count, iu, ju, diff, r2, first, second)


def _free_block(matrix, fixed):
    import numpy as np

    flags = np.asarray(fixed, dtype=bool)
    if not bool(flags.any()):
        return matrix
    chosen = np.flatnonzero(~flags)
    columns = np.concatenate([np.arange(3 * int(atom), 3 * int(atom) + 3) for atom in chosen])
    return matrix[np.ix_(columns, columns)]


def index_at(job: dict, positions) -> tuple[int, float]:
    """Index 1 is a saddle. This matrix is not a search call."""
    import numpy as np

    from optbench.potentials import lennard_jones_hessian
    from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    fixed = np.asarray(job["fixed"], dtype=bool)
    stop = float(job["gtol"])
    kind = str(job["potential"]["kind"])
    if kind == "lennard-jones":
        matrix = lennard_jones_hessian(pos)
        if bool(fixed.any()):
            block = _free_block(matrix, fixed)
            values = np.linalg.eigvalsh(0.5 * (block + block.T))
            tolerance = curvature_tolerance(stop, float(np.linalg.norm(block)))
        else:
            values = np.asarray(
                vibrational_eigenvalues(matrix, pos, rotations=True),
                dtype=np.float64,
            )
            tolerance = curvature_tolerance(stop, float(np.linalg.norm(matrix)))
    elif kind == "morse":
        potential = job["potential"]
        matrix = morse_smooth_hessian(
            pos,
            job["box"],
            job["pbc"],
            float(potential["D"]),
            float(potential["alpha"]),
            float(potential["r0"]),
            float(potential["rc"]),
        )
        block = _free_block(matrix, fixed)
        values = np.linalg.eigvalsh(0.5 * (block + block.T))
        tolerance = curvature_tolerance(stop, float(np.linalg.norm(block)))
    else:
        raise ValueError(kind)
    lowest = float(values[0]) if len(values) else 0.0
    return negative_index(values, tolerance), lowest


def _evaluate_factory(job: dict):
    import numpy as np

    positions = np.asarray(job["positions"], dtype=np.float64).reshape(-1, 3)
    fixed = np.asarray(job["fixed"], dtype=bool)
    free = np.flatnonzero(~fixed)
    if free.size == 0:
        raise ValueError(f"{job['start']} has no free atoms")
    kind = str(job["potential"]["kind"])
    if kind == "lennard-jones":
        from optbench.potentials import lennard_jones

        def evaluate(free_coords):
            pos = positions.copy()
            pos[free] = np.asarray(free_coords, dtype=np.float64).reshape(-1, 3)
            energy, gradient = lennard_jones(pos)
            return float(energy), np.asarray(gradient, dtype=np.float64).reshape(-1, 3)[free].reshape(-1)

    elif kind == "morse":
        from optbench.ase_sella import _morse_arrays

        cell = np.diag(np.asarray(job["box"], dtype=np.float64))
        periodic = np.asarray(job["pbc"], dtype=bool)
        potential = job["potential"]

        def evaluate(free_coords):
            pos = positions.copy()
            pos[free] = np.asarray(free_coords, dtype=np.float64).reshape(-1, 3)
            energy, forces = _morse_arrays(
                pos,
                cell,
                periodic,
                float(potential["D"]),
                float(potential["alpha"]),
                float(potential["r0"]),
                float(potential["rc"]),
                str(potential["form"]),
            )
            gradient = -np.asarray(forces, dtype=np.float64)[free].reshape(-1)
            return float(energy), gradient

    else:
        raise ValueError(kind)
    return evaluate, positions, free


def _record(job: dict, **fields) -> dict:
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": 0,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": 1,
        "converged": False,
        "hessian_calls": 0,
        "steps": 0,
        "lowest_eigenvalue": None,
    }
    record.update(fields)
    return record


def _finish(job, counter, outcome, error, last, steps, hessian_calls) -> dict:
    import numpy as np

    calls = int(counter.calls) if counter is not None else 0
    energy = None
    force_norm = None
    point = None
    full = None
    if last is not None:
        full, energy, gradient = last
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if gradient.size:
            force_norm = float(np.linalg.norm(gradient))
        point = np.asarray(full, dtype=np.float64).reshape(-1)
    finite = (
        point is not None
        and energy is not None
        and force_norm is not None
        and math.isfinite(energy)
        and math.isfinite(force_norm)
        and bool(np.all(np.isfinite(point)))
    )
    cap = int(job["cap"])
    gtol = float(job["gtol"])
    at_force = (
        finite
        and outcome == "converged"
        and force_norm is not None
        and force_norm <= gtol
        and calls < cap
        and not error
    )
    index = None
    lowest = None
    if at_force and full is not None:
        try:
            index, lowest = index_at(job, full)
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            outcome = "crash"
            at_force = False
    if outcome == "crash":
        reasons = ["crash"]
    elif not finite:
        reasons = ["divergence"]
    elif calls >= cap or outcome == "cap":
        reasons = ["cap"]
    elif not at_force:
        reasons = ["divergence"]
    elif index != 1:
        reasons = ["hessian_index"]
    else:
        reasons = []
    failure = classify_failure(reasons)
    return _record(
        job,
        force_calls=calls,
        verify_calls=0 if counter is None else int(counter.verify_calls),
        success=failure is None,
        failure=failure,
        reasons=reasons,
        energy=None if energy is None or not math.isfinite(float(energy)) else float(energy),
        force_norm=force_norm if force_norm is not None and math.isfinite(force_norm) else None,
        hessian_index=index,
        converged=bool(at_force),
        hessian_calls=int(hessian_calls),
        steps=int(steps),
        lowest_eigenvalue=lowest,
        error=error if outcome == "crash" and error else None,
    )


def search_start(job: dict) -> dict:
    """Run one start. OptKing's own convergence test is not the stop."""
    import logging

    scratch = Path(job["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    counter = None
    outcome = "crash"
    error = ""
    last = None
    steps = 0
    hessian_calls = 0
    try:
        os.chdir(scratch)
        logging.disable(logging.CRITICAL)
        import numpy as np
        import qcelemental as qcel
        from optking.exceptions import AlgError, OptError
        from optking.opt_helper import CustomHelper

        logging.disable(logging.CRITICAL)
        evaluate, positions, free = _evaluate_factory(job)
        free_positions = positions[free]
        counter = HarnessCounter(cache=True, cap=int(job["cap"]))
        molecule = qcel.models.Molecule(
            symbols=[str(job["symbol"])] * int(free.size),
            geometry=free_positions.reshape(-1).tolist(),
            fix_com=False,
            fix_orientation=False,
            molecular_charge=0,
            molecular_multiplicity=1,
        )
        helper = CustomHelper(
            molecule,
            {
                "opt_type": "TS",
                "step_type": str(job["step_type"]),
                "opt_coordinates": "CARTESIAN",
                "full_hess_every": int(job["full_hess_every"]),
                "geom_maxiter": int(job["cap"]) + 5,
                "alg_geom_maxiter": int(job["cap"]) + 5,
                "print_lvl": 1,
            },
        )
        opened = np.asarray(helper.geom, dtype=np.float64).reshape(-1)
        if opened.shape != free_positions.reshape(-1).shape or float(np.max(np.abs(opened - free_positions.reshape(-1)))) > 1e-8:
            raise RuntimeError("OptKing moved the input coordinates before the first step")
        stall = 0
        while True:
            geom = np.array(helper.geom, dtype=np.float64, copy=True).reshape(-1)
            try:
                energy, gradient = counted_call(counter, evaluate, geom)
            except ForceCallCap:
                outcome = "cap"
                break
            gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
            force = float(np.linalg.norm(gradient))
            full = positions.copy()
            full[free] = geom.reshape(-1, 3)
            last = (full, float(energy), gradient.copy())
            if not math.isfinite(energy) or not math.isfinite(force) or not bool(np.all(np.isfinite(geom))):
                outcome = "divergence"
                break
            if force <= float(job["gtol"]) and counter.calls < int(job["cap"]):
                outcome = "converged"
                break
            needed = helper.calculations_needed()
            hessian = None
            if "hessian" in needed:
                try:
                    hessian = finite_difference_hessian(
                        lambda trial: counted_call(counter, evaluate, trial),
                        geom,
                        float(job["fd_step"]),
                    )
                except ForceCallCap:
                    outcome = "cap"
                    break
                hessian_calls += 1
            helper.E = float(energy)
            helper.gX = gradient
            helper.HX = None if hessian is None else hessian
            try:
                helper.compute()
                helper.take_step()
            except ForceCallCap:
                outcome = "cap"
                break
            except OptError as exc:
                text = str(exc)
                if "Maximum number of steps" in text:
                    outcome = "cap" if counter.calls >= int(job["cap"]) else "divergence"
                else:
                    outcome = "crash"
                    error = f"{type(exc).__name__}: {exc}"
                break
            except (AlgError, ValueError, np.linalg.LinAlgError) as exc:
                outcome = "crash"
                error = f"{type(exc).__name__}: {exc}"
                break
            steps += 1
            if steps % 25 == 0:
                (scratch / "progress.txt").write_text(
                    f"steps {steps} calls {counter.calls} force {force}\n",
                    encoding="utf-8",
                )
            moved = np.array(helper.geom, dtype=np.float64, copy=True).reshape(-1)
            displacement = float(np.linalg.norm(moved - geom))
            if displacement < STALL_DISPLACEMENT and force > float(job["gtol"]):
                stall += 1
            else:
                stall = 0
            if stall >= STALL_STEPS:
                outcome = "divergence"
                error = error or "cartesian step stayed below 1e-8"
                break
            if counter.calls >= int(job["cap"]):
                outcome = "cap"
                break
    except ForceCallCap:
        outcome = "cap"
    except Exception as exc:
        outcome = "crash"
        error = error or f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
    record = _finish(job, counter, outcome, error, last, steps, hessian_calls)
    if record.get("error") is None:
        record.pop("error", None)
    return record


def _check_potential(problem: dict, potential: dict) -> None:
    """The template potential is the problem potential, including the cutoff."""
    stated = problem.get("potential") or {}
    if str(stated.get("kind")) != str(potential.get("kind")):
        raise ValueError("template potential kind is not the problem potential")
    if potential["kind"] == "lennard-jones":
        pairs = (
            ("epsilon", "epsilon"),
            ("sigma", "sigma"),
            ("rc", "cutoff"),
        )
    elif potential["kind"] == "morse":
        pairs = (
            ("D", "D"),
            ("alpha", "alpha"),
            ("r0", "r0"),
            ("rc", "cutoff"),
        )
        if str(potential.get("form")) != "smooth" or str(stated.get("pair_style")) != "morse/smooth/linear":
            raise ValueError("template Morse form is not morse/smooth/linear")
    else:
        raise ValueError(potential["kind"])
    for left, right in pairs:
        if abs(float(potential[left]) - float(stated[right])) > 1e-12:
            raise ValueError(f"template {left} is not the problem {right}")
    if [bool(value) for value in potential.get("pbc", [])] != [bool(value) for value in stated.get("pbc", [])]:
        raise ValueError("template pbc is not the problem pbc")


def _select(path: Path, select: str) -> bool:
    if select == "initial":
        return path.name.startswith("initial_")
    if select == "con":
        return path.suffix == ".con" and path.name != "reactant.con"
    raise ValueError(f"unsupported input select {select}")


def _sort_key(path: Path) -> tuple:
    digits = "".join(character if character.isdigit() else " " for character in path.stem).split()
    number = int(digits[-1]) if digits else 0
    return (number, path.name)


def load_jobs(root: Path, problem: dict, template: dict, *, geometries: Path | None, work: Path | None, output: Path) -> list[dict]:
    """One job dict per archived start, reactant excluded."""
    import tarfile

    from optbench.campaign import geometry_archive, iter_geometries

    select = str((template.get("input") or {}).get("select") or "con")
    if geometries is None:
        archive = geometry_archive(root, problem)
        dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".extracted"
        recorded = archive.name
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() != recorded:
            raise RuntimeError(f"{dest} was extracted from another archive")
        if not marker.is_file():
            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text(recorded, encoding="utf-8")
        cons = [path for path in dest.rglob("*.con") if _select(path, select)]
    else:
        cons = [path for path in iter_geometries(geometries) if _select(path, select)]
    cons.sort(key=_sort_key)
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected is not None and len(cons) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(cons)}")
    potential = dict(template["potential"])
    _check_potential(problem, potential)
    pbc = [bool(value) for value in potential.get("pbc", [False, False, False])]
    section = template.get("optking") or {}
    step_type = str(section.get("step_type") or "")
    if step_type not in STEP_TYPES:
        raise ValueError(f"{template['id']} step_type must be RS_I_RFO or P_RFO")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    stop = template.get("stop") or {}
    if "force_norm" in stop and abs(float(stop["force_norm"]) - gtol) > 1e-12:
        raise ValueError("template force_norm is not the problem force norm")
    if "max_force_calls" in stop and int(stop["max_force_calls"]) < cap:
        raise ValueError("template max_force_calls is below the problem force-call cap")
    jobs = []
    scratch_root = (work or output.parent / "work") / template["id"].replace("/", "-")
    for index, path in enumerate(cons):
        parsed = parse_con(path.read_text(encoding="utf-8", errors="replace"))
        jobs.append(
            {
                "index": index,
                "start": path.stem,
                "positions": parsed["positions"],
                "fixed": parsed["fixed"],
                "symbol": parsed["symbol"],
                "box": parsed["box"],
                "pbc": pbc,
                "potential": potential,
                "step_type": step_type,
                "full_hess_every": int(section.get("full_hess_every", 0)),
                "fd_step": float(section.get("fd_step", FD_STEP)),
                "gtol": gtol,
                "cap": cap,
                "scratch": str(scratch_root / path.stem),
            }
        )
    return jobs


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, *, partial: bool) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "force_calls_median": summary.get("force_calls_median"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
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
        "n_cap": metrics.get("n_cap"),
        "n_divergence": metrics.get("n_divergence"),
        "n_wrong_stationary": metrics.get("n_wrong_stationary"),
        "n_crash": metrics.get("n_crash"),
    }
    lines = [
        "# counter harness. One force call is one HarnessCounter entry.",
        "# The initial Hessian is a central difference of the harness gradient on the free atoms.",
        "# Each displaced gradient is a force call. Later steps use the Bofill update and cost one gradient.",
        "# The Hessian index is analytic and is not a search call.",
        "# The 2014 mean is the half-even mean of starts at the force stop and under the cap.",
        "# v2 success also requires Hessian index 1.",
        "# A curvature is negative only below -max(force_norm, 1e-8*||H||_F, 1e-6).",
        "# LJ38 removes rigid modes. The Pt island keeps fixed atoms out of the matrix and does not project rigid modes.",
        "# OptKing's own convergence test is not the stop. The stop is the problem Euclidean force norm.",
        "# Each worker uses one BLAS thread.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    if partial:
        lines.append("# partial true")
        rows["partial"] = "true"
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _publish(output: Path, records: list[dict], template: dict, problem: dict, *, version: str, commit: str, cap: int, partial: bool) -> dict:
    ordered = sorted(records, key=lambda row: (row.get("index", 0), str(row.get("start"))))
    public = []
    for record in ordered:
        row = dict(record)
        row.pop("index", None)
        if row.get("error") is None:
            row.pop("error", None)
        public.append(row)
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in public]
    summary = summarize_capped(legacy_counts, cap) if public else {
        "force_calls": None,
        "force_calls_min": None,
        "force_calls_max": None,
        "nfailed": 0,
    }
    successes = [int(record["force_calls"]) for record in public if record["converged"]]
    summary["median"] = statistics.median(successes) if successes else None
    summary["force_calls_median"] = summary["median"]
    if public:
        metrics = score_v2(
            [
                {
                    "force_calls": record["force_calls"],
                    "success": record["success"],
                    "failure": record["failure"],
                }
                for record in public
            ],
            cap,
        )
    else:
        metrics = {}
    section = template.get("optking") or {}
    passed = {
        "opt_type": "TS",
        "step_type": section.get("step_type"),
        "opt_coordinates": "CARTESIAN",
        "full_hess_every": section.get("full_hess_every", 0),
        "fd_step": section.get("fd_step", FD_STEP),
        "gtol": float((problem.get("convergence") or {})["force_norm"]),
        "max_force_calls": cap,
    }
    provenance = collect_provenance(
        code="optking",
        code_version=version,
        commit=commit,
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. The initial Hessian is a counted central difference. "
            "The index Hessian is analytic and is not a search call."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "optking",
                "algorithm": template.get("algorithm"),
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
                    "force_calls_median": summary.get("force_calls_median"),
                    "nfailed": summary.get("nfailed"),
                    "median": summary.get("median"),
                },
                "starts": public,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, partial=partial), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), public)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(public)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics.get("par2_mean")
    summary["v2_nfailed"] = metrics.get("nfailed")
    summary["partial"] = partial
    return summary


def _load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[str(row["start"])] = row
    return done


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
    resume: bool = False,
) -> dict:
    version = installed_version()
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"optking {version} does not match template {template['code_version']}")
    if "767e2c0" not in version:
        raise RuntimeError(f"optking {version} is not pin {PIN}")
    commit = str(template.get("commit") or "")
    if commit != PIN:
        raise RuntimeError(f"{template['id']} commit is not {PIN}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    jobs = load_jobs(root, problem, template, geometries=geometries, work=work, output=output)
    expected = len(jobs)
    if limit > 0:
        jobs = jobs[:limit]
    cap = int(jobs[0]["cap"]) if jobs else int(problem["convergence"]["max_force_calls"])
    starts_path = output.with_suffix(".starts.jsonl")
    done = _load_done(starts_path) if resume else {}
    if not resume and starts_path.exists():
        starts_path.unlink()
        done = {}
    pending = []
    records = []
    for job in jobs:
        previous = done.get(job["start"])
        if previous is not None:
            previous = dict(previous)
            previous["index"] = job["index"]
            records.append(previous)
            continue
        pending.append(job)
    partial = len(jobs) < expected

    def publish() -> dict:
        return _publish(
            output,
            records,
            template,
            problem,
            version=version,
            commit=commit,
            cap=cap,
            partial=partial or len(records) < expected,
        )

    if pending:
        def _note(record: dict) -> None:
            print(
                f"{template['id']} {record['start']} calls={record['force_calls']} "
                f"failure={record.get('failure')} {len(records)}/{len(jobs)}",
                flush=True,
            )

        if workers <= 1:
            for done_count, job in enumerate(pending, 1):
                records.append(search_start(job))
                _note(records[-1])
                if done_count == len(pending) or done_count % 10 == 0:
                    publish()
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(search_start, job) for job in pending]
                finished = 0
                total = len(futures)
                for future in as_completed(futures):
                    records.append(future.result())
                    finished += 1
                    _note(records[-1])
                    if finished == total or finished % 10 == 0:
                        publish()
    summary = publish()
    return summary
