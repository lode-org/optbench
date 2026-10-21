"""AIRSS random structure search on the harness Lennard-Jones potential.

``buildcell`` from AIRSS 0.9.4 draws each cell. The relaxation is the
pp3 two-point step (Barzilai and Borwein), two passes, on
``optbench.potentials.lennard_jones``. pp3's built-in pair potential
subtracts a cutoff shift, so the scored evaluations stay on the spec
potential and the harness counter.

One archived start index is one independent search. The search stops
when the harness energy is at or below the Wales minimum plus the
problem tolerance, or when the harness count reaches the problem
force-call cap.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import CountedPotential
from optbench.potentials import lennard_jones
from optbench.score import half_even, score_v2
from optbench.v2 import collect_provenance

WALES_URL = "https://www-wales.ch.cam.ac.uk/examples/GMIN/LJ38/lowest"
TARBALL_SHA256 = "5593cc4ee360d998764fceb188284fdf0ef781d206754176dcb360c65103ddaa"
# Runaway guard on the number of structures. A hit is a crash. The scored
# cap is the problem max_force_calls.
SAFETY_STRUCTURES = 4000
_PASSES = 2


class BuildFailure(RuntimeError):
    """buildcell returned no usable cell."""


def wales_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "lj38_wales.txt"


def load_wales_coordinates(path: Path | None = None) -> np.ndarray:
    """Cartesian rows from the vendored Wales LJ38 minimum."""
    rows = []
    for line in (path or wales_path()).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
    coords = np.asarray(rows, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"{path or wales_path()} has no coordinates")
    return coords


def reference_energy(coords: np.ndarray | None = None) -> float:
    """Harness energy of the published minimum. The header value is rounded."""
    point = load_wales_coordinates() if coords is None else np.asarray(coords, dtype=np.float64)
    energy, _gradient = lennard_jones(point)
    return float(energy)


def reached_minimum(energy: float, reference: float, tolerance: float) -> bool:
    """True when ``energy`` is at or below the known minimum plus ``tolerance``."""
    return math.isfinite(energy) and energy <= reference + tolerance


def seed_words(start_index: int, structure_index: int) -> tuple[int, ...]:
    """Eight gfortran ``random_seed`` integers for one cell.

    buildcell reads ``#SEED=`` into an array of length 8. A different
    length makes every cell share one seed.
    """
    words = []
    for slot in range(8):
        value = (
            1664525 * (structure_index + 1)
            + 1013904223 * (start_index + 1)
            + 22695477 * (slot + 1)
        ) % 2147483647
        words.append(int(value) or 1)
    return tuple(words)


def find_buildcell() -> Path:
    """The AIRSS ``buildcell`` binary, from the environment or ``PATH``."""
    env = os.environ.get("AIRSS_BUILDCELL")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"AIRSS_BUILDCELL={env} is not an executable")
    found = shutil.which("buildcell")
    if found:
        return Path(found)
    home = Path.home() / "airss" / "bin" / "buildcell"
    if home.is_file() and os.access(home, os.X_OK):
        return home
    raise FileNotFoundError("buildcell is not on PATH and AIRSS_BUILDCELL is unset")


def cell_text(
    *,
    natoms: int,
    box: float,
    posamp: float,
    minsep: float,
    symmops: str,
    nform: int,
    seed: tuple[int, ...],
    element: str = "Al",
) -> str:
    """AIRSS example 1.4, with the lengths passed in.

    Example 1.4 uses sigma 2. The spec potential uses sigma 1, so the
    caller passes lengths scaled by one half and a free cubic box.
    """
    if len(seed) != 8:
        raise ValueError("buildcell expects eight seed integers")
    side = int(box) if float(box).is_integer() else box
    seed_text = " ".join(str(int(word)) for word in seed)
    return (
        "%BLOCK LATTICE_CART\n"
        f"{side} 0 0\n"
        f"0 {side} 0\n"
        f"0 0 {side}\n"
        "#FIX\n"
        "%ENDBLOCK LATTICE_CART\n"
        "%BLOCK POSITIONS_FRAC\n"
        f"{element} 0 0 0 # {element}1 % NUM={int(natoms)}\n"
        "%ENDBLOCK POSITIONS_FRAC\n"
        "FIX_ALL_CELL : true\n"
        f"#MINSEP={minsep}\n"
        "#CLUSTER\n"
        f"#POSAMP={posamp}\n"
        f"#SEED={seed_text}\n"
        f"#SYMMOPS={symmops}\n"
        f"#NFORM={int(nform)}\n"
    )


def positions_from_cell(text: str, natoms: int) -> np.ndarray:
    """Cartesian positions from a buildcell ``LATTICE_ABC`` cell."""
    length = None
    in_abc = False
    in_pos = False
    coords: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("%BLOCK LATTICE_ABC"):
            in_abc = True
            continue
        if upper.startswith("%ENDBLOCK LATTICE_ABC"):
            in_abc = False
            continue
        if in_abc and length is None and stripped and not stripped.startswith("#"):
            length = float(stripped.split()[0])
            continue
        if upper.startswith("%BLOCK POSITIONS_FRAC"):
            in_pos = True
            continue
        if upper.startswith("%ENDBLOCK POSITIONS_FRAC"):
            in_pos = False
            continue
        if in_pos and stripped and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) < 4:
                raise BuildFailure("a position line has fewer than four fields")
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if length is None:
        raise BuildFailure("buildcell wrote no LATTICE_ABC length")
    if len(coords) != natoms:
        raise BuildFailure(f"buildcell wrote {len(coords)} atoms, expected {natoms}")
    return np.asarray(coords, dtype=np.float64) * length


def build_positions(binary: Path, text: str, natoms: int, *, timeout: float = 60.0) -> np.ndarray:
    """Run ``buildcell`` on ``text`` and return Cartesian coordinates."""
    try:
        proc = subprocess.run(
            [str(binary)],
            input=text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildFailure(exc.__class__.__name__) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {proc.returncode}"
        raise BuildFailure(tail)
    return positions_from_cell(proc.stdout, natoms)


def _bb_pass(
    pos: np.ndarray,
    box: float,
    potential: CountedPotential,
    *,
    etol: float,
    maxsteps: int,
) -> tuple[np.ndarray, float, int, bool]:
    """One ``opt_tpsd`` pass. The step is ``abs(xg/gg)`` and the shift is the new cell image.

    ``g`` is the force, the negative of the harness gradient, with the
    mean removed. The returned energy is the last evaluated energy.
    The returned coordinates include the last step.
    """
    pos = np.array(pos, dtype=np.float64, copy=True)
    pos0 = pos.copy()
    g = np.ones_like(pos)
    h = 1.0
    dh = 1e300
    steps = 0
    converged = True
    while abs(dh) > etol:
        steps += 1
        if steps > maxsteps:
            converged = False
            break
        h0 = h
        g0 = g
        energy, grad = potential.energy_forces(pos.reshape(-1))
        g = -np.asarray(grad, dtype=np.float64).reshape(pos.shape)
        g = g - g.mean(axis=0)
        h = float(energy)
        delta_g = g - g0
        xg = float(np.sum((pos - pos0) * delta_g))
        gg = float(np.sum(delta_g * delta_g))
        if abs(xg) > 0.0:
            step = abs(xg / gg)
        else:
            glength = math.sqrt(float(np.sum(g * g)))
            if glength > 1e8:
                step = 1e-2 / math.sqrt(glength)
            else:
                step = 1e-8
        pos0 = pos.copy()
        pos = pos + step * g
        shift = np.floor(pos / box)
        pos0 = (pos0 / box - shift) * box
        pos = (pos / box - shift) * box
        dh = h - h0
    return pos, h, steps, converged


def relax_positions(
    pos: np.ndarray,
    box: float,
    *,
    etol: float = 1e-10,
    maxsteps: int = 9999,
) -> tuple[np.ndarray, float, int, bool]:
    """Two pp3 passes on a counted harness Lennard-Jones potential."""
    potential = CountedPotential(lambda coords: lennard_jones(np.asarray(coords, dtype=np.float64)))
    point = np.asarray(pos, dtype=np.float64)
    converged = True
    energy = 0.0
    for _pass in range(_PASSES):
        point, energy, _steps, ok = _bb_pass(point, box, potential, etol=etol, maxsteps=maxsteps)
        converged = converged and ok
    return point, float(energy), int(potential.calls), converged


def _config(template: dict) -> dict:
    section = template.get("airss") or {}
    required = ("natoms", "box", "posamp", "minsep", "symmops", "nform")
    missing = [key for key in required if section.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{template['id']} needs airss.{', airss.'.join(missing)}")
    config = {
        "natoms": int(section["natoms"]),
        "box": float(section["box"]),
        "posamp": float(section["posamp"]),
        "minsep": float(section["minsep"]),
        "symmops": str(section["symmops"]),
        "nform": int(section["nform"]),
        "etol": float(section.get("etol", 1e-10)),
        "maxsteps": int(section.get("maxsteps", 9999)),
    }
    if section.get("budget_structures") not in (None, ""):
        config["budget_structures"] = int(section["budget_structures"])
    if section.get("starts") not in (None, ""):
        config["starts"] = int(section["starts"])
        config["start_label"] = str(section.get("start_label") or "start")
    return config


def _job(
    template: dict,
    index: int,
    start: str,
    binary: Path,
    reference: float | None,
    tolerance: float | None,
    call_cap: int | None = None,
) -> dict:
    config = _config(template)
    threshold = None if reference is None or tolerance is None else float(reference) + float(tolerance)
    job = {
        "index": index,
        "start": start,
        "buildcell": str(binary),
        "reference_energy": reference,
        "threshold": threshold,
        "safety": SAFETY_STRUCTURES,
        "code": template.get("code", "AIRSS"),
        "code_version": template.get("code_version", ""),
        "commit": template.get("commit", ""),
        "tuning": template.get("tuning", ""),
        "natoms": config["natoms"],
        "box": config["box"],
        "posamp": config["posamp"],
        "minsep": config["minsep"],
        "symmops": config["symmops"],
        "nform": config["nform"],
        "etol": config["etol"],
        "maxsteps": config["maxsteps"],
    }
    if config.get("budget_structures"):
        job["budget"] = int(config["budget_structures"])
    if call_cap is not None:
        job["call_cap"] = int(call_cap)
    return job


def _crash(job: dict, calls: int, energy: float | None, structures: int, error: str) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "error": error,
        "energy": energy,
        "structures": structures,
        "reference_energy": job["reference_energy"],
        "converged": False,
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }


def _capped(job: dict, calls: int, energy: float, structures: int) -> dict:
    """A start that stopped at the force-call cap or the structure budget."""
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "cap",
        "reasons": ["cap"],
        "energy": energy,
        "structures": structures,
        "reference_energy": job.get("reference_energy"),
        "converged": False,
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }


def search_start(job: dict) -> dict:
    """Search one start until the energy window, and return a v2 record.

    With no threshold, stop after ``budget`` structures and keep the lowest energy.
    """
    binary = Path(job["buildcell"])
    calls = 0
    structures = 0
    build_failures = 0
    nonfinite = 0
    best = math.inf
    budget_only = job.get("threshold") is None
    limit = int(job["budget"]) if budget_only else int(job["safety"])
    for structure_index in range(limit):
        text = cell_text(
            natoms=job["natoms"],
            box=job["box"],
            posamp=job["posamp"],
            minsep=job["minsep"],
            symmops=job["symmops"],
            nform=job["nform"],
            seed=seed_words(job["index"], structure_index),
        )
        try:
            pos = build_positions(binary, text, job["natoms"])
        except BuildFailure:
            build_failures += 1
            if build_failures >= 30:
                return _crash(job, calls, None if best is math.inf else best, structures, "buildcell")
            continue
        build_failures = 0
        structures += 1
        _point, energy, used, _ok = relax_positions(
            pos,
            job["box"],
            etol=job["etol"],
            maxsteps=job["maxsteps"],
        )
        calls += used
        if not math.isfinite(energy):
            nonfinite += 1
            if nonfinite >= 20:
                return _crash(job, calls, None if best is math.inf else best, structures, "nonfinite")
            continue
        if energy < best:
            best = energy
        if structures % 100 == 0:
            print(
                f"{job['start']} structures={structures} calls={calls} best={best:.6f}",
                flush=True,
            )
        if not budget_only and reached_minimum(
            energy,
            float(job["reference_energy"]),
            float(job["threshold"]) - float(job["reference_energy"]),
        ):
            return _success(job, calls, energy, structures)
        call_cap = job.get("call_cap")
        if call_cap is not None and calls >= int(call_cap):
            if best is math.inf:
                return _crash(job, calls, None, structures, "nonfinite")
            return _capped(job, calls, best, structures)
    if budget_only and best is not math.inf:
        return _capped(job, calls, best, structures)
    return _crash(job, calls, None if best is math.inf else best, structures, "safety")


def _success(job: dict, calls: int, energy: float, structures: int) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": True,
        "failure": None,
        "reasons": [],
        "energy": energy,
        "structures": structures,
        "reference_energy": job["reference_energy"],
        "converged": True,
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }


def _load_start_names(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path) -> list[str]:
    import tarfile

    from optbench.campaign import geometry_archive, iter_geometries
    from optbench.potentials import read_con

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
    names = []
    natoms = None
    for path in cons:
        pos, _box = read_con(path.read_text(encoding="utf-8", errors="replace"))
        if natoms is None:
            natoms = int(pos.shape[0])
        elif int(pos.shape[0]) != natoms:
            raise ValueError(f"{path} has {pos.shape[0]} atoms, expected {natoms}")
        names.append(path.stem)
    return names


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
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
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "structures_min": summary.get("structures_min"),
        "structures_max": summary.get("structures_max"),
        "partial": "true" if summary.get("partial") else None,
        "cap": summary.get("cap"),
    }
    stated_cap = summary.get("stated_cap")
    if summary.get("partial"):
        lines = [
            "# counter harness. One force call is one Lennard-Jones evaluation inside the two-point step.",
            "# The problem states no energy window. Each start stops after the structure budget or the force-call cap.",
        ]
        if stated_cap is None:
            lines.append("# failure is cap. cap is the largest harness count in this file, so PAR-2 is twice that count.")
        else:
            lines.append(
                f"# failure is cap. cap is {int(stated_cap)} force calls. PAR-2 charges each failure twice that cap."
            )
        lines.append("# force_calls is the half-even mean of those counts. energy_min and energy_max are the best energies.")
    else:
        lines = [
            "# counter harness. One force call is one Lennard-Jones evaluation inside the two-point step.",
            "# buildcell draws the cell. The archived .con files name the starts.",
            "# A start succeeds when the harness energy is at or below the Wales minimum plus energy_above_minimum.",
        ]
        if stated_cap is None:
            lines.append("# The problem states no force-call cap, so par2_mean is unset and the median is the call median.")
        else:
            lines.append(
                f"# The problem force-call cap is {int(stated_cap)}. A start at or above it fails. PAR-2 charges each failure twice that cap."
            )
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _cap_note(budget_only: bool, problem_cap: int | None) -> str:
    if budget_only and problem_cap is None:
        return "The problem states no energy window, so each start stops at the structure budget."
    if budget_only:
        return (
            "The problem states no energy window. Each start stops at the structure budget "
            f"or at {problem_cap} force calls."
        )
    if problem_cap is None:
        return "The problem states no force-call cap."
    return f"The problem force-call cap is {problem_cap}."


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
    from optbench.campaign import write_starts

    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    if template.get("commit") != TARBALL_SHA256:
        raise RuntimeError(f"{template['id']} commit is not the AIRSS 0.9.4 tarball sha256")
    convergence = problem.get("convergence") or {}
    stated_cap = convergence.get("max_force_calls")
    problem_cap = int(stated_cap) if stated_cap not in (None, "") else None
    config = _config(template)
    raw_tolerance = convergence.get("energy_above_minimum")
    if raw_tolerance is None and not config.get("budget_structures"):
        raise ValueError(
            f"{problem['id']} has no energy_above_minimum and {template['id']} has no airss.budget_structures"
        )
    binary = find_buildcell()
    if config.get("starts"):
        label = str(config.get("start_label") or "start")
        names = [f"{label}.{index}" for index in range(1, int(config["starts"]) + 1)]
    else:
        names = _load_start_names(root, problem, geometries, work, output)
    if raw_tolerance is None:
        reference = None
        tolerance = None
    else:
        tolerance = float(raw_tolerance)
        coords = load_wales_coordinates()
        if coords.shape[0] != config["natoms"]:
            raise ValueError(f"Wales file has {coords.shape[0]} atoms, template has {config['natoms']}")
        reference = reference_energy(coords)
    jobs = [
        _job(template, index, name, binary, reference, tolerance, problem_cap)
        for index, name in enumerate(names)
    ]
    if workers <= 1:
        records = []
        total = len(jobs)
        for done, job in enumerate(jobs, 1):
            record = search_start(job)
            records.append(record)
            print(
                f"{template['id']} {done}/{total} start={record['start']} "
                f"calls={record['force_calls']} energy={record['energy']} success={record['success']}",
                flush=True,
            )
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(search_start, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                print(
                    f"{template['id']} {done}/{total} start={record['start']} "
                    f"calls={record['force_calls']} energy={record['energy']} success={record['success']}",
                    flush=True,
                )
    records.sort(key=lambda row: row["index"])
    budget_only = reference is None
    crashes = [row for row in records if row.get("failure") == "crash"]
    missed = [row for row in records if not row["success"]]
    if crashes or (missed and not budget_only and problem_cap is None):
        partial_path = output.with_suffix(".partial.jsonl")
        for row in records:
            row.pop("index", None)
        write_starts(partial_path, records)
        raise RuntimeError(
            f"{len(missed)} of {len(records)} starts missed the minimum; wrote {partial_path}"
        )
    for row in records:
        row.pop("index", None)
    calls = [int(row["force_calls"]) for row in records]
    energies = [float(row["energy"]) for row in records if row.get("energy") is not None]
    structures = [int(row["structures"]) for row in records]
    cap = problem_cap if problem_cap is not None else (max(calls) if budget_only else None)
    summary = {
        "force_calls": half_even(sum(calls) / len(calls)),
        "force_calls_min": min(calls),
        "force_calls_max": max(calls),
        "nfailed": len(missed),
        "energy_min": min(energies) if energies else None,
        "energy_max": max(energies) if energies else None,
        "structures_min": min(structures),
        "structures_max": max(structures),
        "partial": budget_only,
        "cap": cap,
        "stated_cap": problem_cap,
    }
    scored = [
        {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
        for row in records
    ]
    metrics = score_v2(scored, cap)
    passed = dict(config)
    if reference is not None:
        passed["reference_energy"] = reference
        passed["energy_above_minimum"] = tolerance
        passed["wales"] = WALES_URL
    provenance = collect_provenance(
        code=str(template.get("code", "AIRSS")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "measured",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter on optbench.potentials.lennard_jones. "
            "buildcell generates each cell. "
            + _cap_note(budget_only, problem_cap)
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code", "AIRSS"),
                "algorithm": template.get("algorithm", "random structure search"),
                "code_version": template.get("code_version", ""),
                "commit": template.get("commit", ""),
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": passed,
                "provenance": provenance,
                "metrics": metrics,
                "starts": records,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    import json

    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["median"] = metrics["median"]
    summary["v2_nfailed"] = metrics["nfailed"]
    return summary
