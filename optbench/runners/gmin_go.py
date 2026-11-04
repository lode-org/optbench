"""Basin-hopping and parallel-tempering basin-hopping with GMIN.

GMIN's own Lennard-Jones stays in the binary for an unset
``OPTBENCH_POTPORT``. A scored run sets the port. ``LJ`` then returns the
energy and gradient from ``optbench.potentials.lennard_jones`` and each
request is one harness call. The server is the call count. GMIN's NPCALL
is not the score.

The quench RMS is ``sqrt(sum(g^2)/(3N))``, the same reduction as GMIN.
A quench at or below the template ``sloppyconv`` can update the lowest
energy. ``failure_cap`` matches ``max_force_calls`` when the problem
states one. lj38 does not, and its template cap is 2000000.

A start succeeds when the lowest quench energy is within
``energy_above_minimum`` of the harness energy of the Cambridge points
and the call count is under the cap. The harness energy is
``lennard_jones`` on ``codes/gmin/reference/lj<N>.points``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import HarnessCounter
from optbench.failures import classify_failure
from optbench.gmin_pot import lennard_jones_compiled
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts
from optbench.wales import header, snapshot, stamped_sha256

CODE_NAME = "GMIN"
_REVISION_LABEL = "SVN revision:"


def gmin_binary() -> Path:
    env = os.environ.get("OPTBENCH_GMIN", "")
    path = Path(env) if env else Path(shutil.which("GMIN") or "")
    if not path.is_file():
        raise RuntimeError("set OPTBENCH_GMIN to the GMIN binary built by codes/gmin/build-bh.sh")
    return path


def binary_revision(binary: Path) -> str:
    """The revision ``GMIN -v`` prints.

    GMIN's ``display_version.sh`` compiles the ``SVNREV`` file of the build
    directory into that line. ``codes/gmin/build-bh.sh`` copies it there from
    the tarball, and the line is empty when no build did.
    """
    proc = subprocess.run([str(binary), "-v"], capture_output=True, text=True, check=False)
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        _head, label, value = line.partition(_REVISION_LABEL)
        if label:
            return value.strip()
    raise RuntimeError(f"{binary} -v printed no {_REVISION_LABEL} line")


def binary_build(binary: Path) -> dict:
    """``wales_sha256`` and ``wales_svnrev`` of ``binary``.

    The revision is the line ``GMIN -v`` prints. The sha256 is the stamp
    that ``codes/gmin/build-bh.sh`` writes beside the binary.
    """
    return snapshot(stamped_sha256(Path(binary).parent), binary_revision(Path(binary)))


def reference_evaluation(root: Path, natoms: int) -> tuple[float, float]:
    """Harness energy and RMS of the Cambridge point file for this size."""
    path = root / "codes" / "gmin" / "reference" / f"lj{natoms}.points"
    if not path.is_file():
        raise FileNotFoundError(f"no reference points for {natoms} atoms")
    pos = np.loadtxt(path)
    if pos.shape != (natoms, 3):
        raise ValueError(f"{path} has shape {pos.shape}")
    energy, grad = lennard_jones(pos)
    rms = float(np.sqrt(float(np.dot(grad, grad)) / grad.size))
    return float(energy), rms


def _center(pos: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) - np.mean(pos, axis=0)


def _read_geometry(path: Path) -> np.ndarray:
    if path.suffix == ".con":
        pos, _box = read_con(path.read_text(encoding="utf-8", errors="replace"))
        return pos
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"{path} has no coordinates")
    return np.asarray(rows, dtype=np.float64)


def _start_paths(dest: Path, problem_id: str) -> list[Path]:
    if problem_id == "globalopt/lj38":
        files = list(dest.rglob("*.con"))
        return sorted(files, key=lambda path: int(path.stem))
    numbered = []
    for path in dest.rglob("coords.*"):
        suffix = path.name.split(".", 1)[1]
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    numbered.sort()
    return [path for _index, path in numbered]


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    recorded = archive.name
    if marker.is_file():
        previous = marker.read_text(encoding="utf-8").strip()
        if previous != recorded:
            raise RuntimeError(f"{dest} was extracted from {previous}, not {recorded}")
        return
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    marker.write_text(recorded, encoding="utf-8")


# Numbered coords.N from the 2014 website tarballs: 200 for LJ75 and 100
# for LJ98 and LJ150. Used when a problem file leaves archive_path unset.
_ARCHIVE_FALLBACK = {
    "globalopt/lj75": "website/globalopt/lj75/lj75_clusters.tgz",
    "globalopt/lj98": "website/globalopt/lj98/lj98_clusters.tgz",
    "globalopt/lj150": "website/globalopt/lj150/lj150_clusters.tgz",
}


def _problem_for_archive(problem: dict) -> dict:
    if (problem.get("input") or {}).get("archive_path"):
        return problem
    fallback = _ARCHIVE_FALLBACK.get(str(problem.get("id") or ""))
    if not fallback:
        return problem
    copied = dict(problem)
    copied["input"] = {"archive_path": fallback}
    return copied


def load_geometries(root: Path, problem: dict, work: Path, *, center: bool = True) -> list[tuple[str, np.ndarray]]:
    from optbench.campaign import geometry_archive

    archive = geometry_archive(root, _problem_for_archive(problem))
    dest = work / "geometries" / problem["id"].replace("/", "-")
    _extract(archive, dest)
    paths = _start_paths(dest, problem["id"])
    if not paths:
        raise FileNotFoundError(f"{archive} has no starts")
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(paths) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(paths)}")
    loaded = []
    for path in paths:
        pos = _read_geometry(path)
        if center:
            pos = _center(pos)
        # coords.1 has stem "coords". Keep the numeric name. A .con stem is the start index.
        label = path.stem if path.suffix == ".con" else path.name
        loaded.append((label, pos))
    return loaded


def _section(template: dict) -> dict:
    section = template.get("gmin") or {}
    method = str(section.get("method") or "")
    if method not in {"bh", "bhpt"}:
        raise ValueError(f"{template['id']} gmin.method must be bh or bhpt")
    replicas = int(section.get("replicas") or (4 if method == "bhpt" else 1))
    if method == "bh" and replicas != 1:
        raise ValueError(f"{template['id']} serial basin-hopping uses one replica")
    if method == "bhpt" and replicas < 2:
        raise ValueError(f"{template['id']} BHPT needs at least two replicas")
    return section


def _archive_lines(job: dict) -> list[str]:
    """Archive move keywords. Absent keys leave the generic data file unchanged."""
    lines = []
    if job.get("fixboth"):
        lines.append("FIXBOTH")
    if job.get("permdist") not in (None, ""):
        lines.append(f"PERMDIST {float(job['permdist']):.8f}")
    if job.get("newrestart") not in (None, ""):
        lines.append(f"NEWRESTART {int(job['newrestart'])}")
    if job.get("sort"):
        lines.append("SORT")
    symmetrise = str(job.get("symmetrise") or "").strip()
    if symmetrise:
        lines.append(f"SYMMETRISE {symmetrise}")
    avoid = str(job.get("avoid") or "").strip()
    if avoid:
        lines.append(f"AVOID {avoid}")
    symmetrisecsm = str(job.get("symmetrisecsm") or "").strip()
    if symmetrisecsm:
        lines.append(f"SYMMETRISECSM {symmetrisecsm}")
    return lines


def _data_text(job: dict) -> str:
    # This binary is built with GMIN_MPI. mc.F90 sets the replica index only
    # inside IF (MPIT), and the serial DO loop is compiled out. MPI is required
    # for one replica as well as for BHPT. It resets STEP and TEMP, so it is first.
    lines = ["MPI"]
    if int(job["replicas"]) > 1:
        lines.append(
            f"BHPT {job['pttmin']:.8f} {job['pttmax']:.8f} {job['exchprob']:.8f}"
        )
    lines.extend(
        [
            f"SLOPPYCONV {job['sloppyconv']:.8f}",
            f"TIGHTCONV {job['tightconv']:.8f}",
            f"EDIFF {job['ediff']:.8f}",
            f"MAXIT {int(job['maxit'])} {int(job['maxit2'])}",
            f"SAVE {int(job.get('save') or 2)}",
            "CENTRE",
            *( [f"RADIUS {float(job['radius']):.8f}"] if float(job.get("radius") or 0) > 0 else [] ),
            f"TEMPERATURE {job['temperature']:.8f}",
            f"STEP {job['step']:.8f} {job['astep']:.8f}",
            *_archive_lines(job),
            (
                f"STEPS {int(job['steps'])} {float(job['tfac']):.1f}"
                if job.get("tfac") not in (None, "")
                else f"STEPS {int(job['steps'])}"
            ),
            *([f"RANSEED {int(job['seed'])}"] if job.get("emit_seed", True) else []),
            f"TARGET {job['reference']:.16f}",
            f"MAX_NPCALL {int(job['cap'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def _coords_text(coords: list[list[float]], replicas: int) -> str:
    block = "\n".join(
        f"{row[0]:20.10f} {row[1]:20.10f} {row[2]:20.10f}" for row in coords
    )
    return ("\n".join([block] * replicas)) + "\n"


def _command(binary: str, replicas: int) -> list[str]:
    # mpirun -np 1 as well. A direct launch of the MPI binary leaves the
    # replica index unset unless OpenMPI singleton initialises the same way.
    # Yield while a replica waits on the harness socket. Otherwise the ranks
    # spin and the potential server loses the core.
    mpirun = shutil.which("mpirun")
    if not mpirun:
        raise RuntimeError("GMIN MPI binary needs mpirun")
    return [
        mpirun,
        "--oversubscribe",
        "--mca",
        "mpi_yield_when_idle",
        "1",
        "-np",
        str(replicas),
        "--bind-to",
        "none",
        "-x",
        "OPTBENCH_POTPORT",
        "-x",
        "OPTBENCH_POTHOST",
        "-x",
        "OMP_NUM_THREADS",
        binary,
    ]


def _classify(job: dict, snapshot: dict, returncode: int | None, error: str) -> dict:
    calls = int(snapshot.get("calls") or 0)
    cap = int(job["cap"])
    cap_hit = bool(snapshot.get("cap_hit")) or calls >= cap
    energy = snapshot.get("quench_energy")
    finite = energy is not None and math.isfinite(float(energy))
    signaled = returncode is not None and returncode < 0
    reasons: list[str] = []
    if error and not cap_hit:
        reasons.append("crash")
    elif signaled and not cap_hit:
        reasons.append("crash")
    elif not finite:
        reasons.append("cap" if cap_hit else "divergence")
    elif cap_hit:
        reasons.append("cap")
    elif float(energy) > float(job["reference"]) + float(job["tolerance"]):
        reasons.append("energy")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if not finite else float(energy),
        "lowest_energy": snapshot.get("lowest_energy"),
        "quench_rms": snapshot.get("quench_rms"),
        "quench_calls": snapshot.get("quench_calls"),
        "reference_energy": float(job["reference"]),
        "tolerance": float(job["tolerance"]),
        "rule_stated": bool(job["rule_stated"]),
        "cap_hit": cap_hit,
        "returncode": returncode,
        "replicas": int(job["replicas"]),
        "server_errors": int(snapshot.get("errors") or 0),
    }
    if error:
        record["error"] = error
    return record


def run_one(job: dict) -> dict:
    """One GMIN start. The harness server lives in this process."""
    from optbench.gmin_pot import GminServer

    work = Path(job["work"])
    stored = work / "record.json"
    if job.get("resume") and stored.is_file():
        return json.loads(stored.read_text(encoding="utf-8"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").write_text(_data_text(job), encoding="utf-8")
    coords = job["coords"]
    (work / "coords").write_text(_coords_text(coords, int(job["replicas"])), encoding="utf-8")
    counter = HarnessCounter(cache=False, cap=int(job["cap"]))
    server = GminServer(lennard_jones_compiled, counter, sloppy_rms=float(job["sloppyconv"]))
    server.start()
    env = os.environ.copy()
    env["OPTBENCH_POTHOST"] = "127.0.0.1"
    env["OPTBENCH_POTPORT"] = str(server.port)
    env["OMP_NUM_THREADS"] = "1"
    error = ""
    returncode: int | None = None
    try:
        with (work / "stdout.txt").open("w", encoding="utf-8") as stdout, (
            work / "stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                _command(job["binary"], int(job["replicas"])),
                cwd=work,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                returncode = int(proc.wait(timeout=float(job["timeout"])))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
                returncode = -9
                error = "timeout"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        returncode = returncode if returncode is not None else -1
    snapshot = server.snapshot()
    server.stop()
    record = _classify(job, snapshot, returncode, error)
    stored.write_text(json.dumps(record), encoding="utf-8")
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, partial: bool, wales: dict) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": metrics.get("nfailed"),
        "median": summary.get("median"),
        "code": CODE_NAME,
        "code_version": _code_version(wales),
        "commit": wales["wales_sha256"],
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "problem": template.get("problem"),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "rate_low": metrics.get("rate_low"),
        "rate_high": metrics.get("rate_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "cap": summary.get("cap"),
    }
    if partial:
        rows["partial"] = "true"
    lines = [
        *header(wales),
        "# counter harness. One force call is one GMIN potential request to lennard_jones.",
        "# commit is the sha256 of the Wales tarball the binary was built from.",
        "# energy is the lowest quench at or below SLOPPYCONV. lowest_energy on each start is the lowest finite evaluation.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _code_version(wales: dict) -> str:
    return wales["wales_svnrev"][:8]


def _write_result(output: Path, problem: dict, template: dict, records: list[dict], partial: bool, passed: dict) -> dict:
    cap = int(passed["cap"])
    wales = snapshot(passed["wales_sha256"], passed["wales_svnrev"])
    ordered = sorted(records, key=lambda row: int(row["index"]))
    for record in ordered:
        record.pop("index", None)
        record.update(wales)
    capped_counts = [cap if not record["success"] else int(record["force_calls"]) for record in ordered]
    summary = summarize_capped(capped_counts, cap)
    successes = [int(record["force_calls"]) for record in ordered if record["success"]]
    import statistics

    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in ordered if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    summary["cap"] = cap
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in ordered
        ],
        cap,
    )
    provenance = collect_provenance(
        code=CODE_NAME,
        code_version=_code_version(wales),
        commit=wales["wales_sha256"],
        parameters=passed,
    )
    rule = (
        "The problem energy_above_minimum is the success window around the harness energy of the Cambridge points."
        if passed["rule_stated"]
        else "The problem file states no success rule. Success on this partial row uses the template EDIFF around the harness energy of the Cambridge points."
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "archive",
        "problem": problem["id"],
        "scoring": "par2-rate-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": rule,
        "runs": [
            {
                "entry": output.stem,
                "code": CODE_NAME,
                "algorithm": template.get("algorithm", ""),
                "code_version": _code_version(wales),
                "commit": wales["wales_sha256"],
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": passed,
                "provenance": provenance,
                "metrics": metrics,
                "starts": ordered,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, partial, wales), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), ordered)
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["nfailed"] = metrics["nfailed"]
    return summary


def plan(
    root: Path,
    problem: dict,
    template: dict,
    *,
    scratch: Path,
    limit: int = 0,
    resume: bool = False,
) -> tuple[list[dict], dict]:
    """The jobs ``run_template`` runs, one per archived start in archive order, and its parameters."""
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    section = _section(template)
    convergence = problem.get("convergence") or {}
    stated_tol = convergence.get("energy_above_minimum")
    ediff = float(section["ediff"])
    if stated_tol is not None and abs(float(stated_tol) - ediff) > 1e-12:
        raise ValueError("template ediff is not the problem energy_above_minimum")
    cap = template.get("failure_cap")
    if cap is None:
        raise ValueError(f"{template['id']} needs failure_cap")
    cap = int(cap)
    stated_cap = convergence.get("max_force_calls")
    if stated_cap is not None and int(stated_cap) != cap:
        raise ValueError("template failure_cap is not the problem max_force_calls")
    binary = gmin_binary()
    wales = binary_build(binary)
    loaded = load_geometries(root, problem, scratch, center=bool(section.get("center", True)))
    if limit > 0:
        loaded = loaded[: int(limit)]
    natoms = int(loaded[0][1].shape[0])
    reference, reference_rms = reference_evaluation(root, natoms)
    sloppy = float(section["sloppyconv"])
    if reference_rms > sloppy:
        raise RuntimeError(
            f"reference LJ{natoms} RMS {reference_rms} is above sloppyconv {sloppy}"
        )
    tolerance = float(stated_tol) if stated_tol is not None else ediff
    rule_stated = stated_tol is not None
    replicas = int(section["replicas"])
    timeout = float(section.get("timeout") or (cap * 0.005 + 900))
    passed = {
        "method": str(section["method"]),
        "replicas": replicas,
        "temperature": float(section["temperature"]),
        "pttmin": float(section["pttmin"]),
        "pttmax": float(section["pttmax"]),
        "exchprob": float(section["exchprob"]),
        "step": float(section["step"]),
        "astep": float(section["astep"]),
        "sloppyconv": sloppy,
        "tightconv": float(section["tightconv"]),
        "ediff": ediff,
        "maxit": int(section["maxit"]),
        "maxit2": int(section["maxit2"]),
        "radius": float(section["radius"]),
        "steps": int(section["steps"]),
        "seed_base": int(section["seed"]),
        "cap": cap,
        "reference_energy": reference,
        "reference_rms": reference_rms,
        "tolerance": tolerance,
        "rule_stated": rule_stated,
        "potential": "lennard_jones",
        "epsilon": 1.0,
        "sigma": 1.0,
        "cutoff": 40.0,
        "fixboth": bool(section.get("fixboth", False)),
        "permdist": section.get("permdist"),
        "newrestart": section.get("newrestart"),
        "sort": bool(section.get("sort", False)),
        "symmetrise": str(section.get("symmetrise") or ""),
        "avoid": str(section.get("avoid") or ""),
        "symmetrisecsm": str(section.get("symmetrisecsm") or ""),
        "save": int(section.get("save") or 2),
        "tfac": section.get("tfac"),
        "emit_seed": bool(section.get("emit_seed", True)),
        **wales,
    }
    jobs = []
    lennard_jones_compiled(np.zeros((1, 3)))
    for index, (name, pos) in enumerate(loaded):
        if pos.shape != (natoms, 3):
            raise ValueError(f"{name} has {pos.shape[0]} atoms, not {natoms}")
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": pos.tolist(),
                "work": str(scratch / "starts" / f"{index:04d}"),
                "binary": str(binary),
                "replicas": replicas,
                "cap": cap,
                "sloppyconv": sloppy,
                "reference": reference,
                "tolerance": tolerance,
                "rule_stated": rule_stated,
                "resume": bool(resume),
                "timeout": timeout,
                "pttmin": passed["pttmin"],
                "pttmax": passed["pttmax"],
                "exchprob": passed["exchprob"],
                "temperature": passed["temperature"],
                "step": passed["step"],
                "astep": passed["astep"],
                "tightconv": passed["tightconv"],
                "ediff": ediff,
                "maxit": passed["maxit"],
                "maxit2": passed["maxit2"],
                "radius": passed["radius"],
                "steps": passed["steps"],
                "seed": int(section["seed"]) if not passed["emit_seed"] else int(section["seed"]) + index,
                "save": passed["save"],
                "tfac": passed["tfac"],
                "emit_seed": passed["emit_seed"],
                "fixboth": passed["fixboth"],
                "permdist": passed["permdist"],
                "newrestart": passed["newrestart"],
                "sort": passed["sort"],
                "symmetrise": passed["symmetrise"],
                "avoid": passed["avoid"],
                "symmetrisecsm": passed["symmetrisecsm"],
            }
        )
    return jobs, passed


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    jobs, passed = plan(root, problem, template, scratch=scratch, limit=limit, resume=resume)
    if workers <= 1:
        records = []
        for done, job in enumerate(jobs, 1):
            records.append(run_one(job))
            print(f"{template['id']} {done}/{len(jobs)}", flush=True)
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                print(f"{template['id']} {done}/{total}", flush=True)
    stated_n = (problem.get("sample") or {}).get("starts")
    complete = (
        passed["rule_stated"]
        and stated_n is not None
        and len(records) == int(stated_n)
        and int(limit) <= 0
    )
    return _write_result(output, problem, template, records, partial=not complete, passed=passed)
