"""Steepest descent on the harness Morse bulk potential.

The client is ``eonclient`` with ``potential = ext_pot``. One force call
is one ``HarnessCounter`` charge on that socket. The potential is
``morse_bulk``: depth 0.7102, alpha 1.6047, r0 2.897, cutoff 7, energy
shifted to zero at the cutoff, orthorhombic minimum image.

The step and the stop are the archived r2030 config:
``opt_method = sd``, ``sd_alpha = 0.04``, ``max_move = 0.2``,
``converged_force = 1e-3``. ``sd_twopoint`` stays unset. ``max_iterations``
stays unset, so the client keeps its constructor limit of 1000. The
problem cap, 10000, is the harness ceiling used for PAR-2. A v2 success
is under that ceiling, at the force stop, and Hessian index 0. The
Hessian is computed after the client returns and is not charged.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.campaign import format_ini
from optbench.eon_harness import _run_client, bridge_binary
from optbench.failures import classify_failure
from optbench.potentials import morse_bulk, morse_bulk_hessian, read_con
from optbench.potserver import PotServer, remote_count
from optbench.score import score_v2, summarize_capped
from optbench.stationary import negative_index
from optbench.v2 import write_starts

EIGENVALUE_TOL = 1e-6

_CLIENTS = {
    "svn": {
        "env": "EON_SVN_CLIENT",
        "default": Path.home() / "opt" / "eon-svn" / "bin" / "eonclient",
        "flag": "-v",
        "token": "r2549",
    },
    "develop": {
        "env": "EON_DEVELOP_CLIENT",
        "default": Path.home() / "opt" / "eon-develop" / "bin" / "eonclient",
        "flag": "--version",
        "token": "500350d9",
    },
}


def _client(family: str) -> str:
    spec = _CLIENTS[family]
    env = os.environ.get(spec["env"])
    path = env or (str(spec["default"]) if spec["default"].is_file() else "")
    if not path or not Path(path).is_file():
        raise RuntimeError(f"set {spec['env']} to the {family} eonclient")
    proc = subprocess.run(
        [path, spec["flag"]],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = proc.stdout or ""
    if spec["token"] not in text:
        raise RuntimeError(f"{path} printed {text!r}, expected {spec['token']}")
    return path


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, partial: bool) -> str:
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
        "reducer": template.get("reducer", ""),
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
    }
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one ext_pot evaluation.",
        "# The potential is morse_bulk, cutoff 7, shifted, orthorhombic minimum image.",
        "# Archived r2030 config: opt_method sd, sd_alpha 0.04, max_move 0.2, converged_force 0.001.",
        "# sd_twopoint and max_iterations are unset. The constructor iteration limit is 1000.",
        "# v2 success is under the problem cap, at the force stop, and Hessian index 0.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _geometry_paths(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path) -> list[Path]:
    import tarfile

    from optbench.campaign import geometry_archive, iter_geometries

    if geometries is not None:
        return list(iter_geometries(geometries))
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
    return cons


def _evaluate(fallback: np.ndarray):
    def evaluate(positions, atomic_numbers=None, box=None):
        del atomic_numbers
        coords = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        lengths = np.asarray(fallback, dtype=np.float64)
        if box is not None and len(box) >= 9:
            candidate = np.array([float(box[0]), float(box[4]), float(box[8])], dtype=np.float64)
            if np.all(candidate > 0.0):
                lengths = candidate
        energy, gradient = morse_bulk(coords, lengths)
        forces = -np.asarray(gradient, dtype=np.float64).reshape(-1)
        return float(energy), tuple(float(value) for value in forces)

    return evaluate


def _ini_text(template: dict, wrapper: Path) -> str:
    sections = json.loads(json.dumps(template["ini"]))
    optimizer = sections.get("Optimizer") or {}
    if str(optimizer.get("opt_method") or "") != "sd":
        raise ValueError(f"{template['id']} is not steepest descent")
    if "max_iterations" in optimizer:
        raise ValueError(f"{template['id']} sets max_iterations; the archived config leaves the constructor limit")
    if "sd_twopoint" in optimizer or "sd_twopoint" in (sections.get("SD") or {}):
        raise ValueError(f"{template['id']} sets sd_twopoint; that is the other Morse row")
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    return format_ini(sections)


def _min_job(job: dict) -> dict:
    cap = int(job["cap"])
    gtol = float(job["gtol"])
    text = Path(job["con"]).read_text(encoding="utf-8", errors="replace")
    positions, box = read_con(text)
    work = Path(job["work"])
    server = PotServer(_evaluate(box), cap=cap).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    error = ""
    calls = 0
    verify = 0
    try:
        found = _run_client(
            job["client"],
            job["ini"],
            work,
            [(job["con"], "pos.con")],
            [],
            3600,
        )
        calls, verify = remote_count(server.host, server.port)
        if found["returncode"] != 0 and not found["fields"]:
            error = (found["stderr"] or "client exited")[:500]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        server.stop()
        os.environ.pop("OPTBENCH_POTSERVER", None)
    calls = min(int(calls), cap)
    # MinimizationJob writes the relaxed geometry to min.con and leaves pos.con at the start.
    final = work / "min.con"
    if not final.is_file():
        final = work / "pos.con"
    energy = None
    force_norm = None
    hessian_index = None
    if final.is_file():
        try:
            point, final_box = read_con(final.read_text(encoding="utf-8", errors="replace"))
            energy, gradient = morse_bulk(point, final_box)
            gradient = np.asarray(gradient, dtype=np.float64)
            force_norm = float(np.linalg.norm(gradient))
            hessian = morse_bulk_hessian(point, final_box)
            hessian_index = negative_index(np.linalg.eigvalsh(hessian).tolist(), EIGENVALUE_TOL)
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
    finite = (
        energy is not None
        and force_norm is not None
        and math_finite(energy)
        and math_finite(force_norm)
    )
    at_force = finite and force_norm is not None and force_norm <= gtol and calls < cap and not error
    reasons: list[str] = []
    if error and calls < cap:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif calls >= cap:
        reasons.append("cap")
    elif not at_force:
        reasons.append("force_norm")
    elif hessian_index != 0:
        reasons.append("hessian_index")
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": int(verify),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": force_norm,
        "hessian_index": hessian_index,
        "expected_index": 0,
        "problem": job["problem"],
        "template": job["template"],
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": "default",
        "error": error,
    }


def math_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


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
    family = str(template.get("client_family") or "")
    if family not in _CLIENTS:
        raise ValueError(f"{template['id']} needs client_family svn or develop")
    if str(template.get("tuning") or "") != "default":
        raise ValueError(f"{template['id']} is not a stock run")
    client = _client(family)
    wrapper = bridge_binary()
    ini = _ini_text(template, wrapper)
    gtol = float((problem.get("convergence") or {})["force_norm"])
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    paths = _geometry_paths(root, problem, geometries, work, output)
    if limit:
        paths = paths[: int(limit)]
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    jobs = []
    for index, path in enumerate(paths):
        jobs.append(
            {
                "index": index,
                "start": path.stem,
                "con": str(path),
                "work": str(scratch / path.stem),
                "client": client,
                "ini": ini,
                "cap": cap,
                "gtol": gtol,
                "problem": problem["id"],
                "template": template["id"],
                "code": template.get("code"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
            }
        )
    if workers <= 1:
        records = [_min_job(job) for job in jobs]
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_min_job, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                print(f"{template['id']} {done}/{total}", flush=True)
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    partial = bool(limit) or len(records) < len(_geometry_paths(root, problem, geometries, work, output))
    legacy_counts = [cap if not record["success"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["success"]]
    summary["median"] = statistics.median(successes) if successes else None
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in records
        ],
        cap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, partial), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary
