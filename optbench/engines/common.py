"""Campaign files shared by the saddle engines.

A start is ``optbench.start/v2``. The score is ``score_v2`` on the harness
count. Hessian finite differences run in verification mode and are not
search cost. A failure is one of crash, divergence, cap, or
wrong_stationary.
"""

from __future__ import annotations

import json
import statistics
import tomllib
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from optbench.campaign import geometry_archive, load_template, resolve_problem
from optbench.failures import classify_failure
from optbench.pes import evaluate, load_pairs
from optbench.rgsaddle_band import vibrational_index
from optbench.score import score_v2
from optbench.stationary import numerical_hessian
from optbench.v2 import collect_provenance, write_starts

_FD_STEP = 1e-5
_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def registry_pin(root: Path, code_id: str) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == code_id:
            return row
    raise ValueError(f"codes/registry.toml has no {code_id} entry")


def require_pin(template: dict, pin: dict) -> None:
    if template.get("code") != pin["id"]:
        raise ValueError(f"{template['id']} code is not {pin['id']}")
    if template.get("commit") != pin["commit"]:
        raise ValueError(f"{template['id']} commit does not match the {pin['id']} registry")


def band_settings(problem: dict) -> dict:
    """Image count, cap, and potential from a NEB or double-ended problem."""
    band = problem.get("band") or {}
    convergence = problem.get("convergence") or {}
    if "potential" not in band:
        raise ValueError(f"{problem['id']} has no band potential")
    if convergence.get("max_force_calls") is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    force = convergence.get("force_norm", convergence.get("gradient_norm"))
    if force is None:
        raise ValueError(f"{problem['id']} has no force norm")
    sample = problem.get("sample") or {}
    if "pairs" not in sample:
        raise ValueError(f"{problem['id']} has no sample.pairs")
    return {
        "potential": str(band["potential"]),
        "n_interior": int(band["interior_images"]),
        "max_move": float(band["max_move"]),
        "force_norm": float(force),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(sample["pairs"]),
        "geometry": str(problem["geometry"]["kind"]),
    }


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    root = Path(root)
    key = (str(root.resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        loaded = load_pairs(
            geometry_archive(root, problem),
            settings["geometry"],
            settings["pairs"],
        )
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def pair_seed(label: str, base: int) -> int:
    """Positive 31-bit seed. The same label and base always return the same value."""
    digest = zlib.crc32(label.encode("utf-8")) & 0x7FFFFFFF
    return (int(base) + int(digest)) % 2147483646 + 1


def initial_push(reactant: np.ndarray, product: np.ndarray, free: np.ndarray, step: float) -> np.ndarray:
    """One free atom, unit step from the reactant toward the product.

    The product is not a band endpoint. It only chooses the push direction.
    """
    delta = np.asarray(product, dtype=np.float64) - np.asarray(reactant, dtype=np.float64)
    mask = np.asarray(free, dtype=bool)
    delta[~mask] = 0.0
    norms = np.linalg.norm(delta, axis=1)
    push = np.zeros_like(delta)
    if not np.any(mask):
        raise ValueError("the pair has no free atom")
    index = int(np.argmax(norms))
    if norms[index] < 1e-12:
        free_index = int(np.flatnonzero(mask)[0])
        push[free_index, 0] = float(step)
        return push
    push[index] = delta[index] / norms[index] * float(step)
    return push


def lattice(box, positions: np.ndarray) -> np.ndarray:
    """Diagonal cell. A missing box is a large orthorhombic cell around the cluster."""
    cell = np.zeros((3, 3), dtype=np.float64)
    if box is None:
        span = float(np.max(np.abs(positions))) if positions.size else 1.0
        length = max(80.0, 4.0 * span + 10.0)
        cell[0, 0] = cell[1, 1] = cell[2, 2] = length
        return cell
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)
    if lengths.size == 3:
        cell[0, 0], cell[1, 1], cell[2, 2] = (float(value) for value in lengths)
        return cell
    if lengths.size == 9:
        return lengths.reshape(3, 3).astype(np.float64, copy=False)
    raise ValueError(f"box has length {lengths.size}")


def project_rigid(kind: str, free: np.ndarray) -> bool:
    mask = np.asarray(free, dtype=bool)
    return bool(np.all(mask) and mask.size >= 2 and kind == "lj")


def judge_saddle(
    name: str,
    point: np.ndarray | None,
    energy: float | None,
    gradient: np.ndarray | None,
    *,
    counter,
    cap: int,
    force_norm: float,
    kind: str,
    free: np.ndarray,
    box,
    prefix: list[str],
    expected_index: int = 1,
    keep_mode: bool = False,
    rotations: int | None = None,
) -> dict:
    """Free-atom force stop and Hessian index. The Hessian is verification.

    ``expected_index`` is 1 for a transition state and 0 for a minimum.

    Fixed atoms stay at ``point``. The force norm and the Hessian use the
    free block, which is the block the band campaign scores.
    """
    reasons = list(prefix)
    calls = int(counter.calls)
    verify_calls = int(counter.verify_calls)
    force = None
    hessian_index = None
    lowest_mode = None
    finite_point = point is not None and np.all(np.isfinite(point))
    if not finite_point and "crash" not in reasons and "cap" not in reasons:
        reasons.append("divergence")
    if calls >= cap and "cap" not in reasons:
        reasons.append("cap")
    if finite_point and gradient is not None and "crash" not in reasons:
        coords = np.asarray(point, dtype=np.float64).reshape((-1, 3))
        mask = np.asarray(free, dtype=bool).reshape(-1)
        free_grad = np.asarray(gradient, dtype=np.float64).reshape((-1, 3))[mask]
        force = float(np.linalg.norm(free_grad))
    if (
        finite_point
        and gradient is not None
        and "crash" not in reasons
        and "cap" not in reasons
        and "divergence" not in reasons
    ):
        if energy is None or not np.isfinite(energy) or force is None or not np.isfinite(force):
            reasons.append("divergence")
        elif force > force_norm:
            reasons.append("force_norm")
        else:
            coords = np.asarray(point, dtype=np.float64).reshape((-1, 3))
            mask = np.asarray(free, dtype=bool).reshape(-1)

            def gradient_of(values):
                full = np.array(coords, copy=True)
                full[mask] = np.asarray(values, dtype=np.float64).reshape((-1, 3))
                _energy, grad = evaluate(kind, full, box)
                return np.asarray(grad, dtype=np.float64)[mask].reshape(-1)

            with counter.verification():
                hessian = numerical_hessian(gradient_of, coords[mask].reshape(-1).tolist(), step=_FD_STEP)
            matrix = np.asarray(hessian, dtype=np.float64)
            hessian_index = vibrational_index(
                matrix,
                coords[mask],
                project_rigid(kind, free),
                rotations,
            )
            if keep_mode:
                _values, vectors = np.linalg.eigh(matrix)
                lowest_mode = np.asarray(vectors[:, 0], dtype=np.float64)
            if hessian_index != int(expected_index):
                reasons.append("hessian_index")
            verify_calls = int(counter.verify_calls)
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": calls,
        "verify_calls": verify_calls,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": force,
        "hessian_index": hessian_index,
        "expected_index": int(expected_index),
        "converged": failure is None,
        "lowest_mode": None if lowest_mode is None else lowest_mode.tolist(),
    }


def judge_mode(name: str, calls: int, cap: int, overlap: float, target: float, error: str) -> dict:
    """Lowest-eigenvector start. Success is overlap with the reference mode.

    The Hessian of the unmoved geometry is not the score.
    """
    reasons: list[str] = []
    if error and "force-call cap" not in error:
        reasons.append("crash")
    matched = float(overlap) >= float(target) and not reasons
    if calls >= cap:
        reasons.append("cap")
        matched = False
    if not matched and "crash" not in reasons and "cap" not in reasons:
        reasons.append("mode_overlap")
    if matched:
        reasons = []
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "overlap": float(overlap),
        "overlap_target": float(target),
        "hessian_index": None,
        "expected_index": None,
        "error": error,
        "converged": failure is None,
    }


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, *, images: int | None, comment: str,
         counter: str = "harness") -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [line for line in comment.strip().splitlines() if line]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        if images:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
            lines.append("call_unit total")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row.get('hessian_index') == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {template.get('problem') or records[0].get('problem', '')}")
    lines.append(f"tuning {template['tuning']}")
    lines.append(f"counter {counter}")
    if images is not None:
        lines.append(f"images {images}")
    return "\n".join(lines) + "\n"


def write_campaign(
    output: Path,
    records: list[dict],
    template: dict,
    problem: dict,
    settings: dict,
    parameters: dict,
    *,
    images: int | None,
    comment: str,
    counter: str = "harness",
) -> dict:
    """Write ``benchmark.dat``, the starts file, and the v2 score."""
    if counter not in ("harness", "client"):
        raise ValueError("counter must be harness or client")
    if any(row.get("counter", counter) != counter for row in records):
        raise ValueError("campaign records disagree on the counter")
    output.parent.mkdir(parents=True, exist_ok=True)
    stamped = []
    for record in records:
        row = dict(record)
        row["problem"] = problem["id"]
        row["template"] = template["id"]
        row["code"] = template["code"]
        row["code_version"] = template["code_version"]
        row["commit"] = template["commit"]
        row["algorithm"] = template["algorithm"]
        row["tuning"] = template["tuning"]
        stamped.append(row)
    text = _dat(stamped, {**template, "problem": problem["id"]}, images=images, comment=comment, counter=counter)
    output.write_text(text, encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), stamped)
    score = score_v2(
        [
            {
                "force_calls": row["force_calls"],
                "success": row["success"],
                "failure": row["failure"],
            }
            for row in stamped
        ],
        settings["cap"],
    )
    successes = [int(row["force_calls"]) for row in stamped if row["success"]]
    success_only = {
        "successes": len(successes),
        "nfailed": len(stamped) - len(successes),
        "n": len(stamped),
        "transition_states": sum(1 for row in stamped if row.get("hessian_index") == 1),
    }
    if successes:
        success_only["force_calls"] = int(round(statistics.fmean(successes)))
        success_only["force_calls_min"] = min(successes)
        success_only["force_calls_max"] = max(successes)
        success_only["force_calls_median"] = statistics.median(successes)
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": counter,
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score,
        "success_mean": success_only,
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": success_only.get("force_calls"),
        "nfailed": success_only["nfailed"],
        "starts": success_only["n"],
        "successes": success_only["successes"],
        "output": str(output),
    }


def run_jobs(jobs: list[dict], execute, workers: int) -> list[dict]:
    """One search per process. A reused process would keep a native singleton."""
    if len(jobs) <= 1:
        records = [execute(job) for job in jobs]
    else:
        import multiprocessing

        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        ) as pool:
            records = list(pool.map(execute, jobs))
    order = {job["name"]: index for index, job in enumerate(jobs)}
    records.sort(key=lambda row: order[row["start"]])
    return records


def run_start(jobs: list[dict], execute, start: str) -> dict:
    """The campaign job named ``start``, alone in a spawned process.

    A campaign of more than one job gives each job a fresh spawned
    process, so one start here sees the same process state.
    """
    import multiprocessing

    chosen = [job for job in jobs if str(job["name"]) == str(start)]
    if len(chosen) != 1:
        raise KeyError(f"{len(chosen)} jobs are named {start}")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        return pool.submit(execute, chosen[0]).result()


def prepared(root: Path, problem_name: str, template_name: str, code_id: str) -> tuple[dict, dict, dict]:
    root = Path(root)
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    pin = registry_pin(root, code_id)
    require_pin(template, pin)
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    return problem, template, band_settings(problem)
