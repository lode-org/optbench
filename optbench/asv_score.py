"""Read asv result files and score the quadratic sample.

The force-call counts, the success bit, and the failure-class bits are
the track values asv saved. This module reads the optimizers asv already stored.
Caps, sigmas, and the profile points stay on the benchmark classes and
in ``optbench.statistics`` because a result row does not store them.
That gap is ``docs/asv-gaps.md``.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import sys
from importlib import metadata
from pathlib import Path

from .failures import FAILURES
from .noise import KAPPA, WINDOW, noise_floor, stop_threshold
from .profile import performance_profile, profile_cost
from .score import score_v2
from .quadratic import ENERGY_TOL, FORCE_NORM, RMSD_TOL
from .statistics import (
    COMBINED_RESAMPLES,
    DIMER_CAP,
    DIMER_SAMPLES,
    DIMER_SIGMA,
    DISPLACEMENT_CAP,
    DISPLACEMENT_SAMPLES,
    DISPLACEMENT_SIGMA,
    DISPLACEMENT_STEP,
    FAILURE_ORDER,
    FAILURE_ROW_RESAMPLES,
    KEPT_CAP,
    NOISE_CAP,
    NOISE_RESAMPLES,
    PROFILE_RESAMPLES,
    PROFILE_SEED,
    SEED,
    SIGMA_LEVELS,
    _problem_points,
)
from .v2 import collect_provenance, summarise_run

TRACKS = (
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)


def _root_on_path(root: Path) -> None:
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def _quadratic(root: Path):
    _root_on_path(root)
    import benchmarks.quadratic as quadratic

    return quadratic


_UNWRAP_MISSING = object()
_UNWRAP_CACHE: dict[str, object] = {}


def _cacheable(value) -> bool:
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, tuple):
        return all(_cacheable(item) for item in value)
    return False


def _unwrap(value):
    """asv stores parameter values with ``repr`` quotes inside the JSON string.

    The same quoted strings repeat for every start. An immutable parse is
    remembered. A list or a dict is parsed again, so callers do not share it.
    """
    if not isinstance(value, str):
        return value
    cached = _UNWRAP_CACHE.get(value, _UNWRAP_MISSING)
    if cached is not _UNWRAP_MISSING:
        return cached
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value
    if _cacheable(parsed):
        _UNWRAP_CACHE[value] = parsed
    return parsed


def decoded_params(params: list) -> list[list]:
    """The parameter axes of one result row, as the benchmark received them.

    asv-runner saves each value as ``repr(value)``. A string parameter
    ``min01`` is the JSON string ``"'min01'"``.
    """
    return [[_unwrap(value) for value in axis] for axis in params]


def _count(value: float) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _track_index(result) -> dict[str, str]:
    """Map ``Class.method`` to the one result key that ends with it.

    Built once per file. A class name is the component before the method,
    so each tail is scanned a single time instead of once per track.
    """
    cached = getattr(result, "_optbench_track_index", None)
    if cached is not None:
        return cached
    found: dict[str, str] = {}
    ambiguous: set[str] = set()
    for key in result.get_all_result_keys():
        text = str(key)
        dot = text.rfind(".")
        if dot <= 0:
            continue
        tail = text[text.rfind(".", 0, dot) + 1 :]
        if tail in found or tail in ambiguous:
            found.pop(tail, None)
            ambiguous.add(tail)
            continue
        found[tail] = text
    result._optbench_track_index = found
    return found


def _key(result, class_name: str, method: str) -> str:
    suffix = f"{class_name}.{method}"
    found = _track_index(result)
    key = found.get(suffix)
    if key is None:
        matches = [item for item in result.get_all_result_keys() if str(item).endswith(suffix)]
        raise SystemExit(
            f"{suffix} matched {matches or 'nothing'} in commit {result.commit_hash}"
        )
    return key


def _missing(value) -> bool:
    if value is None:
        return True
    # A NaN is the only float that is not equal to itself.
    return value != value


def _column(result, key, params):
    """Values for one track, in the stored parameter order.

    ``get_result_value`` rebuilds that order. No public accessor returns
    the stored list, so a matching list is copied and the caller cannot
    alias asv's own list.
    """
    stored_params = result.get_result_params(key)
    values = result._results[key]
    if stored_params == params and isinstance(values, list):
        return list(values)
    values = result.get_result_value(key, params)
    if not isinstance(values, list):
        return [values]
    return values


def _combo_at(index: int, axes: list) -> tuple:
    """The parameter tuple at ``index``, last axis varying fastest."""
    chosen = []
    for axis in reversed(axes):
        size = len(axis)
        chosen.append(axis[index % size])
        index //= size
    chosen.reverse()
    return tuple(chosen)


def _track_columns(result, class_name: str):
    """Aligned track columns for one class, after a single key scan."""
    index = _track_index(result)
    force_key = _key(result, class_name, "track_force_calls")
    params = result.get_result_params(force_key)
    if params:
        stored_axes = [tuple(str(_unwrap(part)) for part in axis) for axis in params]
        width = 1
        for axis in stored_axes:
            width *= len(axis)
        combos = None
    else:
        stored_axes = []
        width = 1
        combos = [()]
    columns = {}
    for method in TRACKS:
        key = index.get(f"{class_name}.{method}")
        if not key:
            _key(result, class_name, method)
        values = _column(result, key, params)
        if len(values) != width:
            raise SystemExit(f"{class_name}.{method} has {len(values)} values for {width} parameters")
        for value in values:
            if value is None or value != value:
                raise SystemExit(f"{class_name} {method} was skipped or failed")
        columns[method] = values
    if combos is None:
        combos = list(itertools.product(*stored_axes))
    return force_key, params, combos, columns


def track_rows(result, class_name: str) -> list[dict]:
    """One dict per parameter combination, in asv's product order."""
    force_key, params, combos, columns = _track_columns(result, class_name)
    version = result.benchmark_version.get(force_key)
    force = columns["track_force_calls"]
    success_col = columns["track_success"]
    fail_cols = tuple((name, columns[f"track_failure_{name}"]) for name in FAILURES)
    rows = []
    append = rows.append
    for index, combo in enumerate(combos):
        success = success_col[index]
        if success == 1.0 or success == 1:
            for name, column in fail_cols:
                if column[index] == 1.0 or column[index] == 1:
                    shown = _combo_at(index, params) if params else ()
                    raise SystemExit(f"{class_name} {shown} is a success with failure tracks {[name]}")
            failure = None
            ok = True
        elif success == 0.0 or success == 0:
            failure = None
            for name, column in fail_cols:
                if column[index] == 1.0 or column[index] == 1:
                    if failure is not None:
                        shown = _combo_at(index, params) if params else ()
                        raise SystemExit(f"{class_name} {shown} failure tracks are {[failure, name]}")
                    failure = name
            if failure is None:
                shown = _combo_at(index, params) if params else ()
                raise SystemExit(f"{class_name} {shown} failure tracks are []")
            ok = False
        else:
            shown = _combo_at(index, params) if params else ()
            raise SystemExit(f"{class_name} {shown} success track is {float(success)}")
        append(
            {
                "params": combo,
                "force_calls": _count(force[index]),
                "success": ok,
                "failure": failure,
                "version": version,
            }
        )
    return rows


def load_quadratic_result(results_dir: Path, machine: str | None = None):
    """The one result file that holds ``QuadraticDisplacement`` tracks.

    ``results_dir`` has one directory per asv machine name. ``machine``
    picks one of them. Without it the tracks have to be under a single
    machine. Either way that machine holds one quadratic result.
    """
    from asv.results import iter_results

    found = []
    for result in iter_results(str(results_dir)):
        keys = list(result.get_all_result_keys())
        if any(key.endswith("QuadraticDisplacement.track_force_calls") for key in keys):
            found.append(result)
    if not found:
        raise SystemExit(
            f"{results_dir} has no QuadraticDisplacement tracks. "
            "Run asv run without --quick on the commit that contains benchmarks/quadratic.py."
        )
    machines = sorted({result.params.get("machine", "") for result in found})
    if machine:
        found = [result for result in found if result.params.get("machine") == machine]
        if not found:
            raise SystemExit(
                f"{results_dir} has no QuadraticDisplacement tracks for machine {machine}. "
                f"Machines with them: {machines}"
            )
    elif len(machines) != 1:
        raise SystemExit(f"quadratic tracks are under machines {machines}. Pass --machine with one of them.")
    commits = {result.commit_hash for result in found}
    if len(commits) != 1 or len(found) != 1:
        raise SystemExit(
            f"quadratic tracks are in {len(found)} files for commits {sorted(commits)}"
        )
    return found[0]


def _start(row: dict, name: str) -> dict:
    seed, start = row["params"]
    return {
        "name": name,
        "seed": seed,
        "start": start,
        "force_calls": row["force_calls"],
        "success": row["success"],
        "failure": row["failure"],
    }


def _record(row: dict) -> dict:
    return {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}


def _noise_classes(module) -> list[type]:
    found = [value for value in vars(module).values() if isinstance(value, type) and hasattr(value, "sigma")]
    by_sigma = {cls.sigma: cls for cls in found}
    missing = [sigma for sigma in SIGMA_LEVELS if sigma not in by_sigma]
    if missing or len(by_sigma) != len(SIGMA_LEVELS):
        raise SystemExit(f"noise benchmarks do not match SIGMA_LEVELS, missing {missing}")
    return [by_sigma[sigma] for sigma in SIGMA_LEVELS]


def _failure_classes(module) -> list[type]:
    found = [value for value in vars(module).values() if isinstance(value, type) and hasattr(value, "kind")]
    by_kind = {cls.kind: cls for cls in found}
    missing = [kind for kind in FAILURE_ORDER if kind not in by_kind]
    if missing:
        raise SystemExit(f"failure benchmarks missing {missing}")
    return [by_kind[kind] for kind in FAILURE_ORDER]


def statistics_from_asv(root: Path, results_dir: Path, machine: str | None = None) -> dict:
    """Build the validation document from saved asv tracks."""
    root = Path(root)
    results_dir = Path(results_dir)
    module = _quadratic(root)
    result = load_quadratic_result(results_dir, machine)
    displacement_rows = track_rows(result, "QuadraticDisplacement")
    dimer_rows = track_rows(result, "QuadraticDimer")
    if len(displacement_rows) != DISPLACEMENT_SAMPLES or len(dimer_rows) != DIMER_SAMPLES:
        raise SystemExit("the saved seed grid is not the v2 sample")
    displacement = [
        _start(row, f"seed-{int(row['params'][0]) - SEED}") for row in displacement_rows
    ]
    dimer = [_start(row, f"seed-{int(row['params'][0]) - SEED}") for row in dimer_rows]
    profile_classes = (
        module.QuadraticProfileStep005,
        module.QuadraticProfileStep001,
        module.QuadraticProfileCap30,
    )
    codes = {}
    par2 = {}
    for cls in profile_classes:
        rows = track_rows(result, cls.__name__)
        codes[cls.code_name] = [_start(row, f"problem-{row['params'][1]}") for row in rows]
        par2[cls.code_name] = score_v2(
            [_record(row) for row in rows],
            cls.cap,
            resamples=PROFILE_RESAMPLES,
            seed=SEED,
        )
    profile_costs = {
        code: [profile_cost(start) for start in starts] for code, starts in codes.items()
    }
    profile = performance_profile(profile_costs)
    noise_runs = []
    for cls in _noise_classes(module):
        rows = track_rows(result, cls.__name__)
        if len(rows) != 1:
            raise SystemExit(f"{cls.__name__} should be one start")
        row = rows[0]
        components = 2
        sigma = cls.sigma
        start = _start(row, f"sigma-{sigma}")
        noise_runs.append(
            {
                "sigma": sigma,
                "kappa": KAPPA,
                "window": WINDOW,
                "components": components,
                "noise_floor": noise_floor(sigma, components, KAPPA),
                "force_threshold": stop_threshold(FORCE_NORM, sigma, components, KAPPA),
                "energy_tol": max(ENERGY_TOL, KAPPA * sigma),
                "rmsd_tol": max(RMSD_TOL, KAPPA * sigma),
                "start": start,
                "metrics": score_v2(
                    [_record(row)],
                    NOISE_CAP,
                    resamples=NOISE_RESAMPLES,
                    seed=SEED,
                ),
                "benchmark_version": row["version"],
            }
        )
    failure_classes = []
    failure_rows = {}
    for cls in _failure_classes(module):
        rows = track_rows(result, cls.__name__)
        if len(rows) != 1:
            raise SystemExit(f"{cls.__name__} should be one start")
        row = rows[0]
        if row["failure"] != cls.kind:
            raise SystemExit(f"{cls.__name__} track says {row['failure']}, class says {cls.kind}")
        failure_rows[cls.kind] = row
        start = _start(row, cls.kind)
        failure_classes.append(
            {
                "cap": cls.cap,
                "start": start,
                "metrics": score_v2(
                    [_record(row)],
                    cls.cap,
                    resamples=FAILURE_ROW_RESAMPLES,
                    seed=SEED,
                ),
                "benchmark_version": row["version"],
            }
        )
    kept_rows = track_rows(result, "QuadraticKept")
    if len(kept_rows) != 1:
        raise SystemExit("QuadraticKept should be one start")
    kept = _start(kept_rows[0], "finished-under-cap")
    cap_hit = _start(failure_rows["cap"], "cap-hit")
    combined = score_v2(
        [_record(kept_rows[0]), _record(failure_rows["cap"])],
        KEPT_CAP,
        resamples=COMBINED_RESAMPLES,
        seed=SEED,
    )
    provenance = collect_provenance(
        code="optbench",
        code_version=metadata.version("optbench"),
        commit=result.commit_hash,
        parameters={
            "displacement_sigma": DISPLACEMENT_SIGMA,
            "dimer_sigma": DIMER_SIGMA,
            "seed": SEED,
            "samples": DISPLACEMENT_SAMPLES,
            "noise_sigma": list(SIGMA_LEVELS),
            "kappa": KAPPA,
            "window": WINDOW,
            "profile_seed": PROFILE_SEED,
            "asv": metadata.version("asv"),
            "asv_runner": metadata.version("asv-runner"),
            "env_name": result._env_name,
            "python": result._python,
            "date": result.date,
            "displacement_version": displacement_rows[0]["version"],
            "dimer_version": dimer_rows[0]["version"],
        },
        machine=dict(result.params),
    )
    return {
        "schema": "optbench.validation/v2",
        "status": "quadratic",
        "source": "asv",
        "problem": "quadratic/bowl",
        "note": (
            "Track values from results/asv for one commit. "
            "Not a Lennard-Jones campaign. Caps are the benchmark class attributes."
        ),
        "provenance": provenance,
        "displacement": {
            "sigma": DISPLACEMENT_SIGMA,
            "samples": DISPLACEMENT_SAMPLES,
            "seed": SEED,
            "step": DISPLACEMENT_STEP,
            "cap": DISPLACEMENT_CAP,
            "starts": displacement,
            "metrics": summarise_run(displacement, DISPLACEMENT_CAP),
        },
        "dimer": {
            "sigma": DIMER_SIGMA,
            "samples": DIMER_SAMPLES,
            "seed": SEED,
            "cap": DIMER_CAP,
            "x0": [0.4, -0.3],
            "starts": dimer,
            "metrics": summarise_run(dimer, DIMER_CAP),
        },
        "profile": {
            "problems": [list(point) for point in _problem_points()],
            "calls": {code: [start["force_calls"] for start in starts] for code, starts in codes.items()},
            "success": {code: [start["success"] for start in starts] for code, starts in codes.items()},
            "par2": par2,
            "dolan_more": {
                "taus": profile["taus"],
                "rho": profile["rho"],
                "solved": profile["solved"],
                "area_log10": profile["area_log10"],
                "reference": profile["reference"],
            },
        },
        "noise": {"sigma_levels": list(SIGMA_LEVELS), "kappa": KAPPA, "window": WINDOW, "runs": noise_runs},
        "failures": {
            "classes": failure_classes,
            "combined_cap": KEPT_CAP,
            "combined_starts": [kept, cap_hit],
            "metrics": combined,
        },
    }


def write_statistics_from_asv(
    path, root: Path = Path("."), results_dir: Path | None = None, machine: str | None = None
) -> dict:
    root = Path(root)
    dest_results = Path(results_dir) if results_dir is not None else root / "results" / "asv"
    document = statistics_from_asv(root, dest_results, machine)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document
