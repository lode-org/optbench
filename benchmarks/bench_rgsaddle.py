"""rgsaddle band entries and min-mode searches as airspeed-velocity benchmarks.

One class is one problem and one template. Parameters are the seed and
the archived start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start, and it is read at import.

``time_search`` is the timed search. It writes a sidecar. The ``track_*``
methods read that sidecar, so a full ``asv run`` records force calls,
success, the four failure classes, and wall time from the same search.
``setup`` does not search: the parent ``TrackedSearch.setup`` would run
the band again before the timer. A track whose sidecar is missing
searches, so a track-only run is a real number.

The seed is the string ``"0"``. The band does not draw random numbers.
The parameter is there so every entry has the same seed and start axes.
Sidecars live under ``results/rgsaddle-asv`` and are not the published
store. The published numbers are the asv result file.

Min-mode classes use ``template_id``. The default grid is seed ``"0"``
and the first geometry. ``OPTBENCH_ASV_FULL=1`` at import selects every
start (200 on LJ38, 49 on the Pt island). Counting is
``RgpotCounter.calculate`` on a ``HarnessCounter``. A missing
``RGSADDLE_LIB`` is a ``RuntimeError``. Nonlinear CG is not a row.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]

PROBLEMS = (
    ("Heptamer", "neb/Pt-heptamer"),
    ("PtIsland", "double-ended-saddle-search/pt-island"),
    ("Lj38De", "double-ended-saddle-search/lj38"),
    ("Lj38TwoTs", "double-ended-saddle-search/lj38_2ts_separation"),
)

TEMPLATES = (
    ("Neb", "rgsaddle/neb"),
    ("CiNeb", "rgsaddle/ci-neb"),
    ("EwCiNeb", "rgsaddle/ew-ci-neb"),
    ("IdppCiNeb", "rgsaddle/idpp-ci-neb"),
    ("SidppCiNeb", "rgsaddle/sidpp-ci-neb"),
    ("OciNeb", "rgsaddle/oci-neb"),
)

TRACKS = (
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)

_FAILURES = ("cap", "wrong_stationary", "divergence", "crash")
CLASS_PROBLEM: dict[str, str] = {}
CLASS_TEMPLATE: dict[str, str] = {}
CLASS_CAP: dict[str, int] = {}


def result_root() -> Path:
    """Directory that holds ``results/rgsaddle-asv`` sidecars."""
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def sidecar_path(problem_name: str, template_name: str, seed: str, start: str) -> Path:
    """One JSON record for one asv parameter pair.

    The problem is part of the path. LJ38 and the two-saddle set both
    name their first geometry ``pair_00``.
    """
    slug = problem_name.replace("/", "-") + "--" + template_name.replace("/", "-")
    return result_root() / "results" / "rgsaddle-asv" / slug / str(seed) / f"{start}.json"


def all_starts(problem_name: str) -> list[str]:
    """Every archived pair name for ``problem_name``, in archive order."""
    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.pes import load_pairs

    problem = resolve_problem(ROOT, problem_name)
    pairs = load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )
    return [pair.name for pair in pairs]


def problem_cap(problem_name: str) -> int:
    from optbench.campaign import resolve_problem

    return int(resolve_problem(ROOT, problem_name)["convergence"]["max_force_calls"])


def selected_starts(names: list[str]) -> list[str]:
    """The first geometry, or every name when ``OPTBENCH_ASV_FULL=1``."""
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return list(names)
    return list(names[:1])


def _write_sidecar(problem_name: str, template_name: str, seed: str, start: str, record: dict) -> None:
    path = sidecar_path(problem_name, template_name, seed, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _failure_of(record: dict) -> str | None:
    if record.get("success"):
        return None
    label = record.get("failure")
    if label in _FAILURES:
        return str(label)
    from optbench.failures import classify_failure

    return classify_failure([str(reason) for reason in record.get("reasons") or []]) or "crash"


def _expected_track(record: dict, method: str) -> float:
    if method == "track_force_calls":
        return float(record["force_calls"])
    if method == "track_success":
        return 1.0 if record["success"] else 0.0
    failure = _failure_of(record)
    name = method.removeprefix("track_failure_")
    return 1.0 if failure == name else 0.0


def _bind(class_name: str, problem: str, template: str, starts: list[str], cap: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = 1800
        problem_name = problem
        template_name = template

        def setup(self, seed, start):
            from optbench.rgsaddle_band import library_path

            if library_path() is None:
                raise NotImplementedError("librgsaddle.so is not available")
            path = sidecar_path(self.problem_name, self.template_name, str(seed), str(start))
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.record = None

        def measure(self, seed, start):
            from optbench.rgsaddle_band import search_named

            try:
                record = search_named(ROOT, self.problem_name, self.template_name, str(start), str(seed))
            except Exception as exc:
                record = {
                    "schema": "optbench.start/v2",
                    "start": str(start),
                    "seed": str(seed),
                    "force_calls": 0,
                    "verify_calls": 0,
                    "counter": "harness",
                    "comparable": True,
                    "success": False,
                    "failure": "crash",
                    "reasons": ["crash"],
                    "crash": type(exc).__name__,
                    "problem": self.problem_name,
                    "template": self.template_name,
                }
            record["force_calls"] = float(record["force_calls"])
            record["success"] = bool(record["success"])
            record["failure"] = _failure_of(record)
            if record["success"]:
                record["failure"] = None
            return record

        def time_search(self, seed, start):
            print(f"start {self.template_name} {start}", file=sys.stderr, flush=True)
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            _write_sidecar(self.problem_name, self.template_name, str(seed), str(start), record)
            self.record = record
            print(
                f"done {self.template_name} {start} calls={record['force_calls']} "
                f"success={record['success']} failure={record['failure']}",
                file=sys.stderr,
                flush=True,
            )

        def _ready(self, seed, start) -> dict:
            record = getattr(self, "record", None)
            if record is None:
                record = self.measure(seed, start)
                _write_sidecar(self.problem_name, self.template_name, str(seed), str(start), record)
                self.record = record
            return record

        def track_force_calls(self, seed, start) -> float:
            return float(self._ready(seed, start)["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "crash" else 0.0

    _Benchmark.cap = cap
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    CLASS_PROBLEM[class_name] = problem
    CLASS_TEMPLATE[class_name] = template
    CLASS_CAP[class_name] = cap
    return _Benchmark


def _asv_column(document: dict, benchmark: str, column: str):
    row = document["results"][benchmark]
    index = document["result_columns"].index(column)
    if index >= len(row) or row[index] is None:
        raise KeyError(f"{benchmark} has no {column}")
    return row[index]


def verify_asv_tracks(document: dict) -> None:
    """Every rgsaddle ``track_*`` value equals the sidecar from that search."""
    for name in document["results"]:
        parts = name.split(".")
        if len(parts) < 2 or parts[-1] not in TRACKS:
            continue
        class_name = parts[-2]
        if class_name not in CLASS_TEMPLATE:
            continue
        method = parts[-1]
        params = _asv_column(document, name, "params")
        values = _asv_column(document, name, "result")
        pairs = list(itertools.product(*params))
        if len(values) != len(pairs):
            raise ValueError(f"{name} has {len(values)} results for {len(pairs)} parameters")
        template_name = CLASS_TEMPLATE[class_name]
        for (seed, start), value in zip(pairs, values, strict=True):
            if value is None or value != value:
                raise ValueError(f"{name}({seed}, {start}) has no result")
            record = json.loads(
                sidecar_path(CLASS_PROBLEM[class_name], template_name, seed, start).read_text(encoding="utf-8")
            )
            expected = _expected_track(record, method)
            if float(value) != expected:
                raise ValueError(f"{name}({seed}, {start}) is {value}, the sidecar has {expected}")


def attach_wall_seconds(document: dict, directory: Path) -> int:
    """Copy each ``time_search`` sample onto the sidecar as ``wall_seconds``.

    The sample is asv's timed call. Returns how many sidecars were updated.
    """
    updated = 0
    for name, row in document["results"].items():
        parts = name.split(".")
        if len(parts) < 2 or parts[-1] != "time_search":
            continue
        class_name = parts[-2]
        if class_name not in CLASS_TEMPLATE:
            continue
        index = document["result_columns"].index("result")
        if index >= len(row) or row[index] is None:
            raise KeyError(f"{name} has no result")
        params = row[document["result_columns"].index("params")]
        values = row[index]
        template_name = CLASS_TEMPLATE[class_name]
        for (seed, start), value in zip(itertools.product(*params), values, strict=True):
            if value is None or value != value:
                raise ValueError(f"{name}({seed}, {start}) has no wall time")
            path = (
                directory
                / (CLASS_PROBLEM[class_name].replace("/", "-") + "--" + template_name.replace("/", "-"))
                / str(seed)
                / f"{start}.json"
            )
            if not path.is_file():
                path = sidecar_path(CLASS_PROBLEM[class_name], template_name, seed, start)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["wall_seconds"] = float(value)
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            updated += 1
    return updated


for _problem_prefix, _problem_name in PROBLEMS:
    _names = all_starts(_problem_name)
    _cap = problem_cap(_problem_name)
    _selected = selected_starts(_names)
    for _template_prefix, _template_name in TEMPLATES:
        _class_name = f"{_problem_prefix}Rgsaddle{_template_prefix}"
        globals()[_class_name] = _bind(_class_name, _problem_name, _template_name, _selected, _cap)

CACHE = Path(os.environ.get("OPTBENCH_RGSADDLE_CACHE", "/tmp/optbench-asv-rgsaddle"))
RUNNER_REVISION = "harness-rgpot-1"


def lj38_start_names() -> list[str]:
    return [f"{index:04d}" for index in range(200)]


def pt_start_names() -> list[str]:
    return [f"initial_{index}" for index in range(49)]


def selected_lj38() -> list[str]:
    names = lj38_start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def selected_pt() -> list[str]:
    names = pt_start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def _require_library() -> None:
    if not os.environ.get("RGSADDLE_LIB"):
        raise RuntimeError("set RGSADDLE_LIB to the pinned librgsaddle.so")


def _cache_path(template_id: str, start: str, seed: str) -> Path:
    safe = template_id.replace("/", "-")
    return CACHE / RUNNER_REVISION / safe / f"{start}__seed{seed}.json"


def _execute(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    os.environ["RAYON_NUM_THREADS"] = "1"
    _require_library()
    from optbench.campaign import _check_stop, load_template, resolve_problem
    from optbench.rgsaddle_minmode import _dispatch, _geometries, _spec_from_geometry

    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, problem_id)
    cap = _check_stop(problem, template)
    work = CACHE / "geometries" / problem_id.replace("/", "-")
    rows = _geometries(ROOT, problem, work)
    row = next((item for item in rows if item["name"] == start), None)
    if row is None:
        raise RuntimeError(f"{problem_id} has no start {start}")
    optimizer = (template.get("ini") or {}).get("Optimizer") or {}
    convergence = problem.get("convergence") or {}
    force_tol = float(optimizer.get("converged_force", convergence.get("force_norm", 1e-3)))
    spec = _spec_from_geometry(row, problem, template, cap, force_tol)
    record = _dispatch(spec)
    record["seed"] = seed
    record["template"] = template_id
    record["problem"] = problem_id
    record.pop("x", None)
    path = _cache_path(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    path = _cache_path(template_id, start, seed)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _execute(template_id, problem_id, start, seed)


def _cap(template_id: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_id)
    return _check_stop(resolve_problem(ROOT, template["problem"]), template)


def _bind_minmode(class_name: str, template_id: str, problem_id: str, starts: list[str], limit: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]
        timeout = limit

        def setup(self, seed, start):
            _require_library()

        def measure(self, seed, start):
            return _execute(template_id, problem_id, start, seed)

        def time_search(self, seed, start):
            self.record = self.measure(seed, start)

        def track_force_calls(self, seed, start) -> float:
            return float(_load(template_id, problem_id, start, seed)["force_calls"])

        def track_success(self, seed, start) -> float:
            record = _load(template_id, problem_id, start, seed)
            return 1.0 if record["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "crash" else 0.0

        track_force_calls.unit = "force calls"
        track_success.unit = "fraction"
        track_failure_cap.unit = "count"
        track_failure_wrong_stationary.unit = "count"
        track_failure_divergence.unit = "count"
        track_failure_crash.unit = "count"

    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.cap = _cap(template_id)
    _Benchmark.template_id = template_id
    _Benchmark.problem_id = problem_id
    return _Benchmark


# (class, template, problem, starts, timeout seconds)
_ROWS = [
    ("RgsaddleLj38DimerLbfgs", "rgsaddle/dimer-lbfgs", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38DimerFire", "rgsaddle/dimer-fire", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38Lanczos", "rgsaddle/lanczos", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38DimerRotskip", "rgsaddle/dimer-rotskip", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38Prfo", "rgsaddle/prfo", "tsopt/LJ38", selected_lj38(), 7200),
    ("RgsaddlePtDimerLbfgs", "rgsaddle/pt-dimer-lbfgs", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtDimerFire", "rgsaddle/pt-dimer-fire", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtLanczos", "rgsaddle/pt-lanczos", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtDimerRotskip", "rgsaddle/pt-dimer-rotskip", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddleModeDimer", "rgsaddle/mode-dimer", "lowest-eigenvector/lj38", selected_lj38(), 1800),
    ("RgsaddleModeLanczos", "rgsaddle/mode-lanczos", "lowest-eigenvector/lj38", selected_lj38(), 1800),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind_minmode(_class_name, _template_id, _problem_id, _starts, _timeout)
