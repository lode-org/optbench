"""LAMMPS climbing-image NEB as airspeed-velocity benchmarks.

One class is one problem and one template. Parameters are the seed and
the archived start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start, and it is read at import.

``time_search`` is the timed search. It writes a sidecar. The ``track_*``
methods read that sidecar, so a full ``asv run`` records force calls,
success, the four failure classes, and wall time from the same search.
``setup`` does not search. A track whose sidecar is missing searches, so
a track-only run is a real number.

The seed is the string ``"0"``. The band does not draw random numbers.
Sidecars live under ``results/lammpsneb-asv`` and are not the published
store. The published numbers are the asv result file.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]

PROBLEMS = (
    ("Heptamer", "neb/Pt-heptamer"),
    ("PtIsland", "double-ended-saddle-search/pt-island"),
    ("Lj38De", "double-ended-saddle-search/lj38"),
    ("Lj38TwoTs", "double-ended-saddle-search/lj38_2ts_separation"),
)

TEMPLATES = (
    ("CiQuickmin", "lammpsneb/ci-quickmin"),
    ("CiFire", "lammpsneb/ci-fire"),
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
    """Directory that holds ``results/lammpsneb-asv`` sidecars."""
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def sidecar_path(problem_name: str, template_name: str, seed: str, start: str) -> Path:
    slug = problem_name.replace("/", "-") + "--" + template_name.replace("/", "-")
    return result_root() / "results" / "lammpsneb-asv" / slug / str(seed) / f"{start}.json"


def all_starts(problem_name: str) -> list[str]:
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
        timeout = 3600
        problem_name = problem
        template_name = template

        def setup(self, seed, start):
            from optbench.lammps_neb import lammps_executable

            if lammps_executable() is None:
                raise NotImplementedError("the LAMMPS binary is not available")
            path = sidecar_path(self.problem_name, self.template_name, str(seed), str(start))
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.record = None

        def measure(self, seed, start):
            from optbench.lammps_neb import search_named

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
            # setup loads a sidecar from an earlier attempt. Timing that
            # load again would replace the search, so keep the record.
            if getattr(self, "record", None) is not None:
                return
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
    """Every LAMMPS ``track_*`` value equals the sidecar from that search."""
    from optbench.asv_score import decoded_params

    for name in document["results"]:
        parts = name.split(".")
        if len(parts) < 2 or parts[-1] not in TRACKS:
            continue
        class_name = parts[-2]
        if class_name not in CLASS_TEMPLATE:
            continue
        method = parts[-1]
        params = decoded_params(_asv_column(document, name, "params"))
        values = _asv_column(document, name, "result")
        pairs = list(itertools.product(*params))
        if len(values) != len(pairs):
            raise ValueError(f"{name} has {len(values)} results for {len(pairs)} parameters")
        template_name = CLASS_TEMPLATE[class_name]
        for (seed, start), value in zip(pairs, values, strict=True):
            if value is None or value != value:
                raise ValueError(f"{name}({seed}, {start}) has no result")
            record = json.loads(
                sidecar_path(CLASS_PROBLEM[class_name], template_name, str(seed), str(start)).read_text(
                    encoding="utf-8"
                )
            )
            expected = _expected_track(record, method)
            if float(value) != expected:
                raise ValueError(f"{name}({seed}, {start}) is {value}, the sidecar has {expected}")


for _problem_prefix, _problem_name in PROBLEMS:
    _names = all_starts(_problem_name)
    _cap = problem_cap(_problem_name)
    _selected = selected_starts(_names)
    for _template_prefix, _template_name in TEMPLATES:
        _class_name = f"{_problem_prefix}Lammpsneb{_template_prefix}"
        globals()[_class_name] = _bind(_class_name, _problem_name, _template_name, _selected, _cap)
