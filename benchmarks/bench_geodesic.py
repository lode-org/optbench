"""Geodesic initialiser ablation as airspeed-velocity benchmarks.

One class is one problem and one template. Parameters are the seed and
the archived start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start, and it is read at import.

``setup`` does not search. ``time_search`` runs the search once and
writes a sidecar. The ``track_*`` methods read that sidecar. A track
whose sidecar is missing searches, so a track-only run is a real number.

The seed is ``"0"``. Sidecars live under ``results/geodesic-asv`` and
are not the published store. The published numbers are the asv result
file and the campaign files.
"""

from __future__ import annotations

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

RGSADDLE_PROBLEMS = (
    ("Heptamer", "neb/Pt-heptamer"),
    ("PtIsland", "double-ended-saddle-search/pt-island"),
    ("Lj38De", "double-ended-saddle-search/lj38"),
    ("Lj38TwoTs", "double-ended-saddle-search/lj38_2ts_separation"),
)

RGSADDLE_TEMPLATES = (
    ("CiNeb", "geodesic/ci-neb"),
    ("CiNebLbfgs", "geodesic/ci-neb-lbfgs"),
)

ASE_TEMPLATES = (
    ("Lj38De", "geodesic/ase-de-lj38-default", "AseDefault"),
    ("Lj38TwoTs", "geodesic/ase-de-2ts-default", "AseDefault"),
    ("PtIsland", "geodesic/ase-de-pt-default", "AseDefault"),
    ("Heptamer", "geodesic/ase-heptamer-default", "AseDefault"),
    ("Heptamer", "geodesic/ase-heptamer-tuned", "AseTuned"),
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
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def sidecar_path(problem_name: str, template_name: str, seed: str, start: str) -> Path:
    slug = problem_name.replace("/", "-") + "--" + template_name.replace("/", "-")
    return result_root() / "results" / "geodesic-asv" / slug / str(seed) / f"{start}.json"


def selected_starts(names: list[str]) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return list(names)
    return list(names[:1])


def _timeout(problem_name: str) -> int:
    if problem_name == "double-ended-saddle-search/lj38_2ts_separation":
        return 14400
    if problem_name == "double-ended-saddle-search/pt-island":
        return 7200
    return 3600


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


def _rgsaddle_starts(problem_name: str) -> list[str]:
    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.pes import load_pairs

    problem = resolve_problem(ROOT, problem_name)
    pairs = load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )
    return [pair.name for pair in pairs]


def _ase_starts(template_name: str) -> tuple[str, list[str], int]:
    from optbench.ase_sella import discover, materialize
    from optbench.campaign import load_template, resolve_problem

    template = load_template(ROOT, template_name)
    problem = resolve_problem(ROOT, template["problem"])
    material = materialize(ROOT, template, problem, Path("/tmp/optbench-geometry"))
    jobs = discover(material, template)
    if not jobs:
        raise RuntimeError(f"{template_name} has no starts")
    return template["problem"], [name for name, _paths, _seed in jobs], int(template["stop"]["max_force_calls"])


def _problem_cap(problem_name: str) -> int:
    from optbench.campaign import resolve_problem

    return int(resolve_problem(ROOT, problem_name)["convergence"]["max_force_calls"])


def _bind(class_name: str, problem: str, template: str, starts: list[str], cap: int, backend: str):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = _timeout(problem)
        problem_name = problem
        template_name = template
        problem_id = problem
        template_id = template

        def setup(self, seed, start):
            if backend == "rgsaddle":
                from optbench.rgsaddle_band import library_path

                if library_path() is None:
                    raise NotImplementedError("librgsaddle.so is not available")
            path = sidecar_path(self.problem_name, self.template_name, str(seed), str(start))
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.record = None

        def measure(self, seed, start):
            try:
                if backend == "rgsaddle":
                    from optbench.rgsaddle_band import search_named

                    record = search_named(ROOT, self.problem_name, self.template_name, str(start), str(seed))
                else:
                    from optbench.ase_sella import run_point
                    from optbench.failures import failure_of

                    raw = run_point(ROOT, self.template_name, str(start), int(seed))
                    calls = float(raw["force_calls"])
                    reasons = [str(reason) for reason in raw.get("reasons") or []]
                    if "error" in reasons and "crash" not in reasons:
                        reasons = ["crash", *reasons]
                    success = bool(raw.get("success")) and calls < cap
                    if not success and not reasons:
                        reasons = ["cap"] if calls >= cap else ["divergence"]
                    judged = {"force_calls": calls, "success": success, "reasons": reasons}
                    record = {
                        "schema": "optbench.start/v2",
                        "start": str(start),
                        "seed": str(seed),
                        "force_calls": calls,
                        "success": success,
                        "failure": failure_of(judged, cap),
                        "reasons": reasons,
                        "counter": "harness",
                        "comparable": True,
                        "problem": self.problem_name,
                        "template": self.template_name,
                        "wall_seconds": raw.get("wall_seconds"),
                    }
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


for _prefix, _problem in RGSADDLE_PROBLEMS:
    _names = selected_starts(_rgsaddle_starts(_problem))
    _cap = _problem_cap(_problem)
    for _template_prefix, _template in RGSADDLE_TEMPLATES:
        _class_name = f"{_prefix}Geodesic{_template_prefix}"
        globals()[_class_name] = _bind(_class_name, _problem, _template, _names, _cap, "rgsaddle")

for _prefix, _template, _label in ASE_TEMPLATES:
    _problem, _names, _cap = _ase_starts(_template)
    _class_name = f"{_prefix}Geodesic{_label}"
    globals()[_class_name] = _bind(
        _class_name,
        _problem,
        _template,
        selected_starts(_names),
        _cap,
        "ase",
    )
