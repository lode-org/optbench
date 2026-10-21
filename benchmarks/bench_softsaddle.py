"""SoftSaddle lowest-mode rows for airspeed velocity.

One class is one template. The default grid is the first LJ38 geometry.
``OPTBENCH_ASV_FULL=1`` selects all 200 starts, and it is read at import.
``time_search`` writes a sidecar. The ``track_*`` methods read that
sidecar, so one full run records force calls, success, the four failure
classes, and wall time from the same search. Import does not load the
shared libraries. A missing library is a ``RuntimeError`` from ``setup``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OPTBENCH_SOFTSADDLE_CACHE", "/tmp/optbench-asv-softsaddle"))


def lj38_start_names() -> list[str]:
    return [f"{index:04d}" for index in range(200)]


def selected_lj38() -> list[str]:
    names = lj38_start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def _cap(template_id: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_id)
    return int(_check_stop(resolve_problem(ROOT, template["problem"]), template))


def _solver(template_id: str) -> str:
    from optbench.campaign import load_template

    main = (load_template(ROOT, template_id).get("ini") or {}).get("Main") or {}
    return str(main.get("solver", "lanczos"))


def _cache_path(template_id: str, start: str, seed: str) -> Path:
    safe = template_id.replace("/", "-")
    return CACHE / safe / f"{start}__seed{seed}.json"


def _execute(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    from optbench.campaign import _check_stop, load_template, resolve_problem
    from optbench.rgsaddle_minmode import _geometries
    from optbench.softsaddle_run import library_path, mode_spec, run_mode_point

    solver = _solver(template_id)
    if library_path(solver) is None:
        raise RuntimeError(f"softsaddle {solver} library is not built")
    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, problem_id)
    cap = _check_stop(problem, template)
    work = CACHE / "geometries" / problem_id.replace("/", "-")
    rows = _geometries(ROOT, problem, work)
    row = next((item for item in rows if item["name"] == start), None)
    if row is None:
        raise RuntimeError(f"{problem_id} has no start {start}")
    index = rows.index(row)
    spec = mode_spec(ROOT, problem, template, row, cap, CACHE / "work" / template_id.replace("/", "-"), index)
    record = run_mode_point(spec)
    record["seed"] = seed
    record["template"] = template_id
    record["problem"] = problem_id
    path = _cache_path(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def _load(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    path = _cache_path(template_id, start, seed)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _execute(template_id, problem_id, start, seed)


def _bind(class_name: str, template_id: str, problem_id: str, starts: list[str], limit: int):
    solver = _solver(template_id)

    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]
        timeout = limit

        def setup(self, seed, start):
            from optbench.softsaddle_run import library_path

            if library_path(solver) is None:
                raise RuntimeError(f"softsaddle {solver} library is not built")

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


_ROWS = [
    ("SoftsaddleModeLanczos", "softsaddle/mode-lanczos", "lowest-eigenvector/lj38", selected_lj38(), 1800),
    ("SoftsaddleModeDavidson", "softsaddle/mode-davidson", "lowest-eigenvector/lj38", selected_lj38(), 1800),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind(_class_name, _template_id, _problem_id, _starts, _timeout)
