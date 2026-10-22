"""PyAMFF training, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` selects the
harness initialization ``seed_base + start``. The default grid is start
``0``. ``OPTBENCH_ASV_FULL=1`` selects every initialization. It is read
at import.

``setup`` raises ``NotImplementedError`` when PyAMFF is not installed, and
asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-pyamff"))


def _ready() -> bool:
    try:
        import pyamff  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def selected_indices(count: int) -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Pyamff" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem
    from optbench.runners.pyamff_ml import _full_n, _stop

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "pyamff":
            continue
        problem = resolve_problem(ROOT, str(template["problem"]))
        cap, tol, coeff = _stop(problem, template)
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": str(problem["id"]),
                "cap": cap,
                "tol": tol,
                "force_coefficient": coeff,
                "n": _full_n(problem),
            }
        )
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the harness seed {SEED}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.pyamff_ml import _job, _run_one, prepare

    problem = resolve_problem(ROOT, spec["problem_id"])
    template = load_template(ROOT, spec["template_id"])
    meta = prepare(ROOT, problem)
    path = _cache_path(spec["template_id"], SEED, str(int(start)))
    job = _job(template, problem, meta, int(start), path)
    record = _run_one(job)
    record["seed"] = SEED
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], SEED, str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _PyamffSearch(TrackedSearch):
    timeout = 21600

    def setup(self, seed, start):
        if not _ready():
            raise NotImplementedError("pyamff is not installed in this environment")
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.spec, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _load(self.spec, seed, start)
        return self.record

    def track_force_calls(self, seed, start) -> float:
        return float(self._stored(seed, start)["force_calls"])

    def track_success(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["success"] else 0.0

    def track_failure_cap(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "cap" else 0.0

    def track_failure_wrong_stationary(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "wrong_stationary" else 0.0

    def track_failure_divergence(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "divergence" else 0.0

    def track_failure_crash(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "crash" else 0.0


_PyamffSearch.track_force_calls.unit = "force calls"
_PyamffSearch.track_success.unit = "fraction"
_PyamffSearch.track_failure_cap.unit = "count"
_PyamffSearch.track_failure_wrong_stationary.unit = "count"
_PyamffSearch.track_failure_divergence.unit = "count"
_PyamffSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])

    class _Benchmark(_PyamffSearch):
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]
        cap = bound["cap"]
        param_names = ["seed", "start"]
        params = [[SEED], starts]

    _Benchmark.__name__ = bound["class_name"]
    _Benchmark.__qualname__ = bound["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
