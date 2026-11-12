"""geomeTRIC minimization, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` is the
archived geometry. The default grid is that seed and start ``0``.
``OPTBENCH_ASV_FULL=1`` selects every archived start. It is read at import.

``time_search`` is the minimization. Track methods read the record that
call wrote. ``setup`` raises ``NotImplementedError`` when geomeTRIC is
not installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"
_FULL = {"minimization/lj38": 1000, "minimization/morse-bulk": 100}


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-geometric"))


def _ready() -> bool:
    try:
        import geometric  # noqa: F401
    except ImportError:
        return False
    return True


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Geometric" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    for template in list_templates(ROOT):
        if template.get("engine") != "geometric":
            continue
        problem_id = str(template["problem"])
        section = template.get("geometric") or {}
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "coordsys": str(section.get("coordsys")),
                "element": str(section.get("element")),
                "neighbors": int(section.get("neighbors") or 0),
                "cap": 10000,
                "n": _FULL[problem_id] if full else 1,
            }
        )
    return found


def _geometries(problem_id: str):
    if problem_id not in _GEOMETRIES:
        from optbench.campaign import resolve_problem
        from optbench.runners.rgmin_min import _load_geometries

        problem = resolve_problem(ROOT, problem_id)
        work = Path("/tmp/optbench-asv-geo")
        _GEOMETRIES[problem_id] = _load_geometries(ROOT, problem, None, work, work / "unused.dat")
    return _GEOMETRIES[problem_id]


_GEOMETRIES: dict[str, list] = {}


def _job(spec: dict, start: str) -> dict:
    from optbench.campaign import resolve_problem
    from optbench.runners.geometric_min import _kind

    index = int(start)
    geometries = _geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, box = geometries[index]
    kind = _kind(spec["problem_id"])
    problem = resolve_problem(ROOT, spec["problem_id"])
    convergence = problem["convergence"]
    return {
        "index": index,
        "start": name,
        "coords": coords.tolist(),
        "box": None if kind == "lj" or box is None else box.tolist(),
        "kind": kind,
        "coordsys": spec["coordsys"],
        "neighbors": spec["neighbors"],
        "element": spec["element"],
        "gtol": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
    }


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{int(start)}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.runners.geometric_min import _minimize_job

    record = _minimize_job(_job(spec, start))
    record["seed"] = SEED
    path = _cache_path(spec["template_id"], SEED, str(int(start)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], SEED, str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _GeometricSearch(TrackedSearch):
    def setup(self, seed, start):
        if not _ready():
            raise NotImplementedError("geometric is not installed in this environment")
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


_GeometricSearch.track_force_calls.unit = "force calls"
_GeometricSearch.track_success.unit = "fraction"
_GeometricSearch.track_failure_cap.unit = "count"
_GeometricSearch.track_failure_wrong_stationary.unit = "count"
_GeometricSearch.track_failure_divergence.unit = "count"
_GeometricSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_GeometricSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = limit
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
