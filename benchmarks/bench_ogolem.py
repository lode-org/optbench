"""OGOLEM genetic algorithm, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` is the
archived geometry. The OGOLEM RNG seed is the start index plus one, not
this parameter.

The default grid is that seed and start ``0``. ``OPTBENCH_ASV_FULL=1``
selects every archived start. It is read at import.

``time_search`` is one JVM. ``setup`` raises ``NotImplementedError``
when the jars are absent, and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-ogolem"))


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Ogolem" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "ogolem":
            continue
        cap = int(template["failure_cap"])
        section = template.get("ogolem") or {}
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": str(template["problem"]),
                "cap": cap,
                "pool": int(section.get("pool_size", 100)),
                "iterations": int(section.get("glob_iterations", 19900)),
                "cell": float(section.get("cell_bohr", 60)),
                "threads": int(section.get("threads", 1)),
                "code_version": str(template.get("code_version") or ""),
                "commit": str(template.get("commit") or ""),
                "tuning": str(template.get("tuning") or ""),
                "code": str(template.get("code") or "ogolem"),
                "n": 1,
            }
        )
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        from optbench.campaign import resolve_problem
        from optbench.runners.ogolem_go import load_starts

        counts: dict[str, int] = {}
        for spec in found:
            problem_id = spec["problem_id"]
            if problem_id not in counts:
                problem = resolve_problem(ROOT, problem_id)
                template = next(row for row in list_templates(ROOT) if row["id"] == spec["template_id"])
                counts[problem_id] = len(load_starts(ROOT, problem, template, _cache_dir() / "geometries"))
            spec["n"] = counts[problem_id]
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _job(spec: dict, start: str) -> dict:
    from optbench.campaign import list_templates, resolve_problem
    from optbench.runners.ogolem_go import _threshold, load_starts, ogolem_home

    index = int(start)
    problem = resolve_problem(ROOT, spec["problem_id"])
    template = next(row for row in list_templates(ROOT) if row["id"] == spec["template_id"])
    loaded = load_starts(ROOT, problem, template, _cache_dir() / "geometries")
    if index < 0 or index >= len(loaded):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords = loaded[index]
    return {
        "index": index,
        "start": name,
        "coords": coords.reshape(-1, 3).tolist(),
        "cap": spec["cap"],
        "threshold": _threshold(problem, template),
        "pool": spec["pool"],
        "iterations": spec["iterations"],
        "cell": spec["cell"],
        "threads": spec["threads"],
        "seed": index + 1,
        "home": str(ogolem_home()),
        "work": str(_cache_dir() / "work" / spec["class_name"] / f"{index:04d}-{name}"),
        "wall": 86400,
        "trace": False,
        "code": spec["code"],
        "code_version": spec["code_version"],
        "commit": spec["commit"],
        "tuning": spec["tuning"],
        "template": spec["template_id"],
        "problem": spec["problem_id"],
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.runners.ogolem_go import execute_start

    record = execute_start(_job(spec, start))
    record.pop("traces", None)
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


class _OgolemSearch(TrackedSearch):
    def setup(self, seed, start):
        from optbench.runners.ogolem_go import jars_ready

        if not jars_ready():
            raise NotImplementedError("ogolem jars are not installed")
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


_OgolemSearch.track_force_calls.unit = "force calls"
_OgolemSearch.track_success.unit = "fraction"
_OgolemSearch.track_failure_cap.unit = "count"
_OgolemSearch.track_failure_wrong_stationary.unit = "count"
_OgolemSearch.track_failure_divergence.unit = "count"
_OgolemSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])

    class _Benchmark(_OgolemSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 86400
        cap = bound["cap"]
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
