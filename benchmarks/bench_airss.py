"""AIRSS global optimisation, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` selects the
archived start order. The default grid is start ``0``, which keeps
``asv check`` on one parameter. ``OPTBENCH_ASV_FULL=1`` selects every
start of the template: the template's ``airss.starts`` when it states
them, otherwise the problem's ``sample.starts``. It is read at import.

``time_search`` is one random-structure search. Its job is the one that
``optbench.runners.airss_go.plan`` gives ``optbench run`` for that start.
Track methods read the record that call wrote, or run the search when
there is none. ``setup`` raises ``NotImplementedError`` when
``buildcell`` is absent, and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-airss"))


def _templates() -> list[dict]:
    from optbench.campaign import list_templates

    return [template for template in list_templates(ROOT) if template.get("engine") == "airss"]


def start_count(template: dict) -> int:
    """The number of starts ``optbench run`` gives ``template``."""
    from optbench.campaign import resolve_problem

    stated = (template.get("airss") or {}).get("starts")
    if stated:
        return int(stated)
    return int(resolve_problem(ROOT, str(template["problem"]))["sample"]["starts"])


def selected_starts(template_id: str) -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") != "1":
        return ["0"]
    from optbench.campaign import load_template

    return [str(index) for index in range(start_count(load_template(ROOT, template_id)))]


def _slug(template_id: str) -> str:
    return template_id.replace("/", "-")


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    return _cache_dir() / _slug(template_id) / f"{seed}-{int(start)}.json"


def _compute(template_id: str, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.airss_go import plan, search_start

    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, str(template["problem"]))
    jobs, _reference, _tolerance = plan(
        ROOT,
        problem,
        template,
        geometries=None,
        output=_cache_dir() / f"{_slug(template_id)}.dat",
        work=_cache_dir() / "geometries",
    )
    index = int(start)
    if index < 0 or index >= len(jobs):
        raise IndexError(f"{template_id} start {start} is outside the {len(jobs)} starts")
    record = search_start(jobs[index])
    record["seed"] = SEED
    record.pop("index", None)
    path = _cache_path(template_id, SEED, str(index))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(template_id: str, seed: str, start: str) -> dict:
    path = _cache_path(template_id, SEED, str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(template_id, seed, start)


class _AirssSearch(TrackedSearch):
    """One random structure search. ``time_search`` is the timed search."""

    param_names = ["seed", "start"]
    timeout = 3600
    cap = None

    def setup(self, seed, start):
        from optbench.runners.airss_go import find_buildcell

        try:
            find_buildcell()
        except FileNotFoundError as exc:
            raise NotImplementedError(str(exc)) from exc
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.template_id, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _load(self.template_id, seed, start)
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


_AirssSearch.track_force_calls.unit = "force calls"
_AirssSearch.track_success.unit = "fraction"
_AirssSearch.track_failure_cap.unit = "count"
_AirssSearch.track_failure_wrong_stationary.unit = "count"
_AirssSearch.track_failure_divergence.unit = "count"
_AirssSearch.track_failure_crash.unit = "count"


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Airss" + "".join(part.capitalize() for part in slug.replace("_", "-").split("-"))


def _benchmark(template: dict):
    bound = template

    class _Benchmark(_AirssSearch):
        params = [[SEED], selected_starts(str(bound["id"]))]
        template_id = str(bound["id"])
        problem_id = str(bound["problem"])

    name = _class_name(str(template["id"]))
    _Benchmark.__name__ = name
    _Benchmark.__qualname__ = name
    return _Benchmark


for _template in _templates():
    globals()[_class_name(str(_template["id"]))] = _benchmark(_template)
