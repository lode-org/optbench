"""asv classes for the campaign engines in ``optbench.engines``.

One class is one (template, problem) cell. A template that names a
problem is that problem's cell. A template without a problem runs on
every problem with a ``[band]`` table, when its engine builds band jobs.

``plan`` in the engine module is the function ``run`` calls, so a class
runs the campaign's own job for its start. The job runs alone in a
spawned process, as each campaign job did.

The parameters are the seed and the start. Seed ``0`` is the only
campaign seed. The start is the campaign job name, which is the
``start`` field of the committed ``.starts.jsonl``. The default grid is
the first start. ``OPTBENCH_ASV_FULL=1`` selects every start of the
problem. It is read at import.

asv runs each track in its own process. ``time_search`` runs the search
and stores its record under ``OPTBENCH_ASV_CACHE``. A track reads that
record, or runs the search when there is none, so every track is the
value of one search. The cache key includes the template, problem, and
registry row, so a record from another pin is not read. When the engine
module has ``build_stamp``, each record names that build, and a record
from another build is run again.

The campaign has no wall-clock stop. The problem's force-call cap ends
every search, so the class timeout is unbounded. asv's forkserver
launcher accepts that; its spawn launcher cannot wait without a bound.

``saddle-search/pt-island`` is not a cell. Its sample is one catalog
walk, and each search starts from the catalog the earlier ones built.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from pathlib import Path

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]

def _scratch_cache(name: str) -> Path:
    chosen = os.environ.get("OPTBENCH_ASV_CACHE")
    if chosen:
        return Path(chosen)
    root = os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP")
    if not root:
        import tempfile

        root = tempfile.gettempdir()
    return Path(root) / name


SEED = "0"
WALK_PROBLEM = "saddle-search/pt-island"
# plan builds band jobs from cached_pairs in these engines only.
BAND_ENGINES = ("compass", "partn", "tsase")
_NAMES: dict[str, list[str]] = {}


def full_grid() -> bool:
    return os.environ.get("OPTBENCH_ASV_FULL") == "1"


def cache_dir() -> Path:
    return _scratch_cache("optbench-asv-engines")


def _slug(text: str) -> str:
    return text.replace("/", "-")


def camel(text: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[/\-_.]", text) if part)


def _band_problems(root: Path) -> list[str]:
    found = []
    for path in sorted((root / "problems").rglob("*.toml")):
        problem = tomllib.loads(path.read_text(encoding="utf-8"))
        if problem.get("band"):
            found.append(str(problem["id"]))
    return found


def cells(root: Path = ROOT, engine: str | None = None) -> list[dict]:
    """Every (template, problem) cell of the engines that plan per-start jobs."""
    from optbench.campaign import list_templates
    from optbench.engines import ENGINES

    root = Path(root)
    band = None
    found = []
    for template in list_templates(root):
        name = str(template.get("engine", ""))
        module = ENGINES.get(name)
        if module is None or not hasattr(module, "plan"):
            continue
        if engine is not None and name != engine:
            continue
        stated = template.get("problem")
        if stated:
            problems = [str(stated)]
        elif name in BAND_ENGINES:
            band = _band_problems(root) if band is None else band
            problems = band
        else:
            problems = []
        for problem_id in problems:
            if problem_id == WALK_PROBLEM:
                continue
            class_name = camel(str(template["id"])) + ("" if stated else camel(problem_id))
            found.append(
                {
                    "class_name": class_name,
                    "engine": name,
                    "code": str(template["code"]),
                    "template_id": str(template["id"]),
                    "problem_id": problem_id,
                }
            )
    return found


def start_names(root: Path, problem_id: str) -> list[str]:
    """The campaign job names for ``problem_id``, in campaign order."""
    if problem_id not in _NAMES:
        from optbench.campaign import resolve_problem
        from optbench.engines.common import band_settings, cached_pairs
        from optbench.engines.geometries import load_starts

        problem = resolve_problem(root, problem_id)
        if problem.get("band"):
            names = list(cached_pairs(root, problem, band_settings(problem)))
        else:
            work = cache_dir() / "geometries" / _slug(problem_id)
            names = [str(row["name"]) for row in load_starts(root, problem, work)]
        _NAMES[problem_id] = names
    return _NAMES[problem_id]


def problem_cap(root: Path, problem_id: str) -> int:
    from optbench.campaign import resolve_problem

    stated = (resolve_problem(root, problem_id).get("convergence") or {}).get("max_force_calls")
    if stated is None:
        raise ValueError(f"{problem_id} has no force-call cap")
    return int(stated)


def _key(cell: dict) -> str:
    from optbench.campaign import load_template, resolve_problem
    from optbench.engines.common import registry_pin

    template = load_template(ROOT, cell["template_id"])
    parts = {
        "template": template,
        "problem": resolve_problem(ROOT, cell["problem_id"]),
        "pin": registry_pin(ROOT, str(template["code"])),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def record_path(cell: dict, seed: str, start: str) -> Path:
    return (
        cache_dir()
        / "records"
        / _slug(cell["template_id"])
        / _slug(cell["problem_id"])
        / _key(cell)
        / f"{seed}-{start}.json"
    )


def compute(cell: dict, seed: str, start: str) -> dict:
    """The campaign job for ``start``, run once and stored."""
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the campaign seed {SEED}")
    from optbench.engines import ENGINES
    from optbench.engines.common import run_start

    path = record_path(cell, SEED, str(start))
    path.unlink(missing_ok=True)
    planned = ENGINES[cell["engine"]].plan(ROOT, cell["problem_id"], cell["template_id"], cache_dir() / "work")
    record = run_start(planned["jobs"], planned["execute"], str(start))
    record.update(planned.get("wales") or {})
    record.setdefault("problem", cell["problem_id"])
    record.setdefault("template", cell["template_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)
    return record


def load(cell: dict, seed: str, start: str) -> dict:
    path = record_path(cell, SEED, str(start))
    if str(seed) == SEED and path.is_file():
        from optbench.engines import ENGINES
        from optbench.wales import same_build

        record = json.loads(path.read_text(encoding="utf-8"))
        build_stamp = getattr(ENGINES[cell["engine"]], "build_stamp", None)
        if build_stamp is None or same_build(record, build_stamp()):
            return record
    return compute(cell, seed, start)


class EngineSearch(_TrackedSearch):
    """Tracks for one campaign job. ``cell`` is set by ``bind``."""

    cell: dict = {}

    def setup(self, seed, start):
        from optbench.engines import ENGINES

        reason = ENGINES[self.cell["engine"]].available()
        if reason:
            raise NotImplementedError(reason)
        self.record = None

    def time_search(self, seed, start):
        self.record = compute(self.cell, seed, start)

    def _stored(self, seed, start) -> dict:
        if getattr(self, "record", None) is None:
            self.record = load(self.cell, seed, start)
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


EngineSearch.track_force_calls.unit = "force calls"
EngineSearch.track_success.unit = "fraction"
EngineSearch.track_failure_cap.unit = "count"
EngineSearch.track_failure_wrong_stationary.unit = "count"
EngineSearch.track_failure_divergence.unit = "count"
EngineSearch.track_failure_crash.unit = "count"


def bind(cell: dict, root: Path = ROOT):
    names = start_names(root, cell["problem_id"])
    bound = dict(cell)

    class _Benchmark(EngineSearch):
        param_names = ["seed", "start"]
        params = [[SEED], names if full_grid() else names[:1]]
        timeout = math.inf
        cell = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]
        cap = problem_cap(root, bound["problem_id"])

    _Benchmark.__name__ = cell["class_name"]
    _Benchmark.__qualname__ = cell["class_name"]
    return _Benchmark


def install_benchmarks(engine: str, namespace: dict) -> None:
    """Define one asv class per cell of ``engine`` in ``namespace``."""
    for cell in cells(ROOT, engine):
        namespace[cell["class_name"]] = bind(cell)
