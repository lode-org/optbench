"""AIRSS global optimisation, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` selects the
archived start order. The default grid is start ``0``, which keeps
``asv check`` on one parameter. ``OPTBENCH_ASV_FULL=1`` selects every
archived start. It is read at import.

``time_search`` is one random-structure search. Track methods read the
record that call wrote. ``setup`` raises ``NotImplementedError`` when
``buildcell`` is absent, and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-airss"))


def problem_cap() -> int:
    data = tomllib.loads((ROOT / "problems" / "globalopt" / "lj38.toml").read_text(encoding="utf-8"))
    return int(data["convergence"]["max_force_calls"])


def selected_starts() -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") != "1":
        return ["0"]
    data = tomllib.loads((ROOT / "problems" / "globalopt" / "lj38.toml").read_text(encoding="utf-8"))
    count = int(data["sample"]["starts"])
    return [str(index) for index in range(count)]


def _cache_path(seed: str, start: str) -> Path:
    return _cache_dir() / "lj38" / f"{seed}-{int(start)}.json"


def _compute(seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.airss_go import _job, _load_start_names, find_buildcell, reference_energy, search_start

    template = load_template(ROOT, "airss/lj38")
    problem = resolve_problem(ROOT, str(template["problem"]))
    names = _load_start_names(ROOT, problem, None, _cache_dir() / "geometries", _cache_dir() / "lj38.dat")
    index = int(start)
    if index < 0 or index >= len(names):
        raise IndexError(f"airss/lj38 start {start} is outside the archive")
    tolerance = float(problem["convergence"]["energy_above_minimum"])
    job = _job(
        template,
        index,
        names[index],
        find_buildcell(),
        reference_energy(),
        tolerance,
        problem_cap(),
    )
    record = search_start(job)
    record["seed"] = SEED
    record.pop("index", None)
    path = _cache_path(SEED, str(index))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(seed: str, start: str) -> dict:
    path = _cache_path(SEED, str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(seed, start)


class AirssLj38(TrackedSearch):
    """LJ38 random structure search. ``time_search`` is the timed search."""

    param_names = ["seed", "start"]
    params = [[SEED], selected_starts()]
    timeout = 3600
    template_id = "airss/lj38"
    problem_id = "globalopt/lj38"
    cap = problem_cap()

    def setup(self, seed, start):
        from optbench.runners.airss_go import find_buildcell

        try:
            find_buildcell()
        except FileNotFoundError as exc:
            raise NotImplementedError(str(exc)) from exc
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _load(seed, start)
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


AirssLj38.track_force_calls.unit = "force calls"
AirssLj38.track_success.unit = "fraction"
AirssLj38.track_failure_cap.unit = "count"
AirssLj38.track_failure_wrong_stationary.unit = "count"
AirssLj38.track_failure_divergence.unit = "count"
AirssLj38.track_failure_crash.unit = "count"
