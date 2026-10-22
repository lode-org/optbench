"""Lennard-Jones 38 with Gaussian force noise, one class per template.

The default grid is seed ``0`` and start ``0``. ``OPTBENCH_ASV_FULL=1``
selects every archived start. It is read at import. ``time_search`` is
the minimization. Track methods read the record that call wrote.
"""

from __future__ import annotations

import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-noise"))


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Noise" + "".join(part.capitalize() for part in slug.split("-"))


def _ready(family: str) -> bool:
    try:
        if family == "rgmin":
            import rgmin  # noqa: F401
        elif family == "ase":
            import ase  # noqa: F401
        elif family == "sella":
            import sella  # noqa: F401
        elif family == "geometric":
            import geometric  # noqa: F401
        elif family == "gmin":
            from optbench.gmin_lbfgs import available

            return available()
        elif family == "vcsqnm":
            import sqnm.sqnm  # noqa: F401
        elif family == "pele":
            import pele  # noqa: F401
        else:
            return False
    except ImportError:
        return False
    return True


def _specs() -> list[dict]:
    from optbench.noise_catalog import specs

    found = []
    for template in specs():
        solver = template["solver"]
        found.append(
            {
                "class_name": _class_name(template["id"]),
                "template_id": template["id"],
                "problem_id": template["problem"],
                "family": solver["family"],
                "cap": 10000,
                "n": 1000 if os.environ.get("OPTBENCH_ASV_FULL") == "1" else 1,
                "template": template,
            }
        )
    return found


def _job(spec: dict, start: str) -> dict:
    from optbench.campaign import resolve_problem
    from optbench.noise import KAPPA, window_energy_tolerance
    from optbench.runners.rgmin_min import _load_geometries

    index = int(start)
    problem = resolve_problem(ROOT, spec["problem_id"])
    geometries = _load_geometries(ROOT, problem, None, Path("/tmp/optbench-asv-geo"), Path("/tmp/optbench-asv-geo/unused.dat"))
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, _box = geometries[index]
    template = spec["template"]
    solver = template["solver"]
    sigma = float(template["noise"]["sigma"])
    convergence = problem["convergence"]
    options = {key: value for key, value in solver.items() if key not in {"family", "method", "name"}}
    return {
        "index": index,
        "start": name,
        "coords": [float(value) for value in coords],
        "family": solver["family"],
        "method": str(solver.get("method") or ""),
        "optimizer": str(solver.get("name") or ""),
        "options": options,
        "sigma": sigma,
        "seed": int(template["noise"]["seed"]),
        "clean_tol": float(convergence["force_norm"]),
        "energy_tol": window_energy_tolerance(float(convergence["energy_tol"]), sigma, KAPPA),
        "cap": int(convergence["max_force_calls"]),
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.runners.noise_min import _minimize_job

    record = _minimize_job(_job(spec, start))
    record["seed"] = SEED
    path = _cache_dir() / spec["template_id"].replace("/", "-") / f"{SEED}-{int(start)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(__import__("json").dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


class _NoiseSearch(TrackedSearch):
    def setup(self, seed, start):
        if not _ready(self.spec["family"]):
            raise NotImplementedError(f"{self.spec['family']} is not installed in this environment")
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.spec, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            path = _cache_dir() / self.spec["template_id"].replace("/", "-") / f"{SEED}-{int(start)}.json"
            if path.is_file():
                self.record = __import__("json").loads(path.read_text(encoding="utf-8"))
            else:
                self.record = _compute(self.spec, seed, start)
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


_NoiseSearch.track_force_calls.unit = "force calls"
_NoiseSearch.track_success.unit = "fraction"
_NoiseSearch.track_failure_cap.unit = "count"
_NoiseSearch.track_failure_wrong_stationary.unit = "count"
_NoiseSearch.track_failure_divergence.unit = "count"
_NoiseSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])

    class _Benchmark(_NoiseSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = bound["cap"]
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
