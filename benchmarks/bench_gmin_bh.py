"""GMIN basin-hopping, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` selects the
archived geometry. The GMIN random seed is the template base plus that
index.

The default grid is seed ``0`` and archived start ``0``.
``OPTBENCH_ASV_FULL=1`` selects every start the problem states in
``sample.starts``. It is read at import.

``time_search`` runs the start and stores its record under
``OPTBENCH_ASV_CACHE``. asv runs each track in its own process, so a
track reads that record, or runs the start when there is none. The
record path includes a hash of the template and problem, so a record
from another template is not read. A record also names the
``wales_sha256`` and ``wales_svnrev`` of the binary that ran it, and a
record from another build is run again.

``setup`` raises ``NotImplementedError`` when ``OPTBENCH_GMIN`` is unset
and ``GMIN`` is not on ``PATH``. asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
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


def _binary_ready() -> bool:
    if os.environ.get("OPTBENCH_GMIN"):
        return Path(os.environ["OPTBENCH_GMIN"]).is_file()
    return shutil.which("GMIN") is not None


def _cache_dir() -> Path:
    return _scratch_cache("optbench-asv-gmin-bh")


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Gmin" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "gmin-bh":
            continue
        problem = resolve_problem(ROOT, str(template["problem"]))
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": str(template["problem"]),
                "cap": int(template["failure_cap"]),
                "n": int(problem["sample"]["starts"]),
            }
        )
    return found


def _key(spec: dict) -> str:
    from optbench.campaign import load_template, resolve_problem

    parts = {
        "template": load_template(ROOT, spec["template_id"]),
        "problem": resolve_problem(ROOT, spec["problem_id"]),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _cache_path(spec: dict, start: str) -> Path:
    return _cache_dir() / spec["template_id"].replace("/", "-") / _key(spec) / f"{SEED}-{int(start)}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not {SEED}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.gmin_go import plan, run_one

    path = _cache_path(spec, start)
    path.unlink(missing_ok=True)
    work = _cache_dir() / "work" / spec["template_id"].replace("/", "-")
    jobs, passed = plan(
        ROOT,
        resolve_problem(ROOT, spec["problem_id"]),
        load_template(ROOT, spec["template_id"]),
        scratch=work,
    )
    index = int(start)
    if index < 0 or index >= len(jobs):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    record = run_one(jobs[index])
    record["wales_sha256"] = passed["wales_sha256"]
    record["wales_svnrev"] = passed["wales_svnrev"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    from optbench.runners.gmin_go import binary_build, gmin_binary
    from optbench.wales import same_build

    path = _cache_path(spec, start)
    if str(seed) == SEED and path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if same_build(record, binary_build(gmin_binary())):
            return record
    return _compute(spec, seed, start)


class _GminSearch(_TrackedSearch):
    def setup(self, seed, start):
        if not _binary_ready():
            raise NotImplementedError("GMIN binary is not available")
        if str(seed) != SEED:
            raise ValueError(f"seed {seed} is not {SEED}")
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


_GminSearch.track_force_calls.unit = "force calls"
_GminSearch.track_success.unit = "fraction"
_GminSearch.track_failure_cap.unit = "count"
_GminSearch.track_failure_wrong_stationary.unit = "count"
_GminSearch.track_failure_divergence.unit = "count"
_GminSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    # The class body treats any assigned name as local, so the outer spec
    # is copied into names that the class only reads.
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_GminSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        # run_one stops GMIN at the job timeout that plan derives from the cap.
        timeout = math.inf
        cap = limit
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
