"""GMIN global optimisation from the archived data files, one class per row.

The rows are ``optbench.runners.gmin_paper.ROWS``, the rows of the
``gmin-paper-*`` campaign files. Each runs the serial GMIN with its own
potential and the ``data`` file of
``archive/entries/globalopt/<problem>/<entry>``. That file's ``STEPS`` and
``TARGET`` are the budget and the stop.

``param_names`` are the seed and the archived coordinates index. Seed
``0`` is the archived file, and GMIN reads its random seed from ``data``.
The default grid is seed ``0`` and the first index of the row.
``OPTBENCH_ASV_FULL=1`` selects every index the row runs. It is read at
import.

The force count is the function-call field of the ``Target hit after``
line, the ``ncalls`` of the campaign records. The lj98 and lj150 files
report the same field as energy calls. A start with no such line is a
``cap`` failure when GMIN exits 0 with no error. It is a ``crash`` when
GMIN exits nonzero, or when it stops on a ``STOP`` line because it cannot
read the data file. Its force count is NaN, because GMIN prints no count
without that line.

``time_search`` runs the start and stores its record under
``OPTBENCH_ASV_CACHE``. asv runs each track in its own process, so a
track reads that record, or runs the start when there is none. The
record path includes a hash of the row, its data file, and the problem.
A record also names the ``wales_sha256`` and ``wales_svnrev`` of the
binary that ran it, and a record from another build is run again.

``setup`` raises ``NotImplementedError`` when ``OPTBENCH_GMIN_SERIAL``
does not name a file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
    from optbench.runners.gmin_paper import serial_binary

    try:
        serial_binary()
    except RuntimeError:
        return False
    return True


def _cache_dir() -> Path:
    return _scratch_cache("optbench-asv-gmin-paper")


def selected_indices(row: dict) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(int(row["start"]), int(row["stop"]) + 1)]
    return [str(row["start"])]


def _class_name(campaign: str) -> str:
    return "".join(part.capitalize() for part in campaign.split("-"))


def _specs() -> list[dict]:
    from optbench.runners.gmin_paper import ROWS, campaign_name

    return [
        {
            "class_name": _class_name(campaign_name(row)),
            "campaign": campaign_name(row),
            "problem_id": f"globalopt/{row['problem']}",
            "row": dict(row),
        }
        for row in ROWS
    ]


def _key(spec: dict) -> str:
    from optbench.runners.gmin_paper import _problem_dict

    row = spec["row"]
    data = ROOT / "archive" / "entries" / "globalopt" / row["problem"] / row["entry"] / "data"
    parts = {
        "row": row,
        "data": data.read_text(encoding="utf-8"),
        "problem": _problem_dict(ROOT, row["problem"]),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _cache_path(spec: dict, start: str) -> Path:
    return _cache_dir() / spec["campaign"] / _key(spec) / f"{SEED}-{int(start)}.json"


def _record(found: dict) -> dict:
    if int(found["success"]) == 1:
        return {"force_calls": float(found["ncalls"]), "success": True, "failure": None}
    failure = "cap" if int(found["returncode"]) == 0 and not found.get("error") else "crash"
    return {"force_calls": math.nan, "success": False, "failure": failure}


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not {SEED}")
    from optbench.runners.gmin_go import binary_build
    from optbench.runners.gmin_paper import prepare_jobs, run_one, serial_binary

    wales = binary_build(serial_binary())
    path = _cache_path(spec, start)
    path.unlink(missing_ok=True)
    work = _cache_dir() / "work" / spec["campaign"]
    row = spec["row"]
    index = int(start)
    if not int(row["start"]) <= index <= int(row["stop"]):
        raise IndexError(f"{spec['campaign']} start {start} is outside the archive")
    # Reading every tarball member of a 10000-start row takes minutes.
    (job,) = prepare_jobs(ROOT, dict(row, start=index, stop=index), work)
    found = run_one(dict(job, resume=False, wales=wales))
    record = dict(_record(found), ncalls=found["ncalls"], quenches=found["quenches"], **wales)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    from optbench.runners.gmin_go import binary_build
    from optbench.runners.gmin_paper import serial_binary
    from optbench.wales import same_build

    path = _cache_path(spec, start)
    if str(seed) == SEED and path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if same_build(record, binary_build(serial_binary())):
            return record
    return _compute(spec, seed, start)


class _GminPaperSearch(_TrackedSearch):
    def setup(self, seed, start):
        if not _binary_ready():
            raise NotImplementedError("serial GMIN binary is not available")
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


_GminPaperSearch.track_force_calls.unit = "force calls"
_GminPaperSearch.track_success.unit = "fraction"
_GminPaperSearch.track_failure_cap.unit = "count"
_GminPaperSearch.track_failure_wrong_stationary.unit = "count"
_GminPaperSearch.track_failure_divergence.unit = "count"
_GminPaperSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    # The class body treats any assigned name as local, so the outer spec
    # is copied into names that the class only reads.
    bound = spec
    starts = selected_indices(spec["row"])

    class _Benchmark(_GminPaperSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        # GMIN stops at the TARGET or the STEPS of the archived data file.
        timeout = math.inf
        spec = bound
        campaign = bound["campaign"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
