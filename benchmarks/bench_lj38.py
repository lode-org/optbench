"""One eonclient subprocess per Lennard-Jones 38 start.

The default grid is one seed and the first archive geometry. All 1000
starts are selected when ``OPTBENCH_ASV_FULL=1`` at import time. The
client runs only when ``OPTBENCH_ASV_EON=1``. Otherwise ``measure``
raises ``NotImplementedError`` and asv-runner skips the parameter.

The seed is on the grid so a later seeded displacement uses the same
parameter names. This client invocation does not draw from it.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT
    / "inputs"
    / "sha256"
    / "86bf774955b906c486d37282c48e42628e8fe4af3579222f6e07fcf5913157b1"
    / "lj38-clusters.tgz"
)
TEMPLATES = (
    ("Lj38EonCg", "eon-client/cg", 1800),
    ("Lj38EonFire", "eon-client/fire", 1800),
    ("Lj38EonLbfgsM20", "eon-client/lbfgs-m20", 600),
    ("Lj38EonQm", "eon-client/qm", 3600),
    ("Lj38EonSd", "eon-client/sd", 3600),
    ("Lj38EonSdTwopoint", "eon-client/sd-twopoint", 1800),
)


def start_names() -> list[str]:
    with tarfile.open(ARCHIVE) as packed:
        return sorted(name for name in packed.getnames() if name.endswith(".con"))


def selected_starts() -> list[str]:
    names = start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def client_path() -> str:
    env = os.environ.get("EONCLIENT")
    if env:
        return env
    found = shutil.which("eonclient")
    if not found:
        raise RuntimeError("put eonclient on PATH or set EONCLIENT")
    return found


def _ini(template_name: str) -> str:
    from optbench.campaign import format_ini, load_template

    return format_ini(load_template(ROOT, template_name)["ini"])


def _cap(template_name: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_name)
    problem_name = template.get("problem", "minimization/lj38")
    return _check_stop(resolve_problem(ROOT, problem_name), template)


_STARTS = selected_starts()


def _bind(template_name: str, limit: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], _STARTS]
        timeout = limit

        def measure(self, seed, start):
            if os.environ.get("OPTBENCH_ASV_EON") != "1":
                raise NotImplementedError("eonclient benchmarks run when OPTBENCH_ASV_EON=1")
            self.work = Path(tempfile.mkdtemp(prefix="optbench-lj38-"))
            ini = _ini(self.template_name)
            from optbench.campaign import run_client

            with tarfile.open(ARCHIVE) as packed:
                packed.extract(start, self.work, filter="data")
            calls = run_client(
                client_path(),
                ini,
                str(self.work / start),
                str(self.work / "run"),
            )
            if calls is None:
                raise RuntimeError("eonclient returned no force-call count")
            success = calls < self.cap
            return {
                "force_calls": float(calls),
                "success": success,
                "failure": None if success else "cap",
            }

        def teardown(self, seed, start):
            work = getattr(self, "work", None)
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)

    _Benchmark.template_name = template_name
    _Benchmark.cap = _cap(template_name)
    return _Benchmark


for _class_name, _template_name, _timeout in TEMPLATES:
    _cls = _bind(_template_name, _timeout)
    _cls.__name__ = _class_name
    _cls.__qualname__ = _class_name
    globals()[_class_name] = _cls
