"""One skipped benchmark for every problem and registry code without a class.

A real code copies ``benchmarks/quadratic.py`` and implements
``measure``. These classes are the placeholder so the grid exists. There
is one class per registry code, and its problems are the ones no class
covers yet. They have ``track_force_calls`` and ``time_search`` only.
The success and failure tracks are on the quadratic classes, which are
the runs that produce numbers. Multiplying those tracks across every
unimplemented pair would spawn a process per track.

``setup`` raises ``NotImplementedError``. asv-runner records that as a
skip (NaN) and does not save an exit code. See ``docs/asv-gaps.md``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Engines whose cells are classes in benchmarks/bench_<engine>.py.
ENGINE_CLASSES = ("compass", "tsase", "partn", "lammps", "dlfind", "pysisyphus", "optim")
# Registry code ids whose template classes are in benchmarks/bench_<module>.py.
TEMPLATE_MODULES = (("airss", "airss"),)


def problem_ids() -> list[str]:
    found = []
    for path in sorted((ROOT / "problems").rglob("*.toml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("id = "):
                found.append(line.split("=", 1)[1].strip().strip('"'))
                break
        else:
            raise ValueError(f"{path} has no id")
    return found


def code_ids() -> list[str]:
    data = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    return [row["id"] for row in data["code"]]


def covered() -> set[tuple[str, str]]:
    """(problem, code) cells that a class in this directory measures."""
    from optbench.asv_engines import cells

    found = set()
    for engine in ENGINE_CLASSES:
        found.update((cell["problem_id"], cell["code"]) for cell in cells(ROOT, engine))
    for code, module in TEMPLATE_MODULES:
        found.update((problem, code) for problem in template_problems(module))
    return found


def template_problems(module: str) -> set[str]:
    """Problems of the template classes in ``benchmarks/bench_<module>.py``."""
    import importlib

    from optbench.campaign import load_template

    found = set()
    for value in vars(importlib.import_module(f"benchmarks.bench_{module}")).values():
        template = getattr(value, "template_id", None) or getattr(value, "template_name", None)
        if isinstance(value, type) and template:
            found.add(str(load_template(ROOT, str(template))["problem"]))
    return found


def pending_cells() -> dict[str, list[str]]:
    """Problems without a class, per registry code, in registry order."""
    done = covered()
    problems = problem_ids()
    return {
        code: [problem for problem in problems if (problem, code) not in done]
        for code in code_ids()
    }


class _Pending:
    param_names = ["problem", "code", "seed", "start"]
    repeat = 1
    rounds = 1
    number = 1
    min_run_count = 1
    sample_time = 0
    warmup_time = 0
    timeout = 60

    def setup(self, problem, code, seed, start):
        raise NotImplementedError(
            f"no benchmark for {problem} with {code}; copy benchmarks/quadratic.py"
        )

    def track_force_calls(self, problem, code, seed, start) -> float:
        raise NotImplementedError(problem)

    def time_search(self, problem, code, seed, start):
        raise NotImplementedError(problem)


_Pending.track_force_calls.unit = "force calls"


def _class_name(code: str) -> str:
    return "Pending" + "".join(part[:1].upper() + part[1:] for part in re.split(r"[^0-9A-Za-z]+", code))


def _bind(code: str, problems: list[str]):
    class _Benchmark(_Pending):
        params = [problems, [code], ["0"], ["0"]]

    _Benchmark.__name__ = _class_name(code)
    _Benchmark.__qualname__ = _Benchmark.__name__
    return _Benchmark


for _code, _problems in pending_cells().items():
    if _problems:
        globals()[_class_name(_code)] = _bind(_code, _problems)
