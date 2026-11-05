"""Start asv with a record of the BLAS pools its benchmark server gets.

usage: python -m optbench.asv_launch <asv arguments>

asv copies its own environment into the benchmark server, and the server
imports numpy before any benchmark module (``optbench.blas_threads``).
NumPy's pool is therefore sized before the suite loads, and no asv
result file records the pools.

A probe imports ``asv_runner.benchmarks`` first, as the server does,
and then numpy and scipy.linalg, under this process's environment. It
runs the interpreter of ``-E existing:<python>`` when that is given, and
this one otherwise. ``OPTBENCH_ASV_RECORD=<file>`` writes the thread
variables, the pools, and the asv arguments as JSON. That file is the
``--run-env`` of ``results/ase-sella-full/summarize.py``. asv then runs
in this process with the same arguments.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import blas_threads


def interpreter(args: list[str]) -> str:
    """The Python of an ``existing`` environment, or this one."""
    for index, arg in enumerate(args):
        value = None
        if arg in {"-E", "--environment"} and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith("--environment="):
            value = arg.split("=", 1)[1]
        elif arg.startswith("-E") and len(arg) > 2:
            value = arg[2:]
        if value and value.startswith("existing:"):
            return value.split(":", 1)[1]
    return sys.executable


def probe(python: str) -> list[dict]:
    """BLAS pools of a fresh interpreter that imports in the server's order."""
    source = Path(blas_threads.__file__).read_text(encoding="utf-8")
    done = subprocess.run(
        [python, "-c", source, "--asv-order"],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    pools = probe(interpreter(args))
    record = {"variables": blas_threads.variables(), "blas": pools, "asv": args}
    print("optbench.asv_launch " + json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
    target = os.environ.get("OPTBENCH_ASV_RECORD")
    if target:
        Path(target).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from asv.main import main as asv_main

    sys.argv = ["asv", *args]
    asv_main()


if __name__ == "__main__":
    main()
