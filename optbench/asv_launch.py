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

``machine`` with ``--yes`` or with any field writes the whole entry
itself. asv 0.6.6 stores only the fields that differ from the values it
detects, so ``asv machine --yes --machine NAME`` leaves out ``arch``,
``cpu``, ``num_cpu``, ``os``, and ``ram``. Each field here is the given
value, else the stored one, else the one asv detects.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import blas_threads

MACHINE_FIELDS = ("machine", "os", "arch", "cpu", "num_cpu", "ram")


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


def machine_entry(stored: dict | None, given: dict[str, str]) -> dict[str, str]:
    """Every field asv records: the given value, else stored, else detected."""
    from asv.machine import Machine

    entry = Machine.get_defaults()
    entry.update({key: value for key, value in (stored or {}).items() if key in MACHINE_FIELDS})
    entry.update(given)
    return entry


def machine_file() -> Path:
    from asv.machine import MachineCollection

    return Path(MachineCollection.get_machine_file_path())


@contextmanager
def locked(path: Path, mode: int):
    """Hold ``fcntl.flock`` on ``<path>.lock`` for the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a", encoding="utf-8") as handle:
        fcntl.flock(handle, mode)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_machines(path: Path) -> dict:
    if not path.is_file():
        return {"version": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def write_machines(path: Path, machines: dict) -> None:
    """Replace the file in one rename, so a reader never sees half of it."""
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    )
    with handle:
        json.dump(machines, handle, indent=4, sort_keys=True)
    os.replace(handle.name, path)


def machine_fields(args: list[str]) -> dict[str, str] | None:
    """The fields of an ``asv machine`` command, or ``None`` for asv's prompt."""
    parser = argparse.ArgumentParser(add_help=False)
    for field in MACHINE_FIELDS:
        parser.add_argument("--" + field)
    parser.add_argument("--yes", action="store_true")
    known, _rest = parser.parse_known_args(args)
    given = {field: getattr(known, field) for field in MACHINE_FIELDS if getattr(known, field) is not None}
    if not given and not known.yes:
        return None
    return given


def write_machine(given: dict[str, str], path: Path | None = None) -> dict[str, str]:
    """Merge one whole entry into the machine file and return it."""
    from asv.machine import Machine

    path = machine_file() if path is None else path
    name = given.get("machine") or Machine.get_defaults()["machine"]
    with locked(path, fcntl.LOCK_EX):
        machines = read_machines(path)
        entry = machine_entry(machines.get(name), {**given, "machine": name})
        machines[name] = entry
        machines.setdefault("version", 1)
        write_machines(path, machines)
    return entry


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["machine"]:
        given = machine_fields(args[1:])
        if given is not None:
            entry = write_machine(given)
            print(json.dumps(entry, indent=4, sort_keys=True))
            return
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
