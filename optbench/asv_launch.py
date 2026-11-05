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

Before the probe, every variable in ``optbench.blas_threads.PINNED`` is
set in this process, so the server starts with every BLAS pool at one
thread. ``asv.conf.json`` cannot carry them. A matrix ``env_nobuild``
value is part of the environment name, and so of every result file name,
and ``-E existing`` ignores the matrix. The launch stops if any probed
pool reports more than one thread.

``machine`` with ``--yes`` or with any field writes the whole entry
itself. asv 0.6.6 stores only the fields that differ from the values it
detects, so ``asv machine --yes --machine NAME`` leaves out ``arch``,
``cpu``, ``num_cpu``, ``os``, and ``ram``. Each field here is the given
value, else the stored one, else the one asv detects.

``run``, ``continuous``, and ``find`` load one machine entry, and asv
then writes the whole machine file back, truncating it first. A second
asv that starts in that window reads an empty file. Each of these
commands here gets a private file that holds only the entry it loads,
and asv writes that one. The shared file is read under a shared
``flock`` on ``<file>.lock``. ``machine`` and ``update`` change the shared
file, and they run under the exclusive lock. A bare ``asv`` takes no lock.
The private entry has every field, so ``results/<machine>/machine.json``
and each result's ``params`` do too, whoever wrote the shared entry.
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
MACHINE_READERS = ("run", "continuous", "find")
MACHINE_WRITERS = ("machine", "update")


def subcommand(args: list[str]) -> str | None:
    """The asv subcommand, after asv's global options."""
    index = 0
    while index < len(args) and args[index].startswith("-"):
        index += 2 if args[index] == "--config" else 1
    return args[index] if index < len(args) else None


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


def machine_name(args: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--machine", "-m")
    known, _rest = parser.parse_known_args(args)
    return known.machine


def stored_entry(machines: dict, name: str | None) -> tuple[str, dict]:
    """The entry ``MachineCollection.load`` returns for ``--machine name``."""
    from asv.machine import Machine

    entries = {key: value for key, value in machines.items() if key != "version"}
    detected = Machine.get_unique_machine_name()
    wanted = name or detected
    if wanted in entries:
        return wanted, entries[wanted]
    if len(entries) == 1 and wanted == detected:
        return next(iter(entries.items()))
    raise SystemExit(
        f"optbench.asv_launch: no machine entry {wanted!r}."
        f" Run python -m optbench.asv_launch machine --yes --machine {wanted}."
    )


@contextmanager
def own_machine_file(args: list[str]):
    """Point asv at a private file that holds the one entry it loads.

    A stored field the entry lacks is filled from detection, so an entry
    that ``asv machine --machine NAME`` saved as just the name still gives
    the run its ``arch``, ``cpu``, ``num_cpu``, ``os``, and ``ram``.
    """
    from asv.machine import MachineCollection

    shared = machine_file()
    with locked(shared, fcntl.LOCK_SH):
        machines = read_machines(shared)
    name, stored = stored_entry(machines, machine_name(args))
    entry = machine_entry(stored, {"machine": str(stored.get("machine") or name)})
    original = MachineCollection.__dict__["get_machine_file_path"]
    with tempfile.TemporaryDirectory(prefix="optbench-asv-machine-") as directory:
        private = Path(directory) / shared.name
        write_machines(private, {name: entry, "version": 1})
        MachineCollection.get_machine_file_path = staticmethod(lambda: str(private))
        try:
            yield private
        finally:
            MachineCollection.get_machine_file_path = original


def run_asv(args: list[str]) -> None:
    from asv.main import main as asv_main

    sys.argv = ["asv", *args]
    asv_main()


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    command = subcommand(args)
    if command == "machine":
        given = machine_fields(args)
        if given is not None:
            entry = write_machine(given)
            print(json.dumps(entry, indent=4, sort_keys=True))
            return
    blas_threads.pin()
    pools = probe(interpreter(args))
    record = {"variables": blas_threads.variables(), "blas": pools, "asv": args}
    print("optbench.asv_launch " + json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
    target = os.environ.get("OPTBENCH_ASV_RECORD")
    if target:
        Path(target).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wide = [pool for pool in pools if pool["threads"] != 1]
    if wide:
        raise SystemExit(f"optbench.asv_launch: BLAS runs more than one thread under {blas_threads.PINNED}: {wide}")
    if command in MACHINE_WRITERS:
        with locked(machine_file(), fcntl.LOCK_EX):
            run_asv(args)
    elif command in MACHINE_READERS:
        with own_machine_file(args):
            run_asv(args)
    else:
        run_asv(args)


if __name__ == "__main__":
    main()
