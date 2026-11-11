"""The asv machine record that a new result carries.

asv 0.6.6 names a machine after the node name of the host and fills
``os``, ``arch``, ``cpu``, ``num_cpu`` and ``ram`` from this process.
Here ``machine`` is ``host_label`` and every other field is
``unspecified``. ``results/asv`` holds one directory per label.
"""

from __future__ import annotations

import json
from pathlib import Path

from .v2 import UNSPECIFIED, check_label, host_label

FIELDS = ("os", "arch", "cpu", "num_cpu", "ram")


def machine_record(label: str | None = None) -> dict[str, str]:
    """``machine`` is ``label`` or ``host_label()``, and the rest are ``unspecified``."""
    name = host_label() if label is None else check_label(label, "machine")
    return {"machine": name, **{field: UNSPECIFIED for field in FIELDS}}


def write_machine_file(label: str | None = None, path: Path | None = None) -> dict[str, str]:
    """Store ``machine_record`` in asv's machine file under its label.

    The file is ``~/.asv-machine.json`` unless ``path`` is given. Other
    entries in it stay. ``asv run --machine <label>`` reads this entry.
    """
    from asv.machine import MachineCollection

    record = machine_record(label)
    MachineCollection.save(record["machine"], record, _path=None if path is None else str(path))
    return record


def machine_dir(results_dir: Path, label: str | None = None) -> Path:
    """``results_dir/<label>``, with the ``machine.json`` asv reads beside the result files.

    An existing ``machine.json`` that is not this record, such as an
    archived one, is left alone and raises ``ValueError``.
    """
    record = {**machine_record(label), "version": 1}
    path = Path(results_dir) / record["machine"]
    machine = path / "machine.json"
    if machine.exists():
        if json.loads(machine.read_text(encoding="utf-8")) != record:
            raise ValueError(f"{machine} is not the {record['machine']} record. Use another label.")
        return path
    path.mkdir(parents=True, exist_ok=True)
    machine.write_text(json.dumps(record, indent=4, sort_keys=True), encoding="utf-8")
    return path
