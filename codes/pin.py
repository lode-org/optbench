"""Print one field of a ``codes/registry.toml`` entry.

The build scripts read every pin here, so the registry is the only place a
pin is written.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def field(code_id: str, name: str) -> str:
    registry = tomllib.loads((Path(__file__).resolve().parent / "registry.toml").read_text(encoding="utf-8"))
    for row in registry["code"]:
        if row["id"] == code_id:
            if name not in row:
                raise SystemExit(f"codes/registry.toml {code_id} has no {name}")
            return str(row[name])
    raise SystemExit(f"codes/registry.toml has no {code_id}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: pin.py <code id> <field>")
    print(field(sys.argv[1], sys.argv[2]))
