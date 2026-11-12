"""Read a peak-memory value written by asv_bench_memray."""

from __future__ import annotations

import json
from pathlib import Path


def peak_bytes(results_dir: Path = Path("results/asv-memray")) -> tuple[str, float]:
    matches = [
        path
        for path in results_dir.glob("*/*-*.json")
        if path.name != "machine.json"
    ]
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "none"
        raise SystemExit(f"expected one memray result, found {found}")
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    columns = data["result_columns"]
    result_index = columns.index("result")
    rows = data["results"]
    names = [name for name in rows if "ray_kept" in name]
    if len(names) != 1:
        raise SystemExit(f"expected one ray_kept row, found {names}")
    value = rows[names[0]][result_index]
    if isinstance(value, list):
        if len(value) != 1:
            raise SystemExit(f"expected one peak sample, found {value}")
        value = value[0]
    return names[0], float(value)


def write_peak(path: Path = Path("results/asv-memray/peak.txt")) -> None:
    name, value = peak_bytes(path.parent)
    path.write_text(f"{name} {value:.0f} bytes\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"), end="")
