"""Start geometries for problems that are not reactant/product pairs.

Transition-state optimisation and the lowest-eigenvector sample use the
same archives as the rgsaddle min-mode campaigns. Minimisation uses
every ``.con`` in the problem archive. A job dict is plain lists so a
process pool can pickle it.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

from optbench.surfaces import read_con, read_mode, species_labels


def pes_kind(problem: dict) -> str:
    """``optbench.pes.evaluate`` name for this problem."""
    ident = str(problem.get("id", ""))
    spec = problem.get("potential") or {}
    kind = str(spec.get("kind", ""))
    if ident == "minimization/morse-bulk" or kind == "morse":
        return "morse_pt"
    if kind in {"lennard-jones", "lj", ""} and ident != "minimization/morse-bulk":
        return "lj"
    raise ValueError(f"{ident or 'problem'} has no engine potential")


def point_settings(problem: dict) -> dict:
    """Cap, force stop, and potential. A catalog problem may omit the cap."""
    convergence = problem.get("convergence") or {}
    stated = convergence.get("max_force_calls")
    force = convergence.get("force_norm", convergence.get("gradient_norm", 1e-3))
    return {
        "potential": pes_kind(problem),
        "cap": None if stated is None else int(stated),
        "force_norm": float(force),
        "overlap": float((problem.get("sample") or {}).get("overlap", 0.99)),
    }


def _stored(root: Path, archive_path: str) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(archive_path)


def _extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.is_dir():
        return archive
    if not tarfile.is_tarfile(archive):
        return archive.parent
    marker = dest / ".extracted"
    if marker.is_file():
        return dest
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    marker.write_text("ok\n", encoding="utf-8")
    return dest


def _box(kind: str, cell: np.ndarray) -> list[float] | None:
    if kind == "lj":
        return None
    lengths = np.asarray(cell, dtype=np.float64).reshape(-1)
    if lengths.size < 3 or float(np.max(np.abs(lengths[:3]))) <= 0.0:
        return None
    return [float(lengths[0]), float(lengths[1]), float(lengths[2])]


def _row(name: str, path: Path, kind: str, mode: Path | None = None) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    positions, fixed, cell = read_con(text)
    reference = None
    if mode is not None:
        full = read_mode(mode.read_text(encoding="utf-8", errors="replace"))
        piece = full.reshape(-1, 3)[~fixed].reshape(-1)
        scale = float(np.linalg.norm(piece))
        if scale < 1e-14:
            raise ValueError(f"{mode} has no free-atom direction")
        reference = (piece / scale).tolist()
    return {
        "name": name,
        "positions": np.asarray(positions, dtype=np.float64).tolist(),
        "fixed": np.asarray(fixed, dtype=bool).tolist(),
        "box": _box(kind, cell),
        "reference": reference,
        "symbols": species_labels(text),
    }


def load_starts(root: Path, problem: dict, work: Path) -> list[dict]:
    """One dict per sample start, in problem order."""
    root = Path(root)
    work = Path(work)
    kind = pes_kind(problem)
    ident = problem["id"]
    archive_path = (problem.get("input") or {}).get("archive_path")
    if ident == "saddle-search/pt-island":
        paths = (problem.get("input") or {}).get("geometries") or []
        con_path = next(path for path in paths if str(path).endswith(".con"))
        return [_row("pt-island", _stored(root, con_path), kind)]
    if not archive_path:
        raise ValueError(f"{ident} has no input.archive_path")
    extracted = _extract(_stored(root, archive_path), work / "geometries")
    rows: list[dict] = []
    if ident == "tsopt/LJ38":
        folder = extracted / "coords-con"
        for index in range(200):
            rows.append(_row(f"{index:04d}", folder / f"{index:04d}.con", kind))
    elif ident == "tsopt/pt-island":
        folder = extracted / "pt-island-con"
        for index in range(49):
            rows.append(_row(f"initial_{index}", folder / f"initial_{index}.con", kind))
    elif ident == "lowest-eigenvector/lj38":
        folder = extracted / "coords-con"
        for index in range(200):
            rows.append(
                _row(
                    f"{index:04d}",
                    folder / f"{index:04d}.con",
                    kind,
                    folder / f"{index:04d}_mode",
                )
            )
    elif problem.get("suite") == "minimization":
        cons = sorted(path for path in extracted.rglob("*.con") if path.is_file())
        if not cons:
            raise FileNotFoundError(f"{archive_path} has no .con files")
        rows = [_row(path.stem, path, kind) for path in cons]
    else:
        raise ValueError(f"no geometry layout for {ident}")
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(rows) != int(expected):
        raise RuntimeError(f"{ident} has {len(rows)} geometries, the problem states {expected}")
    missing = [row["name"] for row in rows if not row["positions"]]
    if missing:
        raise FileNotFoundError(missing[0])
    return rows
