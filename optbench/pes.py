"""Pair potentials and endpoint readers for the band campaigns.

Morse platinum matches the eOn ``morse_pt`` parameters that the LAMMPS
twin reproduces: ``pair_style morse 9.5``, ``pair_coeff * * 0.7102
1.6047 2.897``, ``pair_modify shift yes``. Lennard-Jones is the
archive ``lj/cut`` with ``epsilon = sigma = 1`` and cutoff 40, with no
shift. A force call is one energy-and-gradient evaluation of every
pair inside the cutoff.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MORSE_D = 0.7102
MORSE_ALPHA = 1.6047
MORSE_R0 = 2.897
MORSE_CUTOFF = 9.5
LJ_EPSILON = 1.0
LJ_SIGMA = 1.0
LJ_CUTOFF = 40.0


@dataclass(frozen=True)
class Pair:
    """One reactant/product pair. Coordinates are ``(n_atoms, 3)``."""

    name: str
    reactant: np.ndarray
    product: np.ndarray
    free: np.ndarray
    box: np.ndarray | None


def _mic(diff: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return diff
    return diff - box * np.round(diff / box)


def morse_pt(coords: np.ndarray, box: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Shifted Morse energy and Cartesian gradient.

    ``coords`` is ``(n_atoms, 3)``. The gradient is ``dE/dx``, the
    negative of the force.
    """
    xyz = np.asarray(coords, dtype=np.float64)
    n = xyz.shape[0]
    diff = _mic(xyz[:, None, :] - xyz[None, :, :], box)
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    inside = (r > 1e-12) & (r < MORSE_CUTOFF)
    if not np.any(inside):
        return 0.0, np.zeros_like(xyz)
    r = r[inside]
    dr = r - MORSE_R0
    x = np.exp(-MORSE_ALPHA * dr)
    raw = MORSE_D * (x * x - 2.0 * x)
    rc = MORSE_CUTOFF - MORSE_R0
    xc = np.exp(-MORSE_ALPHA * rc)
    shift = MORSE_D * (xc * xc - 2.0 * xc)
    energy = float(np.sum(raw - shift))
    dedr = 2.0 * MORSE_ALPHA * MORSE_D * x * (1.0 - x)
    ii = iu[0][inside]
    jj = iu[1][inside]
    unit = diff[ii, jj] / r[:, None]
    contrib = dedr[:, None] * unit
    grad = np.zeros_like(xyz)
    np.add.at(grad, ii, contrib)
    np.add.at(grad, jj, -contrib)
    return energy, grad


def lennard_jones(coords: np.ndarray, box: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """12-6 Lennard-Jones energy and Cartesian gradient. No shift."""
    xyz = np.asarray(coords, dtype=np.float64)
    n = xyz.shape[0]
    diff = _mic(xyz[:, None, :] - xyz[None, :, :], box)
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    inside = (r > 1e-12) & (r < LJ_CUTOFF)
    if not np.any(inside):
        return 0.0, np.zeros_like(xyz)
    r = r[inside]
    inv_r = 1.0 / r
    sr2 = inv_r * inv_r
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = float(4.0 * LJ_EPSILON * np.sum(sr12 - sr6))
    dedr = 24.0 * LJ_EPSILON * inv_r * (sr6 - 2.0 * sr12)
    ii = iu[0][inside]
    jj = iu[1][inside]
    contrib = dedr[:, None] * diff[ii, jj] * inv_r[:, None]
    grad = np.zeros_like(xyz)
    np.add.at(grad, ii, contrib)
    np.add.at(grad, jj, -contrib)
    return energy, grad


def energy_weighted_springs(energies: np.ndarray, k_min: float = 0.97, k_max: float = 9.7) -> np.ndarray:
    """Henkelman segment constants, one per consecutive pair of images.

    This is the eOn weighting: the higher energy of the two images
    sets the segment, ``k`` is ``k_max`` at the path maximum and
    ``k_min`` at or below the higher endpoint.
    """
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("springs need one energy per image")
    reference = float(max(values[0], values[-1]))
    highest = float(np.max(values))
    constants = np.full(values.size - 1, k_min, dtype=np.float64)
    span = highest - reference
    if span < 1e-10:
        return constants
    for index in range(1, values.size):
        ei = float(max(values[index], values[index - 1]))
        if ei <= reference:
            continue
        alpha = min(1.0, max(0.0, (highest - ei) / span))
        constants[index - 1] = (1.0 - alpha) * k_max + alpha * k_min
    return constants


def idpp_objective(
    coords: np.ndarray,
    target: np.ndarray,
    box: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Image-dependent pair potential and its Cartesian gradient.

    Smidstrup, Pedersen, Stokbro, and Jónsson, J. Chem. Phys. 140,
    214106 (2014): ``S = sum_{i<j} d_ij^{-4} (r_ij - d_ij)^2``.
    ``target`` is the interpolated distance matrix. The gradient is
    ``dS/dx``.
    """
    xyz = np.asarray(coords, dtype=np.float64)
    dist, diff = _pair_geometry(xyz, box)
    n = xyz.shape[0]
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    d = np.asarray(target, dtype=np.float64)[iu]
    ok = r > 1e-12
    weight = np.zeros_like(r)
    weight[ok] = np.maximum(d[ok], 1e-8) ** -4
    gap = r - d
    energy = float(np.sum(weight * gap * gap))
    dedr = np.zeros_like(r)
    dedr[ok] = 2.0 * weight[ok] * gap[ok] / r[ok]
    ii = iu[0]
    jj = iu[1]
    contrib = dedr[:, None] * diff[ii, jj]
    grad = np.zeros_like(xyz)
    np.add.at(grad, ii, contrib)
    np.add.at(grad, jj, -contrib)
    return energy, grad


def pair_distances(coords: np.ndarray, box: np.ndarray | None = None) -> np.ndarray:
    """Minimum-image distances. The diagonal is zero."""
    return _pair_geometry(np.asarray(coords, dtype=np.float64), box)[0]


def _pair_geometry(coords: np.ndarray, box: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    diff = _mic(coords[:, None, :] - coords[None, :, :], box)
    return np.linalg.norm(diff, axis=-1), diff


def evaluate(kind: str, coords: np.ndarray, box: np.ndarray | None) -> tuple[float, np.ndarray]:
    if kind == "morse_pt":
        return morse_pt(coords, box)
    if kind == "lj":
        return lennard_jones(coords, None)
    raise ValueError(f"unknown potential {kind}")


def free_gradient(kind: str, free_flat: np.ndarray, template: np.ndarray, free: np.ndarray, box: np.ndarray | None):
    """Energy and the free-atom gradient of one geometry.

    Fixed atoms stay at ``template``. The returned gradient is flat,
    in free-atom order, and is ``dE/dx``.
    """
    full = np.array(template, dtype=np.float64, copy=True)
    full[free] = np.asarray(free_flat, dtype=np.float64).reshape((-1, 3))
    energy, grad = evaluate(kind, full, box)
    return float(energy), grad[free].reshape(-1)


def interpolate_free(
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    n_interior: int,
    box: np.ndarray | None,
) -> np.ndarray:
    """Linear path of free coordinates, minimum-imaged when ``box`` is set.

    The array has shape ``(n_interior + 2, n_free * 3)``. Endpoints are
    the free atoms of the reactant and of the minimum-image product.
    """
    if n_interior < 1:
        raise ValueError("a band needs at least one interior image")
    delta = _mic(product - reactant, box)
    total = n_interior + 2
    frames = []
    for index in range(total):
        moved = reactant + (index / (total - 1)) * delta
        frames.append(moved[free].reshape(-1))
    return np.vstack(frames)


def read_con(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """eOn ``.con`` frame: positions, free mask, orthorhombic box lengths.

    The layout is the one ``aselite.read_con`` uses. A non-zero fourth
    field freezes that atom. Angles other than 90 degrees are refused
    because the campaign cell is orthorhombic.
    """
    lines = text.splitlines()
    if len(lines) < 12:
        raise ValueError("con file is too short")
    lengths = np.array([float(value) for value in lines[2].split()], dtype=np.float64)
    angles = np.array([float(value) for value in lines[3].split()], dtype=np.float64)
    if lengths.shape != (3,) or angles.shape != (3,):
        raise ValueError("con box is not three lengths and three angles")
    if np.any(np.abs(angles - 90.0) > 1e-6):
        raise ValueError("con cell is not orthorhombic")
    n_types = int(lines[6].split()[0])
    counts = [int(value) for value in lines[7].split()]
    if len(counts) < n_types:
        raise ValueError("con type counts do not match the type line")
    counts = counts[:n_types]
    cursor = 9
    positions = []
    free = []
    for type_index, count in enumerate(counts):
        if cursor >= len(lines):
            raise ValueError("con file ended before a species name")
        cursor += 2
        for _ in range(count):
            if cursor >= len(lines):
                raise ValueError("con file ended inside coordinates")
            parts = lines[cursor].split()
            if len(parts) < 4:
                raise ValueError(f"con coordinate {cursor + 1} has no fixed flag")
            positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
            free.append(parts[3] == "0")
            cursor += 1
        if type_index + 1 < n_types:
            continue
    coords = np.asarray(positions, dtype=np.float64)
    if coords.shape[0] != sum(counts):
        raise ValueError("con atom count does not match the header")
    box = lengths if np.all(lengths > 0.0) else None
    return coords, np.asarray(free, dtype=bool), box


def read_xyz(text: str) -> np.ndarray:
    """XYZ block. The first line is the atom count. Every atom is free.

    The second line is the comment and may be blank, so it is not dropped.
    """
    lines = text.splitlines()
    count = int(lines[0].split()[0])
    body = lines[2 : 2 + count]
    coords = []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError("xyz coordinate is missing x y z")
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(coords) != count:
        raise ValueError(f"xyz announced {count} atoms and held {len(coords)}")
    return np.asarray(coords, dtype=np.float64)


def _members(archive: Path) -> dict[str, str]:
    found = {}
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            found[member.name] = extracted.read().decode("utf-8", errors="replace")
    return found


def _require_count(pairs: list[Pair], expected: int | None, label: str) -> list[Pair]:
    if expected is not None and len(pairs) != expected:
        raise ValueError(f"{label} has {len(pairs)} pairs, the problem says {expected}")
    if not pairs:
        raise ValueError(f"{label} has no pairs")
    return pairs


def load_pairs(archive: Path, kind: str, expected: int | None) -> list[Pair]:
    """Every reactant/product pair in one geometry archive."""
    texts = _members(archive)
    if kind == "con-minima-pairs":
        return _require_count(_con_minima(texts), expected, kind)
    if kind == "con-reactant-products":
        return _require_count(_con_reactant_products(texts), expected, kind)
    if kind == "xyz-pairs":
        return _require_count(_xyz_pairs(texts), expected, kind)
    raise ValueError(f"unknown geometry kind {kind}")


def _con_minima(texts: dict[str, str]) -> list[Pair]:
    groups: dict[str, dict[str, str]] = {}
    for name, text in texts.items():
        if not name.endswith(".con"):
            continue
        path = Path(name)
        if path.name not in ("a.con", "b.con"):
            continue
        groups.setdefault(path.parent.as_posix(), {})[path.name] = text
    pairs = []
    for folder in sorted(groups):
        ends = groups[folder]
        if "a.con" not in ends or "b.con" not in ends:
            continue
        reactant, free_a, box = read_con(ends["a.con"])
        product, free_b, _box_b = read_con(ends["b.con"])
        if reactant.shape != product.shape or not np.array_equal(free_a, free_b):
            raise ValueError(f"{folder} endpoints do not match")
        pairs.append(Pair(Path(folder).name, reactant, product, free_a, box))
    return pairs


def _con_reactant_products(texts: dict[str, str]) -> list[Pair]:
    reactant_name = next((name for name in texts if Path(name).name == "reactant.con"), None)
    if reactant_name is None:
        raise ValueError("con-reactant-products has no reactant.con")
    reactant, free, box = read_con(texts[reactant_name])
    products = []
    for name, text in texts.items():
        stem = Path(name).name
        if not stem.startswith("product_") or not stem.endswith(".con"):
            continue
        index = int(stem[len("product_") : -len(".con")])
        products.append((index, text))
    pairs = []
    for index, text in sorted(products):
        product, free_b, _box = read_con(text)
        if product.shape != reactant.shape or not np.array_equal(free, free_b):
            raise ValueError(f"product_{index} does not match the reactant")
        fixed_shift = float(np.max(np.abs(product[~free] - reactant[~free]))) if np.any(~free) else 0.0
        if fixed_shift > 1e-4:
            raise ValueError(f"product_{index} moves a fixed atom by {fixed_shift}")
        pairs.append(Pair(f"product_{index}", reactant, product, free, box))
    return pairs


def _xyz_pairs(texts: dict[str, str]) -> list[Pair]:
    starts: dict[int, str] = {}
    ends: dict[int, str] = {}
    for name in texts:
        stem = Path(name).name
        if stem.startswith("start_") and stem.endswith(".xyz"):
            starts[int(stem[len("start_") : -len(".xyz")])] = name
        elif stem.startswith("end_") and stem.endswith(".xyz"):
            ends[int(stem[len("end_") : -len(".xyz")])] = name
    missing = sorted(set(starts) ^ set(ends))
    if missing:
        raise ValueError(f"xyz pairs are missing an endpoint at {missing[:5]}")
    pairs = []
    for index in sorted(starts):
        reactant = read_xyz(texts[starts[index]])
        product = read_xyz(texts[ends[index]])
        if reactant.shape != product.shape:
            raise ValueError(f"pair {index} endpoints differ in length")
        free = np.ones(len(reactant), dtype=bool)
        pairs.append(Pair(f"pair_{index:02d}", reactant, product, free, None))
    return pairs
