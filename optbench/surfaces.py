"""Lennard-Jones and smoothed Morse surfaces for a counted saddle search.

The Lennard-Jones pair is the ASE form used by the Sella LJ38 entry:
``4 ε [(σ/r)^12 - (σ/r)^6]`` inside the cutoff, with no shift.
The Morse pair is LAMMPS ``morse/smooth/linear``: the Morse energy
``D (e^{-2α(r-r0)} - 2 e^{-α(r-r0)})`` minus its value at the cutoff
and minus ``(r - rc)`` times the Morse derivative at the cutoff, so
the energy and the radial force are both zero at ``rc``.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

Array = np.ndarray


def read_con(text: str) -> tuple[Array, Array, Array]:
    """Return positions ``(N, 3)``, a fixed mask, and the orthorhombic cell.

    The fixed mask is true where the eOn flag is non-zero. The cell is
    the box lengths on the line above the angle line.
    """
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("coordinate"))
    count = int(float(lines[header - 3].split()[0]))
    positions = np.empty((count, 3), dtype=np.float64)
    fixed = np.zeros(count, dtype=bool)
    for row, line in enumerate(lines[header + 1 : header + 1 + count]):
        parts = line.split()
        positions[row] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if len(parts) > 3:
            fixed[row] = float(parts[3]) != 0.0
    cell = np.zeros(3, dtype=np.float64)
    for index in range(header):
        parts = lines[index].split()
        if len(parts) < 3:
            continue
        try:
            lengths = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            continue
        nxt = lines[index + 1].split() if index + 1 < len(lines) else []
        if len(nxt) >= 3:
            try:
                angles = [float(nxt[0]), float(nxt[1]), float(nxt[2])]
            except ValueError:
                continue
            if all(abs(angle - 90.0) < 1e-6 for angle in angles):
                cell = np.asarray(lengths, dtype=np.float64)
                break
    if len(positions) != count:
        raise ValueError(f"expected {count} coordinates")
    return positions, fixed, cell


def read_mode(text: str) -> Array:
    """Reference mode. A leading ``#eigenvalue`` line is skipped."""
    values = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.append(float(stripped.split()[0]))
    mode = np.asarray(values, dtype=np.float64)
    scale = np.linalg.norm(mode)
    if not np.isfinite(scale) or scale < 1e-14:
        raise ValueError("reference mode has no direction")
    return mode / scale


def minimum_image(delta: Array, cell: Array, pbc: tuple[bool, bool, bool]) -> Array:
    """Fold Cartesian displacements into an orthorhombic cell."""
    out = np.array(delta, dtype=np.float64, copy=True)
    for axis, periodic in enumerate(pbc):
        length = float(cell[axis]) if cell is not None else 0.0
        if periodic and length > 0.0:
            out[..., axis] -= length * np.round(out[..., axis] / length)
    return out


def lennard_jones(
    positions: Array,
    *,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    cutoff: float = 40.0,
    cell: Array | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[float, Array]:
    """Energy and ``dE/dx`` for a cutoff Lennard-Jones cluster."""
    pos = np.asarray(positions, dtype=np.float64)
    count = len(pos)
    delta = minimum_image(pos[:, None, :] - pos[None, :, :], cell if cell is not None else np.zeros(3), pbc)
    distance2 = np.einsum("ija,ija->ij", delta, delta)
    upper = np.triu_indices(count, k=1)
    pair2 = distance2[upper]
    cutoff2 = cutoff * cutoff
    chosen = (pair2 > 0.0) & (pair2 < cutoff2)
    gradient = np.zeros_like(pos)
    if not np.any(chosen):
        return 0.0, gradient
    radius2 = pair2[chosen]
    radius = np.sqrt(radius2)
    ratio = sigma / radius
    sixth = ratio**6
    twelfth = sixth * sixth
    energy = float(4.0 * epsilon * np.sum(twelfth - sixth))
    slope = 24.0 * epsilon / radius * (sixth - 2.0 * twelfth)
    left = upper[0][chosen]
    right = upper[1][chosen]
    vector = (slope / radius)[:, None] * delta[left, right]
    np.add.at(gradient, left, vector)
    np.add.at(gradient, right, -vector)
    return energy, gradient


def _morse_numpy(
    positions: Array,
    *,
    depth: float,
    alpha: float,
    r0: float,
    cutoff: float,
    cell: Array | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[float, Array]:
    """NumPy pair sum. The compiled loop is the same formula."""
    pos = np.asarray(positions, dtype=np.float64)
    count = len(pos)
    alpha_dr = -alpha * (cutoff - r0)
    at_cut = math.exp(alpha_dr)
    offset = depth * (at_cut * at_cut - 2.0 * at_cut)
    der_cut = -2.0 * alpha * depth * (at_cut * at_cut - at_cut)
    delta = minimum_image(pos[:, None, :] - pos[None, :, :], cell if cell is not None else np.zeros(3), pbc)
    distance2 = np.einsum("ija,ija->ij", delta, delta)
    upper = np.triu_indices(count, k=1)
    pair2 = distance2[upper]
    chosen = (pair2 > 0.0) & (pair2 < cutoff * cutoff)
    gradient = np.zeros_like(pos)
    if not np.any(chosen):
        return 0.0, gradient
    radius = np.sqrt(pair2[chosen])
    decay = np.exp(-alpha * (radius - r0))
    morse = depth * (decay * decay - 2.0 * decay)
    energy = float(np.sum(morse - offset - (radius - cutoff) * der_cut))
    slope = -2.0 * alpha * depth * (decay * decay - decay) - der_cut
    left = upper[0][chosen]
    right = upper[1][chosen]
    vector = (slope / radius)[:, None] * delta[left, right]
    np.add.at(gradient, left, vector)
    np.add.at(gradient, right, -vector)
    return energy, gradient


_MORSE_LIB = None
_MORSE_TRIED = False


def _morse_path(source: Path) -> Path:
    """``morse_pair.c`` built once per source and flags, published whole."""
    from optbench.fastpot import _link

    flags = "-O3 -march=native -fPIC -shared"
    digest = hashlib.sha256(source.read_bytes() + flags.encode()).hexdigest()[:16]
    library = Path(tempfile.gettempdir()) / f"optbench-morse-{digest}.so"
    if not library.is_file():
        _link(
            ["gcc", *flags.split(), str(source), "-lm"],
            library,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return library


def _morse_library():
    """Compile the pair loop once. A missing compiler keeps the NumPy sum."""
    global _MORSE_LIB, _MORSE_TRIED
    if _MORSE_TRIED:
        return _MORSE_LIB
    _MORSE_TRIED = True
    source = Path(__file__).with_name("morse_pair.c")
    if not source.is_file():
        return None
    try:
        library = _morse_path(source)
    except (OSError, subprocess.CalledProcessError):
        return None
    import ctypes

    loaded = ctypes.CDLL(str(library))
    loaded.morse_pair_energy.restype = ctypes.c_double
    loaded.morse_pair_energy.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _MORSE_LIB = loaded
    return loaded


def morse_smooth_linear(
    positions: Array,
    *,
    depth: float,
    alpha: float,
    r0: float,
    cutoff: float,
    cell: Array | None = None,
    pbc: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[float, Array]:
    """Energy and ``dE/dx`` for LAMMPS ``morse/smooth/linear``.

    ``der_at_cutoff`` in the LAMMPS pair is the Morse ``dE/dr`` at the
    cutoff. Subtracting it from the slope, and subtracting
    ``(r - rc) * der_at_cutoff`` from the energy, puts both at zero
    on the cutoff sphere.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    library = _morse_library()
    if library is None:
        return _morse_numpy(
            pos, depth=depth, alpha=alpha, r0=r0, cutoff=cutoff, cell=cell, pbc=pbc
        )
    count = int(pos.shape[0])
    gradient = np.zeros_like(pos)
    lengths = np.zeros(3, dtype=np.float64) if cell is None else np.asarray(cell, dtype=np.float64)
    import ctypes

    energy = library.morse_pair_energy(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        count,
        float(depth),
        float(alpha),
        float(r0),
        float(cutoff),
        float(lengths[0]),
        float(lengths[1]),
        float(lengths[2]),
        int(bool(pbc[0])),
        int(bool(pbc[1])),
        int(bool(pbc[2])),
    )
    return float(energy), gradient


def load_con(path: Path) -> tuple[Array, Array, Array]:
    return read_con(path.read_text(encoding="utf-8", errors="replace"))


def potential_from_problem(problem: dict):
    """Return ``(positions -> (energy, gradient))`` bound to the problem potential."""
    spec = problem.get("potential") or {}
    kind = str(spec.get("kind", "lennard-jones"))
    pbc_raw = spec.get("pbc") or [False, False, False]
    pbc = (bool(pbc_raw[0]), bool(pbc_raw[1]), bool(pbc_raw[2]))

    def evaluate(positions: Array, cell: Array) -> tuple[float, Array]:
        if kind == "morse":
            return morse_smooth_linear(
                positions,
                depth=float(spec["D"]),
                alpha=float(spec["alpha"]),
                r0=float(spec["r0"]),
                cutoff=float(spec.get("cutoff", 9.5)),
                cell=cell,
                pbc=pbc,
            )
        return lennard_jones(
            positions,
            epsilon=float(spec.get("epsilon", 1.0)),
            sigma=float(spec.get("sigma", 1.0)),
            cutoff=float(spec.get("cutoff", 40.0)),
            cell=cell,
            pbc=pbc,
        )

    return evaluate
