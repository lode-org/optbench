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
import os
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
    """Energy and ``dE/dx`` for a cutoff Lennard-Jones cluster.

    The squared distance is ``(x² + z²) + y²``, the order
    ``np.einsum("ija,ija->ij")`` used when the committed rows were made.
    ``(σ/r)^6`` is ``shared_pow``, the value ``ratio**6`` has on hosts
    where NumPy runs its AVX-512 power loop.
    """
    pos = np.asarray(positions, dtype=np.float64)
    count = len(pos)
    delta = minimum_image(pos[:, None, :] - pos[None, :, :], cell if cell is not None else np.zeros(3), pbc)
    distance2 = (delta[..., 0] * delta[..., 0] + delta[..., 2] * delta[..., 2]) + delta[..., 1] * delta[..., 1]
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
    sixth = shared_pow(ratio, 6.0)
    twelfth = sixth * sixth
    energy = float(4.0 * epsilon * np.sum(twelfth - sixth))
    slope = 24.0 * epsilon / radius * (sixth - 2.0 * twelfth)
    left = upper[0][chosen]
    right = upper[1][chosen]
    vector = (slope / radius)[:, None] * delta[left, right]
    np.add.at(gradient, left, vector)
    np.add.at(gradient, right, -vector)
    return energy, gradient


# x86-64-v2 is the x86-64 psABI micro-architecture level of SSE3, SSSE3,
# SSE4.1, SSE4.2, POPCNT, CMPXCHG16B and LAHF-SAHF, with no AVX or FMA
# (psABI section 3.1.1, table 3.1; GCC x86 Options, -march). Every host
# at that level runs the same instructions. -ffp-contract=off keeps
# a*b+c two rounded operations and -fno-fast-math keeps IEEE order and
# rounding (GCC Optimize Options).
NATIVE_FLAGS = (
    "-std=c99",
    "-march=x86-64-v2",
    "-O2",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fPIC",
    "-shared",
)
_NATIVE_SOURCES = ("morse_pair.c", "svml_exp.c", "svml_pow.c")
_NATIVE_HEADERS = ("svml_exp.h", "svml_pow.h")
_NATIVE_LIB = None


def native_library():
    """The Morse pair loop, and the exp and pow the surfaces share.

    The Morse loop and its NumPy reference call this exp, and the
    Lennard-Jones surface calls this pow, so a compiler is required:
    np.exp, np.power and libm pick a code path by CPU and give different
    last bits.
    """
    global _NATIVE_LIB
    if _NATIVE_LIB is not None:
        return _NATIVE_LIB
    import ctypes

    here = Path(__file__).parent
    sources = [here / name for name in _NATIVE_SOURCES]
    digest = hashlib.sha256(" ".join(NATIVE_FLAGS).encode())
    for path in [*sources, *(here / name for name in _NATIVE_HEADERS)]:
        digest.update(path.read_bytes())
    library = Path(tempfile.gettempdir()) / f"optbench-surfaces-{digest.hexdigest()[:16]}.so"
    if not library.is_file():
        partial = library.with_name(f"{library.name}.{os.getpid()}.tmp")
        try:
            subprocess.run(
                ["gcc", *NATIVE_FLAGS, "-o", str(partial), *map(str, sources), "-lm"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError("the surfaces need gcc to build their exp, pow and Morse pair loop") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"gcc could not build the surface library: {error.stderr.strip()}") from error
        partial.replace(library)
    loaded = ctypes.CDLL(str(library))
    double = ctypes.c_double
    pointer = ctypes.POINTER(double)
    loaded.optbench_exp.restype = double
    loaded.optbench_exp.argtypes = [double]
    loaded.optbench_exp_array.restype = None
    loaded.optbench_exp_array.argtypes = [pointer, pointer, ctypes.c_long]
    loaded.optbench_pow_scalar_array.restype = None
    loaded.optbench_pow_scalar_array.argtypes = [pointer, double, pointer, ctypes.c_long]
    loaded.morse_pair_energy.restype = ctypes.c_int
    loaded.morse_pair_energy.argtypes = [
        pointer,
        pointer,
        ctypes.c_int,
        *[double] * 7,
        *[ctypes.c_int] * 3,
        pointer,
    ]
    _NATIVE_LIB = loaded
    return loaded


def shared_exp(values: Array) -> Array:
    """``exp`` of a float64 array with the pair loop's routine."""
    import ctypes

    given = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty_like(given)
    pointer = ctypes.POINTER(ctypes.c_double)
    native_library().optbench_exp_array(
        given.ctypes.data_as(pointer), out.ctypes.data_as(pointer), given.size
    )
    return out


def shared_pow(values: Array, exponent: float) -> Array:
    """``values ** exponent`` for a float64 array and a scalar exponent.

    The result is NumPy's float64 power loop on AVX-512 hosts, on every
    host: its fast paths for exponents -1, 0, 0.5, 1 and 2, and the SVML
    power routine otherwise.
    """
    import ctypes

    given = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty_like(given)
    pointer = ctypes.POINTER(ctypes.c_double)
    native_library().optbench_pow_scalar_array(
        given.ctypes.data_as(pointer), float(exponent), out.ctypes.data_as(pointer), given.size
    )
    return out


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
    """NumPy pair sum, the reference the compiled loop equals bit for bit.

    The squared distance is summed as ``(x² + z²) + y²``, the order
    ``np.einsum("ija,ija->ij")`` used when the committed rows were made.
    """
    pos = np.asarray(positions, dtype=np.float64)
    count = len(pos)
    at_cut = float(native_library().optbench_exp(-alpha * (cutoff - r0)))
    offset = depth * (at_cut * at_cut - 2.0 * at_cut)
    der_cut = -2.0 * alpha * depth * (at_cut * at_cut - at_cut)
    delta = minimum_image(pos[:, None, :] - pos[None, :, :], cell if cell is not None else np.zeros(3), pbc)
    distance2 = (delta[..., 0] * delta[..., 0] + delta[..., 2] * delta[..., 2]) + delta[..., 1] * delta[..., 1]
    upper = np.triu_indices(count, k=1)
    pair2 = distance2[upper]
    chosen = (pair2 > 0.0) & (pair2 < cutoff * cutoff)
    gradient = np.zeros_like(pos)
    if not np.any(chosen):
        return 0.0, gradient
    radius = np.sqrt(pair2[chosen])
    decay = shared_exp(-alpha * (radius - r0))
    morse = depth * (decay * decay - 2.0 * decay)
    energy = float(np.sum(morse - offset - (radius - cutoff) * der_cut))
    slope = -2.0 * alpha * depth * (decay * decay - decay) - der_cut
    left = upper[0][chosen]
    right = upper[1][chosen]
    vector = (slope / radius)[:, None] * delta[left, right]
    np.add.at(gradient, left, vector)
    np.add.at(gradient, right, -vector)
    return energy, gradient


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
    import ctypes

    count = int(pos.shape[0])
    gradient = np.zeros_like(pos)
    lengths = np.zeros(3, dtype=np.float64) if cell is None else np.asarray(cell, dtype=np.float64)
    energy = ctypes.c_double()
    status = native_library().morse_pair_energy(
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
        ctypes.byref(energy),
    )
    if status != 0:
        raise MemoryError("the Morse pair loop could not allocate its pair buffers")
    return float(energy.value), gradient


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
