"""Shifted free-boundary Lennard-Jones, the rgpot ``LJClusterPot`` kernel.

eonclient's ``ljcluster`` potential is ``rgpot::LJClusterPot`` with
``u0 = 1``, ``psi = 1``, and ``cutoff = 15``. The pair energy is the
12-6 term minus its value at the cutoff, and the cell is ignored.
``dx`` is ``r_i - r_j``. Pairs at or beyond the cutoff are left out.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from fractions import Fraction


def lj_shift(u0: float, psi: float, cutoff: float) -> float:
    """Potential at the cutoff. Subtracting it makes that point zero.

    ``(psi / cutoff)^6`` is rounded once from its exact rational value,
    so no libm ``pow`` chosen by CPU can change it.
    """
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    ratio = float(Fraction(psi / cutoff) ** 6)
    return 4.0 * u0 * ratio * (ratio - 1.0)


def _lj_cluster_python(
    positions: Sequence[float],
    atomic_numbers: Sequence[int] | None = None,
    box: Sequence[float] | None = None,
    *,
    u0: float = 1.0,
    psi: float = 1.0,
    cutoff: float = 15.0,
) -> tuple[float, tuple[float, ...]]:
    """Energy and flat forces for one free cluster.

    ``positions`` is ``x1, y1, z1, x2, ...`` in Angstrom. Atomic numbers
    and the box are accepted so the function has the rgpot ``calculate``
    shape; the kernel does not use them.
    """
    del atomic_numbers, box
    count = len(positions)
    if count % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    atoms = count // 3
    forces = [0.0] * count
    if atoms < 2:
        return 0.0, tuple(forces)
    cutoff_u = lj_shift(u0, psi, cutoff)
    cutoff2 = cutoff * cutoff
    psi2 = psi * psi
    energy = 0.0
    for i in range(atoms):
        ix = 3 * i
        for j in range(i + 1, atoms):
            jx = 3 * j
            dx = float(positions[ix]) - float(positions[jx])
            dy = float(positions[ix + 1]) - float(positions[jx + 1])
            dz = float(positions[ix + 2]) - float(positions[jx + 2])
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0 or r2 >= cutoff2:
                continue
            inv_r2 = 1.0 / r2
            sr2 = psi2 * inv_r2
            attraction = sr2 * sr2 * sr2
            well = 4.0 * u0 * attraction
            energy += well * (attraction - 1.0) - cutoff_u
            scale = 6.0 * well * inv_r2 * (2.0 * attraction - 1.0)
            fx = scale * dx
            fy = scale * dy
            fz = scale * dz
            forces[ix] += fx
            forces[ix + 1] += fy
            forces[ix + 2] += fz
            forces[jx] -= fx
            forces[jx + 1] -= fy
            forces[jx + 2] -= fz
    return energy, tuple(forces)


def read_component_coordinates(text: str) -> list[float]:
    """Flatten ``x y z`` rows after ``Coordinates of Component``."""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("Coordinates of Component"))
    coords: list[float] = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            break
        try:
            coords.extend((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            break
    if not coords:
        raise ValueError("con file has no coordinates")
    return coords


def lj_cluster(
    positions: Sequence[float],
    atomic_numbers: Sequence[int] | None = None,
    box: Sequence[float] | None = None,
    *,
    u0: float = 1.0,
    psi: float = 1.0,
    cutoff: float = 15.0,
) -> tuple[float, tuple[float, ...]]:
    """Energy and flat forces. The native kernel matches ``_lj_cluster_python``."""
    count = len(positions)
    if count % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    if os.environ.get("OPTBENCH_FASTPOT") != "0" and count >= 6:
        from optbench.fastpot import lj_cluster_forces

        got = lj_cluster_forces(positions, u0, psi, cutoff, lj_shift(u0, psi, cutoff))
        if got is not None:
            return got
    return _lj_cluster_python(
        positions,
        atomic_numbers,
        box,
        u0=u0,
        psi=psi,
        cutoff=cutoff,
    )
