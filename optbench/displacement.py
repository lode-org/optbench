"""The listed-atom displacement of an eOn saddle search.

``displace.ListedAtoms`` draws the epicenter uniformly from the listed
free atoms. ``Displace.get_displacement`` then moves the epicenter and
every atom closer than ``displace_radius`` to it in the reactant
(minimum image). Each free atom of that set gets a normal deviate of
standard deviation ``displace_magnitude`` on every coordinate. The
min-mode seed is a second, independent draw of the same kind on the
same atoms, scaled to unit length: ``[Saddle Search] random_mode``
defaults to true and the archived config.ini does not set it.

That is the 2014 code (eOn bf9b5cae), which produced the archived
searches. eOn has mixed a void bias into the displacement since commit
ac3def14 (2019-03-23), so this module has none.
"""

from __future__ import annotations

import numpy as np


def neighbours(reference, lengths, radius: float, atoms) -> dict[int, list[int]]:
    """Atoms closer than ``radius`` to each of ``atoms``."""
    points = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    box = np.asarray(lengths, dtype=np.float64)
    periodic = box > 0.0
    found = {}
    for atom in atoms:
        delta = points - points[int(atom)]
        delta[:, periodic] -= box[periodic] * np.round(delta[:, periodic] / box[periodic])
        distance = np.linalg.norm(delta, axis=1)
        found[int(atom)] = [int(other) for other in np.flatnonzero(distance < float(radius)) if other != int(atom)]
    return found


def displace(rng: np.random.Generator, free, listed, near: dict[int, list[int]], magnitude: float):
    """Return the epicenter, the ``(N, 3)`` displacement and the unit mode seed."""
    movable = np.asarray(free, dtype=bool)
    epicenter = int(listed[int(rng.integers(0, len(listed)))])
    moved = [atom for atom in [epicenter, *near[epicenter]] if movable[atom]]
    displacement = np.zeros((len(movable), 3), dtype=np.float64)
    for atom in moved:
        displacement[atom] = rng.normal(scale=float(magnitude), size=3)
    mode = np.zeros_like(displacement)
    for atom in moved:
        mode[atom] = rng.normal(scale=float(magnitude), size=3)
    return epicenter, displacement, mode / np.linalg.norm(mode)
