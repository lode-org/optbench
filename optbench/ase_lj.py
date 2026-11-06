"""ASE's shifted Lennard-Jones calculator with the same bits on every host.

``ase.calculators.lj.LennardJones`` (ASE 3.29.0) raises each pair's
``σ²/r²`` to the third power with ``np.power``, which NumPy dispatches by
CPU, and takes the cutoff shift from libm ``pow``. On the committed LJ38
starts its energies and forces change with the host's vector extensions.
This calculator does ASE's operations in ASE's order, with the third
power from :func:`optbench.surfaces.shared_pow` and the cutoff powers
rounded once from their exact rational values. It therefore gives ASE's
bits on hosts where NumPy runs its AVX-512 power loop, and those bits on
every other host.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones as AseLennardJones
from ase.neighborlist import NeighborList

from optbench.surfaces import shared_pow


class LennardJones(AseLennardJones):
    """Energy and forces of a finite cluster, shifted to zero at ``rc``.

    The parameters and defaults are ASE's. The smooth cutoff and periodic
    cells are not supported, and the stress is not computed.
    """

    implemented_properties = ["energy", "energies", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.parameters.smooth:
            raise NotImplementedError("the smooth cutoff is not supported")
        if self.atoms.pbc.any():
            raise NotImplementedError("periodic cells are not supported")

        natoms = len(self.atoms)
        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        rc = self.parameters.rc
        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)
        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell
        ratio = Fraction(sigma / rc)
        e0 = 4 * epsilon * (float(ratio**12) - float(ratio**6))
        sigma2 = sigma * sigma
        rc2 = rc * rc

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            vectors = positions[neighbors] + np.dot(offsets, cell) - positions[ii]
            r2 = (vectors**2).sum(1)
            c6 = shared_pow(sigma2 / r2, 3.0)
            c6[r2 > rc2] = 0.0
            c12 = c6**2
            pair_energies = 4 * epsilon * (c12 - c6)
            pair_forces = -24 * epsilon * (2 * c12 - c6) / r2
            pair_energies -= e0 * (c6 != 0.0)
            energies[ii] += 0.5 * pair_energies.sum()
            forces[ii] += (pair_forces[:, np.newaxis] * vectors).sum(axis=0)

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["energies"] = energies
        self.results["free_energy"] = energy
        self.results["forces"] = forces
