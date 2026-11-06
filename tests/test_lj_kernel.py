"""The Lennard-Jones kernels give the same bits on any host.

``surfaces.lennard_jones`` and ``ase_lj.LennardJones`` take their powers
from the vendored SVML power routine instead of ``np.power``, which
NumPy dispatches by CPU. The pair loops and the GMIN kernel are plain
IEEE operations built for the x86-64-v2 baseline with no FMA
contraction. The digests are each kernel's energy and gradient on every
committed LJ38 start, so a host or a CPU path that changes a bit fails
here.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_eon_develop import _lj_cut_python, lj_cut
from optbench import ase_lj, fastpot, gmin_pot, lj, pes, potentials, surfaces
from optbench.campaign import geometry_archive

ROOT = Path(__file__).resolve().parents[1]
CON_ARCHIVES = (
    "website/tsopt/LJ38/coords-con.tar.gz",
    "website/lowest-eigenvector/lj38/coords-con.tar.gz",
    "website/minimization/lj38/lj38-clusters.tgz",
)
XYZ_ARCHIVES = (
    "website/double-ended-saddle-search/lj38/coords.tgz",
    "website/double-ended-saddle-search/lj38_2ts_separation/coords.tgz",
)
STARTS = 1602


def _ase(calculator, positions: np.ndarray) -> tuple[float, np.ndarray]:
    from ase import Atoms

    atoms = Atoms(symbols=["Ar"] * len(positions), positions=positions, pbc=False)
    atoms.calc = calculator(epsilon=1.0, sigma=1.0, rc=40.0)
    return atoms.get_potential_energy(), atoms.get_forces()


KERNELS = {
    "surfaces.lennard_jones": lambda pos: surfaces.lennard_jones(pos),
    "potentials.lennard_jones": lambda pos: potentials.lennard_jones(pos),
    "pes.lennard_jones": lambda pos: pes.lennard_jones(pos),
    "lj.lj_cluster": lambda pos: lj.lj_cluster(pos.ravel().tolist()),
    "lj_cut": lambda pos: lj_cut(pos.ravel().tolist(), epsilon=1.0, sigma=1.0, cutoff=40.0),
    "gmin_pot.lennard_jones_compiled": lambda pos: gmin_pot.lennard_jones_compiled(pos),
    "ase_lj.LennardJones": lambda pos: _ase(ase_lj.LennardJones, pos),
}
DIGESTS = {
    "surfaces.lennard_jones": "6cd5812079be60f4918f3a59c59fea0d7129cee15d22033a3e4b184f373840bc",
    "potentials.lennard_jones": "efaa319caaafbfc6dad0609d39f1b3c9209598be073c715451428573c3302013",
    "pes.lennard_jones": "296d44a0e93ae54ea7cc6b2c2c77315285e934c0947969bc711e55c85359d3d9",
    "lj.lj_cluster": "9cee6a3369f8b989f2bf9860f2245bb4001860d6b8c31890fa582d8802ac86cf",
    "lj_cut": "2b612182fff31f3ad18f7253930580cd3679e1fa78416618c8a002695a6584ce",
    "gmin_pot.lennard_jones_compiled": "418d146ecfe93531de15acc7ce19e343ba69029a95a63caf4f63d81a46c8fd63",
    "ase_lj.LennardJones": "9a8344edbf249ec7b04314097fc62a6f143a8ad403e101946d14660b3d2e76da",
}
# rgpot LJClusterPot cutoff 15 and the other cutoffs the harness passes.
SHIFTS = {
    15.0: "-0x1.790fc2e5a701ap-22",
    40.0: "-0x1.0c6f7a0a455f6p-30",
    2.5: "-0x1.0b5600734bfa5p-6",
    3.0: "-0x1.6719c71783017p-8",
}


@functools.cache
def _starts() -> tuple[tuple[str, np.ndarray], ...]:
    found = []
    for path in CON_ARCHIVES + XYZ_ARCHIVES:
        suffix = ".con" if path in CON_ARCHIVES else ".xyz"
        stored = geometry_archive(ROOT, {"id": path, "input": {"archive_path": path}})
        texts = {}
        with tarfile.open(stored) as packed:
            for member in packed:
                if member.isfile() and member.name.endswith(suffix):
                    texts[member.name] = packed.extractfile(member).read().decode("utf-8")
        for name in sorted(texts):
            positions = surfaces.read_con(texts[name])[0] if suffix == ".con" else pes.read_xyz(texts[name])
            found.append((f"{path}:{name}", positions))
    return tuple(found)


def _bits(energy, gradient) -> bytes:
    values = np.ascontiguousarray(np.asarray(gradient, dtype=np.float64))
    return np.float64(energy).tobytes() + values.tobytes()


def digests() -> tuple[int, dict[str, str]]:
    found = {name: hashlib.sha256() for name in KERNELS}
    count = 0
    for _name, positions in _starts():
        count += 1
        for kernel, evaluate in KERNELS.items():
            found[kernel].update(_bits(*evaluate(np.asarray(positions, dtype=np.float64))))
    return count, {name: digest.hexdigest() for name, digest in found.items()}


def test_the_builds_are_the_fixed_baseline():
    assert "-march=x86-64-v2" in fastpot._CFLAGS
    assert {"-O2", "-ffp-contract=off", "-fno-fast-math"} <= set(fastpot._CFLAGS)
    loose = {"-march=native", "-O3", "-Ofast", "-ffast-math", "-fopenmp"}
    assert loose.isdisjoint(fastpot._CFLAGS)


def test_every_kernel_has_its_digest_on_every_committed_start():
    assert digests() == (STARTS, DIGESTS)


def test_the_pair_loops_equal_their_python_references():
    assert fastpot.kernels() is not None
    differ = []
    for name, positions in _starts():
        flat = np.asarray(positions, dtype=np.float64).ravel().tolist()
        pairs = (
            (lj.lj_cluster(flat), lj._lj_cluster_python(flat)),
            (
                lj_cut(flat, epsilon=1.0, sigma=1.0, cutoff=40.0),
                _lj_cut_python(flat, epsilon=1.0, sigma=1.0, cutoff=40.0),
            ),
        )
        for native, python in pairs:
            if _bits(*native) != _bits(*python):
                differ.append(name)
    assert differ == []


def test_the_cutoff_shift_rounds_the_sixth_power_once():
    for cutoff, expected in SHIFTS.items():
        ratio = float(Fraction(1.0 / cutoff) ** 6)
        assert lj.lj_shift(1.0, 1.0, cutoff) == 4.0 * ratio * (ratio - 1.0)
        assert lj.lj_shift(1.0, 1.0, cutoff).hex() == expected


def _numpy_power_is_svml() -> bool:
    from numpy._core._multiarray_umath import __cpu_features__

    disabled = set(os.environ.get("NPY_DISABLE_CPU_FEATURES", "").replace(",", " ").split())
    return bool(__cpu_features__.get("X86_V4")) and "X86_V4" not in disabled


def test_shared_pow_is_numpy_power_where_numpy_runs_svml():
    if not _numpy_power_is_svml():
        pytest.skip("np.power runs SVML only on AVX-512 hosts with the X86_V4 dispatch")
    differ = 0
    for _name, positions in _starts():
        delta = positions[:, None, :] - positions[None, :, :]
        upper = np.triu_indices(len(positions), k=1)
        ratio = 1.0 / np.sqrt(np.einsum("ija,ija->ij", delta, delta)[upper])
        differ += int((surfaces.shared_pow(ratio, 6.0) != ratio**6).sum())
    rng = np.random.default_rng(20261103)
    values = rng.uniform(1.0e-3, 4.0, 100_000)
    for exponent in (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0, -6.0, 2.5):
        differ += int((surfaces.shared_pow(values, exponent) != np.power(values, exponent)).sum())
    assert differ == 0


def test_the_ase_calculator_is_ase_where_numpy_runs_svml():
    if not _numpy_power_is_svml():
        pytest.skip("np.power runs SVML only on AVX-512 hosts with the X86_V4 dispatch")
    from ase.calculators.lj import LennardJones

    differ = [
        name
        for name, positions in _starts()
        if _bits(*_ase(ase_lj.LennardJones, positions)) != _bits(*_ase(LennardJones, positions))
    ]
    assert differ == []


@pytest.mark.parametrize("change", ["smooth", "pbc"])
def test_the_ase_calculator_refuses_what_it_does_not_reproduce(change):
    from ase import Atoms

    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], cell=[10.0] * 3, pbc=change == "pbc")
    atoms.calc = ase_lj.LennardJones(rc=3.0, smooth=change == "smooth")
    with pytest.raises(NotImplementedError):
        atoms.get_forces()


def _dispatch_targets() -> str:
    from numpy._core._multiarray_umath import __cpu_dispatch__

    return " ".join(__cpu_dispatch__)


@pytest.mark.parametrize(
    "toggles",
    [
        {"GLIBC_TUNABLES": "glibc.cpu.hwcaps=-AVX2,-FMA,-AVX512F"},
        {"NPY_DISABLE_CPU_FEATURES": "dispatch"},
        {"GLIBC_TUNABLES": "glibc.cpu.hwcaps=-AVX2,-FMA,-AVX512F", "NPY_DISABLE_CPU_FEATURES": "dispatch"},
    ],
    ids=["glibc-baseline", "numpy-baseline", "both"],
)
def test_cpu_dispatch_does_not_change_a_bit(toggles):
    env = dict(os.environ)
    for key, value in toggles.items():
        env[key] = _dispatch_targets() if value == "dispatch" else value
    script = (
        "import json, runpy; "
        f"print(json.dumps(runpy.run_path({str(Path(__file__))!r})['digests']()))"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    count, found = json.loads(done.stdout)
    assert (count, found) == (STARTS, DIGESTS)
