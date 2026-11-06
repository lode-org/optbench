"""The Morse pair loop equals its NumPy reference bit for bit on any host.

Both call the one vendored exp, built for the x86-64-v2 baseline with
no FMA contraction, so neither glibc's nor NumPy's choice of code path
by CPU can change a bit. The digest is the kernel's output on every
committed Morse start, so a host that differs fails here.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import geometry_archive
from optbench.surfaces import NATIVE_FLAGS, _morse_numpy, morse_smooth_linear, read_con

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = (
    "website/tsopt/pt-island/pt-island-con.tgz",
    "website/double-ended-saddle-search/pt-island/pt-island-con.tgz",
    "website/saddle-search/pt-island/pt-island.con",
    "website/hTST/Pt/eon-dimer-cg/pos.con",
)
PARAMETERS = {"depth": 0.7102, "alpha": 1.6047, "r0": 2.897, "cutoff": 9.5}
DIGEST = "ceca24dffa39abb7dfe8b7f4f4b60988328c515d462669c4e007bfd0c774ae53"


def _starts():
    for path in ARCHIVES:
        stored = geometry_archive(ROOT, {"id": path, "input": {"archive_path": path}})
        if not tarfile.is_tarfile(stored):
            yield path, stored.read_text(encoding="utf-8")
            continue
        with tarfile.open(stored) as packed:
            for member in packed.getmembers():
                if member.isfile() and member.name.endswith(".con"):
                    yield f"{path}:{member.name}", packed.extractfile(member).read().decode("utf-8")


def _bits(energy: float, gradient: np.ndarray) -> bytes:
    return np.float64(energy).tobytes() + np.ascontiguousarray(gradient).tobytes()


def digests() -> tuple[str, str, list[str]]:
    kernel = hashlib.sha256()
    reference = hashlib.sha256()
    differ = []
    for name, text in _starts():
        positions, _fixed, cell = read_con(text)
        for pbc in ((True, True, False), (False, False, False)):
            got = _bits(*morse_smooth_linear(positions, cell=cell, pbc=pbc, **PARAMETERS))
            want = _bits(*_morse_numpy(positions, cell=cell, pbc=pbc, **PARAMETERS))
            kernel.update(got)
            reference.update(want)
            if got != want:
                differ.append(f"{name} pbc={pbc}")
    return kernel.hexdigest(), reference.hexdigest(), differ


def test_the_build_is_the_fixed_baseline():
    assert "-march=x86-64-v2" in NATIVE_FLAGS
    assert {"-O2", "-ffp-contract=off", "-fno-fast-math"} <= set(NATIVE_FLAGS)
    loose = {"-march=native", "-O3", "-Ofast", "-ffast-math", "-fopenmp"}
    assert loose.isdisjoint(NATIVE_FLAGS)


def test_the_pair_loop_equals_the_reference_on_every_committed_start():
    kernel, reference, differ = digests()
    assert differ == []
    assert kernel == reference == DIGEST


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
        "import runpy, sys; "
        f"found = runpy.run_path({str(Path(__file__))!r})['digests'](); "
        "print(found[0], found[1], len(found[2]))"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.split() == [DIGEST, DIGEST, "0"]
