"""eOn SVN templates stay on the r2549 pin and the problem grids."""

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tomllib

from benchmarks.bench_eon_develop import lj_cut
from benchmarks.bench_eon_svn import (
    MORSE_A,
    MORSE_CUTOFF,
    MORSE_D,
    MORSE_RE,
    SPECS,
    _morse_fast,
    _morse_path,
)
from optbench.campaign import _check_stop, load_template, resolve_problem
from optbench.potentials import lennard_jones

ROOT = Path(__file__).resolve().parents[1]


def _eon_morse(positions, box=None):
    """Python copy of r2549 Morse::force, for the C kernel check."""
    import math

    atoms = len(positions) // 3
    bx = by = bz = 0.0
    if box is not None:
        bx, by, bz = box[0], box[4], box[8]
    gap_c = 1.0 - math.exp(-MORSE_A * (MORSE_CUTOFF - MORSE_RE))
    ecut = MORSE_D * (gap_c * gap_c - 1.0)
    energy = 0.0
    forces = [0.0] * len(positions)
    for i in range(atoms):
        ix = 3 * i
        for j in range(i + 1, atoms):
            jx = 3 * j
            dx = positions[ix] - positions[jx]
            dy = positions[ix + 1] - positions[jx + 1]
            dz = positions[ix + 2] - positions[jx + 2]
            if bx > 0.0:
                dx -= bx * math.floor(dx / bx + 0.5)
            if by > 0.0:
                dy -= by * math.floor(dy / by + 0.5)
            if bz > 0.0:
                dz -= bz * math.floor(dz / bz + 0.5)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                continue
            r = math.sqrt(r2)
            if r >= MORSE_CUTOFF:
                continue
            gap = 1.0 - math.exp(-MORSE_A * (r - MORSE_RE))
            energy += MORSE_D * (gap * gap - 1.0) - ecut
            bond = 2.0 * MORSE_D * gap * (gap - 1.0) * MORSE_A
            scale = bond / r
            fx, fy, fz = scale * dx, scale * dy, scale * dz
            forces[ix] += fx
            forces[ix + 1] += fy
            forces[ix + 2] += fz
            forces[jx] -= fx
            forces[jx + 1] -= fy
            forces[jx + 2] -= fz
    return energy, tuple(forces)


def test_c_morse_matches_r2549_morse():
    positions = []
    for i in range(12):
        positions.extend((0.4 * i, 0.15 * (i % 3), 0.2 * (i % 5)))
    box = [8.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 12.0]
    energy, forces = _eon_morse(positions, box)
    other, fast = _morse_fast(positions, box=box)
    assert abs(energy - other) < 1e-9
    assert len(fast) == len(forces)
    for left, right in zip(fast, forces, strict=True):
        assert abs(left - right) < 1e-8
    bare, _bare_forces = _morse_fast(positions)
    assert abs(bare - _eon_morse(positions)[0]) < 1e-9


def test_concurrent_morse_builds_share_one_library(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTBENCH_FASTPOT_DIR", str(tmp_path))
    workers = 6
    gate = threading.Barrier(workers)

    def build(_index):
        gate.wait()
        return _morse_path()

    with ThreadPoolExecutor(workers) as pool:
        paths = set(pool.map(build, range(workers)))
    assert len(paths) == 1
    (library,) = paths
    assert [entry.name for entry in tmp_path.iterdir()] == [library.name]
    assert ctypes.CDLL(str(library)).morse_eon_flat


def test_lj_cut_matches_the_noise_kernel():
    positions = [0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.2, 1.3, 0.4]
    energy, forces = lj_cut(positions, cutoff=40.0)
    other, gradient = lennard_jones(positions)
    assert abs(energy - other) < 1e-9
    assert len(forces) == len(gradient)
    for left, right in zip(forces, gradient, strict=True):
        assert abs(left + float(right)) < 1e-8


def test_svn_templates_match_the_problem_grids():
    by_template = {spec["template"]: spec for spec in SPECS}
    assert len(by_template) == 21
    expected = {
        "eon-svn/ts-lj38-dimer": 200,
        "eon-svn/ts-lj38-lanczos": 200,
        "eon-svn/ts-pt-dimer": 49,
        "eon-svn/ts-pt-lanczos": 49,
        "eon-svn/de-lj38-ci": 50,
        "eon-svn/de-lj38-neb": 50,
        "eon-svn/de-pt-ci": 59,
        "eon-svn/de-pt-neb": 59,
        "eon-svn/de-2ts-ci": 51,
        "eon-svn/de-2ts-neb": 51,
        "eon-svn/neb-pt-ci": 13,
        "eon-svn/neb-pt-neb": 13,
        "eon-svn/low-lj38-dimer": 200,
        "eon-svn/low-lj38-lanczos": 200,
        "eon-svn/lj38-noise-0": 1000,
        "eon-svn/lj38-noise-1e-4": 1000,
        "eon-svn/lj38-noise-1e-3": 1000,
        "eon-svn/lj38-noise-1e-2": 1000,
        "eon-svn/saddle-pt-dimer": 10,
        "eon-svn/saddle-pt-dimer-improved": 10,
        "eon-svn/saddle-pt-lanczos": 2,
    }
    assert set(by_template) == set(expected)
    for template_id, count in expected.items():
        spec = by_template[template_id]
        template = load_template(ROOT, template_id)
        assert template["commit"] == "r2549"
        assert template["code"] == "Eon svn"
        assert template["engine"] == "eon-client"
        assert template["problem"]
        _check_stop(resolve_problem(ROOT, template["problem"]), template)
        if spec["kind"] == "akmc":
            assert len(spec["campaign_seeds"]) == count
            assert spec["campaign_starts"] == ["pt-island"]
        else:
            assert spec["campaign_seeds"]
            assert len(spec["campaign_starts"]) == count
        optimizer = template["ini"].get("Optimizer") or {}
        if optimizer.get("opt_method") == "lbfgs":
            assert "LBFGS" not in template["ini"]
            assert optimizer["lbfgs_memory"] == 20
            assert "lbfgs_max_inverse_curvature" in optimizer
        assert "ci_mmf" not in template["ini"].get("Nudged Elastic Band", {})
        assert "energy_weighted" not in template["ini"].get("Nudged Elastic Band", {})


def test_registry_pins_r2549():
    data = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in data["code"] if item["id"] == "eon-svn")
    assert row["commit"] == "r2549"
    assert row["name"] == "Eon svn"
