"""OGOLEM row text, archives, and the harness socket."""

import math
import socket
from pathlib import Path

import pytest

from optbench.campaign import load_template, resolve_problem
from optbench.potentials import lennard_jones
from optbench.runners.ogolem_go import (
    ANGTOBOHR,
    AR_SIGMA_BOHR,
    COMMIT,
    LJ38_STOP_ENERGY,
    XYZ_SCALE,
    HarnessLJServer,
    _force_cap,
    _partial_reason,
    _repeated_energy,
    _threshold,
    _write_outputs,
    center_cluster,
    classify_start,
    load_starts,
    ogo_text,
)

ROOT = Path(__file__).resolve().parents[1]

def _exchange(server: HarnessLJServer, line: str) -> str:
    host, port = server.server_address
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        reader = sock.makefile("rwb")
        reader.write(b"hello\n")
        reader.flush()
        assert reader.readline() == b"ok\n"
        reader.write(line.encode("ascii") + b"\n")
        reader.flush()
        return reader.readline().decode("ascii")
def test_server_matches_lennard_jones_and_caches_and_caps():
    server = HarnessLJServer(cap=2, threshold=None)
    try:
        first = _exchange(server, "calculate 2 0.0 0.0 0.0 1.5 0.0 0.0")
        parts = first.split()
        energy, gradient = lennard_jones([0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
        assert parts[0] == "ok"
        assert int(parts[1]) == 1
        assert float(parts[2]) == pytest.approx(energy)
        assert [float(value) for value in parts[3:]] == pytest.approx(gradient.tolist())
        repeat = _exchange(server, "calculate 2 0.0 0.0 0.0 1.5 0.0 0.0")
        assert int(repeat.split()[1]) == 1
        second = _exchange(server, "calculate 2 0.0 0.0 0.0 2.5 0.0 0.0")
        assert int(second.split()[1]) == 2
        capped = _exchange(server, "calculate 2 0.0 0.0 0.0 3.5 0.0 0.0")
        assert capped.startswith("cap 2")
        assert server.counter.calls == 2
    finally:
        server.close()

def test_lj38_stop_is_the_archived_eon_threshold():
    template = load_template(ROOT, "ogolem/lj38")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    assert template["commit"] == COMMIT
    assert template["failure_cap"] == 2000000
    assert float(template["reference_energy"]) + float(problem["convergence"]["energy_above_minimum"]) == pytest.approx(
        LJ38_STOP_ENERGY
    )

def test_larger_clusters_use_the_cambridge_window_and_the_problem_cap():
    expected = {
        "lj75": (-397.492331, 1_000_000, 200),
        "lj98": (-543.665361, 1_000_000_000, 100),
        "lj150": (-893.310258, 10_000_000, 100),
    }
    for slug, (energy, cap, starts) in expected.items():
        problem = resolve_problem(ROOT, f"globalopt/{slug}")
        template = load_template(ROOT, f"ogolem/{slug}")
        assert float(problem["reference"]["energy"]) == energy
        assert int(problem["sample"]["starts"]) == starts
        assert _force_cap(problem, template) == cap
        assert _threshold(problem, template) == pytest.approx(energy + 0.001)
    lj38 = resolve_problem(ROOT, "globalopt/lj38")
    assert _threshold(lj38, load_template(ROOT, "ogolem/lj38")) == pytest.approx(LJ38_STOP_ENERGY)
    assert _force_cap(lj38, load_template(ROOT, "ogolem/lj38")) == 2_000_000

def test_archive_counts_do_not_double_count_xyz(tmp_path: Path):
    lj38 = load_starts(ROOT, resolve_problem(ROOT, "globalopt/lj38"), load_template(ROOT, "ogolem/lj38"), tmp_path)
    assert len(lj38) == 100
    assert [name for name, _pos in lj38].count("0.con") == 1
    assert all(name.endswith(".con") for name, _pos in lj38)
    lj75 = load_starts(ROOT, resolve_problem(ROOT, "globalopt/lj75"), load_template(ROOT, "ogolem/lj75"), tmp_path)
    lj98 = load_starts(ROOT, resolve_problem(ROOT, "globalopt/lj98"), load_template(ROOT, "ogolem/lj98"), tmp_path)
    lj150 = load_starts(ROOT, resolve_problem(ROOT, "globalopt/lj150"), load_template(ROOT, "ogolem/lj150"), tmp_path)
    assert len(lj75) == 200
    assert len(lj98) == 100
    assert len(lj150) == 100
    assert lj75[0][1].shape[0] == 75
    assert lj98[0][1].shape[0] == 98
    assert lj150[0][1].shape[0] == 150

def test_length_unit_is_the_argon_sigma():
    # Contact at this sigma is 2**(1/6)*6.434516991 bohr, above 1.2*(1.85+1.85).
    assert AR_SIGMA_BOHR == 6.434516991
    assert XYZ_SCALE * ANGTOBOHR == pytest.approx(AR_SIGMA_BOHR)
    assert load_template(ROOT, "ogolem/lj38")["ogolem"]["cell_bohr"] == 60

def test_center_and_ogo_text_follow_the_sweden_example():
    import numpy as np

    shifted = np.array([[50.0, 51.0, 52.0], [52.0, 51.0, 52.0]])
    centered = center_cluster(shifted)
    assert centered.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])
    text = ogo_text(atoms=38, pool=100, iterations=19900, cell=18, seed=3, threshold=LJ38_STOP_ENERGY)
    assert "Backend=xyz:optbench" in text
    assert "CellSize=18;18;18" in text
    assert "PoolSize=100" in text
    assert "NumberOfGlobIterations=19900" in text
    assert "cutstyle=2" in text
    assert "RandomNumberGenerator=javarng:seed=3" in text
    assert f"AcceptableFitness={LJ38_STOP_ENERGY:.16g}" in text
    plain = ogo_text(atoms=75, pool=100, iterations=19900, cell=18, seed=1, threshold=None)
    assert "AcceptableFitness" not in plain
    with pytest.raises(ValueError):
        ogo_text(atoms=2, pool=1, iterations=0, cell=18, seed=1, threshold=None)

def test_classify_start_uses_the_energy_window_and_the_cap():
    assert classify_start(exit_code=0, calls=10, energy=-173.928, cap=100, threshold=LJ38_STOP_ENERGY, hit_cap=False)[0]
    success, failure, _reasons = classify_start(
        exit_code=0, calls=10, energy=-170.0, cap=100, threshold=LJ38_STOP_ENERGY, hit_cap=False
    )
    assert success is False
    assert failure == "wrong_stationary"
    _success, failure, _reasons = classify_start(
        exit_code=0, calls=100, energy=-173.928, cap=100, threshold=LJ38_STOP_ENERGY, hit_cap=False
    )
    assert failure == "cap"
    _success, failure, _reasons = classify_start(
        exit_code=0, calls=10, energy=-400.0, cap=100, threshold=None, hit_cap=False
    )
    assert failure == "wrong_stationary"
    _success, failure, _reasons = classify_start(
        exit_code=1, calls=0, energy=None, cap=100, threshold=LJ38_STOP_ENERGY, hit_cap=False
    )
    assert failure == "crash"
    _success, failure, _reasons = classify_start(
        exit_code=0, calls=4, energy=math.nan, cap=100, threshold=LJ38_STOP_ENERGY, hit_cap=False
    )
    assert failure == "divergence"

def test_partial_reason_counts_starts_on_the_lowest_energy():
    problem = {"id": "globalopt/lj75"}
    records = [
        {"energy": -396.2822488255712, "force_calls": 1000000},
        {"energy": -396.2822488255711, "force_calls": 1000000},
        {"energy": -392.6594685996733, "force_calls": 1000000},
    ]
    reason = _partial_reason(problem, records, True, 200)
    assert reason == (
        "scored 3 of 200 starts; lowest energy -396.28224883; "
        "2 of 3 starts at -396.28224883; mean calls 1000000.0; median calls 1000000.0"
    )
    hit = dict(records[0])
    hit["energy"] = -397.4913461429096
    hit["force_calls"] = 544484
    mixed = _partial_reason(problem, [hit, *records], True, 200)
    assert mixed == (
        "scored 4 of 200 starts; lowest energy -397.49134614; "
        "2 of 4 starts at -396.28224883; mean calls 886121.0; median calls 1000000.0"
    )

def test_a_complete_grid_records_a_repeated_energy_that_is_not_the_lowest(tmp_path: Path):
    records = [
        {"energy": -397.4913461429096, "force_calls": 544484, "success": True, "failure": None, "start": "coords.1", "index": 1},
        {"energy": -396.2822488255712, "force_calls": 1000000, "success": False, "failure": "cap", "start": "coords.2", "index": 2},
        {"energy": -396.2822488255711, "force_calls": 1000000, "success": False, "failure": "cap", "start": "coords.3", "index": 3},
        {"energy": -396.2822488255712, "force_calls": 1000000, "success": False, "failure": "cap", "start": "coords.4", "index": 4},
    ]
    assert _repeated_energy(records) == "3 of 4 starts at -396.28224883; lowest energy -397.49134614"
    assert _repeated_energy(records[1:]) == ""
    problem = {"id": "globalopt/lj75", "sample": {"starts": 4}}
    template = {
        "id": "ogolem/lj75",
        "code": "ogolem",
        "code_version": "2016.CURRENT",
        "commit": COMMIT,
        "algorithm": "genetic algorithm",
        "problem": "globalopt/lj75",
        "tuning": "default",
        "failure_cap": 1000000,
        "note": "published settings",
    }
    output = tmp_path / "ogolem-lj75.dat"
    _write_outputs(output, records, template, problem, 1000000, True)
    text = output.read_text(encoding="utf-8")
    assert "partial true" not in text
    assert "finding 3 of 4 starts at -396.28224883; lowest energy -397.49134614" in text
