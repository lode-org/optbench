"""OGOLEM harness socket and Sweden input."""

import math
import socket

import pytest

from optbench.potentials import lennard_jones
from optbench.runners.ogolem_go import (
    LJ38_STOP_ENERGY,
    HarnessLJServer,
    center_cluster,
    classify_start,
    ogo_text,
)

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
