"""OGOLEM Sweden input text and failure classes."""

import math

import pytest

from optbench.runners.ogolem_go import (
    LJ38_STOP_ENERGY,
    center_cluster,
    classify_start,
    ogo_text,
)

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
