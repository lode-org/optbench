"""rgmin ASV classes follow the shared parameter and track names."""

import pytest

import re

from benchmarks.bench_rgmin import (
    SEED,
    asv_benchmark_pattern,
    selected_indices,
    RgminLj38Bb,
    RgminLj38Lbfgs,
    RgminMorseLbfgs,
    _specs,
)
from optbench.failures import FAILURES


def test_rgmin_asv_params_are_seed_and_start_index(monkeypatch):
    assert RgminLj38Lbfgs.param_names == ["seed", "start"]
    assert RgminLj38Lbfgs.params[0] == [SEED]
    assert RgminLj38Lbfgs.params[1] == ["0"]
    assert RgminMorseLbfgs.params[1] == ["0"]
    assert RgminLj38Lbfgs.repeat == 1
    assert RgminLj38Lbfgs.rounds == 1
    assert RgminLj38Lbfgs.number == 1
    assert RgminLj38Lbfgs.warmup_time == 0
    assert RgminLj38Lbfgs.sample_time == 0
    assert RgminLj38Lbfgs.cap == 10000
    assert RgminLj38Lbfgs.track_force_calls.unit == "force calls"
    assert RgminLj38Lbfgs.track_success.unit == "fraction"
    for cause in FAILURES:
        method = getattr(RgminLj38Lbfgs, f"track_failure_{cause}")
        assert method.unit == "count"
    assert selected_indices(1000) == ["0"]
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert selected_indices(1000) == [str(index) for index in range(1000)]
    assert selected_indices(100) == [str(index) for index in range(100)]


def test_asv_class_filter_is_anchored_and_literal():
    names = [spec["class_name"] for spec in _specs()]
    assert len(names) == 36
    assert len(set(names)) == 36
    for name in names:
        pattern = re.compile(asv_benchmark_pattern(name))
        assert pattern.search(f"bench_rgmin.{name}.time_search('0', '0')")
        for other in names:
            if other != name:
                assert pattern.search(f"bench_rgmin.{other}.time_search('0', '0')") is None


def test_default_lj38_bb_does_not_leave_the_cutoff():
    """An uncapped ``istep * g`` trial on cluster 0000 is a vacuum in two calls.

    The start sits in the repulsive core. rgmin's default session takes one
    trial of ``-istep * g`` with ``istep`` 0.1 and ``Accept::None``. Without
    the per-atom cap that step moves atoms past the cutoff of 40, the next
    evaluation is exactly zero, and a zero Hessian is index 0. The LJ38
    default row sets ``maxstep`` 0.2, the same cap as eOn ``max_move`` and
    the rgmin ASE wrapper.
    """
    pytest.importorskip("rgmin")
    bench = RgminLj38Bb()
    bench.setup(SEED, "0")
    bench.time_search(SEED, "0")
    record = bench.record
    assert record["force_calls"] > 10
    assert record["energy"] < -10.0
    assert bench.track_force_calls(SEED, "0") == record["force_calls"]


def test_lj38_lbfgs_index_drops_rigid_rotation_and_keeps_a_real_saddle():
    """Start 1 is a minimum whose raw negative curvature is rotational.

    Start 827 keeps one vibrational eigenvalue below -1 after the same
    projection. Both finals are the capped L-BFGS row.
    """
    pytest.importorskip("rgmin")
    from benchmarks.bench_rgmin import _job, _specs
    from optbench.runners.rgmin_min import _minimize_job

    spec = next(row for row in _specs() if row["template_id"] == "rgmin/lj38-lbfgs")
    minimum = _minimize_job(_job(spec, "1"))
    assert minimum["force_calls"] == 375
    assert minimum["energy"] == pytest.approx(-169.737668749, abs=1e-6)
    assert minimum["hessian_index"] == 0
    assert minimum["success"] is True
    assert minimum["min_eigenvalues"][0] > -0.01

    saddle = _minimize_job(_job(spec, "827"))
    assert saddle["force_calls"] == 133
    assert saddle["energy"] == pytest.approx(-160.842180602, abs=1e-6)
    assert saddle["hessian_index"] == 1
    assert saddle["failure"] == "wrong_stationary"
    assert saddle["min_eigenvalues"][0] < -1.0


def test_one_lbfgs_start_is_a_harness_count():
    pytest.importorskip("rgmin")
    bench = RgminLj38Lbfgs()
    bench.setup(SEED, "0")
    bench.time_search(SEED, "0")
    calls = bench.track_force_calls(SEED, "0")
    assert calls == int(calls)
    assert 0 < calls < 10000
    assert bench.record["counter"] == "harness"
    assert bench.record["failure"] in (None, *FAILURES)
    assert bench.track_success(SEED, "0") == (1.0 if bench.record["success"] else 0.0)
    assert bench.track_failure_cap(SEED, "0") == (1.0 if bench.record["failure"] == "cap" else 0.0)
