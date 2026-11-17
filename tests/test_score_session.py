"""A deferred score batch matches the inline scorer and stays in-process when small."""

import time

import pytest

from optbench.board import (
    _ScoreSession,
    _blank_row,
    _make_start,
    _pack_score,
    _score_jobs,
    _score_many,
    _score_records,
    _stop_pool,
)


def test_a_small_batch_matches_the_inline_scorer() -> None:
    records = [{"force_calls": 10.0, "success": True}]
    cap = 100
    session = _ScoreSession()
    try:
        batch = session.batch()
        batch.add((records, cap))
        row = _blank_row(entry="one")
        rows: list[dict] = []
        unparsed: list[dict] = []
        batch.hold([("score", row, records, cap)], rows, unparsed, None)
        assert session.pool is None
        session.finish()
    finally:
        session.close()
    direct = _score_many([(records, cap)])[0]
    assert rows[0]["par2_mean"] == direct["par2_mean"]
    assert rows[0]["median"] == direct["median"]
    assert rows[0]["n"] == direct["n"]
    assert unparsed == []


def test_a_long_list_keeps_its_place_among_short_ones(monkeypatch) -> None:
    """A long list is sent on its own, ahead of the short lists waiting with it.

    The summaries still come back in the order the lists were added.
    """
    monkeypatch.setattr("optbench.board._SCORE_HEAVY", 1000)

    def one(cost: float) -> list[dict]:
        return [{"force_calls": cost, "success": True}]

    def many(count: int) -> list[dict]:
        return [{"force_calls": float(i + 1), "success": True} for i in range(count)]

    jobs = [
        (one(1.0), 100),
        (one(2.0), 100),
        (many(2000), 100),
        (one(3.0), 100),
        (one(4.0), 100),
        (many(2000), 50),
        (one(5.0), 100),
    ]
    session = _ScoreSession()
    try:
        batch = session.batch()
        for job in jobs:
            batch.add(job)
        found = batch.collect()
    finally:
        session.close()
    direct = [_score_records(records, cap) for records, cap in jobs]
    assert [item["n"] for item in found] == [len(records) for records, _cap in jobs]
    assert found == direct


def test_an_empty_score_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="records is empty"):
        _score_records([], 10)


def test_a_failure_needs_a_cap() -> None:
    with pytest.raises(ValueError, match="a failure needs a cap"):
        _score_records([{"force_calls": 1.0, "success": False}], None)


def test_a_success_cannot_carry_a_failure_class() -> None:
    record = {"force_calls": 1.0, "success": True, "failure": "cap"}
    with pytest.raises(ValueError, match="a success cannot carry a failure class"):
        _score_records([record], 10)
    with pytest.raises(ValueError, match="a success cannot carry a failure class"):
        _score_records([(1.0, True, "cap", None, None)], 10)


def test_a_success_on_the_cap_keeps_its_count_and_a_failure_there_costs_twice_the_cap() -> None:
    summary = _score_records([{"force_calls": 10.0, "success": True}], 10)
    assert summary["nfailed"] == 0
    assert summary["par_applied"] is True
    assert summary["par2_mean"] == 10.0
    assert summary["success_only_mean"] == 10.0
    assert summary["n"] == 1
    assert summary["median"] == 10.0
    assert summary["success_rate"] == 1.0
    assert summary["rate_high"] == 1.0
    failed = _score_records([{"force_calls": 10.0, "success": False}], 10)
    assert failed["par2_mean"] == 20.0
    assert failed["median"] == 20.0
    assert failed["success_rate"] == 0.0


def test_an_unlabeled_failure_is_divergence_under_the_cap_and_cap_at_it() -> None:
    under = _score_records([{"force_calls": 3.0, "success": False}], 10)
    assert under["n_divergence"] == 1
    assert under["n_cap"] == 0
    assert under["par2_mean"] == 20.0
    reached = _score_records([{"force_calls": 10.0, "success": False}], 10)
    assert reached["n_cap"] == 1
    assert reached["n_divergence"] == 0
    assert reached["par2_mean"] == 20.0


def test_no_cap_leaves_the_par_mean_unset() -> None:
    summary = _score_records(
        [
            {"force_calls": 2.0, "success": True},
            {"force_calls": 4.0, "success": True},
        ],
        None,
    )
    assert summary["par_applied"] is False
    assert summary["par2_mean"] is None
    assert summary["success_only_mean"] == 3.0
    assert summary["mean_calls"] == 3.0
    assert summary["nfailed"] == 0


def test_every_failure_has_no_success_mean_and_a_zero_rate_floor() -> None:
    summary = _score_records(
        [
            {"force_calls": 1.0, "success": False, "failure": "crash"},
            {"force_calls": 2.0, "success": False, "failure": "divergence"},
        ],
        5,
    )
    assert summary["success_only_mean"] is None
    assert summary["nfailed"] == 2
    assert summary["rate_low"] == 0.0
    assert summary["n_crash"] == 1
    assert summary["n_divergence"] == 1
    assert summary["par2_mean"] == 10.0


def test_a_reason_list_and_a_packed_reason_list_name_the_same_class() -> None:
    cap = 10
    as_dict = _score_records([{"force_calls": 4.0, "success": False, "reasons": ["crash"]}], cap)
    packed = _score_records([(4.0, False, None, ["crash"], None)], cap)
    assert as_dict["n_crash"] == packed["n_crash"] == 1
    assert as_dict["par2_mean"] == packed["par2_mean"] == 20.0


def test_the_same_records_return_the_same_bootstrap_interval() -> None:
    records = [
        {"force_calls": 1.0, "success": True},
        {"force_calls": 9.0, "success": False, "failure": "cap"},
    ]
    first = _score_records(records, 10)
    second = _score_records(records, 10)
    assert first["median"] == second["median"] == 10.5
    assert first["median_low"] == second["median_low"]
    assert first["median_high"] == second["median_high"]


def test_two_samples_of_one_shape_match_a_fresh_score() -> None:
    from optbench.score import _index_block, score_v2

    first = [{"force_calls": float(value), "success": True} for value in (1, 4, 9, 2, 6)]
    second = [{"force_calls": float(value), "success": True} for value in (8, 3, 3, 7, 1)]
    cap = 20

    def fresh(records):
        if hasattr(_index_block, "_reuse"):
            del _index_block._reuse
        return score_v2(records, cap)

    alone_first = fresh(first)
    alone_second = fresh(second)
    if hasattr(_index_block, "_reuse"):
        del _index_block._reuse
    together_first = score_v2(first, cap)
    together_second = score_v2(second, cap)
    assert together_first == alone_first
    assert together_second == alone_second


def test_packed_fields_score_the_same() -> None:
    records = [
        _make_start(10.0, True, "a", per_image=1.0),
        _make_start(40.0, False, "b", per_image=2.0, failure="cap"),
        {"force_calls": 12.0, "success": False, "failure": None, "reasons": ["crash"], "per_image": 3.0},
    ]
    cap = 30
    assert _score_records(_pack_score(records), cap) == _score_records(records, cap)


def test_one_bad_list_does_not_drop_the_rest_of_the_message() -> None:
    jobs = [
        ([{"force_calls": 1.0, "success": True}], 10),
        ([{"success": True}], 10),
        ([{"force_calls": 4.0, "success": True}], 10),
    ]
    results = _score_jobs(jobs)
    assert results[0]["successes"] == 1
    assert isinstance(results[1], KeyError)
    assert results[2]["successes"] == 1
    assert results[2]["mean_calls"] == 4.0


def test_a_failed_score_message_keeps_the_lists_around_it() -> None:
    session = _ScoreSession()
    batch = session.batch()
    batch._seq = 3

    class _Future:
        def __init__(self, value, fail: bool = False) -> None:
            self.value = value
            self.fail = fail

        def result(self):
            if self.fail:
                raise RuntimeError("worker died")
            return self.value

    batch.futures = [
        ([0], _Future([{"n": 1}])),
        ([1], _Future(None, fail=True)),
        ([2], _Future([{"n": 3}])),
    ]
    found = batch.collect()
    assert found[0] == {"n": 1}
    assert isinstance(found[1], RuntimeError)
    assert str(found[1]) == "worker died"
    assert found[2] == {"n": 3}


def test_a_score_worker_that_exits_keeps_the_finished_list() -> None:
    import os
    from concurrent.futures import ProcessPoolExecutor

    def _die(_jobs):
        os._exit(1)

    session = _ScoreSession()
    session.pool = ProcessPoolExecutor(max_workers=1)
    try:
        batch = session.batch()
        batch._seq = 2
        good = session.pool.submit(
            _score_jobs,
            [([{"force_calls": 2.0, "success": True}], 10)],
        )
        dead = session.pool.submit(_die, [])
        batch.futures = [([0], good), ([1], dead)]
        found = batch.collect()
        assert found[0]["successes"] == 1
        assert isinstance(found[1], Exception)
    finally:
        session.close()


def test_closing_the_score_pool_stops_a_running_worker() -> None:
    from concurrent.futures import ProcessPoolExecutor

    session = _ScoreSession()
    session.pool = ProcessPoolExecutor(max_workers=1)
    session.pool.submit(time.sleep, 30)
    started = time.monotonic()
    session.close()
    assert time.monotonic() - started < 5
    assert session.pool is None


def test_stop_pool_reaps_the_child() -> None:
    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor(max_workers=1)
    pool.submit(time.sleep, 30)
    processes = list(pool._processes.values())
    assert processes and processes[0].is_alive()
    _stop_pool(pool)
    assert processes[0].exitcode is not None
    assert not processes[0].is_alive()
