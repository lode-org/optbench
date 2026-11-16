"""A deferred score batch matches the inline scorer and stays in-process when small."""

from optbench.board import _ScoreSession, _blank_row, _make_start, _pack_score, _score_many, _score_records


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


def test_packed_fields_score_the_same() -> None:
    records = [
        _make_start(10.0, True, "a", per_image=1.0),
        _make_start(40.0, False, "b", per_image=2.0, failure="cap"),
        {"force_calls": 12.0, "success": False, "failure": None, "reasons": ["crash"], "per_image": 3.0},
    ]
    cap = 30
    assert _score_records(_pack_score(records), cap) == _score_records(records, cap)
