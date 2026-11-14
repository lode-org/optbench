"""A deferred score batch matches the inline scorer and stays in-process when small."""

from optbench.board import _ScoreSession, _blank_row, _score_many


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
