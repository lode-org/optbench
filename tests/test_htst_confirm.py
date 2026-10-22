"""Job ids 0-7 are reused by every batch. The score follows each batch."""

from optbench.htst_confirm import BARRIERS, score_rows

POTSERVER = 1712105


def _process_table() -> list[str]:
    header = "proc # saddle energy prefactor product product energy product prefactor barrier rate repeats"
    rows = [header]
    for number, barrier in enumerate(BARRIERS, start=1):
        rows.append(f"{number} -1 1 -1 -1 1 {barrier:.5f} 1 0")
    return rows


def _search(wuid: int, sad: int, result: str) -> list[str]:
    return [str(wuid), "random", "0.60100", "1.0", str(sad), "1", "0", result]


def _batch(order: list[int], sad_for, result_for) -> list[list[str]]:
    return [_search(wuid, sad_for(wuid), result_for(wuid)) for wuid in order]


def _rows(order_for_batch) -> list[list[str]]:
    """15 reference saddles, then a reused id whose calls sit outside that window."""
    rows = []
    # Batch 0, ids 0-7: reference processes 1-8.
    rows.extend(
        _batch(
            order_for_batch(),
            lambda _wuid: 10,
            lambda wuid: f"good-{wuid + 1}",
        )
    )
    # Batch 1, ids 0-7: processes 9-15, then one search that is not a reference.
    rows.extend(
        _batch(
            order_for_batch(),
            lambda wuid: 10 if wuid < 7 else 1000,
            lambda wuid: f"good-{wuid + 9}" if wuid < 7 else "missed",
        )
    )
    # Batch 2 reuses ids 0-7. None of these are reference saddles.
    rows.extend(
        _batch(
            order_for_batch(),
            lambda _wuid: 1000,
            lambda _wuid: "missed",
        )
    )
    return rows


def test_reused_job_ids_do_not_change_the_batch_score():
    processes = _process_table()
    completed = _rows(lambda: list(range(7, -1, -1)))
    already_ordered = _rows(lambda: list(range(8)))

    by_completion = score_rows(completed, processes, batch=8, potserver=POTSERVER)
    by_seed_file = score_rows(already_ordered, processes, batch=8, potserver=POTSERVER)
    # One chunk sorted by job id is the sort that mixes every batch together.
    mixed = score_rows(completed, processes, batch=len(completed), potserver=999)

    assert by_completion["barrier_jobs"] == 15
    assert by_completion["barrier_force_calls"] == 150
    assert by_completion["potserver"] == POTSERVER
    assert by_seed_file["barrier_jobs"] == by_completion["barrier_jobs"]
    assert by_seed_file["barrier_force_calls"] == by_completion["barrier_force_calls"]
    assert by_seed_file["potserver"] == POTSERVER
    assert mixed["barrier_force_calls"] != by_completion["barrier_force_calls"]
    assert mixed["potserver"] == 999
