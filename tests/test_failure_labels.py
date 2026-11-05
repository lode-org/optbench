"""A failure class comes from the asv track when one exists, and is marked when the board infers it."""

from pathlib import Path

from optbench import board
from optbench.asv_score import track_rows

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = "double-ended-saddle-search/pt-island"
STEM = "ase-3.29.0-de-pt-island-default"
TRACK = ("ade7c5132e47bb4008b9d4c8ddb26492e9e7287d", "AseDePtIslandDefault")


def _problem() -> dict:
    return board.load_all(ROOT / "problems")[PROBLEM]


def _campaign_row(cap: int, force_tol: float | None) -> dict:
    group = board._campaign_groups(ROOT / "results" / "campaign")[STEM]
    records, _source = board._load_group_records(group, cap, force_tol)
    row = board._blank_row(problem=PROBLEM, template="ase/de-pt-island-default", source="campaign", cap=cap)
    board._apply_summary(row, board._score_records(records, cap), records)
    return row


def _asv_row(cap: int, force_tol: float | None) -> tuple[dict, list[dict]]:
    from asv.results import iter_results

    commit, class_name = TRACK
    result = next(item for item in iter_results(str(ROOT / "results" / "asv")) if item.commit_hash == commit)
    tracked = track_rows(result, class_name)
    records = []
    for item in tracked:
        raw = {"force_calls": item["force_calls"], "success": item["success"], "failure": item["failure"]}
        record = board._record_from_raw(raw, cap, force_tol)
        record["start"] = tuple(item["params"])
        records.append(record)
    row = board._blank_row(problem=PROBLEM, template="ase/de-pt-island-default", source="asv", cap=cap)
    board._apply_summary(row, board._score_records(records, cap), records)
    return row, tracked


def test_a_start_file_with_no_class_gets_an_inferred_one():
    problem = _problem()
    cap = board._cap_of(problem, None, {})
    force_tol = board._force_tol(problem, None)
    raw = {"failed": True, "force_calls": 3005, "force_norm": 0.19066037575930364, "start": "product_44"}
    record = board._record_from_raw(raw, cap, force_tol)
    assert record["failure"] == "cap"
    assert record["failure_inferred"] is True
    named = board._record_from_raw({**raw, "failure": "divergence"}, cap, force_tol)
    assert named["failure"] == "divergence"
    assert "failure_inferred" not in named


def test_the_campaign_row_marks_its_inferred_class():
    problem = _problem()
    row = _campaign_row(board._cap_of(problem, None, {}), board._force_tol(problem, None))
    assert row["failures"]["cap"] == 1
    assert row["failures_inferred"] == {"cap": 1}
    assert board.failure_cell(row) == "crash 0, divergence 0, cap 1 (1 inferred), wrong_stationary 0"


def test_the_asv_track_of_the_same_run_gives_the_class():
    problem = _problem()
    cap = board._cap_of(problem, None, {})
    force_tol = board._force_tol(problem, None)
    campaign = _campaign_row(cap, force_tol)
    asv, tracked = _asv_row(cap, force_tol)
    failed = [item for item in tracked if not item["success"]]
    assert [(item["params"][-1].strip("'"), item["failure"], item["force_calls"]) for item in failed] == [
        ("product_44", "divergence", 3005)
    ]
    assert board._dedup_key(campaign) == board._dedup_key(asv)
    merged = board._collapse_run([campaign, asv])
    assert merged["failures"] == {"crash": 0, "divergence": 1, "cap": 0, "wrong_stationary": 0}
    assert "failures_inferred" not in merged
    assert board.failure_cell(merged) == "crash 0, divergence 1, cap 0, wrong_stationary 0"


def test_the_problem_page_says_how_a_class_was_inferred():
    problem = _problem()
    row = _campaign_row(board._cap_of(problem, None, {}), board._force_tol(problem, None))
    row["label"] = "ASE (3.29.0) · FIRE"
    paragraph = board.inferred_paragraph({"id": PROBLEM, "ranked": [row]})
    assert 'data-board-note="inferred"' in paragraph
    assert "a start at or above the cap is cap" in paragraph
    assert "ASE (3.29.0) · FIRE" in paragraph
    assert board.inferred_paragraph({"id": PROBLEM, "ranked": [{"label": "x"}]}) == ""
