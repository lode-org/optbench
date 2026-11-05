import html
from pathlib import Path

from optbench import board
from optbench.check import load_oracle
from optbench.differences import difference_errors, difference_note, load_differences

ROOT = Path(__file__).resolve().parents[1]
LABELLED = {
    "minimization/lj38": "eon-fire",
    "double-ended-saddle-search/pt-island": "eon-lanczos",
}


def _archive_rows(problem_id: str) -> list[dict]:
    problems = board.load_all(ROOT / "problems")
    registry = board._registry(ROOT)
    full_n = {problem_id: board.stated_full_n(ROOT, problems[problem_id])}
    rows = [row for row in board._archive_rows(ROOT, problems, registry, full_n) if row["problem"] == problem_id]
    for row in rows:
        board._annotate_row(row, problems[problem_id], registry)
    return rows


def test_every_difference_names_its_archived_and_printed_values_and_evidence():
    entries = load_oracle(ROOT / "oracle" / "entries.jsonl")
    assert difference_errors(ROOT, entries) == []
    assert {item["path"].split("/")[-2] for item in load_differences(ROOT)} == set(LABELLED.values())


def test_only_the_two_rows_that_differ_from_the_paper_carry_a_note():
    for problem_id, entry in LABELLED.items():
        rows = _archive_rows(problem_id)
        labelled = [row["entry"] for row in rows if row.get("difference")]
        assert labelled == [entry]
        assert all("difference" not in row for row in rows if row["entry"] != entry)


def test_the_note_is_printed_beside_the_row():
    rows = _archive_rows("minimization/lj38")
    problem = {"id": "minimization/lj38", "ranked": rows}
    paragraph = board.difference_paragraph(problem)
    assert 'data-board-note="differences"' in paragraph
    fire = next(row for row in rows if row.get("difference"))
    assert html.escape(board.difference_text(fire)) in paragraph
    assert "645, 207 and 2963" in paragraph
    assert board.difference_paragraph({"id": "x", "ranked": [{"label": "a"}]}) == ""


def test_a_note_applies_only_while_the_archive_has_its_values(tmp_path: Path):
    (tmp_path / "oracle").mkdir()
    (tmp_path / "oracle" / "differences.toml").write_text(
        '[[difference]]\npath = "a/benchmark.dat"\ntable = "Table 1"\nnote = "differs"\nevidence = "report.sh"\n'
        '[difference.archived]\nforce_calls = "731"\n[difference.printed]\nforce_calls = "645"\n'
        '[[difference]]\npath = "b/benchmark.dat"\ntable = "Table 2"\nnote = "differs"\nevidence = ""\n'
        '[difference.archived]\nforce_calls = "10"\n[difference.printed]\nforce_calls = "10"\n',
        encoding="utf-8",
    )
    differences = load_differences(tmp_path)
    assert difference_note("a/benchmark.dat", {"force_calls": "731"}, differences) == "differs"
    assert difference_note("a/benchmark.dat", {"force_calls": "645"}, differences) == ""
    entries = [{"path": "a/benchmark.dat", "fields": {"force_calls": "645"}}]
    assert difference_errors(tmp_path, entries) == [
        "oracle/differences.toml a/benchmark.dat expects force_calls '731', the archive has '645'",
        "oracle/differences.toml b/benchmark.dat has no evidence",
        "oracle/differences.toml b/benchmark.dat prints the archived values",
        "oracle/differences.toml b/benchmark.dat names no oracle entry",
    ]
