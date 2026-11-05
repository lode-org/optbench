"""Archived rows are scored with the harness failure rule where the entry records failures."""

import json
from pathlib import Path

from optbench import board
from optbench.board import build_board, failure_cell, failure_paragraph, ranked_cells
from optbench.site_check import _check_budget, _parse_tables

ROOT = Path(__file__).resolve().parents[1]
PIN = "a" * 40


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _toy(root: Path, entries: list[tuple[str, dict]]) -> dict:
    _write(
        root / "problems" / "minimization" / "toy.toml",
        [
            'id = "minimization/toy"',
            'title = "Toy"',
            'suite = "minimization"',
            'problem = "toy"',
            'contract = "stated"',
            "",
            "[sample]",
            "starts = 4",
            "",
            "[convergence]",
            "force_norm = 0.01",
            "max_force_calls = 100",
            "",
            "[metrics]",
            'required = ["force_calls"]',
        ],
    )
    _write(
        root / "templates" / "toy" / "fast.toml",
        [
            'id = "toy/fast"',
            'engine = "toy"',
            'problem = "minimization/toy"',
            'code = "toy"',
            'code_version = "1.0"',
            f'commit = "{PIN}"',
            'tuning = "default"',
        ],
    )
    _write(root / "codes" / "registry.toml", ["[[code]]", 'id = "toy"', 'name = "toy"', f'commit = "{PIN}"'])
    _write(
        root / "results" / "campaign" / "toy-fast.dat",
        [
            "force_calls 1",
            "nfailed 0",
            "code toy",
            "code_version 1.0",
            f"commit {PIN}",
            "template toy/fast",
            "problem minimization/toy",
            "counter harness",
            "tuning default",
        ],
    )
    starts = [
        {"force_calls": calls, "success": True, "start": str(index), "counter": "harness"}
        for index, calls in enumerate((40, 60, 50, 50))
    ]
    _write(root / "results" / "campaign" / "toy-fast.starts.jsonl", [json.dumps(row) for row in starts])
    lines = [
        json.dumps(
            {
                "suite": "minimization",
                "problem": "toy",
                "entry": entry,
                "path": f"website/minimization/toy/{entry}/benchmark.dat",
                "fields": {"code": "Old", "algorithm": entry, **fields},
                "text": "",
            }
        )
        for entry, fields in entries
    ]
    _write(root / "oracle" / "entries.jsonl", lines)
    return build_board(root)["problems"]["minimization/toy"]


def test_an_archived_failure_costs_twice_the_cap(tmp_path: Path):
    problem = _toy(
        tmp_path,
        [
            ("failed", {"force_calls": "10", "nfailed": "1"}),
            ("clean", {"force_calls": "20", "nfailed": "0"}),
            ("uncounted", {"force_calls": "5"}),
        ],
    )
    rows = {row["entry"] or row["origin"]: row for row in problem["ranked"]}

    assert rows["failed"]["par2_mean"] == (3 * 10 + 1 * 2 * 100) / 4
    assert rows["clean"]["par2_mean"] == 20.0
    assert rows["toy-fast"]["par2_mean"] == 50.0
    assert [row["entry"] for row in problem["ranked"]] == ["uncounted", "clean", "toy-fast", "failed"]
    assert [row["rank_cost"] for row in problem["ranked"]] == [5.0, 20.0, 50.0, 57.5]

    assert failure_cell(rows["failed"]) == "1 failed, class not recorded"
    assert failure_cell(rows["clean"]) == "0 failed"
    assert failure_cell(rows["uncounted"]) == "not recorded"
    assert failure_cell(rows["toy-fast"]) == "crash 0, divergence 0, cap 0, wrong_stationary 0"
    assert dict(ranked_cells(rows["failed"], False))["par2_mean"] == "57.5"
    assert dict(ranked_cells(rows["uncounted"], False))["par2_mean"] == ""

    assert rows["failed"]["failure_note"] == ""
    assert rows["clean"]["failure_note"] == ""
    assert rows["uncounted"]["failure_note"] == "the entry records no failure count"
    text = failure_paragraph(problem)
    assert text.startswith('<p class="bench-gaps" data-board-note="failures">')
    assert f"<li>{rows['uncounted']['label']} (the entry records no failure count)</li>" in text
    assert rows["failed"]["label"] not in text
    assert text in board.render_problem_section(problem, set())


def test_an_archived_run_above_the_cap_is_held_as_another_budget(tmp_path: Path):
    problem = _toy(
        tmp_path,
        [
            ("over", {"force_calls": "30", "nfailed": "0", "force_calls_max": "150"}),
            ("under", {"force_calls": "30", "nfailed": "0", "force_calls_max": "100"}),
        ],
    )

    assert [row["entry"] for row in problem["ranked"]] == ["under", "toy-fast"]
    (row,) = problem["over_budget"]
    assert (row["entry"], row["par2_mean"], row["rank_cost"]) == ("over", None, 30.0)
    assert row["failure_note"] == "its archived maximum, 150, is above the cap of 100"
    assert failure_paragraph(problem) == ""

    html = board.render_problem_section(problem, set())
    assert "<h3>Run under a different budget</h3>" in html
    assert f"{row['label']} (its archived maximum, 150, is above the cap of 100)" in html
    table = next(item for item in _parse_tables(html) if item["attrs"].get("data-board-table") == "budget")
    document = {"problems": {"minimization/toy": problem}}
    assert _check_budget("page", table, document) == []
    head, marker, tail = html.partition('data-board-table="budget"')
    tampered = head + marker + tail.replace('<td data-label="Published" data-field="published">30</td>', '<td data-label="Published" data-field="published">31</td>')
    table = next(item for item in _parse_tables(tampered) if item["attrs"].get("data-board-table") == "budget")
    assert _check_budget("page", table, document) == [
        "page minimization/toy budget row 1 published is '31', board has '30'"
    ]


def test_only_the_two_lj38_band_rows_ran_above_the_cap():
    problems = board.load_all(ROOT / "problems")
    registry = board._registry(ROOT)
    full_n = {ident: board.stated_full_n(ROOT, problem) for ident, problem in problems.items()}
    rows = board._archive_rows(ROOT, problems, registry, full_n)
    for row in rows:
        board._annotate_row(row, problems[row["problem"]], registry)
    held = sorted((row["problem"], row["entry"], row["_archived_max"]) for row in rows if row["_over_budget"])
    assert held == [
        ("double-ended-saddle-search/lj38", "eon-lanczos", 2794.0),
        ("double-ended-saddle-search/lj38", "optim-cl-string", 2792.0),
    ]


def test_no_archived_failure_count_renders_as_zero():
    problems = board.load_all(ROOT / "problems")
    registry = board._registry(ROOT)
    full_n = {ident: board.stated_full_n(ROOT, problem) for ident, problem in problems.items()}
    rows = board._archive_rows(ROOT, problems, registry, full_n)
    for row in rows:
        board._annotate_row(row, problems[row["problem"]], registry)
    failed = [row for row in rows if (row["nfailed"] or 0) > 0]
    assert len(failed) == 15
    for row in failed:
        assert failure_cell(row) == f"{row['nfailed']} failed, class not recorded"
    scored = {(row["problem"], row["entry"]): row["par2_mean"] for row in failed if row["par2_mean"] is not None}
    assert len(scored) == 10
    assert scored[("double-ended-saddle-search/lj38", "eon-ci-neb")] == (43 * 827 + 7 * 2 * 2500) / 50
    assert scored[("minimization/lj38", "eon-sd")] == (904 * 4901 + 96 * 2 * 10000) / 1000
    for row in failed:
        if row["par2_mean"] is None:
            assert row["failure_note"]
        else:
            assert row["rank_cost"] == row["par2_mean"] > row["published"]
