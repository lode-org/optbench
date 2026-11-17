"""The check that gates the published tree: what it reports and the order it stops in.

Each case is a small tree with the layout of the real one. The check goes
through the archive structure first, then the known gaps, then tuned
templates, then the site. It prints what it found and stops at the first
stage that has something to say, so the output is compared line by line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optbench.__main__ import main
from optbench.check import banned_word_errors, check, held_back_errors, load_oracle

ALPHA = "website/minimization/toy/alpha/benchmark.dat"


def _problem(root: Path, name: str = "toy", *, contract: str = "stated", entry_count: int | None = None) -> None:
    lines = [
        f'id = "minimization/{name}"',
        f'title = "{name}"',
        'suite = "minimization"',
        f'problem = "{name}"',
        f'contract = "{contract}"',
    ]
    if entry_count is not None:
        lines.append(f"entry_count = {entry_count}")
    lines += ["", "[metrics]", 'required = ["force_calls"]', ""]
    path = root / "problems" / "minimization" / f"{name}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _row(problem: str = "toy", fields: dict | None = None, duplicates: list | None = None, path: str = ALPHA) -> dict:
    return {
        "suite": "minimization",
        "problem": problem,
        "entry": "alpha",
        "path": path,
        "fields": {"force_calls": "100", "nfailed": "0"} if fields is None else fields,
        "duplicates": duplicates or [],
        "text": "",
        "sha256": "0" * 64,
        "commit": "0" * 8,
    }


def _oracle(root: Path, rows: list[dict], known: list[str] | None = None) -> Path:
    path = root / "oracle" / "entries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if known is not None:
        (root / "oracle" / "archive_gaps.txt").write_text("".join(line + "\n" for line in known), encoding="utf-8")
    return path


def _gate(root: Path, capsys) -> tuple[int, list[str]]:
    code = check(root / "problems", root / "oracle" / "entries.jsonl")
    return code, capsys.readouterr().out.splitlines()


def test_a_tree_with_no_structure_or_gap_to_report_goes_on_to_the_site(tmp_path, capsys):
    _problem(tmp_path)
    _oracle(tmp_path, [_row()])
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines == ["entries 1", "problems 1", "archive_gaps 0", "site 1", "site pages are missing"]


def test_an_oracle_row_for_a_problem_with_no_file_is_a_structural_error(tmp_path, capsys):
    _problem(tmp_path)
    _oracle(tmp_path, [_row(), _row(problem="ghost", path="website/minimization/ghost/alpha/benchmark.dat")])
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines == [
        "entries 2",
        "problems 1",
        "archive_gaps 0",
        "structural 1",
        "minimization/ghost has an oracle row and no problem file",
    ]


def test_a_shared_contract_needs_every_required_key_in_every_row(tmp_path, capsys):
    _problem(tmp_path, contract="shared")
    _oracle(tmp_path, [_row(fields={"nfailed": "0"})])
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines[3:] == ["structural 1", f"contract {ALPHA} missing force_calls"]


def test_the_entry_count_of_a_problem_has_to_match_the_oracle(tmp_path, capsys):
    _problem(tmp_path, entry_count=3)
    _oracle(tmp_path, [_row()])
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines[3:] == ["structural 1", "minimization/toy entry_count is 3, oracle has 1"]


def test_every_structural_error_is_listed_and_nothing_after_them_runs(tmp_path, capsys):
    _problem(tmp_path, contract="shared", entry_count=2)
    _oracle(tmp_path, [_row(fields={"nfailed": "0"})])
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines[3:] == [
        "structural 2",
        f"contract {ALPHA} missing force_calls",
        "minimization/toy entry_count is 2, oracle has 1",
    ]


def test_a_gap_the_list_does_not_know_is_printed_with_a_plus_and_a_stale_one_with_a_minus(tmp_path, capsys):
    _problem(tmp_path)
    _oracle(tmp_path, [_row(fields={"nfailed": "0"})], known=["missing website/minimization/toy/old/benchmark.dat force_calls"])
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines[2:] == [
        "archive_gaps 1",
        "archive gaps do not match oracle/archive_gaps.txt",
        f"+ missing {ALPHA} force_calls",
        "- missing website/minimization/toy/old/benchmark.dat force_calls",
    ]


def test_a_gap_that_is_listed_is_known_and_the_check_goes_on(tmp_path, capsys):
    _problem(tmp_path)
    _oracle(tmp_path, [_row(fields={"nfailed": "0"})], known=[f"missing {ALPHA} force_calls"])
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines == ["entries 1", "problems 1", "archive_gaps 1", "site 1", "site pages are missing"]


def test_a_misspelled_key_and_a_repeated_key_are_gaps_named_as_such(tmp_path, capsys):
    _problem(tmp_path)
    duplicate = {"key": "force_calls", "line": 2, "kept": "100", "also": "120"}
    _oracle(
        tmp_path,
        [_row(fields={"force_calls": "100", "contributer": "Someone"}, duplicates=[duplicate])],
        known=[],
    )
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines[3:] == [
        "archive gaps do not match oracle/archive_gaps.txt",
        f"+ spelling {ALPHA} contributer -> contributor",
        f"+ duplicate {ALPHA} force_calls kept '100' also '120'",
    ]


def test_a_template_named_tuned_stops_the_check_before_the_site(tmp_path, capsys):
    _problem(tmp_path)
    _oracle(tmp_path, [_row()])
    template = tmp_path / "templates" / "toy" / "fast-tuned.toml"
    template.parent.mkdir(parents=True)
    template.write_text('id = "toy/fast-tuned"\nengine = "toy"\n', encoding="utf-8")
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines[3:] == ["tuned 1", "tuned template templates/toy/fast-tuned.toml"]


def test_site_errors_are_counted_and_listed_with_the_held_back_files_and_the_banned_words(tmp_path, capsys):
    _problem(tmp_path)
    _oracle(tmp_path, [_row()])
    (tmp_path / "README.md").write_text("A smoke run, and a smoke test.\n", encoding="utf-8")
    board = tmp_path / "optbench" / "board.py"
    board.parent.mkdir()
    board.write_text("def holdback(rows):\n    return rows\n", encoding="utf-8")
    code, lines = _gate(tmp_path, capsys)
    assert code == 1
    assert lines[3:] == [
        "site 3",
        "site pages are missing",
        "optbench/board.py keeps templates off the ranked table",
        "README.md contains smoke",
    ]


def test_the_command_exits_with_the_status_of_the_check_and_prints_what_it_found(tmp_path, capsys):
    _problem(tmp_path)
    _oracle(tmp_path, [_row()])
    with pytest.raises(SystemExit) as stopped:
        main(["check", "--problems", str(tmp_path / "problems"), "--oracle", str(tmp_path / "oracle" / "entries.jsonl")])
    assert stopped.value.code == 1
    assert capsys.readouterr().out.splitlines()[:3] == ["entries 1", "problems 1", "archive_gaps 0"]


def test_oracle_rows_are_numbered_by_their_line_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "entries.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n\n" + json.dumps({"id": "b"}) + "\n  \n", encoding="utf-8")
    assert [(row["id"], row["_line"]) for row in load_oracle(path)] == [("a", 1), ("b", 3)]


def test_a_findings_ledger_that_omits_templates_is_refused_by_line_and_blank_lines_count(tmp_path):
    ledger = tmp_path / "oracle" / "findings.jsonl"
    ledger.parent.mkdir()
    ledger.write_text(
        json.dumps({"id": "one", "beside": {}}) + "\n\n" + json.dumps({"id": "two", "beside": {"omitted": ["toy/x"]}}) + "\n",
        encoding="utf-8",
    )
    assert held_back_errors(tmp_path) == ["oracle/findings.jsonl line 3 keeps templates off the ranked table"]
    ledger.write_text(json.dumps({"id": "one", "beside": {}}) + "\n", encoding="utf-8")
    assert held_back_errors(tmp_path) == []
    assert held_back_errors(tmp_path / "nowhere") == []


def test_a_banned_word_is_named_once_however_often_it_appears_and_a_path_can_carry_one(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "page.md").write_text("A smoke run. Another smoke run. The legacy table.\n", encoding="utf-8")
    (tmp_path / "legacy-summary.rst").write_text("nothing to see\n", encoding="utf-8")
    errors = banned_word_errors(tmp_path)
    assert "notes/page.md contains smoke, legacy" in errors
    assert "legacy-summary.rst path contains legacy" in errors
    assert len(errors) == 2
