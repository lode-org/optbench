"""Fail when a rendered results table disagrees with the v2 board."""

from __future__ import annotations

import html
from html.parser import HTMLParser
from pathlib import Path

from .board import (
    MissingAsvCommit,
    _registry,
    _show_per_image,
    board_rows,
    budget_per_image,
    build_board,
    difference_rows,
    difference_text,
    family_cells,
    inferred_rows,
    ranked_cells,
    unit_per_image,
)
from .problems import load_all

STALE_CAPTIONS = (
    "Present runs. Fewer force calls first.",
    "Present sources. The cell is the record.",
    "Present codes. Fewer calls is to the left.",
)

def _check_family(label: str, table: dict, document: dict) -> list[str]:
    problem_id = table["attrs"].get("data-problem", "")
    family = table["attrs"].get("data-family", "")
    problem = (document.get("problems") or {}).get(problem_id)
    ranked = [
        row for row in (problem.get("ranked") or [] if problem else []) if str(row.get("family") or "") == family
    ]
    body = _ranked_body(table)
    if len(body) != len(ranked):
        return [f"{label} {problem_id} family {family} rows {len(body)} != board {len(ranked)}"]
    errors: list[str] = []
    for index, (got_row, board_row) in enumerate(zip(body, ranked), start=1):
        got = _cell_map(got_row)
        expected = dict(family_cells(board_row))
        for field, value in expected.items():
            if got.get(field) != value:
                errors.append(
                    f"{label} {problem_id} family {family} row {index} {field} is {got.get(field)!r}, board has {value!r}"
                )
    return errors


def _code_name(row: dict, registry: list[dict]) -> str:
    from .site_pages import code_page_name

    return code_page_name(row, registry)


def _check_code_rows(label: str, table: dict, document: dict) -> list[str]:
    errors: list[str] = []
    for row in table["rows"]:
        problem_id = row["attrs"].get("data-problem", "")
        shown = row["attrs"].get("data-label", "")
        if not problem_id:
            continue
        problem = (document.get("problems") or {}).get(problem_id)
        ranked = board_rows(problem) if problem else []
        board_row = next((item for item in ranked if str(item.get("label") or "") == shown), None)
        got = _cell_map(row)
        if board_row is None:
            errors.append(f"{label} {problem_id} code row {shown!r} is not a ranked board row")
            continue
        expected = dict(ranked_cells(board_row, False))
        for field, value in expected.items():
            if got.get(field) != value:
                errors.append(
                    f"{label} {problem_id} {shown} {field} is {got.get(field)!r}, board has {value!r}"
                )
    return errors


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self._table: dict | None = None
        self._row: dict | None = None
        self._cell: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        if tag == "table":
            self._table = {"attrs": attr, "rows": []}
        elif tag == "tr" and self._table is not None:
            self._row = {"attrs": attr, "cells": []}
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {"attrs": attr, "text": []}

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._cell["text"] = "".join(self._cell["text"]).strip()
            self._row["cells"].append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table["rows"].append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _parse_tables(text: str) -> list[dict]:
    parser = _Tables()
    parser.feed(text)
    return parser.tables


def _cell_map(row: dict) -> dict[str, str]:
    found: dict[str, str] = {}
    for cell in row["cells"]:
        field = cell["attrs"].get("data-field")
        if field:
            found[field] = cell["text"]
    return found


def _ranked_body(table: dict) -> list[dict]:
    return [row for row in table["rows"] if _cell_map(row)]


def _check_ranked(label: str, table: dict, document: dict) -> list[str]:
    problem_id = table["attrs"].get("data-problem", "")
    problem = (document.get("problems") or {}).get(problem_id)
    ranked = list(problem.get("ranked") or []) if problem else []
    body = _ranked_body(table)
    if len(body) != len(ranked):
        return [f"{label} {problem_id} ranked rows {len(body)} != board {len(ranked)}"]
    per_image = _show_per_image(problem) if problem else False
    errors: list[str] = []
    for index, (got_row, board_row) in enumerate(zip(body, ranked), start=1):
        got = _cell_map(got_row)
        expected = dict(ranked_cells(board_row, per_image))
        for field, value in expected.items():
            if got.get(field) != value:
                errors.append(
                    f"{label} {problem_id} row {index} {field} is {got.get(field)!r}, board has {value!r}"
                )
    return errors


def _check_rows(where: str, table: dict, rows: list[dict], per_image: bool) -> list[str]:
    body = _ranked_body(table)
    if len(body) != len(rows):
        return [f"{where} rows {len(body)} != board {len(rows)}"]
    errors: list[str] = []
    for index, (got_row, board_row) in enumerate(zip(body, rows), start=1):
        got = _cell_map(got_row)
        for field, value in ranked_cells(board_row, per_image):
            if got.get(field) != value:
                errors.append(f"{where} row {index} {field} is {got.get(field)!r}, board has {value!r}")
    return errors


def _check_unit(label: str, table: dict, document: dict) -> list[str]:
    problem_id = table["attrs"].get("data-problem", "")
    unit = table["attrs"].get("data-unit", "")
    problem = (document.get("problems") or {}).get(problem_id) or {}
    group = next((item for item in problem.get("other_units") or [] if item.get("unit") == unit), None)
    if group is None:
        return [f"{label} {problem_id} ranks {unit!r} and the board has no such unit table"]
    return _check_rows(f"{label} {problem_id} {unit}", table, list(group.get("rows") or []), unit_per_image(group))


def _check_budget(label: str, table: dict, document: dict) -> list[str]:
    problem_id = table["attrs"].get("data-problem", "")
    problem = (document.get("problems") or {}).get(problem_id) or {}
    rows = list(problem.get("over_budget") or [])
    return _check_rows(f"{label} {problem_id} budget", table, rows, budget_per_image(problem))


def _check_headline(label: str, table: dict, document: dict) -> list[str]:
    problem_id = table["attrs"].get("data-problem", "")
    problem = (document.get("problems") or {}).get(problem_id)
    ranked = list(problem.get("ranked") or []) if problem else []
    body = _ranked_body(table)
    if not ranked:
        if body:
            return [f"{label} {problem_id} headline has a row and the board has no full run"]
        return []
    if len(body) != 1:
        return [f"{label} {problem_id} headline rows {len(body)} != 1"]
    per_image = _show_per_image(problem) if problem else False
    got = _cell_map(body[0])
    expected = dict(ranked_cells(ranked[0], per_image))
    errors = []
    for field, value in expected.items():
        if got.get(field) != value:
            errors.append(f"{label} {problem_id} headline {field} is {got.get(field)!r}, board has {value!r}")
    return errors


def _page_texts(root: Path) -> list[tuple[str, str]]:
    pages: list[Path] = []
    include = root / "docs" / "include"
    for name in ("sota.org", "v2-board.org", "collection.org", "codes.org"):
        pages.append(include / name)
    generated = root / "docs" / "orgmode" / "generated"
    for folder in ("suites", "codes"):
        directory = generated / folder
        if directory.is_dir():
            pages.extend(sorted(directory.rglob("*.org")))
    org = root / "docs" / "orgmode"
    for name in ("index.org", "codes.org", "archive.org", "board.org", "run.org"):
        pages.append(org / name)
    build = root / "docs" / "build"
    if build.is_dir():
        pages.extend(sorted(build.rglob("*.html")))
    found: list[tuple[str, str]] = []
    for path in pages:
        if path.is_file():
            found.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
    return found


def _require(text: str, needle: str, label: str, errors: list[str]) -> None:
    if needle not in text:
        errors.append(f"{label} is missing {needle}")


def _headline_ids(text: str) -> set[str]:
    found: set[str] = set()
    for table in _parse_tables(text):
        if table["attrs"].get("data-board-table") != "headline":
            continue
        problem_id = table["attrs"].get("data-problem", "")
        if problem_id:
            found.add(problem_id)
    return found


def check_site(root: Path) -> list[str]:
    """Compare rendered v2 tables with a fresh board.

    ``docs/build`` is included when that directory exists.
    """
    root = Path(root)
    try:
        document = build_board(root)
    except MissingAsvCommit as exc:
        return [str(exc)]
    problems = load_all(root / "problems") if (root / "problems").is_dir() else {}
    registry = _registry(root)
    errors: list[str] = []
    pages = _page_texts(root)
    if not pages:
        return ["site pages are missing"]
    texts = {label: text for label, text in pages}
    seen_code: set[tuple[str, str]] = set()
    for label, text in pages:
        for caption in STALE_CAPTIONS:
            if caption in text:
                errors.append(f"{label} still has the old caption {caption!r}")
        for table in _parse_tables(text):
            kind = table["attrs"].get("data-board-table", "")
            if kind == "ranked":
                errors.extend(_check_ranked(label, table, document))
            elif kind == "headline":
                errors.extend(_check_headline(label, table, document))
            elif kind == "family":
                errors.extend(_check_family(label, table, document))
            elif kind == "unit":
                errors.extend(_check_unit(label, table, document))
            elif kind == "budget":
                errors.extend(_check_budget(label, table, document))
            elif kind == "code-rows":
                errors.extend(_check_code_rows(label, table, document))
                for row in table["rows"]:
                    problem_id = row["attrs"].get("data-problem", "")
                    shown = row["attrs"].get("data-label", "")
                    if problem_id and shown:
                        seen_code.add((problem_id, shown))

    for problem in problems.values():
        suite_label = f"docs/orgmode/generated/suites/{problem['suite']}/{problem['problem']}.org"
        suite = texts.get(suite_label, "")
        if not suite:
            errors.append(f"{suite_label} is missing")
            continue
        _require(
            suite,
            f'data-board-table="ranked" data-problem="{problem["id"]}"',
            suite_label,
            errors,
        )
        board_problem = (document.get("problems") or {}).get(problem["id"]) or {}
        slug = str(problem["id"]).replace("/", "-")
        from .site_pages import profile_worth_showing

        if profile_worth_showing(board_problem):
            _require(suite, f"{slug}-profile.svg", suite_label, errors)
        harness = [
            row
            for row in board_problem.get("ranked") or []
            if row.get("par2_mean") is not None and row.get("success_rate") is not None
        ]
        if harness:
            _require(suite, f"{slug}-cost.svg", suite_label, errors)
        families = {str(row.get("family") or "") for row in board_problem.get("ranked") or []}
        for family in families:
            _require(
                suite,
                f'data-board-table="family" data-family="{family}" data-problem="{problem["id"]}"',
                suite_label,
                errors,
            )
        unscored = [row for row in board_rows(board_problem) if row.get("failure_note")]
        if unscored:
            _require(suite, 'data-board-note="failures"', suite_label, errors)
        for row in unscored:
            _require(suite, html.escape(f"{row.get('label')} ({row['failure_note']})"), suite_label, errors)
        differs = difference_rows(board_problem)
        if differs:
            _require(suite, 'data-board-note="differences"', suite_label, errors)
        for row in differs:
            _require(suite, html.escape(difference_text(row)), suite_label, errors)
        if inferred_rows(board_problem):
            _require(suite, 'data-board-note="inferred"', suite_label, errors)
        if board_problem.get("over_budget"):
            _require(suite, f'data-board-table="budget" data-problem="{problem["id"]}"', suite_label, errors)
        for group in board_problem.get("other_units") or []:
            _require(
                suite,
                f'data-board-table="unit" data-unit="{html.escape(str(group.get("unit") or ""))}" '
                f'data-problem="{problem["id"]}"',
                suite_label,
                errors,
            )
        for name in ("docs/include/sota.org", "docs/include/collection.org"):
            page = texts.get(name, "")
            if not page:
                errors.append(f"{name} is missing")
                continue
            _require(
                page,
                f'data-board-table="headline" data-problem="{problem["id"]}"',
                name,
                errors,
            )

    board_include = "docs/include/v2-board.org"
    board_text = texts.get(board_include, "")
    if not board_text:
        errors.append(f"{board_include} is missing")
    else:
        found = _headline_ids(board_text)
        for problem_id in (document.get("problems") or {}):
            if problem_id not in found:
                errors.append(f"{board_include} headline is missing {problem_id}")

    for problem_id, problem in (document.get("problems") or {}).items():
        for row in board_rows(problem):
            key = (problem_id, str(row.get("label") or ""))
            if key not in seen_code:
                errors.append(f"code page is missing {problem_id} {key[1]}")
            name = _code_name(row, registry)
            slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
            page = texts.get(f"docs/orgmode/generated/codes/{slug}.org", "")
            if not page:
                errors.append(f"docs/orgmode/generated/codes/{slug}.org is missing")
            else:
                label = str(row.get("label") or "")
                if label not in page and html.escape(label) not in page:
                    errors.append(f"docs/orgmode/generated/codes/{slug}.org is missing {label}")

    build = root / "docs" / "build"
    if build.is_dir():
        index = texts.get("docs/build/index.html", "")
        if not index:
            errors.append("docs/build/index.html is missing")
        elif 'data-bench="suites"' not in index and 'data-board-table="headline"' not in index:
            errors.append("docs/build/index.html has no suite table")
        built_board = "docs/build/board.html"
        board_html = texts.get(built_board, "")
        if not board_html:
            errors.append(f"{built_board} is missing")
        else:
            found = _headline_ids(board_html)
            for problem_id in (document.get("problems") or {}):
                if problem_id not in found:
                    errors.append(f"{built_board} headline is missing {problem_id}")
        for problem in problems.values():
            built = f"docs/build/generated/suites/{problem['suite']}/{problem['problem']}.html"
            page = texts.get(built, "")
            if not page:
                errors.append(f"{built} is missing")
                continue
            _require(
                page,
                f'data-board-table="ranked" data-problem="{problem["id"]}"',
                built,
                errors,
            )
    return errors
