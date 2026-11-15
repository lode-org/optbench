import json
import math
from pathlib import Path

import re

from optbench.board import (
    _GEOMETRY_CACHE,
    _GEOMETRY_TOKENS,
    _PLAIN,
    _apply_summary,
    _first_raw,
    _geometry_key,
    _json_lines,
    _json_lines_each,
    _keyed_costs,
    _load_group_records,
    _record_from_raw,
    _record_plain,
    _score_records,
    _start_record,
)
from optbench.score import normalize_start


def test_json_lines_match_the_stdlib_parser(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(
        b'{"energy": 1.5, "n": 2}\n'
        b"\n"
        b'["not", "a", "dict"]\n'
        b'{"label": "NaN-note", "n": 0}\n'
        b'{"energy": NaN, "force_norm": -Infinity, "ok": false}\n'
    )
    rows = _json_lines(path)
    assert rows[0] == {"energy": 1.5, "n": 2}
    assert rows[1] == {"label": "NaN-note", "n": 0}
    assert len(rows) == 3
    assert math.isnan(rows[2]["energy"])
    assert rows[2]["force_norm"] == float("-inf")
    assert rows[2]["ok"] is False


def _same_error(path: Path, raw: bytes) -> None:
    each_error = file_error = None
    try:
        _json_lines_each(raw)
    except Exception as exc:
        each_error = exc
    try:
        _json_lines(path)
    except Exception as exc:
        file_error = exc
    assert each_error is not None and file_error is not None
    assert type(each_error) is type(file_error)
    assert str(each_error) == str(file_error)


def test_both_json_line_paths_agree(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    good = (
        b'{"n": 1}\n'
        b'["not", "a", "dict"]\n'
        b'{"label": "NaN inside the text"}\n'
    )
    path.write_bytes(good)
    assert _json_lines(path) == _json_lines_each(good)
    assert _json_lines_each(good) == [{"n": 1}, {"label": "NaN inside the text"}]
    raw = good + b"{bad\n"
    path.write_bytes(raw)
    _same_error(path, raw)
    pair = b'{"a": 1},{"b": 2}\n'
    path.write_bytes(pair)
    _same_error(path, pair)


def test_geometry_names_match_the_old_patterns() -> None:
    digits = re.compile(r"\d+")
    indexed = re.compile(r"[A-Za-z][A-Za-z0-9]*[_-](\d+)")

    def old(start):
        if isinstance(start, tuple):
            start = start[-1] if start else ""
        text = str(start if start is not None else "").strip()
        if text.endswith(".con"):
            text = text[: -len(".con")]
        if digits.fullmatch(text):
            return str(int(text))
        match = indexed.fullmatch(text)
        if match:
            return str(int(match.group(1)))
        return text

    names = [
        "0",
        "00",
        "12",
        "cluster_0000.con",
        "pos_0000",
        "pair_03",
        "rep1-search1",
        "product_49",
        "min01",
        "ä_1",
        "A1_02",
        "1_02",
        "",
        "²",
        "a_²",
        ("seed", "pair_07"),
        None,
    ]
    for name in names:
        assert _geometry_key(name) == old(name)


def test_plain_records_match_the_general_normalizer() -> None:
    rows = []
    for calls in (1, 10, 100, 100.5, 0):
        for success in (True, False):
            rows.append({"force_calls": calls, "success": success, "start": "pair_03"})
    rows.extend(
        [
            {"force_calls": 100, "success": True, "start": "pair_01", "transition_found": 1},
            {"force_calls": 100, "success": True, "start": "pair_01", "transition_found": True},
            {"force_calls": 100, "success": True, "start": "pair_01", "transition_found": "1"},
            {"force_calls": 100, "success": True, "start": "pair_01", "transition_found": 0},
            {"force_calls": 100, "success": True, "start": "pair_01", "transition_found": False},
            {"force_calls": 100, "success": True, "start": "pair_01", "transition_found": None},
            {"force_calls": 100, "success": False, "failure": "cap", "start": "s"},
            {"force_calls": 50, "success": False, "failure": "divergence", "geometry": "pos_0004"},
            {"force_calls": 50, "success": True, "start": "kept", "geometry": "other"},
            {"force_calls": 12, "success": True, "hessian_index": 1, "hessian_calls": 3, "start": "h"},
            {"force_calls": 12, "success": False, "hessian_index": None, "hessian_calls": "nope", "failure": "crash", "start": "h2"},
            {"force_calls": float("nan"), "success": True, "start": "n"},
            {"force_calls": float("inf"), "success": True, "start": "i"},
            {"force_calls": float("-inf"), "success": False, "failure": "divergence", "start": "j"},
            {"force_calls": 80, "success": False, "reasons": ["not_connected"], "start": "r"},
            {"force_calls": 80, "success": False, "error": "boom", "start": "e"},
            {"force_calls": 4, "success": True, "per_image": 2, "call_unit": "total", "start": "p"},
            {"force_calls": 9, "success": False, "force_norm": 0.2, "start": "f"},
            {"force_calls": 9, "success": False, "force_norm": 1e-8, "start": "f2"},
        ]
    )
    for raw in rows:
        for cap in (None, 10, 100):
            for force_tol in (None, 1e-4):
                general = _record_from_raw(normalize_start(dict(raw), None), cap, force_tol)
                plain = _record_plain(dict(raw), cap, force_tol)
                if plain is _PLAIN:
                    assert _start_record(dict(raw), None, cap, force_tol) == general
                    continue
                assert plain == general
                assert _start_record(dict(raw), None, cap, force_tol) == general


def test_repeated_start_text_is_one_string(tmp_path: Path) -> None:
    """Equal start and failure strings from separate files are one object.

    The text is unchanged, and the profile token is the one the geometry
    helper produces from that text.
    """
    from optbench.board import _apply_summary, _keyed_costs, _load_group_records, _score_records

    def write(name: str, rows: list[dict]) -> Path:
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    first = write(
        "a.starts.jsonl",
        [
            {"force_calls": 4, "success": True, "start": "cluster_0000"},
            {"force_calls": 9, "success": False, "failure": "cap", "start": "cluster_0001"},
        ],
    )
    second = write(
        "b.starts.jsonl",
        [
            {"force_calls": 5, "success": True, "start": "cluster_0000"},
            {"force_calls": 8, "success": False, "failure": "cap", "start": "cluster_0001"},
        ],
    )
    left, _source = _load_group_records({"starts_jsonl": first}, None, None)
    right, _source = _load_group_records({"starts_jsonl": second}, None, None)
    assert left[0]["start"] == "cluster_0000"
    assert left[0]["start"] is right[0]["start"]
    assert left[1]["start"] is right[1]["start"]
    assert left[1]["failure"] == "cap"
    assert left[1]["failure"] is right[1]["failure"]
    for records in (left, right):
        row: dict = {}
        _apply_summary(row, _score_records(records, 10), records)
        assert _keyed_costs(row) == [
            (_geometry_key(record["start"]), float("inf") if not record["success"] else float(record["force_calls"]))
            for record in records
        ]


def test_the_first_start_does_not_require_the_rest_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / "rows.starts.jsonl"
    path.write_bytes(b'{"template": "demo", "force_calls": 1}\nnot json\n')
    assert _first_raw({"starts_jsonl": path}) == {"template": "demo", "force_calls": 1}


def _geometry_at_base(start) -> str:
    """The geometry token from 4500ce76, with no start/token cache."""
    if isinstance(start, tuple):
        start = start[-1] if start else ""
    text = str(start if start is not None else "").strip()
    if text.endswith(".con"):
        text = text[: -len(".con")]
    if text.isdecimal():
        return str(int(text))
    sep = text.rfind("_")
    dash = text.rfind("-")
    if dash > sep:
        sep = dash
    if sep > 0 and text[sep + 1 :].isdecimal():
        head = text[:sep]
        if (
            head
            and head.isascii()
            and head.isalnum()
            and ("A" <= head[0] <= "Z" or "a" <= head[0] <= "z")
        ):
            return str(int(text[sep + 1 :]))
    return text


def test_a_token_and_a_list_start_keep_the_base_profile(tmp_path: Path) -> None:
    """A start equal to an earlier token is parsed, and a JSON list does not raise.

    ``foo.con.con`` tokens to ``foo.con``. That string is itself a start
    whose token is ``foo``. Remembering the first token must not answer
    the second start. A list start is not a cache key; its token is the
    same text the base geometry produced.
    """
    _GEOMETRY_CACHE.clear()
    _GEOMETRY_TOKENS.clear()
    assert _geometry_key("foo.con.con") == "foo.con"
    assert _geometry_key("foo.con") == "foo"
    assert "foo.con" not in _GEOMETRY_CACHE or _GEOMETRY_CACHE["foo.con"] == "foo"

    rows = [
        {"force_calls": 4, "success": True, "start": "foo.con.con"},
        {"force_calls": 6, "success": True, "start": "foo.con"},
        {"force_calls": 2, "success": True, "start": ["left", "right"]},
    ]
    path = tmp_path / "rows.starts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _GEOMETRY_CACHE.clear()
    _GEOMETRY_TOKENS.clear()
    records, source = _load_group_records({"starts_jsonl": path}, None, None)
    assert source == "starts.jsonl"
    assert [record["start"] for record in records] == [row["start"] for row in rows]
    row = {}
    _apply_summary(row, _score_records(records, None), records)
    expected = [
        (_geometry_at_base(record["start"]), float(record["force_calls"]))
        for record in records
    ]
    assert _keyed_costs(row) == expected
    shared = _geometry_key("cluster_0000")
    assert _geometry_key("pos_0000") is shared
