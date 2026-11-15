import math
from pathlib import Path

import re

from optbench.board import (
    _PLAIN,
    _first_raw,
    _geometry_key,
    _json_lines,
    _json_lines_each,
    _json_lines_joined,
    _record_from_raw,
    _record_plain,
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


def test_both_json_line_paths_agree(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(
        b'{"n": 1}\n'
        b'["not", "a", "dict"]\n'
        b'{"label": "NaN inside the text"}\n'
        b"{bad\n"
    )
    raw = path.read_bytes()
    good = raw[: raw.rfind(b"\n{bad")]
    assert _json_lines_joined(good) == _json_lines_each(good)
    assert _json_lines_each(good) == [{"n": 1}, {"label": "NaN inside the text"}]
    each_error = joined_error = None
    try:
        _json_lines_each(raw)
    except Exception as exc:
        each_error = exc
    try:
        _json_lines_joined(raw)
    except Exception as exc:
        joined_error = exc
    assert each_error is not None and joined_error is not None
    assert type(each_error) is type(joined_error)
    assert str(each_error) == str(joined_error)
    pair = b'{"a": 1},{"b": 2}\n'
    each_error = joined_error = None
    try:
        _json_lines_each(pair)
    except Exception as exc:
        each_error = exc
    try:
        _json_lines_joined(pair)
    except Exception as exc:
        joined_error = exc
    assert each_error is not None and joined_error is not None
    assert type(each_error) is type(joined_error)
    assert str(each_error) == str(joined_error)


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


def test_the_first_start_does_not_require_the_rest_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / "rows.starts.jsonl"
    path.write_bytes(b'{"template": "demo", "force_calls": 1}\nnot json\n')
    assert _first_raw({"starts_jsonl": path}) == {"template": "demo", "force_calls": 1}
