import math
from pathlib import Path

from optbench.board import _json_lines


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
