"""Rows that use different failure caps are ranked inside each cap."""

from __future__ import annotations

import json
from pathlib import Path

from optbench.board import _assign_family_ranks, _rank_key, family_cells, headline_table

ROOT = Path(__file__).resolve().parents[1]


def _row(label: str, cap: int, cost: float) -> dict:
    return {
        "family": "min-mode",
        "cap": cap,
        "rank_cost": cost,
        "label": label,
        "template": label,
        "path": label,
    }


def test_a_new_cap_starts_its_rank_at_one():
    low = _row("lanczos", 30000, 59911.5)
    high = _row("dimer-lbfgs", 48000, 95648.3)
    rows = [high, low]
    rows.sort(key=_rank_key)
    _assign_family_ranks(rows)
    assert [row["label"] for row in rows] == ["lanczos", "dimer-lbfgs"]
    assert [row["family_rank"] for row in rows] == [1, 1]
    assert dict(family_cells(rows[0]))["cap"] == "30000"
    assert dict(family_cells(rows[1]))["cap"] == "48000"


def test_pt_island_restarts_rank_for_each_cap():
    board = json.loads((ROOT / "results" / "board" / "index.json").read_text(encoding="utf-8"))
    ranked = board["problems"]["saddle-search/pt-island"]["ranked"]
    grouped: dict[int, list[int]] = {}
    for row in ranked:
        if row.get("family") != "min-mode" or row.get("cap") is None:
            continue
        grouped.setdefault(int(row["cap"]), []).append(int(row["family_rank"]))
    assert 30000 in grouped
    assert 48000 in grouped
    assert min(grouped[30000]) == 1
    assert min(grouped[48000]) == 1


def test_headline_names_the_lowest_par2_when_caps_differ():
    problem = {
        "id": "saddle-search/toy",
        "ranked": [
            {
                "label": "small-cap",
                "family": "min-mode",
                "cap": 30000,
                "par2_mean": 60000.0,
                "rank_cost": 60000.0,
                "unit": "force calls",
                "n": 10,
            },
            {
                "label": "finished",
                "family": "single-ended",
                "cap": 48000,
                "par2_mean": 100.0,
                "rank_cost": 100.0,
                "unit": "force calls",
                "n": 10,
            },
        ],
    }
    html = headline_table(problem)
    assert 'winner-name">finished' in html
    assert 'data-field="label">finished' in html
