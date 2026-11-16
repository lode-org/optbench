"""Published scores of two short Pt-heptamer campaigns.

The median interval is the fixed bootstrap. A change to the packed fields,
the record class, or the index stream moves these numbers.
"""

import json
from pathlib import Path

from optbench.board import (
    _pack_score,
    _rank_key,
    _record_plain,
    _score_packed,
    _assign_family_ranks,
)
from optbench.site_pages import code_summaries

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "campaign"
# problems/neb/Pt-heptamer.toml
CAP = 10000


def _raws(name: str) -> list[dict]:
    path = CAMPAIGN / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scored(name: str) -> tuple[list, list[tuple], dict]:
    records = [_record_plain(raw, CAP, None) for raw in _raws(name)]
    packed = _pack_score(records)
    return records, packed, _score_packed(packed, CAP)


def test_a_plain_success_packs_its_call_count_and_keeps_the_bootstrap_median():
    records, packed, summary = _scored("eon-svn-r2549-neb-pt-ci.starts.jsonl")
    assert {type(record).__name__ for record in records} == {"_StartPlain"}
    assert packed[0] == (226.0, True, None, None, None)
    assert [item[0] for item in packed] == [
        226.0, 218.0, 314.0, 314.0, 314.0, 442.0, 434.0, 346.0, 370.0, 370.0, 362.0, 338.0, 330.0,
    ]
    assert summary["n"] == 13
    assert summary["nfailed"] == 0
    assert summary["success_rate"] == 1.0
    assert summary["par2_mean"] == 336.7692307692308
    assert summary["median"] == 338.0
    assert summary["median_low"] == 314.0
    assert summary["median_high"] == 370.0
    assert summary["median_calls"] == 338.0
    assert summary["min_calls"] == 218.0
    assert summary["max_calls"] == 442.0


def test_a_divergence_packs_its_failure_and_costs_twice_the_cap():
    records, packed, summary = _scored("rgsaddle-0.2.2-neb-Pt-heptamer--rgsaddle-neb.starts.jsonl")
    assert {type(record).__name__ for record in records} == {"_StartFailed"}
    assert packed[0] == (786.0, False, "divergence", None, None)
    assert {item[2] for item in packed} == {"divergence"}
    assert summary["nfailed"] == 13
    assert summary["n_divergence"] == 13
    assert summary["n_cap"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["par2_mean"] == 20000.0
    assert summary["median_calls"] == 962.0


def test_hessian_and_per_image_choose_their_record_class():
    success = _raws("eon-svn-r2549-neb-pt-ci.starts.jsonl")[0]
    failed = _raws("rgsaddle-0.2.2-neb-Pt-heptamer--rgsaddle-neb.starts.jsonl")[0]
    indexed = dict(success, hessian_index=1)
    indexed_failure = dict(failed, hessian_index=2)
    full = dict(success, per_image=3.5)
    records = [_record_plain(raw, CAP, None) for raw in (indexed, indexed_failure, full)]
    assert [type(record).__name__ for record in records] == [
        "_StartIndex",
        "_StartIndexFailed",
        "_StartFull",
    ]
    assert _pack_score(records) == [
        (226.0, True, None, None, None),
        (786.0, False, "divergence", None, None),
        (226.0, True, None, None, 3.5),
    ]


def test_a_tie_shares_a_rank_and_a_failure_does_not_take_it():
    _records, _rows, solved = _scored("eon-svn-r2549-neb-pt-ci.starts.jsonl")
    _failed_records, _failed_rows, failed = _scored(
        "rgsaddle-0.2.2-neb-Pt-heptamer--rgsaddle-neb.starts.jsonl"
    )
    first = {
        "family": "band",
        "label": "CI-NEB",
        "code": "eOn",
        "rank_cost": solved["par2_mean"],
        "par2_mean": solved["par2_mean"],
        "success_rate": solved["success_rate"],
        "n": solved["n"],
        "nfailed": solved["nfailed"],
    }
    twin = dict(first, label="CI-NEB again")
    divergence = {
        "family": "band",
        "label": "NEB",
        "code": "eOn",
        "rank_cost": solved["min_calls"],
        "par2_mean": failed["par2_mean"],
        "success_rate": failed["success_rate"],
        "n": failed["n"],
        "nfailed": failed["nfailed"],
    }
    rows = sorted([divergence, twin, first], key=_rank_key)
    assert [row["label"] for row in rows] == ["CI-NEB", "CI-NEB again", "NEB"]
    _assign_family_ranks(rows)
    assert [row["family_rank"] for row in rows] == [1, 1, 3]
    (summary,) = code_summaries({"id": "neb/Pt-heptamer"}, [first, divergence], [])
    assert summary["code"] == "eOn"
    assert summary["best"] == solved["par2_mean"]
    assert summary["median"] == (solved["par2_mean"] + solved["min_calls"]) / 2
    assert summary["unsolved"] is False
