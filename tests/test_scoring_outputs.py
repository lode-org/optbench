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
    build_board,
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


_DOCUMENT = None


def _placed(problem_id: str) -> list[tuple]:
    global _DOCUMENT
    if _DOCUMENT is None:
        _DOCUMENT = build_board(ROOT)
    problem = _DOCUMENT["problems"][problem_id]
    return [
        (row["family_rank"], row["success_rate"], row["par2_mean"], row["label"])
        for row in problem["ranked"]
        if problem_id == "neb/Pt-heptamer" or row["family"] == "band"
    ]


def test_pt_heptamer_ranks_by_par2_and_shares_tied_places():
    """Equal PAR-2 shares a rank. The next distinct cost skips the tied place."""
    assert _placed("neb/Pt-heptamer") == [
        (1, 1.0, 318.9230769230769, "geodesic (1.0.0) · CI-NEB-LBFGS geodesic"),
        (1, 1.0, 319.53846153846155, "eOn (develop 500350d9) · LBFGS"),
        (2, 1.0, 336.7692307692308, "Eon svn (r2549) · LBFGS"),
        (2, 1.0, 543.5384615384615, "geodesic (1.0.0) · NEB-FIRE geodesic"),
        (3, 1.0, 561.4615384615385, "COMPASS (1.0.0) · COMPASS"),
        (3, 1.0, 606.9230769230769, "ASE (3.29.0) · NEB-FIRE"),
        (4, 1.0, 678.9230769230769, "Eon svn (r2549) · FIRE"),
        (4, 1.0, 678.9230769230769, "eOn (develop 500350d9) · FIRE"),
        (6, 1.0, 727.5384615384615, "eOn (develop f6817b53, 3.6.0) · CG"),
        (4, 1.0, 732.4615384615385, "rgsaddle (0.2.2) · S-IDPP CI-NEB"),
        (5, 1.0, 735.5384615384615, "rgsaddle (0.2.2) · IDPP CI-NEB"),
        (6, 1.0, 736.7692307692307, "rgsaddle (0.2.2) · CI-NEB"),
        (7, 1.0, 759.6923076923077, "rgsaddle (0.2.2) · CI-NEB energy-weighted"),
        (8, 1.0, 760.7692307692307, "rgsaddle (0.2.2) · OCI-NEB"),
        (9, 1.0, 762.6153846153846, "geodesic (1.0.0) · CI-NEB geodesic"),
        (7, 1.0, 929.3846153846154, "eOn (SVN r2549) · CG"),
        (8, 1.0, 1141.076923076923, "eOn (develop 500350d9) · CG"),
        (9, 1.0, 1322.0, "Eon svn (r2549) · QuickMin"),
        (9, 1.0, 1322.0, "eOn (develop 500350d9) · QuickMin"),
        (10, 1.0, 1398.4615384615386, "LAMMPS (30 Sep 2026) · CI-NEB-fire"),
        (11, 1.0, 1521.1538461538462, "TSASE (1.0) · CI-NEB"),
        (12, 1.0, 3417.6923076923076, "LAMMPS (30 Sep 2026) · CI-NEB-quickmin"),
        (11, 1.0, 3517.6923076923076, "eOn (3.6.0) · LBFGS"),
        (13, 0.0, 20000.0, "TSASE (1.0) · NEB"),
        (13, 0.0, 20000.0, "rgsaddle (0.2.2) · NEB"),
    ]


def test_lj38_neb_ties_share_the_cap_rank():
    """Three unsolved NEB rows share one rank. Nothing is placed after them."""
    band = _placed("double-ended-saddle-search/lj38")
    assert band[-3:] == [
        (24, 0.0, 5000.0, "TSASE (1.0) · NEB"),
        (24, 0.0, 5000.0, "eOn (3.6.0) · energy-weighted CI-NEB"),
        (24, 0.0, 5000.0, "rgsaddle (0.2.2) · NEB"),
    ]
    assert band[-4][0] == 23
    assert all(rank <= 24 for rank, _rate, _par2, _label in band)
