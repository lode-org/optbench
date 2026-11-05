"""A ranked row is one code, one version, and one stock method."""

import tomllib
from pathlib import Path

import pytest

from optbench.board import build_board, rank_violations
from optbench.families import family_of, finite_difference_calls, method_name


def test_fd_calls_use_one_force_call_per_coordinate():
    assert finite_difference_calls(1.0, 114) == 114.0
    assert finite_difference_calls(80.881, 114) == 80.881 * 114
    assert finite_difference_calls(0.0, None) == 0.0
    assert finite_difference_calls(1.0, None) is None


def test_a_parameter_record_is_not_the_method_name():
    blob = '{"counter":"harness","parameters":{"memory":100}}'
    assert method_name({"algorithm": "LBFGS"}, {"method": blob, "algorithm": "LBFGS"}) == "LBFGS"
    assert method_name({"method": "CI-NEB lj/cut", "algorithm": "CI-NEB"}, {"method": blob}) == "CI-NEB lj/cut"

ROOT = Path(__file__).resolve().parents[1]

EON = "eOn (develop 500350d9)"
METHODS = {
    "Lanczos improved": "min-mode",
    "Lanczos": "min-mode",
    "dimer": "min-mode",
    "energy-weighted CI-NEB": "band",
    "CI-NEB lj/cut": "band",
    "CI-NEB ljcluster": "band",
    "doubly nudged CI-NEB": "band",
    "OCI-NEB": "band",
}


_DOCUMENT = None


def _document():
    global _DOCUMENT
    if _DOCUMENT is None:
        _DOCUMENT = build_board(ROOT)
    return _DOCUMENT


def test_lj38_2ts_ranks_each_eon_method_and_keeps_tsase_split():
    document = _document()
    problem = document["problems"]["double-ended-saddle-search/lj38_2ts_separation"]
    labels = [row["label"] for row in problem["ranked"]]
    for method, family in METHODS.items():
        label = f"{EON} · {method}"
        assert label in labels
        row = next(item for item in problem["ranked"] if item["label"] == label)
        assert row["family"] == family
        assert row["origin"] == "harness"
        assert row["hessian_calls"] == 0.0
        assert row["fd_calls"] == 0.0
        assert family_of(method, suite="double-ended-saddle-search", kind=row["kind"]) == family
    assert "eOn (3.5.0) · CI-NEB lj/cut" in labels
    tsase = [row for row in problem["ranked"] if row["label"].startswith("TSASE (1.0) ·")]
    assert {row["label"] for row in tsase} >= {"TSASE (1.0) · CI-NEB", "TSASE (1.0) · NEB"}
    families = {row["family"] for row in problem["ranked"]}
    assert "band" in families and "min-mode" in families
    archived = [row for row in problem["ranked"] if row["origin"] == "archive"]
    assert archived
    problem_file = ROOT / "problems" / "double-ended-saddle-search" / "lj38_2ts_separation.toml"
    cap = tomllib.loads(problem_file.read_text(encoding="utf-8"))["convergence"]["max_force_calls"]
    for row in archived:
        n, nfailed = row["n"], row["nfailed"]
        assert row["par2_mean"] == pytest.approx(((n - nfailed) * row["published"] + nfailed * 2 * cap) / n)
        assert row["rank_cost"] == row["par2_mean"]
    assert any(row["nfailed"] > 0 for row in archived)
    assert rank_violations(document) == []


def test_adjusted_and_short_rows_stay_off_the_rank():
    document = _document()
    for problem in document["problems"].values():
        for row in problem["ranked"]:
            assert row["short"] is False
            tuning = str(row.get("tuning") or row.get("profile") or "").lower()
            assert tuning not in {"tuned", "adjusted"}
            template = str(row.get("template") or "")
            assert not template.endswith("-tuned")
            assert not template.endswith("-adjusted")
            assert "ranked false" not in str(row.get("short_reason") or "")
    withheld = document["withheld"]
    assert any(str(row.get("template") or "").endswith("-tuned") for row in withheld)
