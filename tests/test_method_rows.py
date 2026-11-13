"""A ranked row is one code, one version, and one stock method."""

import tomllib
from pathlib import Path

import pytest

from optbench.board import build_board, rank_violations
from optbench.campaign import load_template
from optbench.families import family_of, finite_difference_calls, method_name, named_template


def test_fd_calls_use_one_force_call_per_coordinate():
    assert finite_difference_calls(1.0, 114) == 114.0
    assert finite_difference_calls(80.881, 114) == 80.881 * 114
    assert finite_difference_calls(0.0, None) == 0.0
    assert finite_difference_calls(1.0, None) is None


def test_eon_band_variants_are_not_one_lbfgs_method():
    names = {
        method_name(load_template(ROOT, template_id))
        for template_id in (
            "eon/develop-de-2ts",
            "eon/develop-de-2ts-accept-energy",
            "eon/develop-de-2ts-oci",
            "eon/develop-de-2ts-ew",
        )
    }
    assert names == {
        "CI-NEB ljcluster",
        "CI-NEB accept energy",
        "OCI-NEB",
        "energy-weighted CI-NEB",
    }
    recorded = {"algorithm": "LBFGS", "code": "eOn", "code_version": "3.6.0"}
    labeled = named_template(recorded, load_template(ROOT, "eon/develop-de-2ts-oci"))
    assert method_name(labeled) == "OCI-NEB"
    assert labeled["code_version"] == "3.6.0"
    assert "method" not in recorded


def test_eon_36_band_methods_stay_on_the_board():
    document = _document()
    expected = {
        "double-ended-saddle-search/lj38_2ts_separation": {
            "OCI-NEB",
            "energy-weighted CI-NEB",
            "CI-NEB ljcluster",
            "CI-NEB accept energy",
        },
        "double-ended-saddle-search/lj38": {
            "OCI-NEB",
            "energy-weighted CI-NEB",
            "CI-NEB lj/cut",
            "CI-NEB accept energy",
        },
        "double-ended-saddle-search/pt-island": {
            "OCI-NEB",
            "energy-weighted CI-NEB",
        },
    }
    for problem_id, methods in expected.items():
        ranked = document["problems"][problem_id]["ranked"]
        found = {row["method"] for row in ranked if str(row.get("code_version") or "") == "3.6.0"}
        missing = methods - found
        assert not missing, (problem_id, missing)


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
    assert not any(str(row.get("template") or "").endswith("-tuned") for row in withheld)
