"""Hand values for scoring, profile alignment, and the rank tables."""

import math

from optbench.board import _aligned_costs, _assign_family_ranks, profile_gap
from optbench.families import row_label, version_text
from optbench.profile import performance_profile
from optbench.score import band_total, score_v2, summarize_divided
from optbench.site_pages import leaderboard_html


def test_a_missing_start_is_not_dropped_from_the_profile():
    full = {"label": "full", "_profile": [("0", 10.0), ("1", 30.0)]}
    one = {"label": "one", "_profile": [("0", 10.0)]}
    costs, reason = _aligned_costs([full, one])
    assert costs is None
    assert reason == "The ranked rows do not share one start set."
    assert profile_gap([full, one]) == (
        "No performance profile. The ranked rows do not share one start set."
    )

    overlap = [
        {"label": "a", "_profile": [("0", 10.0), ("1", 10.0)]},
        {"label": "b", "_profile": [("0", 10.0), ("2", 10.0)]},
    ]
    assert _aligned_costs(overlap)[0] is None


def test_the_same_starts_line_up_by_name():
    rows = [
        {"label": "a", "_profile": [("1", 40.0), ("0", 10.0)]},
        {"label": "b", "_profile": [("0", 10.0), ("1", 20.0)]},
    ]
    costs, reason = _aligned_costs(rows)
    assert reason == ""
    assert costs["a"] == [40.0, 10.0]
    assert costs["b"] == [20.0, 10.0]


def test_a_tie_for_best_shares_the_rank():
    profile = performance_profile({"a": [10.0, 40.0], "b": [10.0, 20.0]})
    assert profile["reference"] == [10.0, 20.0]
    assert profile["ratios"]["a"] == [1.0, 2.0]
    assert profile["ratios"]["b"] == [1.0, 1.0]
    assert profile["rho"]["a"][0] == 0.5
    assert profile["rho"]["b"][0] == 1.0
    assert profile["rho"]["a"][1] == 1.0
    # log10(2) is the width of each doubling. The tie pays half a step at tau 1.
    step = math.log10(2.0)
    assert profile["area_log10"]["b"] == step * 4
    assert profile["area_log10"]["a"] == 0.75 * step + 3 * step

    rows = [
        {"family": "local", "rank_cost": 4.0},
        {"family": "local", "rank_cost": 4.0},
        {"family": "local", "rank_cost": 9.0},
        {"family": "global", "rank_cost": 4.0},
    ]
    _assign_family_ranks(rows)
    assert [row["family_rank"] for row in rows] == [1, 1, 3, 1]

    html = leaderboard_html(
        [
            {"group": "p", "code": "Alpha", "best": 4.0, "median": 4.0, "success": 1.0, "par2": 4.0, "code_href": "a.html"},
            {"group": "p", "code": "Beta", "best": 4.0, "median": 5.0, "success": 1.0, "par2": 4.0, "code_href": "b.html"},
            {"group": "p", "code": "Gamma", "best": 8.0, "median": 8.0, "success": 0.5, "par2": 12.0, "code_href": "c.html"},
        ],
        show_problem=False,
        bench="leader",
    )
    assert html.count('class="rank r1"') == 2
    assert 'class="rank r3"' in html
    assert html.count("is-best") == 2


def test_one_start_par2_is_the_call_or_twice_the_cap():
    done = score_v2([{"force_calls": 4, "success": True}], 10, resamples=4, seed=1)
    assert done["par2_mean"] == 4.0
    assert done["success_rate"] == 1.0
    assert done["n"] == 1
    assert done["median"] == 4.0
    assert done["iqr"] == 0.0
    assert done["rate_high"] == 1.0
    # Wilson interval for 1 of 1. The upper end is fixed at 1.
    z2 = 1.959963984540054 ** 2
    denom = 1.0 + z2
    center = (1.0 + z2 / 2.0) / denom
    margin = z2 / (2.0 * denom)
    assert done["rate_low"] == center - margin

    failed = score_v2(
        [{"force_calls": 7, "success": False, "failure": "cap"}],
        10,
        resamples=4,
        seed=1,
    )
    assert failed["par2_mean"] == 20.0
    assert failed["success_only_mean"] is None
    assert failed["rate_low"] == 0.0
    assert failed["n_cap"] == 1


def test_a_version_does_not_repeat_the_code_or_the_method():
    assert version_text("(develop 500350d9) · LBFGS") == "develop 500350d9"
    assert version_text("eOn (develop 500350d9)") == "develop 500350d9"
    assert version_text("1.0 (98aac60)") == "1.0 (98aac60)"
    assert version_text("3.6.0") == "3.6.0"
    assert row_label("eOn", "(develop 500350d9) · LBFGS", "LBFGS") == "eOn (develop 500350d9) · LBFGS"
    assert row_label("eOn", "eOn (develop 500350d9)", "LBFGS") == "eOn (develop 500350d9) · LBFGS"
    assert row_label("eOn (develop b0bd4d5a)", "3.6.0", "Basin hopping") == (
        "eOn (develop b0bd4d5a, 3.6.0) · Basin hopping"
    )
    assert row_label("EON (SVN r2549)", "r2549", "Basin hopping") == "EON (SVN r2549) · Basin hopping"
    assert row_label("TSASE", "1.0 (98aac60)", "quenchDimer+neb-QM") == (
        "TSASE (1.0 (98aac60)) · quenchDimer+neb-QM"
    )
    assert row_label("Gman", "r235", "GL-BFGS(Hess)") == "Gman (r235) · GL-BFGS(Hess)"


def test_a_half_converts_to_the_even_band_total():
    summary = summarize_divided([2, 3], [False, False], images=4, cap=100)
    assert summary["force_calls"] == 2
    assert summary["force_calls_per_image"] == 1
    assert summary["force_calls_per_image_min"] == 0
    assert summary["force_calls_per_image_max"] == 1
    total, per = band_total(36, images=8, call_unit="per_image")
    assert (total, per) == (288.0, 36.0)
