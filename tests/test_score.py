import subprocess

import pytest

from optbench.score import (
    band_total,
    half_even,
    normalize_start,
    summarize_all,
    summarize_capped,
    summarize_divided,
)


def test_cap_splits_failures_from_the_mean():
    summary = summarize_capped([207, 1153, 10000], cap=10000)
    assert summary["force_calls"] == 680
    assert summary["force_calls_min"] == 207
    assert summary["force_calls_max"] == 1153
    assert summary["nfailed"] == 1


def test_summarize_all_keeps_a_failed_start_in_the_mean():
    summary = summarize_all([34, 68, 1846], nfailed=1)
    assert summary["nfailed"] == 1
    assert summary["force_calls_max"] == 1846
    assert summary["force_calls_min"] == 34
    assert summary["force_calls"] == round((34 + 68 + 1846) / 3)


def test_divided_heptamer_mean_is_the_band_total():
    summary = summarize_divided([80, 160, 80000], [False, False, False], images=8, cap=10000)
    assert summary["nfailed"] == 1
    assert summary["force_calls"] == 120
    assert summary["force_calls_raw"] == 120
    assert summary["force_calls_min"] == 80
    assert summary["force_calls_max"] == 160
    assert summary["force_calls_per_image"] == 15
    assert summary["call_unit"] == "total"


def test_half_even_matches_awk_printf():
    for value in (0.5, 1.5, 2.5, -1.5):
        awk = subprocess.check_output(
            ["awk", "-v", f"x={value}", 'BEGIN{printf "%.0f", x}'],
            text=True,
        )
        assert str(half_even(value)) == awk.strip()


def test_a_summary_of_no_counts_keeps_the_failed_tally():
    summary = summarize_all([], nfailed=3)
    assert summary == {
        "force_calls": None,
        "force_calls_min": None,
        "force_calls_max": None,
        "force_calls_median": None,
        "nfailed": 3,
    }


def test_a_negative_failed_tally_is_refused():
    with pytest.raises(ValueError, match="nfailed must be zero or positive"):
        summarize_all([34, 68], nfailed=-1)


def test_the_median_of_an_even_count_is_the_mean_of_the_middle_pair():
    summary = summarize_all([40, 10, 30, 20], nfailed=0)
    assert summary["force_calls_median"] == 25.0
    assert summarize_all([40, 10, 30], nfailed=0)["force_calls_median"] == 30.0


def test_a_capped_summary_of_no_finished_start_has_no_mean():
    assert summarize_capped([10000, 12000], cap=10000) == {
        "force_calls": None,
        "force_calls_min": None,
        "force_calls_max": None,
        "nfailed": 2,
    }
    assert summarize_capped([], cap=10000)["nfailed"] == 0


@pytest.mark.parametrize("cap", [0, -10000])
def test_a_cap_must_be_positive(cap):
    with pytest.raises(ValueError, match="cap must be positive"):
        summarize_capped([207], cap=cap)
    with pytest.raises(ValueError, match="cap must be positive"):
        summarize_divided([207], [False], images=8, cap=cap)


def test_a_band_summary_leaves_out_flagged_and_capped_starts_and_counts_them():
    counts = [80, 160, 80000, 240, 10000]
    flags = [False, True, False, False, False]
    summary = summarize_divided(counts, flags, images=8, cap=10000)
    # The flagged 160, the capped 80000, and the 10000 that sits on the cap.
    assert summary["nfailed"] == 3
    assert summary["force_calls"] == summary["force_calls_raw"] == 160
    assert (summary["force_calls_min"], summary["force_calls_max"], summary["force_calls_median"]) == (80, 240, 160.0)
    assert summary["force_calls_per_image"] == 20
    assert (
        summary["force_calls_per_image_min"],
        summary["force_calls_per_image_max"],
        summary["force_calls_per_image_median"],
    ) == (10, 30, 20.0)


def test_a_band_summary_of_only_failures_has_no_means_and_stays_in_totals():
    summary = summarize_divided([20000, 30000], [False, False], images=8, cap=10000)
    assert summary["nfailed"] == 2
    assert summary["call_unit"] == "total"
    assert {key: value for key, value in summary.items() if key not in {"nfailed", "call_unit"}} == dict.fromkeys(
        [
            "force_calls",
            "force_calls_min",
            "force_calls_max",
            "force_calls_median",
            "force_calls_raw",
            "force_calls_per_image",
            "force_calls_per_image_min",
            "force_calls_per_image_max",
            "force_calls_per_image_median",
        ]
    )


def test_a_band_summary_rounds_its_mean_half_to_even():
    assert summarize_divided([1, 2], [False, False], images=1, cap=10)["force_calls"] == 2
    assert summarize_divided([2, 3], [False, False], images=1, cap=10)["force_calls"] == 2


def test_a_band_summary_refuses_a_missing_image_count_and_unpaired_flags():
    for images in (0, -8):
        with pytest.raises(ValueError, match="images must be positive"):
            summarize_divided([80], [False], images=images, cap=10000)
    with pytest.raises(ValueError, match="counts and failed must have the same length"):
        summarize_divided([80, 160], [False], images=8, cap=10000)


def test_a_band_total_and_its_per_image_quotient_come_from_either_unit():
    assert band_total(30, images=8, call_unit="per_image") == (240.0, 30.0)
    assert band_total(240, images=8, call_unit="total") == (240.0, 30.0)
    assert band_total(240, images=None, call_unit="total") == (240.0, None)
    assert band_total(240, images=0, call_unit="total") == (240.0, None)


def test_a_band_count_needs_a_known_unit_and_an_image_count_to_scale():
    with pytest.raises(ValueError, match="per-image count has no image count"):
        band_total(30, images=None, call_unit="per_image")
    with pytest.raises(ValueError, match="unknown call unit per_atom"):
        band_total(30, images=8, call_unit="per_atom")


def test_a_client_band_start_keeps_the_counter_total_and_leaves_the_caller_record_alone():
    raw = {"force_calls": 10, "raw": 80, "count_key": "force_calls_neb", "success": True, "start": "min03"}
    start = normalize_start(raw, 8)
    assert start == {
        "force_calls": 80.0,
        "raw": 80,
        "count_key": "force_calls_neb",
        "success": True,
        "start": "min03",
        "call_unit": "total",
        "per_image": 10.0,
    }
    assert raw["force_calls"] == 10
    assert "per_image" not in raw


def test_a_client_band_start_without_a_counter_total_is_scaled_from_its_per_image_count():
    start = normalize_start({"force_calls": 10, "count_key": "force_calls_neb", "success": True}, 8)
    assert (start["force_calls"], start["per_image"]) == (80.0, 10.0)
    unscaled = normalize_start({"force_calls": 10, "raw": 80, "count_key": "force_calls_neb"}, None)
    assert unscaled["force_calls"] == 80.0
    assert "per_image" not in unscaled


def test_a_client_band_start_with_no_count_at_all_is_refused_as_a_value_error():
    with pytest.raises(ValueError, match="no raw count and no force calls"):
        normalize_start({"count_key": "force_calls_neb", "success": True, "start": "min03"}, 8)


def test_a_per_image_start_is_put_back_on_the_band_total():
    labelled = normalize_start({"force_calls": 12.5, "call_unit": "per_image", "success": True}, 8)
    assert (labelled["force_calls"], labelled["per_image"], labelled["call_unit"]) == (100.0, 12.5, "total")
    track = normalize_start({"force_calls": 12.5, "success": True}, 8, per_image_track=True)
    assert (track["force_calls"], track["per_image"]) == (100.0, 12.5)


def test_a_total_start_takes_the_first_count_it_finds():
    assert normalize_start({"force_calls": 90, "raw": 70, "per_search": 5}, 8)["force_calls"] == 90.0
    assert normalize_start({"raw": 70, "per_search": 5}, 8)["force_calls"] == 70.0
    assert normalize_start({"per_search": 5}, 8)["force_calls"] == 5.0
    assert normalize_start({"force_calls": 90}, None) == {"force_calls": 90.0, "call_unit": "total"}


def test_a_start_with_no_count_comes_back_unchanged():
    bare = {"success": True, "start": "min07"}
    assert normalize_start(bare, 8) == bare


def test_reuse_writes_into_the_caller_record_and_the_default_does_not():
    raw = {"force_calls": 10, "call_unit": "per_image"}
    assert normalize_start(dict(raw), 8, reuse=True)["force_calls"] == 80.0
    kept = dict(raw)
    assert normalize_start(kept, 8) is not kept
    assert kept == raw
    assert normalize_start(kept, 8, reuse=True) is kept
    assert kept["force_calls"] == 80.0


def test_a_start_that_cannot_be_scaled_says_why():
    with pytest.raises(ValueError, match="start is not a record"):
        normalize_start([226, True], 8)
    with pytest.raises(ValueError, match="per-image count has no image count"):
        normalize_start({"force_calls": 10, "call_unit": "per_image"}, None)
    with pytest.raises(ValueError, match="per-image count has no force calls"):
        normalize_start({"call_unit": "per_image", "success": True}, 8)
    with pytest.raises(ValueError, match="many"):
        normalize_start({"force_calls": "many"}, 8)
    with pytest.raises(ValueError, match="many"):
        normalize_start({"count_key": "force_calls_neb", "raw": "many"}, 8)
