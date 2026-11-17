"""Rate intervals, quantiles, and the seeded bootstrap behind a published score."""

import random
import statistics

import numpy as np
import pytest

from optbench import score
from optbench.failures import FAILURES
from optbench.score import (
    BOOTSTRAP_SEED,
    _index_block,
    _resample_median_iqr,
    _resample_median_iqr_py,
    bootstrap_interval,
    distribution_summary,
    quantile,
    rate_interval,
    score_v2,
)

# PAR-2 costs of a thirteen-start Pt-heptamer CI-NEB campaign. None reached the cap of 10000.
COSTS = [226.0, 218.0, 314.0, 314.0, 314.0, 442.0, 434.0, 346.0, 370.0, 370.0, 362.0, 338.0, 330.0]
# The same campaign when two starts had hit the cap and each cost twice the cap.
HEAVY = [*COSTS, 20000.0, 20000.0]


@pytest.fixture
def fresh_stream(monkeypatch):
    """Forget the index array a previous sample of the same shape left behind."""
    monkeypatch.setattr(_index_block, "_reuse", None, raising=False)


@pytest.mark.parametrize(
    ("successes", "starts", "low", "high"),
    [
        (0, 10, 0.0, 0.2775),
        (1, 10, 0.0179, 0.4042),
        (5, 10, 0.2366, 0.7634),
        (10, 10, 0.7225, 1.0),
        (20, 20, 0.8389, 1.0),
    ],
)
def test_the_rate_interval_is_the_wilson_interval_at_95_percent(successes, starts, low, high):
    got_low, got_high = rate_interval(successes, starts)
    assert got_low == pytest.approx(low, abs=5e-5)
    assert got_high == pytest.approx(high, abs=5e-5)


def test_no_success_and_all_successes_pin_one_end_of_the_interval():
    for starts in range(1, 61):
        assert rate_interval(0, starts)[0] == 0.0
        assert rate_interval(starts, starts)[1] == 1.0


def test_the_rate_interval_brackets_the_observed_rate_and_mirrors_failures():
    for starts in (1, 2, 7, 13, 50, 59):
        for successes in range(starts + 1):
            low, high = rate_interval(successes, starts)
            assert 0.0 <= low <= successes / starts <= high <= 1.0
            mirror_low, mirror_high = rate_interval(starts - successes, starts)
            assert low == pytest.approx(1.0 - mirror_high, abs=1e-12)
            assert high == pytest.approx(1.0 - mirror_low, abs=1e-12)


def test_the_rate_interval_narrows_as_the_same_rate_is_seen_on_more_starts():
    widths = [high - low for low, high in (rate_interval(n // 2, n) for n in (10, 40, 160, 640))]
    assert widths == sorted(widths, reverse=True)
    assert len(set(widths)) == len(widths)


def test_a_larger_quantile_widens_the_rate_interval():
    narrow = rate_interval(6, 10, z=1.0)
    wide = rate_interval(6, 10, z=2.58)
    assert wide[0] < narrow[0] < 0.6 < narrow[1] < wide[1]


def test_the_rate_interval_rejects_counts_that_cannot_happen():
    for starts in (0, -3):
        with pytest.raises(ValueError, match="starts must be positive"):
            rate_interval(0, starts)
    with pytest.raises(ValueError, match="successes must lie between 0 and starts"):
        rate_interval(-1, 5)
    with pytest.raises(ValueError, match="successes must lie between 0 and starts"):
        rate_interval(6, 5)


@pytest.mark.parametrize("fraction", [0.0, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 1.0])
def test_a_quantile_interpolates_linearly_between_the_sorted_values(fraction):
    shuffled = [HEAVY[index] for index in (4, 14, 0, 9, 2, 11, 13, 1, 7, 3, 12, 5, 10, 8, 6)]
    assert quantile(shuffled, fraction) == pytest.approx(float(np.quantile(shuffled, fraction)), rel=1e-12)


def test_a_quantile_of_one_value_is_that_value_and_an_empty_sample_is_refused():
    for fraction in (0.0, 0.3, 1.0):
        assert quantile([336], fraction) == 336.0
    with pytest.raises(ValueError, match="values is empty"):
        quantile([], 0.5)


def test_the_bootstrap_interval_brackets_its_point_and_repeats_from_its_seed():
    first = bootstrap_interval(COSTS, statistics.fmean, resamples=300, seed=11)
    again = bootstrap_interval(COSTS, statistics.fmean, resamples=300, seed=11)
    other = bootstrap_interval(COSTS, statistics.fmean, resamples=300, seed=12)
    assert first == again
    assert first != other
    point, low, high = first
    assert point == pytest.approx(statistics.fmean(COSTS))
    assert min(COSTS) <= low <= point <= high <= max(COSTS)


def test_a_published_interval_can_be_recomputed_from_its_seed():
    resamples, seed = 250, 20260101
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        draw = [COSTS[rng.randrange(len(COSTS))] for _ in range(len(COSTS))]
        medians.append(statistics.median(draw))
    expected = (float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)))
    point, low, high = bootstrap_interval(COSTS, statistics.median, resamples=resamples, seed=seed)
    assert point == statistics.median(COSTS)
    assert (low, high) == pytest.approx(expected, rel=1e-12)


def test_a_looser_alpha_gives_an_interval_inside_the_stricter_one():
    _, strict_low, strict_high = bootstrap_interval(HEAVY, statistics.fmean, resamples=400, alpha=0.01)
    _, loose_low, loose_high = bootstrap_interval(HEAVY, statistics.fmean, resamples=400, alpha=0.5)
    assert strict_low <= loose_low <= loose_high <= strict_high
    assert strict_low < strict_high


def test_a_sample_of_one_repeated_value_has_no_interval_width():
    assert bootstrap_interval([4.0] * 6, statistics.median, resamples=20, seed=1) == (4.0, 4.0, 4.0)


def test_the_bootstrap_refuses_an_empty_sample_or_no_resamples():
    with pytest.raises(ValueError, match="values is empty"):
        bootstrap_interval([], statistics.fmean)
    for resamples in (0, -5):
        with pytest.raises(ValueError, match="resamples must be positive"):
            bootstrap_interval(COSTS, statistics.fmean, resamples=resamples)


@pytest.mark.parametrize("count", [1, 2, 3, 7, 8, 13, 100])
def test_the_index_stream_is_the_randrange_stream_in_order(count, fresh_stream):
    # 3000 draws run past the 624 words one twist of the generator holds.
    needed = 3000
    rng = random.Random(BOOTSTRAP_SEED)
    expected = [rng.randrange(count) for _ in range(needed)]
    assert _index_block(count, needed, BOOTSTRAP_SEED).tolist() == expected


def test_the_index_stream_is_the_same_when_numpy_will_not_take_the_state(monkeypatch, fresh_stream):
    wanted = _index_block(13, 2500, 99).tolist()

    class Refuses:
        @property
        def state(self):
            return {}

        @state.setter
        def state(self, _value):
            raise ValueError("this generator takes no state")

    monkeypatch.setattr(score.np.random, "MT19937", Refuses)
    assert score._tempered_source(99) is None
    monkeypatch.setattr(_index_block, "_reuse", None, raising=False)
    assert _index_block(13, 2500, 99).tolist() == wanted


def test_a_second_sample_of_one_shape_reuses_the_same_read_only_stream(fresh_stream):
    first = _index_block(13, 400, 5)
    assert _index_block(13, 400, 5) is first
    assert not first.flags.writeable
    assert _index_block(13, 401, 5) is not first
    assert _index_block(13, 401, 5).tolist()[:400] == first.tolist()


@pytest.mark.parametrize("size", [1, 2, 3, 8, 13, 15])
def test_the_vectorised_resampling_matches_one_index_at_a_time(size, fresh_stream):
    values = HEAVY[:size]
    fast = _resample_median_iqr(values, 120, 7)
    slow = _resample_median_iqr_py(values, 120, 7)
    assert fast[0] == pytest.approx(slow[0], rel=1e-12)
    assert fast[1] == pytest.approx(slow[1], rel=1e-12, abs=1e-9)


def test_the_one_at_a_time_resampling_follows_the_documented_recipe():
    values = [30.0, 10.0, 20.0, 50.0, 40.0, 60.0]
    resamples, seed = 80, 3
    rng = random.Random(seed)
    medians, widths = [], []
    for _ in range(resamples):
        draw = sorted(values[rng.randrange(len(values))] for _ in range(len(values)))
        medians.append(statistics.median(draw))
        widths.append(float(np.quantile(draw, 0.75) - np.quantile(draw, 0.25)))
    got_medians, got_widths = _resample_median_iqr_py(values, resamples, seed)
    assert got_medians == sorted(medians)
    assert got_widths == pytest.approx(sorted(widths), rel=1e-12, abs=1e-12)


def test_one_value_resamples_to_itself_with_no_spread(fresh_stream):
    medians, widths = _resample_median_iqr([336.0], 25, 4)
    assert medians == [336.0] * 25
    assert widths == [0.0] * 25
    assert _resample_median_iqr_py([336.0], 25, 4) == (medians, widths)


def test_a_summary_of_costs_orders_its_percentiles_and_names_its_seed():
    summary = distribution_summary(HEAVY, resamples=500, seed=17)
    assert summary["n"] == len(HEAVY)
    assert summary["mean"] == pytest.approx(statistics.fmean(HEAVY))
    assert summary["median"] == statistics.median(HEAVY)
    assert summary["iqr"] == pytest.approx(summary["q75"] - summary["q25"])
    assert summary["q25"] <= summary["median"] <= summary["q75"]
    assert summary["median_low"] <= summary["median"] <= summary["median_high"]
    assert 0.0 <= summary["iqr_low"] <= summary["iqr_high"]
    assert summary["bootstrap_resamples"] == 500
    assert summary["bootstrap_seed"] == 17
    assert distribution_summary(HEAVY, resamples=500, seed=17) == summary


def test_a_summary_of_one_cost_has_no_spread():
    summary = distribution_summary([336.0], resamples=30)
    for key in ("mean", "median", "q25", "q75", "median_low", "median_high"):
        assert summary[key] == 336.0
    for key in ("iqr", "iqr_low", "iqr_high"):
        assert summary[key] == 0.0
    assert summary["n"] == 1


def test_a_summary_matches_the_straight_recomputation_of_its_draws(fresh_stream):
    resamples, seed = 150, 29
    rng = random.Random(seed)
    medians, widths = [], []
    for _ in range(resamples):
        draw = [COSTS[rng.randrange(len(COSTS))] for _ in range(len(COSTS))]
        medians.append(statistics.median(draw))
        widths.append(float(np.quantile(draw, 0.75) - np.quantile(draw, 0.25)))
    summary = distribution_summary(COSTS, resamples=resamples, seed=seed)
    assert summary["median_low"] == pytest.approx(float(np.quantile(medians, 0.025)), rel=1e-12)
    assert summary["median_high"] == pytest.approx(float(np.quantile(medians, 0.975)), rel=1e-12)
    assert summary["iqr_low"] == pytest.approx(float(np.quantile(widths, 0.025)), rel=1e-9)
    assert summary["iqr_high"] == pytest.approx(float(np.quantile(widths, 0.975)), rel=1e-9)


def test_a_summary_refuses_an_empty_sample_or_no_resamples():
    with pytest.raises(ValueError, match="values is empty"):
        distribution_summary([])
    with pytest.raises(ValueError, match="resamples must be positive"):
        distribution_summary(COSTS, resamples=0)


def test_a_score_whose_failure_classes_lose_a_start_is_not_published(monkeypatch):
    records = [
        {"force_calls": 226.0, "success": True},
        {"force_calls": 10000.0, "success": False, "failure": "cap"},
    ]
    assert score_v2(records, 10000, resamples=10)["n_cap"] == 1
    monkeypatch.setattr(score, "failure_tally", lambda _records, _cap: {name: 0 for name in FAILURES})
    with pytest.raises(RuntimeError, match="failure classes dropped a start"):
        score_v2(records, 10000, resamples=10)


def test_a_sample_too_long_for_the_32_bit_stream_is_resampled_one_index_at_a_time(monkeypatch):
    class Huge(list):
        def __len__(self):
            return 1 << 32

    seen = []

    def one_at_a_time(values, resamples, seed):
        seen.append((len(values), resamples, seed))
        return [0.0], [0.0]

    monkeypatch.setattr(score, "_resample_median_iqr_py", one_at_a_time)
    assert _resample_median_iqr(Huge(), 3, 4) == ([0.0], [0.0])
    assert seen == [(1 << 32, 3, 4)]
