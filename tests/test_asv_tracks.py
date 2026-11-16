"""Edge cases for one asv track turned into start rows."""

import pytest

from asv.results import _compatible_results

from optbench.asv_score import TRACKS, _column, track_rows

CLASS_NAME = "Toy"
PREFIX = f"bench.{CLASS_NAME}"


class _Result:
    def __init__(self, columns: dict[str, list], params: list[list[str]] | None = None):
        self.commit_hash = "abc"
        self.benchmark_version = {}
        count = 1
        self._params = [["'0'"], ["'a'", "'b'"]] if params is None else params
        for axis in self._params:
            count *= len(axis)
        self._results = {}
        self._benchmark_params = {}
        for method in TRACKS:
            key = f"{PREFIX}.{method}"
            fill = columns.get(method, 0.0 if method != "track_force_calls" else 3.0)
            if method == "track_success" and method not in columns:
                fill = 1.0
            if isinstance(fill, list):
                values = list(fill)
            else:
                values = [fill] * count
            self._results[key] = values
            self._benchmark_params[key] = self._params
            self.benchmark_version[key] = "v"

    def get_all_result_keys(self):
        return list(self._results)

    def get_result_params(self, key):
        return self._benchmark_params[key]

    def get_result_value(self, key, params):
        return _compatible_results(self._results[key], self._benchmark_params[key], params)


def test_column_copies_the_stored_list_and_follows_a_new_order():
    result = _Result({"track_force_calls": [1.0, 2.0]})
    key = f"{PREFIX}.track_force_calls"
    params = result.get_result_params(key)
    copied = _column(result, key, params)
    assert copied == [1.0, 2.0]
    assert copied is not result._results[key]
    copied[0] = 99.0
    assert result._results[key] == [1.0, 2.0]

    asked = [["'0'"], ["'b'", "'a'"]]
    assert _column(result, key, asked) == result.get_result_value(key, asked) == [2.0, 1.0]


def test_a_clean_track_is_one_row_per_parameter():
    rows = track_rows(_Result({}), CLASS_NAME)
    assert [(row["params"], row["force_calls"], row["success"], row["failure"]) for row in rows] == [
        (("0", "a"), 3, True, None),
        (("0", "b"), 3, True, None),
    ]


def test_a_missing_track_value_is_rejected():
    result = _Result({"track_success": [1.0, None]})
    with pytest.raises(SystemExit, match="was skipped or failed"):
        track_rows(result, CLASS_NAME)


def test_a_success_cannot_carry_a_failure_track():
    result = _Result({"track_failure_cap": [1.0, 0.0]})
    with pytest.raises(SystemExit, match="is a success with failure tracks"):
        track_rows(result, CLASS_NAME)


def test_a_failure_needs_exactly_one_class():
    none = _Result({"track_success": [0.0, 1.0], "track_failure_cap": [0.0, 0.0]})
    with pytest.raises(SystemExit, match="failure tracks are"):
        track_rows(none, CLASS_NAME)
    both = _Result(
        {
            "track_success": [0.0, 1.0],
            "track_failure_cap": [1.0, 0.0],
            "track_failure_crash": [1.0, 0.0],
        }
    )
    with pytest.raises(SystemExit, match="failure tracks are"):
        track_rows(both, CLASS_NAME)


def test_a_success_bit_must_be_zero_or_one():
    result = _Result({"track_success": [0.5, 1.0]})
    with pytest.raises(SystemExit, match="success track is"):
        track_rows(result, CLASS_NAME)


def test_whole_force_counts_are_ints_and_quoted_axes_unwrap():
    result = _Result(
        {"track_force_calls": [4.0, 4.5]},
        params=[["'1'", "'2'"]],
    )
    rows = track_rows(result, CLASS_NAME)
    assert [(row["params"], row["force_calls"], row["version"]) for row in rows] == [
        (("1",), 4, "v"),
        (("2",), 4.5, "v"),
    ]


def test_an_integer_success_bit_is_zero_or_one():
    result = _Result(
        {
            "track_success": [1, 0],
            "track_force_calls": [2.0, 3.5],
            "track_failure_cap": [0, 1],
        },
        params=[["'a'", "'b'"]],
    )
    rows = track_rows(result, CLASS_NAME)
    assert [(row["success"], row["force_calls"], row["failure"]) for row in rows] == [
        (True, 2, None),
        (False, 3.5, "cap"),
    ]


@pytest.mark.parametrize("name", ["crash", "divergence", "cap", "wrong_stationary"])
def test_a_failed_start_records_that_class(name: str):
    result = _Result(
        {"track_success": [0.0], f"track_failure_{name}": [1.0]},
        params=[["'a'"]],
    )
    rows = track_rows(result, CLASS_NAME)
    assert rows[0]["success"] is False
    assert rows[0]["failure"] == name


def test_two_failure_classes_name_the_first_pair_in_order():
    result = _Result(
        {
            "track_success": [0.0],
            "track_failure_crash": [1.0],
            "track_failure_cap": [1.0],
        },
        params=[["'a'"]],
    )
    with pytest.raises(SystemExit, match=r"failure tracks are \['crash', 'cap'\]"):
        track_rows(result, CLASS_NAME)


def test_a_track_with_no_axes_is_one_row():
    result = _Result({}, params=[])
    rows = track_rows(result, CLASS_NAME)
    assert [(row["params"], row["force_calls"], row["success"], row["failure"]) for row in rows] == [
        ((), 3, True, None),
    ]


def test_a_nan_track_value_is_rejected():
    result = _Result({"track_success": [float("nan")]}, params=[["'a'"]])
    with pytest.raises(SystemExit, match="was skipped or failed"):
        track_rows(result, CLASS_NAME)


def test_a_short_column_is_rejected():
    result = _Result({"track_force_calls": [3.0]})
    with pytest.raises(SystemExit, match="has 1 values for 2 parameters"):
        track_rows(result, CLASS_NAME)
