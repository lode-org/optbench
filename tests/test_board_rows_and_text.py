"""How the board reads one start, and the sentences it writes about a problem.

A start comes from a campaign file as a dict, from an asv track as three
numbers, or from an older file in a different layout. The board keeps each as a
small record that reads like the dict it replaces. The text helpers write the
gap sentences, the winner cell and the profile notice that go on the pages, so
their wording and their escaping are checked as published.
"""

from __future__ import annotations

import pytest

from optbench.board import (
    _failure_label,
    _hessian_calls,
    _joined_names,
    _make_start,
    _missing_hessian_index,
    _profile_reason,
    _reason_text,
    _record_from_raw,
    _record_from_track,
    _success_needs_hessian,
    _winner_html,
    board_rows,
    gap_paragraph,
    rank_violations,
)


def test_a_start_record_reads_like_the_dict_it_replaces():
    record = _make_start(120.0, False, "r1", failure="cap")
    assert record["force_calls"] == 120.0
    assert record["failure"] == "cap"
    assert record.get("hessian_index") is None
    assert record.get("hessian_index", 7) == 7
    assert "failure" in record
    assert "hessian_index" not in record
    with pytest.raises(KeyError):
        record["hessian_index"]
    assert record == {"force_calls": 120.0, "success": False, "start": "r1", "call_unit": "total", "failure": "cap"}


def test_a_field_the_record_has_a_slot_for_but_never_set_reads_as_absent():
    record = _make_start(10.0, False, "s", reasons=["crash"])
    assert record.get("per_image", 7) == 7
    assert record.get("per_image") is None
    assert "per_image" not in record
    assert record.get("reasons") == ["crash"]


def test_a_start_with_no_name_still_holds_its_count_outcome_and_name_slot():
    record = _make_start(5.0, True, None, call_unit=None)
    assert record == {"force_calls": 5.0, "success": True, "start": None}
    assert "start" in record
    assert record["start"] is None
    assert "call_unit" not in record


def test_two_records_are_equal_when_every_field_they_hold_is_equal_whatever_their_layout():
    plain = _make_start(1.0, True, "a")
    assert plain == _make_start(1.0, True, "a")
    assert plain != _make_start(1.0, True, "a", hessian_index=1)
    assert plain != _make_start(2.0, True, "a")
    assert _make_start(1.0, True, "a", hessian_index=1) == _make_start(1.0, True, "a", hessian_index=1)
    assert plain != 5
    assert plain != "a"
    assert plain != [1.0, True, "a"]


@pytest.mark.parametrize(
    ("fields", "name"),
    [
        ({}, "_StartPlain"),
        ({"hessian_index": 1}, "_StartIndex"),
        ({"failure": "cap"}, "_StartFailed"),
        ({"hessian_index": 1, "failure": "cap"}, "_StartIndexFailed"),
        ({"per_image": 12.0}, "_StartFull"),
        ({"reasons": ["crash"]}, "_StartFull"),
        ({"inferred": True}, "_StartFull"),
    ],
)
def test_a_start_keeps_only_the_fields_it_has(fields, name):
    assert type(_make_start(10.0, False, "s", **fields)).__name__ == name


def test_a_full_record_keeps_the_per_image_count_the_reasons_and_whether_the_class_was_inferred():
    record = _make_start(10.0, False, "s", per_image=2.5, reasons=["force_norm"], inferred=True, hessian_index=1)
    assert record["per_image"] == 2.5
    assert record["reasons"] == ["force_norm"]
    assert record["failure_inferred"] is True
    assert record["hessian_index"] == 1
    assert _make_start(10.0, False, "s", failure="cap").get("failure_inferred") is None


@pytest.mark.parametrize("calls", [120, 120.0])
def test_a_track_count_that_is_an_int_or_a_float_is_a_float_count(calls):
    record = _record_from_track(calls, True, None, "0000", None)
    assert record["force_calls"] == 120.0
    assert type(record["force_calls"]) is float
    assert (record["success"], record["start"]) == (True, "0000")


@pytest.mark.parametrize("calls", [True, "120", None, float("nan"), float("inf"), [120.0]])
def test_a_track_count_that_is_not_a_finite_number_is_no_start(calls):
    assert _record_from_track(calls, True, None, "0000", None) is None


def test_a_track_start_at_the_cap_is_a_cap_failure_even_if_it_said_success():
    at_cap = _record_from_track(100.0, True, None, "0000", 100)
    assert at_cap["success"] is False
    assert at_cap["failure"] == "cap"
    assert at_cap["failure_inferred"] is True
    under = _record_from_track(99.0, True, None, "0000", 100)
    assert under["success"] is True
    assert "failure" not in under


def test_a_track_failure_class_is_kept_and_an_unknown_one_is_left_to_be_inferred():
    named = _record_from_track(40.0, False, "divergence", "0000", 100)
    assert named["failure"] == "divergence"
    assert "failure_inferred" not in named
    unknown = _record_from_track(40.0, False, "weird", "0000", 100)
    assert "failure" not in unknown
    assert unknown["failure_inferred"] is True
    capped = _record_from_track(100.0, False, "weird", "0000", 100)
    assert capped["failure"] == "cap"
    uncapped = _record_from_track(5000.0, False, None, "0000", None)
    assert "failure" not in uncapped


def _raw(**fields) -> dict:
    return {"force_calls": 100, "success": True, "start": "s1", **fields}


def test_a_row_that_is_not_a_dict_or_has_no_count_is_no_start():
    assert _record_from_raw(["force_calls", 1], None, None) is None
    assert _record_from_raw({"success": True}, None, None) is None
    assert _record_from_raw(_raw(force_calls="many"), None, None) is None
    assert _record_from_raw(_raw(force_calls=float("nan")), None, None) is None
    assert _record_from_raw(_raw(force_calls=float("inf")), None, None) is None


def test_the_count_is_read_from_force_calls_then_raw_then_per_search():
    assert _record_from_raw({"force_calls": 5, "raw": 6, "per_search": 7, "success": True}, None, None)["force_calls"] == 5.0
    assert _record_from_raw({"raw": 6, "per_search": 7, "success": True}, None, None)["force_calls"] == 6.0
    assert _record_from_raw({"per_search": 7, "success": True}, None, None)["force_calls"] == 7.0
    assert _record_from_raw({"force_calls": "8", "success": True}, None, None)["force_calls"] == 8.0


def test_a_row_says_success_by_a_success_flag_a_failed_flag_or_a_reason_and_otherwise_is_no_start():
    assert _record_from_raw({"raw": 9, "success": False}, None, None)["success"] is False
    assert _record_from_raw({"raw": 9, "failed": True}, None, None)["success"] is False
    assert _record_from_raw({"raw": 9, "failed": False}, None, None)["success"] is True
    assert _record_from_raw({"raw": 9, "success": None, "failed": False}, None, None)["success"] is True
    for reason, expected in (("0", True), ("0.0", True), ("cap", False), ("crash", False), ("incomplete", False)):
        assert _record_from_raw({"raw": 9, "reason": reason}, None, None)["success"] is expected
    assert _record_from_raw({"raw": 9}, None, None) is None


def test_a_reason_row_gets_its_failure_class_from_the_cap_or_the_force_and_not_from_a_stated_class():
    row = {"raw": 9, "reason": "cap", "failure": "crash"}
    record = _record_from_raw(row, None, None)
    assert record["success"] is False
    assert "failure" not in record
    assert record["failure_inferred"] is True
    capped = _record_from_raw({"raw": 50, "reason": "cap"}, 50, None)
    assert capped["failure"] == "cap"


def test_a_start_at_the_cap_is_a_failure_unless_a_transition_was_confirmed():
    plain = _record_from_raw(_raw(force_calls=100), 100, None)
    assert plain["success"] is False
    assert plain["failure"] == "cap"
    for confirmed in (1, True, "1"):
        kept = _record_from_raw(_raw(force_calls=100, transition_found=confirmed), 100, None)
        assert kept["success"] is True
    refused = _record_from_raw(_raw(force_calls=100, transition_found=0), 100, None)
    assert refused["success"] is False
    assert _record_from_raw(_raw(force_calls=99), 100, None)["success"] is True


def test_a_failed_row_is_classed_by_what_it_says_then_by_the_error_the_cap_and_the_force():
    named = _record_from_raw(_raw(success=False, failure="wrong_stationary"), None, None)
    assert named["failure"] == "wrong_stationary"
    assert "failure_inferred" not in named
    listed = _record_from_raw(_raw(success=False, reasons=["force_norm", "cap"]), None, None)
    assert listed["reasons"] == ["force_norm", "cap"]
    assert "failure" not in listed
    errored = _record_from_raw(_raw(success=False, error="boom"), 1000, 1e-3)
    assert errored["failure"] == "crash"
    assert errored["failure_inferred"] is True
    capped = _record_from_raw(_raw(success=False, force_calls=1000), 1000, 1e-3)
    assert capped["failure"] == "cap"
    above = _record_from_raw(_raw(success=False, force_norm=0.5), 1000, 1e-3)
    assert above["failure"] == "divergence"
    stopped = _record_from_raw(_raw(success=False, force_norm=1e-4), 1000, 1e-3)
    assert stopped["failure"] == "wrong_stationary"
    at_the_stop = _record_from_raw(_raw(success=False, force_norm=1e-3), 1000, 1e-3)
    assert at_the_stop["failure"] == "wrong_stationary"
    unreadable = _record_from_raw(_raw(success=False, force_norm="high"), 1000, 1e-3)
    assert unreadable["failure"] == "divergence"
    blank = _record_from_raw(_raw(success=False, force_norm=""), 1000, 1e-3)
    assert "failure" not in blank
    nothing = _record_from_raw(_raw(success=False), None, None)
    assert "failure" not in nothing
    assert nothing["failure_inferred"] is True


def test_the_optional_fields_of_a_row_are_carried_when_they_parse():
    record = _record_from_raw(
        _raw(per_image="3.5", call_unit="per_image", hessian_index=1, geometry="pair_07", start=None), None, None
    )
    assert record["per_image"] == 3.5
    assert record["call_unit"] == "per_image"
    assert record["hessian_index"] == 1
    assert record["start"] == "pair_07"
    broken = _record_from_raw(_raw(per_image="n/a"), None, None)
    assert "per_image" not in broken


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"hessian_calls": 12}, 12.0),
        ({"hessian_calls": "7.5"}, 7.5),
        ({"hessian_calls": None}, None),
        ({}, None),
        ({"hessian_calls": "many"}, None),
        ({"hessian_calls": [1]}, None),
        ([("hessian_calls", 1)], None),
    ],
)
def test_the_hessian_calls_of_a_row_are_a_number_or_nothing(raw, expected):
    assert _hessian_calls(raw) == expected


@pytest.mark.parametrize(
    ("success_text", "needs"),
    [
        ("Force norm below the stop and a Hessian index of one.", True),
        ("the hessian index is 1", True),
        ("The index is vibrational: one negative mode.", True),
        ("index 1 at the eigenvalue", True),
        ("Force norm below the stop.", False),
        ("The index of the table is one", False),
        ("The lowest eigenvalue is negative.", False),
        ("One vibrational mode is imaginary.", False),
        ("", False),
    ],
)
def test_a_success_rule_needs_the_hessian_when_it_says_so(success_text, needs):
    problem = {"scoring": {"v2": {"success": success_text}}}
    assert _success_needs_hessian(problem) is needs


def test_a_problem_with_no_scoring_text_does_not_need_the_hessian():
    assert _success_needs_hessian(None) is False
    assert _success_needs_hessian({}) is False
    assert _success_needs_hessian({"scoring": {}}) is False
    assert _success_needs_hessian({"scoring": {"v2": None}}) is False


def test_a_success_with_no_hessian_index_is_found_in_every_record_layout():
    ok = _make_start(10.0, True, "a", hessian_index=1)
    assert _missing_hessian_index([ok]) is False
    assert _missing_hessian_index([_make_start(10.0, True, "a")]) is True
    assert _missing_hessian_index([_make_start(10.0, False, "a")]) is False
    assert _missing_hessian_index([_make_start(10.0, True, "a", failure="cap")]) is True
    assert _missing_hessian_index([_make_start(10.0, False, "a", failure="cap")]) is False
    assert _missing_hessian_index([_make_start(10.0, True, "a", hessian_index=1, failure="cap")]) is False
    assert _missing_hessian_index([_make_start(10.0, True, "a", per_image=1.0)]) is True
    assert _missing_hessian_index([_make_start(10.0, True, "a", per_image=1.0, hessian_index=1)]) is False
    assert _missing_hessian_index([ok, _make_start(10.0, True, "b")]) is True
    assert _missing_hessian_index([]) is False


def test_a_success_dict_is_missing_its_index_when_the_key_is_absent():
    assert _missing_hessian_index([{"success": True}]) is True
    assert _missing_hessian_index([{"success": True, "hessian_index": None}]) is False
    assert _missing_hessian_index([{"success": False}]) is False


def test_an_object_with_attributes_is_missing_its_index_when_it_is_none():
    class Row:
        def __init__(self, success, hessian_index):
            self.success = success
            self.hessian_index = hessian_index

    assert _missing_hessian_index([Row(True, None)]) is True
    assert _missing_hessian_index([Row(True, 1)]) is False
    assert _missing_hessian_index([Row(False, None)]) is False


@pytest.mark.parametrize(
    ("success", "failure", "reasons", "calls", "cap", "label"),
    [
        (True, None, None, 10, 100, None),
        (True, "", None, 10, 100, None),
        (True, "none", None, 10, 100, None),
        (False, "crash", None, 10, 100, "crash"),
        (False, "wrong_stationary", ["crash"], 10, 100, "wrong_stationary"),
        (False, None, ["hessian_index"], 10, 100, "wrong_stationary"),
        (False, None, ["force_norm", "cap"], 10, 100, "cap"),
        (False, None, [], 100, 100, "cap"),
        (False, None, None, 100.0, 100, "cap"),
        (False, None, None, 99, 100, "divergence"),
        (False, None, None, 10**6, None, "divergence"),
        (False, "unknown", None, 99, 100, "divergence"),
    ],
)
def test_a_failed_start_has_one_class_by_the_order_the_board_uses(success, failure, reasons, calls, cap, label):
    assert _failure_label(success, failure, reasons, calls, cap) == label


def test_a_success_that_carries_a_failure_class_is_refused():
    with pytest.raises(ValueError, match="a success cannot carry a failure class"):
        _failure_label(True, "cap", None, 10, 100)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], "These codes"),
        ([{"code": "alpha"}], "alpha"),
        ([{"code": "alpha"}, {"code": "beta"}], "alpha and beta"),
        ([{"code": "alpha"}, {"code": "beta"}, {"code": "gamma"}], "alpha, beta, and gamma"),
        ([{"code": "alpha"}, {"code": "alpha"}, {"code": "beta"}], "alpha and beta"),
        ([{"code": " alpha "}, {"code": "alpha"}], "alpha"),
        ([{"code": "", "template": "tool/variant"}], "tool"),
        ([{"entry": "other-1.0-case"}], "other-1.0-case"),
        ([{"code": None}], "unknown"),
        ([{}, {"code": "alpha"}], "unknown and alpha"),
    ],
)
def test_the_codes_off_a_table_are_named_as_a_short_english_list(rows, expected):
    assert _joined_names(rows) == expected


def _aside(code: str, counter: str = "client") -> dict:
    return {"code": code, "counter": counter}


def test_a_problem_with_every_row_ranked_and_no_gap_has_no_gap_paragraph():
    assert gap_paragraph({"id": "minimization/toy", "ranked": [{"label": "a"}]}) == ""


def test_a_problem_with_no_run_says_so():
    text = gap_paragraph({"id": "minimization/toy", "ranked": []})
    assert text == '<p class="bench-gaps">No run is committed for this problem.</p>'


def test_rows_off_the_harness_counter_are_named_in_one_sentence_each_for_the_two_reasons():
    problem = {
        "id": "minimization/toy",
        "ranked": [{"label": "a"}],
        "not_comparable": [_aside("alpha"), _aside("beta", "unset"), _aside("gamma", "harness"), _aside("delta", "ase")],
    }
    text = gap_paragraph(problem)
    assert text == (
        '<p class="bench-gaps">alpha and beta rows are not yet on the harness counter.</p>'
        '<p class="bench-gaps">gamma and delta rows are off the ranked table.</p>'
    )


@pytest.mark.parametrize("suite", ["neb", "double-ended-saddle-search"])
def test_band_suites_call_their_rows_band_rows(suite):
    problem = {"id": f"{suite}/toy", "ranked": [{"label": "a"}], "not_comparable": [_aside("alpha")]}
    assert gap_paragraph(problem) == '<p class="bench-gaps">alpha band rows are not yet on the harness counter.</p>'


def test_the_client_counter_rows_are_used_when_there_are_no_not_comparable_rows():
    problem = {"id": "minimization/toy", "ranked": [{"label": "a"}], "client_counter": [_aside("alpha", "")]}
    assert gap_paragraph(problem) == '<p class="bench-gaps">alpha rows are not yet on the harness counter.</p>'


def test_a_short_or_withheld_problem_and_unparsed_files_each_add_a_sentence_in_order():
    problem = {"id": "minimization/toy", "ranked": [{"label": "a"}], "short": True}
    short = gap_paragraph(problem, [{"path": "x"}])
    assert short == (
        '<p class="bench-gaps">A run short of the full set, or a run that is not the stock method, '
        "stays off the ranked table.</p>"
        '<p class="bench-gaps">Files with no per-start harness record stay off the ranked table.</p>'
    )
    withheld = gap_paragraph({"id": "minimization/toy", "ranked": [{"label": "a"}], "withheld": ["x"]})
    assert "A run short of the full set" in withheld
    unparsed_only = gap_paragraph({"id": "minimization/toy", "ranked": []}, [{"path": "x"}])
    assert unparsed_only == '<p class="bench-gaps">Files with no per-start harness record stay off the ranked table.</p>'


def test_a_code_name_is_escaped_in_the_gap_sentence():
    problem = {"id": "minimization/toy", "ranked": [{"label": "a"}], "not_comparable": [_aside("a<b>&c")]}
    assert "a&lt;b&gt;&amp;c rows are not yet" in gap_paragraph(problem)
    assert "<b>" not in gap_paragraph(problem)


@pytest.mark.parametrize(
    ("row", "text"),
    [
        ({"short": True, "short_reason": "40 of 100 starts"}, "40 of 100 starts"),
        ({"short": True}, "short of the full set"),
        ({"counter": "client"}, "not yet on the harness counter"),
        ({}, "not yet on the harness counter"),
        ({"counter": "ase"}, "counter ase"),
        ({"counter": "harness"}, "counter harness"),
    ],
)
def test_the_reason_a_row_is_aside_is_its_short_reason_or_its_counter(row, text):
    assert _reason_text(row) == text


def test_the_winner_cell_names_the_label_and_the_first_score_the_row_has():
    row = {"label": "Alpha 1.2", "par2_mean": 1234.5678, "published": 9.0, "rank_cost": 5.0}
    assert _winner_html(row) == (
        '<div class="winner"><p class="winner-name">Alpha 1.2</p>'
        '<p class="winner-score"><small>PAR-2</small>1234.57</p></div>'
    )
    assert "<small>Published</small>9" in _winner_html({"label": "A", "published": 9.0, "rank_cost": 5.0})
    assert "<small>Cost</small>5" in _winner_html({"label": "A", "rank_cost": 5.0})
    assert _winner_html({"label": "A"}) == '<div class="winner"><p class="winner-name">A</p></div>'


def test_the_winner_cell_is_empty_with_no_label_and_escapes_the_label():
    assert _winner_html({"par2_mean": 3.0}) == ""
    assert _winner_html({"label": "  ", "par2_mean": 3.0}) == ""
    assert "&lt;i&gt;" in _winner_html({"label": "<i>x</i>"})


@pytest.mark.parametrize(
    ("message", "text"),
    [
        ("problem minimization/toy has no finite cost", "One start has no finite cost."),
        ("", "The profile could not be built."),
        ("cannot draw an empty profile", "Cannot draw an empty profile."),
        ("Already a sentence!", "Already a sentence!"),
        ("a question?", "A question?"),
        ("problem x has a cost", "Problem x has a cost."),
    ],
)
def test_a_profile_notice_is_a_capitalised_sentence(message, text):
    assert _profile_reason(ValueError(message)) == text


def test_the_rows_of_a_problem_are_the_ranked_rows_then_each_other_unit_in_order():
    problem = {
        "ranked": [{"label": "a"}, {"label": "b"}],
        "other_units": [{"rows": [{"label": "c"}]}, {"rows": []}, {"rows": [{"label": "d"}, {"label": "e"}]}],
    }
    assert [row["label"] for row in board_rows(problem)] == ["a", "b", "c", "d", "e"]
    assert board_rows({}) == []
    assert board_rows({"ranked": None, "other_units": None}) == []
    assert board_rows({"other_units": [{}]}) == []


def test_a_board_with_only_clean_ranked_rows_has_no_violations():
    document = {"problems": {"p": {"id": "minimization/toy", "ranked": [{"label": "a"}, {"label": "b", "short": False}]}}}
    assert rank_violations(document) == []
    assert rank_violations({}) == []
    assert rank_violations({"problems": None}) == []


def test_a_ranked_row_that_is_short_adjusted_or_marked_unranked_is_a_violation_that_names_it():
    document = {
        "problems": {
            "p": {
                "id": "minimization/toy",
                "ranked": [
                    {"label": "short one", "short": True},
                    {"template": "alpha/fast-tuned"},
                    {"entry": "beta-1.0-case", "tuning": "tuned"},
                    {"label": "gamma", "short_reason": "ranked false for this start count"},
                ],
                "other_units": [{"rows": [{"label": "delta", "short": True}]}],
            }
        }
    }
    assert rank_violations(document) == [
        "short row minimization/toy short one",
        "adjusted row minimization/toy alpha/fast-tuned",
        "adjusted row minimization/toy beta-1.0-case",
        "unranked row minimization/toy gamma",
        "short row minimization/toy delta",
    ]


def test_a_row_can_break_more_than_one_rule_and_each_is_reported():
    document = {
        "problems": {
            "p": {"id": "x/y", "ranked": [{"label": "z", "short": True, "tuning": "tuned", "short_reason": "ranked false"}]}
        }
    }
    assert rank_violations(document) == ["short row x/y z", "adjusted row x/y z", "unranked row x/y z"]
