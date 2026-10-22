"""benchmark.dat lines and the barrier-window reducer."""

from optbench.dat import parse
from optbench.eon_htst import assign_delta, parse_searches, window_failure
from optbench.htst_windows import HTST, htst_dat, reduce_htst

SUMMARY = {
    "barrier_jobs": 1.5,
    "barrier_force_calls": 12,
    "barrier_cycles": 1,
    "rate_jobs": None,
    "matched_barriers": 15,
    "reference_barriers": 15,
    "processes": 1,
}
PROBLEM = {"id": "hTST/Pt", "convergence": {"max_force_calls": 20000000}}
SPEC = {"temperature": 700.0, "prefactor": 1.2e12, "rate_ref": 155055901.244}


def _text(template: dict) -> str:
    return htst_dat(
        SUMMARY,
        template=template,
        problem=PROBLEM,
        spec=SPEC,
        searches=3,
        minimize_calls=0,
        unfinished=False,
        comments=["# counter harness"],
    )


def test_a_blank_tuning_is_omitted():
    fields = parse(
        _text(
            {
                "id": "eon-client/htst-pt-dimer-lbfgs",
                "code": "eOn",
                "code_version": "3.6.0",
                "commit": "4c1b814f62f1dcf5fcd0cf18ba0b6df3bcff7a28",
                "algorithm": "Dimer L-BFGS",
                "reducer": "htst-cycles",
            }
        )
    )
    assert "tuning" not in fields
    assert fields["counter"] == "harness"
    assert fields["cap"] == "20000000"
    assert fields["template"] == "eon-client/htst-pt-dimer-lbfgs"


def test_a_set_tuning_is_kept():
    fields = parse(
        _text(
            {
                "id": "eon-client/htst-pt-dimer-lbfgs",
                "code": "eOn",
                "code_version": "3.6.0",
                "commit": "abc",
                "algorithm": "Dimer L-BFGS",
                "tuning": "beland-2015-pt111",
                "reducer": "htst-cycles",
            }
        )
    )
    assert fields["tuning"] == "beland-2015-pt111"


def test_parse_searches_keeps_good_and_repeat_ids():
    text = "\n".join(
        [
            "wuid type barrier max-dist sad-fcs mins-fcs pref-fcs result",
            "----",
            "0 random 0.60100 1 10 2 0 good-4",
            "1 random 0.00000 1 3 1 0 Not",
            "2 random 0.60100 1 8 2 0 repeat-4",
        ]
    )
    rows = parse_searches(text)
    assert rows[0]["process_id"] == 4
    assert rows[0]["barrier"] == 0.601
    assert rows[1]["process_id"] is None
    assert rows[2]["process_id"] == 4
    assert [row["sad_fcs"] + row["mins_fcs"] for row in rows] == [12, 4, 10]


def test_assign_delta_sums_to_the_harness_tally():
    rows = [
        {"sad_fcs": 10, "mins_fcs": 0},
        {"sad_fcs": 30, "mins_fcs": 0},
    ]
    assign_delta(rows, 7)
    assert [row["force_calls"] for row in rows] == [1, 6]
    assert sum(row["force_calls"] for row in rows) == 7


def test_one_search_gets_the_whole_delta():
    rows = [{"sad_fcs": 4, "mins_fcs": 1}]
    assign_delta(rows, 19)
    assert rows[0]["force_calls"] == 19


def test_window_cost_is_the_harness_sum():
    spec = HTST["hTST/Pt"]
    rows = [
        {
            "force_calls": 100 + index,
            "barrier": barrier,
            "process_id": index + 1,
        }
        for index, barrier in enumerate(spec["barriers"])
    ]
    summary = reduce_htst(rows, spec)
    assert summary["barrier_cycles"] == 1
    expected = sum(100 + index for index in range(len(spec["barriers"])))
    assert summary["barrier_force_calls"] == expected
    assert window_failure(float(summary["barrier_force_calls"]), 20000000) is None
