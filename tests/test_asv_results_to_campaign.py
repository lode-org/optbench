"""An asv result file turned into campaign files.

The result files are real ``asv`` result objects saved to disk and read back,
so the tests see the layout asv writes: one list of values per benchmark, a
``null`` for a benchmark asv could not run, and ``NaN`` for a track that had no
value. Nothing reruns a search. The numbers are the ones the tracks stored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from asv.results import Results

from optbench.asv_campaign import STEMS, _flag, _missing, _reason, copy_asv_tree, write_results
from optbench.dat import parse

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c" * 40
PREFIX = "benchmarks.bench_eon_develop."
TS_LJ38 = "EonDevelopTsLj38Lanczos"
TS_PT = "EonDevelopTsPtLanczos"
DE_2TS = "EonDevelopDe2ts"
AKMC = "EonDevelopSaddlePtDimer"
NAN = float("nan")
FAILURES = ("cap", "wrong_stationary", "divergence", "crash")


def _result_file(directory: Path, entries: dict, *, commit: str = COMMIT) -> Path:
    header = {"machine": "optbench-ci", "os": "Linux", "arch": "x86_64", "cpu": "test", "num_cpu": "1", "ram": "1"}
    result = Results(header, [], commit, 0, "3.12", "uv-pip-py3.12", {})
    for key, value in entries.items():
        result._results[key] = value
        result._benchmark_params[key] = []
        result._samples[key] = [None]
        result._stats[key] = [None]
        result._benchmark_version[key] = "v1"
        result._started_at[key] = 0
        result._duration[key] = 0.1
    result.save(str(directory))
    return directory / result._filename


def _tracks(cls: str, calls: float, success: float, *, failed: str | None = None, **more) -> dict:
    """The tracks of one class for its one-start grid."""
    values = {
        "track_force_calls": [calls],
        "track_success": [success],
        **{f"track_failure_{name}": [1.0 if name == failed else 0.0] for name in FAILURES},
        **{f"track_{name}": [value] for name, value in more.items() if name != "time_search"},
    }
    if "time_search" in more:
        values["time_search"] = [more["time_search"]]
    return {f"{PREFIX}{cls}.{name}": value for name, value in values.items()}


def _write(tmp_path: Path, entries: dict, *, commit: str = COMMIT):
    path = _result_file(tmp_path / "asv", entries, commit=commit)
    campaign = tmp_path / "out" / "campaign"
    return write_results(ROOT, path, campaign, workers=1), campaign


def _starts(campaign: Path, stem: str) -> list[dict]:
    text = (campaign / f"{stem}.starts").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _dat(campaign: Path, stem: str) -> dict:
    return parse((campaign / f"{stem}.dat").read_text(encoding="utf-8"))


def test_a_result_file_with_one_class_writes_that_classs_campaign_files(tmp_path: Path):
    entries = _tracks(TS_LJ38, 1234.0, 1.0, time_search=0.25)
    written, campaign = _write(tmp_path, entries, commit="d" * 40)
    stem = STEMS[TS_LJ38]
    assert written == [stem]
    assert sorted(path.name for path in campaign.iterdir()) == [f"{stem}.dat", f"{stem}.starts"]
    fields = _dat(campaign, stem)
    assert fields["force_calls"] == "1234"
    assert fields["nfailed"] == "0"
    assert fields["counter"] == "harness"
    assert fields["published_mean"] == "false"
    assert fields["asv_benchmark"] == f"benchmarks.bench_eon_develop.{TS_LJ38}"
    assert fields["asv_commit"] == "d" * 40
    (row,) = _starts(campaign, stem)
    assert (row["raw"], row["reason"], row["failed"], row["seconds"]) == (1234, "0", False, 0.25)
    assert (row["seed"], row["geometry"], row["count_key"]) == (1, "0000", "harness")


@pytest.mark.parametrize(
    "failure",
    ["cap", "wrong_stationary", "divergence", "crash"],
)
def test_the_failure_track_that_is_set_names_the_reason_of_a_failed_start(tmp_path: Path, failure: str):
    entries = _tracks(TS_LJ38, 40.0, 0.0, failed=failure)
    written, campaign = _write(tmp_path, entries)
    stem = STEMS[TS_LJ38]
    assert written == [stem]
    (row,) = _starts(campaign, stem)
    assert (row["reason"], row["failed"], row["raw"]) == (failure, True, 40)
    assert _dat(campaign, stem)["nfailed"] == "1"


def test_a_search_that_returned_no_count_or_time_is_a_failed_start_with_neither(tmp_path: Path):
    entries = _tracks(TS_LJ38, NAN, 0.0, failed="crash", time_search=NAN)
    written, campaign = _write(tmp_path, entries)
    stem = STEMS[TS_LJ38]
    (row,) = _starts(campaign, stem)
    assert row["raw"] is None
    assert row["force_calls"] is None
    assert row["reason"] == "crash"
    assert "seconds" not in row
    fields = _dat(campaign, stem)
    assert fields["nfailed"] == "1"
    assert "force_calls" not in fields


def test_a_class_asv_could_not_run_is_left_out_and_the_others_are_still_written(tmp_path: Path):
    entries = _tracks(TS_LJ38, 900.0, 1.0)
    entries.update(_tracks(TS_PT, 700.0, 1.0))
    entries[f"{PREFIX}{TS_PT}.track_success"] = None
    entries["benchmarks.other.Unrelated.track_force_calls"] = [5.0]
    written, campaign = _write(tmp_path, entries)
    assert written == [STEMS[TS_LJ38]]
    assert sorted(path.name for path in campaign.iterdir()) == [
        f"{STEMS[TS_LJ38]}.dat",
        f"{STEMS[TS_LJ38]}.starts",
    ]


def test_a_class_with_no_force_call_result_is_left_out(tmp_path: Path):
    entries = _tracks(TS_LJ38, 900.0, 1.0)
    entries[f"{PREFIX}{TS_LJ38}.track_force_calls"] = None
    written, campaign = _write(tmp_path, entries)
    assert written == []
    assert list(campaign.iterdir()) == []


def test_a_result_file_with_no_develop_class_writes_nothing_and_still_makes_the_directory(tmp_path: Path):
    written, campaign = _write(tmp_path, {"benchmarks.other.Unrelated.track_force_calls": [5.0]})
    assert written == []
    assert campaign.is_dir()
    assert list(campaign.iterdir()) == []


@pytest.mark.parametrize("missing", ["cap", "wrong_stationary", "divergence", "crash"])
def test_a_class_with_a_failure_track_missing_is_refused_by_name(tmp_path: Path, missing: str):
    entries = _tracks(TS_LJ38, 900.0, 1.0)
    del entries[f"{PREFIX}{TS_LJ38}.track_failure_{missing}"]
    with pytest.raises(ValueError, match=f"^{TS_LJ38} is missing track_failure_{missing}$"):
        _write(tmp_path, entries)
    assert not (tmp_path / "out" / "campaign" / f"{STEMS[TS_LJ38]}.dat").exists()


@pytest.mark.parametrize("missing", ["unique_saddles", "fraction_connected", "total_searches"])
def test_an_akmc_class_with_one_of_its_three_tracks_missing_is_refused_by_name(tmp_path: Path, missing: str):
    extras = {"unique_saddles": 30.0, "fraction_connected": 0.75, "total_searches": 4000.0}
    entries = _tracks(AKMC, 1000.0, 1.0, **extras)
    del entries[f"{PREFIX}{AKMC}.track_{missing}"]
    with pytest.raises(ValueError, match=f"^{AKMC} is missing track_{missing}$"):
        _write(tmp_path, entries)


def test_a_two_transition_state_class_with_no_extrema_track_is_refused_by_name(tmp_path: Path):
    entries = _tracks(DE_2TS, 5000.0, 1.0)
    with pytest.raises(ValueError, match=f"^{DE_2TS} is missing track_extrema$"):
        _write(tmp_path, entries)


def test_a_two_transition_state_class_keeps_the_extrema_it_found(tmp_path: Path):
    written, campaign = _write(tmp_path, _tracks(DE_2TS, 5000.0, 1.0, extrema=2.0))
    stem = STEMS[DE_2TS]
    assert written == [stem]
    (row,) = _starts(campaign, stem)
    assert row["extrema"] == 2
    fields = _dat(campaign, stem)
    assert (fields["transition_states"], fields["extrema_total"]) == ("1", "2")
    assert fields["force_calls"] == "5000"
    assert fields["asv_benchmark"] == f"benchmarks.bench_eon_develop.{DE_2TS}"


def test_a_two_transition_state_start_with_no_extrema_count_has_no_extrema_and_is_still_written(tmp_path: Path):
    written, campaign = _write(tmp_path, _tracks(DE_2TS, 5000.0, 1.0, extrema=NAN))
    stem = STEMS[DE_2TS]
    assert written == [stem]
    (row,) = _starts(campaign, stem)
    assert "extrema" not in row
    assert (row["raw"], row["reason"]) == (5000, "0")


def test_an_extrema_count_is_rounded_to_a_whole_number(tmp_path: Path):
    written, campaign = _write(tmp_path, _tracks(DE_2TS, 5000.0, 1.0, extrema=1.9999999))
    (row,) = _starts(campaign, STEMS[DE_2TS])
    assert row["extrema"] == 2


def test_an_akmc_class_is_on_the_harness_counter_and_unpublished(tmp_path: Path):
    extras = {"unique_saddles": 30.0, "fraction_connected": 0.75, "total_searches": 4000.0, "time_search": 12.5}
    written, campaign = _write(tmp_path, _tracks(AKMC, 1000.0, 1.0, **extras))
    stem = STEMS[AKMC]
    assert written == [stem]
    fields = _dat(campaign, stem)
    assert (fields["counter"], fields["count_keys"]) == ("harness", "HarnessCounter")
    assert fields["published_mean"] == "false"
    assert fields["force_calls"] == "1000"
    assert fields["unique_saddles"] == "30"
    assert fields["asv_commit"] == COMMIT
    (row,) = _starts(campaign, stem)
    assert (row["reached"], row["reason"], row["failed"]) == (True, "0", False)
    assert (row["unique_saddles"], row["fraction_connected"], row["total_searches"]) == (30.0, 0.75, 4000.0)
    assert row["seconds"] == 12.5


def test_an_akmc_class_that_never_reached_the_target_is_an_incomplete_start_with_no_numbers(tmp_path: Path):
    extras = {"unique_saddles": NAN, "fraction_connected": NAN, "total_searches": NAN, "time_search": NAN}
    written, campaign = _write(tmp_path, _tracks(AKMC, NAN, 0.0, failed="crash", **extras))
    stem = STEMS[AKMC]
    (row,) = _starts(campaign, stem)
    assert row["reached"] is False
    assert row["reason"] == "incomplete"
    assert row["failed"] is True
    assert (row["force_calls"], row["unique_saddles"], row["fraction_connected"], row["total_searches"]) == (
        None,
        None,
        None,
        None,
    )
    assert row["seconds"] is None
    assert _dat(campaign, stem)["nfailed"] == "1"


def test_a_track_with_the_wrong_number_of_values_is_refused_by_name(tmp_path: Path):
    entries = _tracks(TS_LJ38, 900.0, 1.0)
    entries[f"{PREFIX}{TS_LJ38}.track_force_calls"] = [900.0, 901.0]
    with pytest.raises(ValueError, match=f"^{TS_LJ38} force_calls has 2 values, expected 1$"):
        _write(tmp_path, entries)


def test_two_classes_in_one_file_are_written_in_the_order_the_benchmarks_define_them(tmp_path: Path):
    entries = _tracks(TS_PT, 700.0, 1.0)
    entries.update(_tracks(TS_LJ38, 900.0, 1.0))
    written, campaign = _write(tmp_path, entries)
    assert written == [STEMS[TS_LJ38], STEMS[TS_PT]]
    assert _dat(campaign, STEMS[TS_LJ38])["force_calls"] == "900"
    assert _dat(campaign, STEMS[TS_PT])["force_calls"] == "700"


def test_a_count_is_rounded_to_a_whole_number_of_calls(tmp_path: Path):
    written, campaign = _write(tmp_path, _tracks(TS_LJ38, 1233.6, 1.0))
    (row,) = _starts(campaign, STEMS[TS_LJ38])
    assert row["raw"] == 1234
    assert _dat(campaign, STEMS[TS_LJ38])["force_calls"] == "1234"


def test_only_a_value_of_exactly_one_sets_a_flag_and_nan_and_none_are_missing():
    assert _flag(1) is True
    assert _flag(1.0) is True
    assert _flag(0.9999) is False
    assert _flag(2.0) is False
    assert _flag(NAN) is False
    assert _flag(None) is False
    assert _missing(None) is True
    assert _missing(NAN) is True
    assert _missing(0.0) is False
    assert _missing(0) is False


def test_the_reason_follows_a_fixed_order_when_the_tracks_overlap():
    assert _reason(1, 1, 1, 1, 1) == "0"
    assert _reason(0, 1, 1, 1, 1) == "crash"
    assert _reason(0, 1, 1, 1, 0) == "divergence"
    assert _reason(0, 1, 1, 0, 0) == "cap"
    assert _reason(0, 0, 1, 0, 0) == "wrong_stationary"
    assert _reason(NAN, 0, 0, 0, 1) == "crash"
    assert _reason(None, 0, 0, NAN, NAN) is None
    assert _reason(NAN, NAN, NAN, NAN, NAN) is None


def test_copy_asv_tree_copies_the_json_files_in_order_and_leaves_the_rest(tmp_path: Path):
    source = tmp_path / "results"
    machine = source / "optbench-ci"
    machine.mkdir(parents=True)
    files = {
        source / "benchmarks.json": '{"version": 2}\n',
        machine / "machine.json": '{"machine": "optbench-ci"}\n',
        machine / "cccccccc-uv-pip-py3.12.json": '{"commit_hash": "' + COMMIT + '"}\n',
        machine / "notes.txt": "not a result\n",
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    dest = tmp_path / "kept" / "asv"
    copied = copy_asv_tree(source, dest)
    assert copied == [
        dest / "benchmarks.json",
        dest / "optbench-ci" / "cccccccc-uv-pip-py3.12.json",
        dest / "optbench-ci" / "machine.json",
    ]
    for path in copied:
        assert path.read_text(encoding="utf-8") == files[source / path.relative_to(dest)]
    assert not (dest / "optbench-ci" / "notes.txt").exists()
    assert sorted(path.name for path in dest.rglob("*") if path.is_file()) == sorted(path.name for path in copied)


def test_copy_asv_tree_of_an_empty_directory_makes_the_destination_and_copies_nothing(tmp_path: Path):
    source = tmp_path / "results"
    source.mkdir()
    dest = tmp_path / "kept"
    assert copy_asv_tree(source, dest) == []
    assert dest.is_dir()
