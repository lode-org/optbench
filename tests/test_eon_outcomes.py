"""How an eOn harness run is classified, recorded, and published.

The harness counts every force call itself and decides the outcome of a
start from the count, the client's stated reason, and the geometry it left
behind. These tests drive that decision with real templates and problems and
check what reaches ``benchmark.dat`` and the per-start file.
"""

from __future__ import annotations

import itertools
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from benchmarks import bench_eon_develop
from optbench import eon_harness
from optbench.board import stated_full_n
from optbench.campaign import list_templates, load_template, resolve_problem
from optbench.eon_harness import (
    ROOT,
    _cell,
    _earr_min,
    _family_of,
    _files,
    _guard,
    _interpolated,
    _pbc_delta,
    _replace_coords,
    _start_row,
    check_client,
    classify_akmc,
    classify_client,
    classify_flag,
    client_file,
    default_starts,
    grid_points,
    kernel_for,
    selected,
    stated_failure_cap,
    write_campaign,
)

TEMPLATE = "eon-svn/ts-lj38-dimer-ljcut"


def _con(rows, box=(11.08, 9.6, 30.0)) -> str:
    lines = [
        "Generated for a test",
        "0.0000 TIME",
        " ".join(str(length) for length in box),
        "90.0 90.0 90.0",
        "0 0",
        "0 0 0",
        "1",
        str(len(rows)),
        "195.084",
        "Pt",
        "Coordinates of Component 1",
    ]
    for index, (x, y, z, flag) in enumerate(rows):
        lines.append(f"{x} {y} {z} {flag} {index}")
    return "\n".join(lines) + "\n"


def _island(adatom) -> str:
    return _con(
        [
            (0.0, 0.0, 0.0, 1),
            (2.77, 0.0, 0.0, 1),
            (1.385, 2.399, 0.0, 1),
            (*adatom, 0),
        ]
    )


def _rows(text: str) -> list[list[float]]:
    return [[float(value) for value in line.split()] for line in text.splitlines()]


def test_the_client_reason_decides_between_success_divergence_and_crash():
    ran = classify_client(calls=120, reason="0", cap=10000)
    assert ran == {"force_calls": 120.0, "success": True, "failure": None}
    assert classify_client(calls=120, reason="2", cap=10000)["failure"] == "divergence"
    assert classify_client(calls=120, reason=None, cap=10000)["failure"] == "crash"
    assert classify_client(calls=120, reason="", cap=10000)["failure"] == "crash"
    assert classify_client(calls=120, reason="0", cap=10000, crashed=True)["failure"] == "crash"
    assert classify_client(calls=None, reason="0", cap=10000) == {
        "force_calls": 0.0,
        "success": False,
        "failure": "crash",
    }


def test_the_budget_wins_over_what_the_client_says_it_did():
    assert classify_client(calls=10000, reason="0", cap=10000)["failure"] == "cap"
    assert classify_client(calls=12000, reason="2", cap=10000)["failure"] == "cap"
    assert classify_client(calls=10000, reason=None, cap=10000, crashed=True)["failure"] == "cap"
    assert classify_client(calls=9999, reason="0", cap=10000)["success"] is True


def test_an_overlap_verdict_replaces_the_client_reason_under_the_budget_only():
    mode = classify_client(calls=43, reason="5", cap=10000, overlap_ok=True)
    assert mode == {"force_calls": 43.0, "success": True, "failure": None}
    miss = classify_client(calls=43, reason="0", cap=10000, overlap_ok=False)
    assert miss["failure"] == "wrong_stationary"
    assert classify_client(calls=10000, reason="0", cap=10000, overlap_ok=True)["failure"] == "cap"


def test_a_flag_run_is_a_success_a_wrong_stationary_point_a_crash_or_a_cap():
    assert classify_flag(calls=80, cap=1000, ok=True) == {"force_calls": 80.0, "success": True, "failure": None}
    assert classify_flag(calls=80, cap=1000, ok=False)["failure"] == "wrong_stationary"
    assert classify_flag(calls=80, cap=1000, ok=True, crashed=True)["failure"] == "crash"
    assert classify_flag(calls=0, cap=1000, ok=True)["failure"] == "crash"
    assert classify_flag(calls=1000, cap=1000, ok=True)["failure"] == "cap"
    assert classify_flag(calls=1000, cap=1000, ok=False, crashed=True)["failure"] == "cap"
    assert classify_flag(calls=None, cap=1000, ok=True) == {"force_calls": 0.0, "success": False, "failure": "crash"}


def test_an_akmc_run_succeeds_only_when_it_reached_its_saddle_count_under_the_budget():
    assert classify_akmc(calls=900, cap=1000, reached=True, crashed=False)["success"] is True
    assert classify_akmc(calls=900, cap=1000, reached=False, crashed=False)["failure"] == "cap"
    assert classify_akmc(calls=900, cap=1000, reached=True, crashed=True)["failure"] == "crash"
    assert classify_akmc(calls=1000, cap=1000, reached=True, crashed=False)["failure"] == "cap"
    assert classify_akmc(calls=None, cap=1000, reached=True, crashed=False)["failure"] == "crash"


def test_the_harness_and_the_asv_module_classify_every_outcome_the_same_way():
    for calls, reason, overlap, crashed in itertools.product(
        [None, 0, 5, 99, 100, 150],
        [None, "", "0", "2", "5"],
        [None, True, False],
        [False, True],
    ):
        arguments = {"calls": calls, "reason": reason, "cap": 100, "overlap_ok": overlap, "crashed": crashed}
        assert classify_client(**arguments) == bench_eon_develop.classify_client(**arguments)
    for calls, reached, crashed in itertools.product([None, 0, 5, 99, 100, 150], [True, False], [True, False]):
        arguments = {"calls": calls, "cap": 100, "reached": reached, "crashed": crashed}
        assert classify_akmc(**arguments) == bench_eon_develop.classify_akmc(**arguments)


def test_a_zero_harness_count_with_client_calls_is_a_crash_not_a_free_success():
    free = {"force_calls": 0.0, "success": True, "failure": None}
    assert _guard(dict(free), 17) == {"force_calls": 0.0, "success": False, "failure": "crash"}
    assert _guard(dict(free), 0) == free
    assert _guard(dict(free), None) == free
    counted = {"force_calls": 30.0, "success": True, "failure": None}
    assert _guard(dict(counted), 17) == counted


def test_the_lowest_energy_of_a_run_comes_from_its_energy_column(tmp_path):
    path = tmp_path / "earr.dat"
    path.write_text("0 -168.10\n1 -170.25 extra\n\n# a comment line\n2 not-a-number\n3 -169.90\nlonely\n", encoding="utf-8")
    assert _earr_min(path) == -170.25
    assert _earr_min(tmp_path / "missing.dat") is None
    path.write_text("# nothing\n\n", encoding="utf-8")
    assert _earr_min(path) is None


def test_a_start_row_keeps_what_the_board_reads_and_drops_what_is_empty():
    record = {
        "force_calls": 127.0,
        "success": False,
        "failure": "wrong_stationary",
        "client_calls": 127,
        "client_reason": "",
        "overlap": 0.0,
        "k": 0,
        "error": "",
        "hessian_index": 2,
        "force_norm": 0.0004,
        "seconds": 1.5,
        "timed_out": True,
        "kernel": "lj40",
    }
    row = _start_row(3, "0042", record)
    assert row == {
        "call_unit": "total",
        "client_calls": 127,
        "count_key": "harness",
        "counter": "harness",
        "failure": "wrong_stationary",
        "force_calls": 127,
        "force_norm": 0.0004,
        "geometry": "0042",
        "hessian_index": 2,
        "k": 0,
        "overlap": 0.0,
        "raw": 127,
        "seconds": 1.5,
        "seed": 3,
        "success": False,
    }
    ok = _start_row(1, "0001", {"force_calls": 88.4, "success": True, "failure": None})
    assert "failure" not in ok
    assert ok["force_calls"] == ok["raw"] == 88


def test_the_start_grid_is_seed_major_and_one_point_unless_the_full_grid_is_asked_for():
    template = {"kind": "ts-lj38", "seeds": [1, 2], "starts": ["a", "b", "c"]}
    assert grid_points(template, full=True) == [(1, "a"), (1, "b"), (1, "c"), (2, "a"), (2, "b"), (2, "c")]
    assert grid_points(template, full=True, limit=4) == [(1, "a"), (1, "b"), (1, "c"), (2, "a")]
    assert grid_points(template, full=False) == [(1, "a")]
    assert grid_points(template, full=False, limit=0) == [(1, "a")]
    walked = grid_points({"kind": "ts-lj38"}, full=True)
    assert len(walked) == 200
    assert walked[0] == (1, "0000")
    assert walked[-1] == (1, "0199")


@pytest.mark.parametrize(
    ("kind", "count", "first", "last"),
    [
        ("ts-lj38", 200, "0000", "0199"),
        ("ts-pt", 49, "initial_0", "initial_48"),
        ("de-lj38", 50, "pair_00", "pair_49"),
        ("de-lj38-mode", 50, "pair_00", "pair_49"),
        ("de-pt", 59, "product_0", "product_58"),
        ("de-2ts", 51, "pair_00", "pair_50"),
        ("lowest", 200, "0000", "0199"),
        ("basin", 100, "0", "99"),
        ("minima", 100, "0", "99"),
        ("akmc", 1, "0", "0"),
        ("prd", 1, "0", "0"),
    ],
)
def test_each_kind_walks_the_starts_its_problem_names(kind, count, first, last):
    starts = default_starts(kind)
    assert (len(starts), starts[0], starts[-1]) == (count, first, last)


def test_an_unknown_kind_has_no_starts():
    with pytest.raises(ValueError, match="unknown harness kind lattice"):
        default_starts("lattice")


def test_a_template_that_is_not_partial_walks_every_start_its_problem_states():
    found = 0
    for entry in list_templates(ROOT):
        if not str(entry.get("id", "")).startswith("eon"):
            continue
        template = load_template(ROOT, entry["id"])
        if not template.get("kind") or template.get("declare_partial"):
            continue
        try:
            _family_of(template)
        except ValueError:
            continue
        stated = stated_full_n(ROOT, resolve_problem(ROOT, template["problem"]))
        if stated is None:
            continue
        starts = template.get("starts") or default_starts(str(template["kind"]))
        assert len(starts) == stated, template["id"]
        found += 1
    assert found > 50


def test_a_failure_cap_comes_from_the_template_then_from_the_problem():
    assert stated_failure_cap({"id": "t", "problem": "tsopt/LJ38", "failure_cap": "2500"}) == 2500
    assert stated_failure_cap({"id": "t", "problem": "tsopt/LJ38"}) == 10000
    assert stated_failure_cap({"id": "t", "problem": "globalopt/lj38", "failure_cap": ""}) == 2000000


def test_a_template_without_a_cap_on_a_problem_without_one_is_refused():
    with pytest.raises(KeyError, match="t has no failure_cap and hTST/Al states no max_force_calls"):
        stated_failure_cap({"id": "t", "problem": "hTST/Al"})


@pytest.mark.parametrize(
    ("ident", "family"),
    [
        ("eon-svn/ts-lj38-dimer-ljcut", "svn"),
        ("eon-client/ts-lj38-dimer", "client"),
        ("eon-client/de-2ts-500350d9-lanczos", "500350d9"),
        ("eon/500350d9-de-lj38-ci-swdneb", "500350d9"),
        ("eon/develop-f6817b53-de-lj38-cap", "f6817b53"),
    ],
)
def test_the_longest_matching_prefix_names_the_family(ident, family):
    assert _family_of({"id": ident, "kind": "ts-lj38"}) == family


def test_a_template_with_no_kind_or_no_family_is_not_a_harness_template():
    with pytest.raises(ValueError, match="eon-client/ts-lj38-dimer is not a harness template"):
        _family_of({"id": "eon-client/ts-lj38-dimer"})
    with pytest.raises(ValueError, match="eon/develop-b0bd4d5a-ts-lj38-dimer is not a harness template"):
        _family_of({"id": "eon/develop-b0bd4d5a-ts-lj38-dimer", "kind": "ts-lj38"})


def _fake_client(path, banner: str):
    path.write_text(f"#!/bin/sh\necho '{banner}'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _spec(tmp_path, env: str, token: str) -> dict:
    return {
        "prefix": "eon-test/",
        "env": env,
        "default": tmp_path / "default" / "bin" / "eonclient",
        "flag": "--version",
        "token": token,
        "cache": token,
    }


def test_the_client_named_by_the_environment_wins_over_the_default(tmp_path, monkeypatch):
    spec = _spec(tmp_path, "OPTBENCH_TEST_EONCLIENT", "deadbeef")
    monkeypatch.delenv(spec["env"], raising=False)
    assert client_file(spec) == ""
    spec["default"].parent.mkdir(parents=True)
    _fake_client(spec["default"], "default client deadbeef")
    assert client_file(spec) == str(spec["default"])
    chosen = _fake_client(tmp_path / "chosen", "chosen client deadbeef")
    monkeypatch.setenv(spec["env"], str(chosen))
    assert client_file(spec) == str(chosen)


def test_a_client_is_run_once_for_its_version_and_then_trusted(tmp_path, monkeypatch):
    monkeypatch.setattr(eon_harness, "_CLIENT_OK", set())
    spec = _spec(tmp_path, "OPTBENCH_TEST_EONCLIENT", "deadbeef")
    client = _fake_client(tmp_path / "eonclient", "eonclient 3.5.0 (deadbeef)")
    monkeypatch.setenv(spec["env"], str(client))
    assert check_client(spec, "test") == str(client)
    _fake_client(client, "something else entirely")
    assert check_client(spec, "test") == str(client)


def test_a_client_that_prints_another_version_is_refused_with_what_it_printed(tmp_path, monkeypatch):
    monkeypatch.setattr(eon_harness, "_CLIENT_OK", set())
    spec = _spec(tmp_path, "OPTBENCH_TEST_EONCLIENT", "deadbeef")
    client = _fake_client(tmp_path / "eonclient", "eonclient 3.4.0 (0123abcd)")
    monkeypatch.setenv(spec["env"], str(client))
    with pytest.raises(RuntimeError, match=r"printed 'eonclient 3\.4\.0 \(0123abcd\)\\n', expected deadbeef"):
        check_client(spec, "test")


def test_no_client_at_all_asks_for_the_environment_variable(tmp_path, monkeypatch):
    spec = _spec(tmp_path, "OPTBENCH_TEST_EONCLIENT", "deadbeef")
    monkeypatch.delenv(spec["env"], raising=False)
    with pytest.raises(RuntimeError, match="set OPTBENCH_TEST_EONCLIENT to the test eonclient"):
        check_client(spec, "test")


def test_the_full_grid_is_selected_by_the_environment(monkeypatch):
    monkeypatch.delenv("OPTBENCH_ASV_FULL", raising=False)
    assert selected(["a", "b", "c"]) == ["a"]
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert selected(["a", "b", "c"]) == ["a", "b", "c"]
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "0")
    assert selected(["a", "b", "c"]) == ["a"]


def test_a_box_matrix_gives_its_three_diagonal_lengths_or_none_at_all():
    box = [11.08, 0.0, 0.0, 0.0, 9.6, 0.0, 0.0, 0.0, 30.0]
    assert _cell(box).tolist() == [11.08, 9.6, 30.0]
    assert _cell(None).tolist() == [0.0, 0.0, 0.0]
    assert _cell([11.08, 9.6, 30.0]).tolist() == [0.0, 0.0, 0.0]


def test_a_kernel_name_gives_the_surface_it_is_named_for():
    rmin = 2.0 ** (1.0 / 6.0)
    dimer = [0.0, 0.0, 0.0, rmin, 0.0, 0.0]
    box = [30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0]
    for name, tolerance in (("lj40", 1e-12), ("lj_cut", 1e-12), ("ljcluster", 1e-6)):
        energy, forces = kernel_for(name)(dimer, None, box)
        assert energy == pytest.approx(-1.0, abs=tolerance)
        assert max(abs(value) for value in forces) < 1e-9
    equilibrium = [0.0, 0.0, 0.0, 2.897, 0.0, 0.0]
    for name in ("morse", "morse_shift", "morse_cluster"):
        energy, forces = kernel_for(name)(equilibrium, [78, 78], box)
        assert energy == pytest.approx(-0.7102, abs=5e-4)
        assert max(abs(value) for value in forces) < 1e-4


def test_only_the_smoothed_morse_has_no_force_left_at_the_cutoff():
    box = [30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0]
    edge = [0.0, 0.0, 0.0, 9.5 - 1e-9, 0.0, 0.0]
    smooth = kernel_for("morse")(edge, [78, 78], box)
    shifted = kernel_for("morse_shift")(edge, [78, 78], box)
    assert abs(smooth[1][3]) < 1e-8
    assert smooth[0] == pytest.approx(0.0, abs=1e-8)
    x = np.exp(-1.6047 * (9.5 - 2.897))
    assert shifted[1][3] == pytest.approx(-2.0 * 1.6047 * 0.7102 * x * (1.0 - x), rel=1e-6)


def test_an_unknown_kernel_is_named():
    with pytest.raises(ValueError, match="no harness kernel gaussian"):
        kernel_for("gaussian")


def test_a_midpoint_displacement_and_unit_mode_follow_the_short_way_across_the_cell(tmp_path):
    reactant = tmp_path / "reactant.con"
    product = tmp_path / "product.con"
    reactant.write_text(_island((0.2, 5.0, 2.0)), encoding="utf-8")
    product.write_text(_island((10.9, 5.0, 2.0)), encoding="utf-8")
    displacement, mode = _interpolated(reactant, product, tmp_path)
    assert Path(displacement).name == "displacement.con"
    moved = Path(displacement).read_text(encoding="utf-8").splitlines()

    hop = 10.9 - 0.2 - 11.08
    assert hop == pytest.approx(-0.38)
    adatom = [float(value) for value in moved[14].split()[:3]]
    assert adatom == pytest.approx([0.2 + 0.5 * hop, 5.0, 2.0], abs=1e-8)
    assert moved[11].split()[:3] == ["0.00000000", "0.00000000", "0.00000000"]
    assert moved[14].split()[3:] == ["0", "3"]
    values = np.array(_rows(mode))
    assert values.shape == (4, 3)
    assert values[3].tolist() == pytest.approx([-1.0, 0.0, 0.0], abs=1e-15)
    assert not values[:3].any()
    assert np.linalg.norm(values) == pytest.approx(1.0, abs=1e-15)


def test_a_displacement_between_the_same_geometry_or_unequal_atom_counts_is_refused(tmp_path):
    reactant = tmp_path / "reactant.con"
    product = tmp_path / "product.con"
    reactant.write_text(_island((0.2, 5.0, 2.0)), encoding="utf-8")
    product.write_text(_island((0.2, 5.0, 2.0)), encoding="utf-8")
    with pytest.raises(ValueError, match="matches"):
        _interpolated(reactant, product, tmp_path)
    product.write_text(_con([(0.0, 0.0, 0.0, 1), (2.77, 0.0, 0.0, 1)]), encoding="utf-8")
    with pytest.raises(ValueError, match="has 2 atoms and .* has 4"):
        _interpolated(reactant, product, tmp_path)


def test_a_periodic_delta_is_folded_into_half_a_cell_and_an_open_cell_is_left_alone():
    delta = np.array([[10.7, -10.7, 5.0], [0.3, 0.0, 0.0]])
    folded = _pbc_delta(delta, np.array([11.0, 11.0, 30.0]))
    assert np.allclose(folded, [[-0.3, 0.3, 5.0], [0.3, 0.0, 0.0]], atol=1e-12)
    assert _pbc_delta(delta, np.zeros(3)).tolist() == delta.tolist()


def test_new_coordinates_replace_the_rows_and_keep_the_species_and_flags():
    text = _island((0.2, 5.0, 2.0))
    updated = _replace_coords(text, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.5, 0.25, 0.125]]))
    lines = updated.splitlines()
    assert lines[9] == "Pt"
    assert lines[10] == "Coordinates of Component 1"
    assert lines[11].split() == ["1.00000000", "2.00000000", "3.00000000", "1", "0"]
    assert lines[14].split() == ["0.50000000", "0.25000000", "0.12500000", "0", "3"]
    with pytest.raises(ValueError, match="con file has 4 coordinates, received 3"):
        _replace_coords(text, np.zeros((3, 3)))


@pytest.mark.parametrize(
    ("kind", "start", "files"),
    [
        ("ts-lj38", "0007", {"pos.con", "displacement.con", "direction.dat"}),
        ("ts-pt", "initial_3", {"pos.con", "displacement.con", "direction.dat"}),
        ("basin", "4", {"pos.con"}),
        ("minima", "4", {"pos.con"}),
        ("de-lj38", "pair_03", {"reactant.con", "product.con"}),
        ("de-lj38-mode", "pair_03", {"pos.con", "displacement.con", "direction.dat"}),
        ("de-2ts", "pair_03", {"reactant.con", "product.con"}),
        ("prd", "0", {"pos.con"}),
    ],
)
def test_each_kind_stages_the_input_files_its_client_reads(tmp_path, kind, start, files):
    copies, texts = _files(kind, start, tmp_path / "src")
    staged = {name for _source, name in copies} | {name for name, _text in texts}
    assert staged == files
    for source, _name in copies:
        assert os.path.isfile(source)


def test_the_unit_mode_of_a_double_ended_start_is_the_difference_between_its_endpoints(tmp_path):
    copies, texts = _files("de-lj38-mode", "pair_03", tmp_path / "src")
    mode = np.array(_rows(dict(texts)["direction.dat"]))
    assert mode.shape == (38, 3)
    assert np.linalg.norm(mode) == pytest.approx(1.0, abs=1e-12)
    reactant = next(source for source, name in copies if name == "pos.con")
    displacement = next(source for source, name in copies if name == "displacement.con")
    from optbench.surfaces import read_con

    left, _fixed, cell = read_con(open(reactant, encoding="utf-8").read())
    middle, _fixed, _cell_middle = read_con(open(displacement, encoding="utf-8").read())
    shift = middle - left
    assert np.allclose(shift / np.linalg.norm(shift), mode, atol=1e-6)


def test_an_unknown_kind_has_no_input_files(tmp_path):
    with pytest.raises(ValueError, match="unknown harness kind lattice"):
        _files("lattice", "0", tmp_path)


def _complete_rows(count: int, calls=(226, 218, 314, 442, 434, 346, 370, 362, 338, 330)):
    rows = []
    for index in range(count):
        record = classify_client(calls=calls[index % len(calls)], reason="0", cap=10000)
        record["client_calls"] = int(record["force_calls"])
        rows.append((1, f"{index:04d}", record))
    return rows


def test_a_campaign_file_holds_the_mean_of_the_successes_and_the_failures_it_counted(tmp_path):
    template = load_template(ROOT, TEMPLATE)
    rows = _complete_rows(5, calls=(100, 110, 120, 130, 140))
    rows.append((1, "0005", classify_client(calls=10000, reason="2", cap=10000)))
    rows.append((1, "0006", {"force_calls": 0.0, "success": False, "failure": "crash", "timed_out": True}))
    rows.append((1, "0007", {"force_calls": 0.0, "success": False, "failure": "crash", "uncached": True}))
    output = tmp_path / "campaign" / "run.dat"
    summary = write_campaign(template, rows, output)

    assert summary["starts"] == 6
    assert summary["nfailed"] == 1
    assert summary["partial"] is True
    assert summary["template"] == TEMPLATE
    assert summary["problem"] == "tsopt/LJ38"
    lines = output.read_text(encoding="utf-8").splitlines()
    for expected in (
        "force_calls 120",
        "force_calls_min 100",
        "force_calls_max 140",
        "force_calls_median 120.0",
        "call_unit total",
        "nfailed 1",
        "counter harness",
        "count_keys HarnessCounter",
        "partial true",
    ):
        assert expected in lines
    starts = [json.loads(line) for line in output.with_suffix(".starts").read_text(encoding="utf-8").splitlines()]
    assert [row["geometry"] for row in starts] == [f"{index:04d}" for index in range(6)]
    assert [row["success"] for row in starts] == [True] * 5 + [False]
    assert starts[5]["failure"] == "cap"
    assert "failure" not in starts[0]


def test_a_campaign_that_holds_every_start_of_its_problem_is_not_partial(tmp_path):
    template = load_template(ROOT, TEMPLATE)
    stated = stated_full_n(ROOT, resolve_problem(ROOT, template["problem"]))
    assert stated == 200
    summary = write_campaign(template, _complete_rows(200), tmp_path / "full.dat")
    assert summary["partial"] is False
    assert summary["nfailed"] == 0
    assert summary["starts"] == 200
    assert "partial true" not in (tmp_path / "full.dat").read_text(encoding="utf-8")
    declared = dict(template, declare_partial=True)
    summary = write_campaign(declared, _complete_rows(200), tmp_path / "declared.dat")
    assert summary["partial"] is True
    assert "partial true" in (tmp_path / "declared.dat").read_text(encoding="utf-8").splitlines()


def test_a_campaign_on_a_problem_with_no_stated_size_is_always_partial(tmp_path):
    template = dict(load_template(ROOT, TEMPLATE), problem="saddle-search/pt-island")
    summary = write_campaign(template, _complete_rows(3), tmp_path / "open.dat")
    assert summary["partial"] is True


def test_a_campaign_of_failures_states_no_call_counts(tmp_path):
    template = load_template(ROOT, TEMPLATE)
    rows = [(1, "0000", classify_client(calls=10000, reason="0", cap=10000)), (1, "0001", classify_client(calls=77, reason="2", cap=10000))]
    summary = write_campaign(template, rows, tmp_path / "failed.dat")
    assert summary["nfailed"] == 2
    assert "force_calls" not in summary
    text = (tmp_path / "failed.dat").read_text(encoding="utf-8")
    assert "nfailed 2" in text.splitlines()
    assert not [line for line in text.splitlines() if line.startswith("force_calls")]


def test_a_campaign_of_nothing_but_timeouts_writes_nothing(tmp_path):
    template = load_template(ROOT, TEMPLATE)
    rows = [(1, "0000", {"force_calls": 5.0, "success": False, "failure": "crash", "timed_out": True})]
    output = tmp_path / "nothing" / "run.dat"
    assert write_campaign(template, rows, output) is None
    assert not output.exists()
    assert not output.with_suffix(".starts").exists()
