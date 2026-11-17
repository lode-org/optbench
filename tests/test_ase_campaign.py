"""How an ASE campaign finds its starts, scores them, and writes what it found.

The starts of the minimization problem are the archived LJ38 clusters. Those
runs are real: ASE relaxes the archived geometries and the harness counts the
calls. Their exact counts depend on the ASE release, so the tests compare a
run with itself, with the rows it wrote, and with the other ways of running
the same start, and never with a remembered number.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms

from optbench import ase_sella
from optbench.ase_lj import LennardJones
from optbench.ase_sella import (
    _baseline_parameters,
    _build_optimizer,
    _describe_precon,
    _jsonable,
    _load_rows,
    _lookup_stored,
    _nat_key,
    _plain,
    _snapshot_optimizer,
    _summarize,
    _v2_records,
    _write_dat,
    discover,
    entry_id,
    format_metric,
    materialize,
    raw_cap,
    run_many,
    run_point,
    run_template,
    write_problem_record,
    write_rows,
)
from optbench.campaign import load_template
from optbench.dat import parse_lenient

pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")

ROOT = Path(__file__).resolve().parents[1]
MINIMIZE = "ase/lj38-bfgs-default"


def _touch(base: Path, *names: str) -> Path:
    for name in names:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    return base


def _names(jobs) -> list[str]:
    return [name for name, _paths, _seed in jobs]


def test_starts_are_taken_in_natural_order_and_only_the_selected_kind(tmp_path):
    material = _touch(
        tmp_path,
        "cluster_10.con",
        "cluster_2.con",
        "cluster_1.con",
        "cluster_3.xyz",
        "initial_1.con",
        "reactant.con",
        "notes.txt",
    )
    for select, expected in (
        ("cluster", ["cluster_1.con", "cluster_2.con", "cluster_10.con"]),
        ("initial", ["initial_1.con"]),
        ("con", ["cluster_1.con", "cluster_2.con", "cluster_10.con", "initial_1.con"]),
    ):
        jobs = discover(material, {"task": "minimize", "input": {"select": select}})
        assert _names(jobs) == expected
        assert all(seed is None for _name, _paths, seed in jobs)
    assert _names(discover(material, {"task": "minimize"})) == _names(
        discover(material, {"task": "minimize", "input": {"select": "con"}})
    )


def test_one_geometry_file_is_one_start_named_after_the_file(tmp_path):
    path = _touch(tmp_path, "pt-island.con") / "pt-island.con"
    ((name, paths, seed),) = discover(path, {"task": "minimize", "input": {"select": "file"}})
    assert (name, paths, seed) == ("pt-island.con", [path], None)


@pytest.mark.parametrize("task", ["dimer", "sella-search"])
def test_a_search_that_varies_the_seed_has_one_start_per_seed_on_one_geometry(tmp_path, task):
    material = _touch(tmp_path, "island.xyz", "island.con", "other.con")
    block = {"searches": 3}
    template = {"task": task, "dimer": block, "sella_search": block}
    jobs = discover(material, template)
    assert _names(jobs) == ["seed-0000", "seed-0001", "seed-0002"]
    assert [seed for _name, _paths, seed in jobs] == [0, 1, 2]
    assert {tuple(paths) for _name, paths, _seed in jobs} == {(material / "island.con",)}
    single = discover(material / "island.con", {"task": task})
    assert _names(single) == ["seed-0000"]


def test_a_band_between_minima_is_one_start_per_directory_that_holds_both_ends(tmp_path):
    material = _touch(
        tmp_path,
        "min_10/a.con",
        "min_10/b.con",
        "min_2/a.con",
        "min_2/b.con",
        "min_3/a.con",
        "other/a.con",
        "other/b.con",
    )
    jobs = discover(material, {"task": "neb", "neb": {"layout": "minima"}})
    assert _names(jobs) == ["min_2", "min_10"]
    assert jobs[0][1] == [material / "min_2" / "a.con", material / "min_2" / "b.con"]


def test_a_band_from_one_reactant_runs_to_each_product_and_prefers_the_con_file(tmp_path):
    material = _touch(
        tmp_path,
        "reactant.con",
        "product_2.xyz",
        "product_2.con",
        "product_10.xyz",
        "product_1.con",
        "start_0.con",
    )
    jobs = discover(material, {"task": "neb", "neb": {"layout": "reactant-products"}})
    assert _names(jobs) == ["product_1", "product_2", "product_10"]
    assert jobs[1][1] == [material / "reactant.con", material / "product_2.con"]
    assert jobs[2][1] == [material / "reactant.con", material / "product_10.xyz"]


def test_an_indexed_band_needs_a_start_and_an_end_and_honours_the_count(tmp_path):
    material = _touch(
        tmp_path,
        "start_0.con",
        "end_0.con",
        "start_1.xyz",
        "start_1.con",
        "end_1.con",
        "start_2.con",
        "end_3.con",
        "reactant.con",
    )
    jobs = discover(material, {"task": "neb", "neb": {"layout": "indexed"}})
    assert _names(jobs) == ["0000", "0001"]
    assert jobs[1][1] == [material / "start_1.con", material / "end_1.con"]
    assert _names(discover(material, {"task": "sella-neb", "neb": {"layout": "indexed", "count": 1}})) == ["0000"]


def test_numbers_inside_a_name_sort_as_numbers():
    names = ["cluster_10.con", "cluster_9.con", "cluster_100.con", "cluster_1.con"]
    assert [path.name for path in sorted((Path(name) for name in names), key=_nat_key)] == [
        "cluster_1.con",
        "cluster_9.con",
        "cluster_10.con",
        "cluster_100.con",
    ]


def _pack(path: Path, members: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as packed:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            packed.addfile(info, io.BytesIO(data))
    return path


def _root_with_manifest(tmp_path: Path, rows: list[str]) -> Path:
    root = tmp_path / "root"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_an_archive_is_unpacked_once_into_a_directory_named_after_its_store(tmp_path):
    root = _root_with_manifest(
        tmp_path,
        [
            "",
            json.dumps({"path": "website/other.tgz", "stored": "inputs/other/blob"}),
            json.dumps({"path": "website/x/clusters.tgz", "stored": "inputs/abc123/blob"}),
        ],
    )
    _pack(root / "inputs" / "abc123" / "blob", {"cluster_0.con": "first\n"})
    cache = tmp_path / "cache"
    template = {"id": "t/x", "input": {"archive_path": "website/x/clusters.tgz"}}
    found = materialize(root, template, {}, cache)
    assert found == cache / "abc123"
    assert (found / "cluster_0.con").read_text(encoding="utf-8") == "first\n"
    assert (found / ".complete").is_file()
    (found / "cluster_0.con").write_text("edited\n", encoding="utf-8")
    again = materialize(root, template, {}, cache)
    assert again == found
    assert (found / "cluster_0.con").read_text(encoding="utf-8") == "edited\n"


def test_a_stored_file_that_is_not_an_archive_is_used_where_it_lies(tmp_path):
    root = _root_with_manifest(tmp_path, [json.dumps({"path": "website/x/one.con", "stored": "inputs/aa/blob"})])
    (root / "inputs" / "aa").mkdir()
    (root / "inputs" / "aa" / "blob").write_text("not a tarball\n", encoding="utf-8")
    problem = {"input": {"archive_path": "website/x/one.con"}}
    found = materialize(root, {"id": "t/x"}, problem, tmp_path / "cache")
    assert found == root / "inputs" / "aa" / "blob"
    assert not (tmp_path / "cache").exists()


def test_the_template_names_the_archive_before_the_problem_does(tmp_path):
    root = _root_with_manifest(
        tmp_path,
        [
            json.dumps({"path": "from-template", "stored": "inputs/t/blob"}),
            json.dumps({"path": "from-problem", "stored": "inputs/p/blob"}),
        ],
    )
    for stored in ("t", "p"):
        (root / "inputs" / stored).mkdir()
        (root / "inputs" / stored / "blob").write_text(stored, encoding="utf-8")
    both = materialize(root, {"id": "t/x", "input": {"archive_path": "from-template"}}, {"input": {"archive_path": "from-problem"}}, tmp_path / "c")
    assert both.parent.name == "t"
    fallback = materialize(root, {"id": "t/x"}, {"input": {"archive_path": "from-problem"}}, tmp_path / "c")
    assert fallback.parent.name == "p"


def test_a_start_with_no_archive_or_an_unlisted_one_is_refused(tmp_path):
    root = _root_with_manifest(tmp_path, [json.dumps({"path": "listed", "stored": "inputs/aa/blob"})])
    with pytest.raises(FileNotFoundError, match="t/x has no geometry archive"):
        materialize(root, {"id": "t/x"}, {}, tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="website/missing.tgz"):
        _lookup_stored(root, "website/missing.tgz")


def _row(start: str, calls: int, *, failed: bool = False, **extra) -> dict:
    return {"start": start, "force_calls": calls, "failed": failed, **extra}


def test_a_capped_reducer_charges_a_failure_and_a_spent_budget_the_cap_and_leaves_them_out_of_the_mean():
    template = {"stop": {"max_force_calls": 500}, "reducer": "capped"}
    rows = [_row("a", 100), _row("b", 200), _row("c", 50, failed=True), _row("d", 500)]
    summary = _summarize(rows, template)
    assert summary["force_calls"] == 150
    assert summary["force_calls_min"] == 100
    assert summary["force_calls_max"] == 200
    assert summary["force_calls_median"] == 150.0
    assert summary["nfailed"] == 2
    nothing = _summarize([_row("a", 500, failed=True)], template)
    assert nothing["force_calls"] is None
    assert nothing["force_calls_median"] is None
    assert nothing["nfailed"] == 1


def test_the_all_reducer_keeps_every_count_and_only_counts_the_failures():
    template = {"stop": {"max_force_calls": 500}, "reducer": "all"}
    rows = [_row("a", 100), _row("b", 200, failed=True), _row("c", 300)]
    summary = _summarize(rows, template)
    assert summary["force_calls"] == 200
    assert summary["force_calls_min"] == 100
    assert summary["force_calls_max"] == 300
    assert summary["nfailed"] == 1


def test_the_per_image_reducer_divides_a_band_total_by_its_moving_images():
    template = {"stop": {"max_force_calls": 500}, "reducer": "per_image", "neb": {"moving_images": 5}}
    rows = [_row("a", 100), _row("b", 200), _row("c", 500, failed=True)]
    summary = _summarize(rows, template)
    assert summary["force_calls"] == 150
    assert summary["force_calls_per_image"] == 30
    assert summary["force_calls_per_image_min"] == 20
    assert summary["force_calls_per_image_max"] == 40
    assert summary["call_unit"] == "total"
    assert summary["nfailed"] == 1


def test_a_reducer_the_campaign_does_not_know_is_refused():
    with pytest.raises(ValueError, match="median"):
        _summarize([_row("a", 1)], {"stop": {"max_force_calls": 5}, "reducer": "median"})


def test_a_success_at_the_cap_is_not_a_success_and_a_failure_keeps_why_it_failed():
    template = {"stop": {"max_force_calls": 500}}
    rows = [
        {"force_calls": 100, "success": True},
        {"force_calls": 500, "success": True},
        {"force_calls": 120, "success": False, "reasons": ["cap", "force_norm"], "failure": "cap"},
        {"force_calls": 130, "success": False, "failure": "crash"},
        {"force_calls": 140, "success": False, "failure": "something else"},
        {"force_calls": 150},
    ]
    records, cap = _v2_records(rows, template)
    assert cap == 500
    assert records == [
        {"force_calls": 100.0, "success": True},
        {"force_calls": 500.0, "success": False},
        {"force_calls": 120.0, "success": False, "reasons": ["cap", "force_norm"]},
        {"force_calls": 130.0, "success": False, "failure": "crash"},
        {"force_calls": 140.0, "success": False},
        {"force_calls": 150.0, "success": False},
    ]


def test_the_entry_id_of_a_template_is_the_name_of_its_published_campaign_file():
    from optbench.campaign import list_templates

    published: dict[str, list[str]] = {}
    for path in sorted((ROOT / "results" / "campaign").glob("*.dat")):
        fields, _duplicates = parse_lenient(path.read_text(encoding="utf-8"))
        if "template" in fields:
            published.setdefault(fields["template"], []).append(path.stem)
    checked = []
    for template in list_templates(ROOT):
        family = template["engine"] in {"ase", "sella"} or ase_sella._geodesic_ase(template)
        if family and len(published.get(template["id"], [])) == 1:
            assert entry_id(template) == published[template["id"]][0], template["id"]
            checked.append(template["id"])
    assert {name.split("/")[0] for name in checked} >= {"ase", "geodesic"}


def test_the_cap_is_the_raw_harness_budget():
    assert raw_cap({"stop": {"max_force_calls": 2500}}) == 2500
    assert raw_cap(load_template(ROOT, "ase/de-lj38-default")) == 2500


def test_metrics_print_whole_when_whole_and_not_at_all_when_not_finite():
    assert format_metric(None) is None
    assert format_metric(float("nan")) is None
    assert format_metric(float("-inf")) is None
    assert format_metric(True) == "true"
    assert format_metric(False) == "false"
    assert format_metric(463.0) == "463"
    assert format_metric(0.34238022749) == "0.3423802275"
    assert format_metric(7) == "7"
    assert format_metric("text") == "text"


def test_plain_and_json_values_keep_their_kind_and_stringify_the_rest():
    assert _plain(True) is True
    assert _plain(np.float64(1.5)) == 1.5
    assert type(_plain(np.float64(1.5))) is float
    assert _plain("x") == "x"
    nested = {"a": (1, 2.5, np.float64(3.0)), 4: [None, True, "s", object]}
    converted = _jsonable(nested)
    assert converted["a"] == [1, 2.5, 3.0]
    assert converted["4"][:3] == [None, True, "s"]
    assert isinstance(converted["4"][3], str)
    json.dumps(converted)


def test_the_precon_a_run_reports_is_its_name_and_its_amplitude():
    from ase.optimize.precon import Exp

    assert _describe_precon(None) == "none"
    assert _describe_precon("Exp") == "Exp"
    assert _describe_precon(Exp(A=3.5)) == "Exp A=3.5"


def _atoms_with_potential() -> Atoms:
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=40.0)
    return atoms


def test_a_snapshot_reports_the_defaults_the_request_and_what_the_optimizer_holds():
    spec = {"name": "FIRE", "parameters": {"dt": 0.2, "maxstep": 0.05}}
    optimizer = _build_optimizer(spec, _atoms_with_potential())
    parameters = _snapshot_optimizer(optimizer, spec)
    assert parameters["dt"] == 0.2
    assert parameters["maxstep"] == 0.05
    assert parameters["Nmin"] == 5
    assert parameters["finc"] == 1.1
    plain = _snapshot_optimizer(_build_optimizer({"name": "BFGS"}, _atoms_with_potential()), {"name": "BFGS"})
    assert plain["maxstep"] == 0.2
    assert plain["alpha"] == 70.0


def test_a_preconditioned_optimizer_names_its_preconditioner_and_renames_maxstep():
    spec = {"name": "PreconFIRE", "parameters": {"maxstep": 0.05, "precon": {"name": "Exp", "A": 3.5}}}
    optimizer = _build_optimizer(spec, _atoms_with_potential())
    parameters = _snapshot_optimizer(optimizer, spec)
    assert parameters["precon"] == "Exp A=3.5"
    assert parameters["maxmove"] == 0.05
    assert "maxstep" not in parameters


def test_the_baseline_of_a_minimizer_is_its_defaults_with_the_request_laid_over_them():
    template = {"task": "minimize", "optimizer": {"name": "BFGS", "parameters": {"maxstep": 0.1}}}
    parameters = _baseline_parameters(template)
    assert parameters == {"maxstep": 0.1, "alpha": 70.0}
    fire = _baseline_parameters(
        {"task": "minimize", "optimizer": {"name": "PreconFIRE", "parameters": {"maxstep": 0.07, "precon": {"A": 2.0}}}}
    )
    assert fire["maxmove"] == 0.07
    assert "maxstep" not in fire
    assert fire["precon"] == "Exp A=2.0"
    assert _baseline_parameters({"task": "minimize", "optimizer": {"name": "Unlisted"}}) == {}


def test_the_baseline_of_sella_follows_the_requested_order_before_the_template_one():
    minimum = _baseline_parameters(
        {"task": "sella", "optimizer": {"parameters": {"order": 0}}, "sella": {"order": 1, "v0": "none"}}
    )
    assert minimum["order"] == 0
    assert minimum["method"] == "qn"
    assert minimum["eig"] is False
    assert "v0" not in minimum
    saddle = _baseline_parameters({"task": "sella", "optimizer": {"parameters": {}}, "sella": {"order": 1, "v0": "reactant"}})
    assert saddle["order"] == 1
    assert saddle["method"] == "prfo"
    assert saddle["eig"] is True
    assert saddle["v0"] == "reactant"


def test_the_baseline_of_a_dimer_and_a_search_name_what_is_not_the_library_default():
    dimer = _baseline_parameters({"task": "dimer", "optimizer": {"parameters": {"f_rot_min": 0.2}}})
    assert dimer["f_rot_min"] == 0.2
    assert dimer["eigenmode_method"] == "dimer"
    assert dimer["dimer_separation"] == 0.0001
    search = _baseline_parameters(
        {
            "task": "sella-search",
            "optimizer": {"name": "Sella", "parameters": {"order": 1}},
            "sella_search": {"searches": 16, "gauss_std": 0.25},
        }
    )
    assert search["v0"] == "none"
    assert search["gauss_std"] == 0.25
    assert search["method"] == "prfo"
    default_std = _baseline_parameters({"task": "sella-search", "optimizer": {"name": "Sella"}})
    assert default_std["gauss_std"] == 0.1


def test_the_baseline_of_a_band_adds_the_band_settings_and_a_sella_band_adds_its_reserve():
    neb = {
        "moving_images": 5,
        "spring": 0.1,
        "climb": False,
        "method": "improvedtangent",
        "interpolate": "idpp",
        "climb_after": 0.5,
    }
    band = _baseline_parameters({"task": "neb", "optimizer": {"name": "FIRE", "parameters": {}}, "neb": neb})
    assert band["spring"] == 0.1
    assert band["climb"] is False
    assert band["moving_images"] == 5
    assert band["interpolate"] == "idpp"
    assert band["climb_after"] == 0.5
    assert band["dt"] == 0.1
    refined = _baseline_parameters(
        {
            "task": "sella-neb",
            "optimizer": {"name": "Sella", "parameters": {"order": 1}},
            "neb": {**neb, "band_optimizer": "LBFGS", "band_parameters": {"memory": 20}},
            "sella": {"order": 1, "reserve": 0.3},
        }
    )
    assert refined["band_optimizer"] == "LBFGS"
    assert refined["band_parameters"] == {"memory": 20}
    assert refined["reserve"] == 0.3
    assert refined["v0"] == "none"
    assert refined["order"] == 1
    assert refined["moving_images"] == 5


@pytest.fixture
def labelled(monkeypatch):
    monkeypatch.setenv("BENCH_HOST_LABEL", "bench-host")


def _fields(path: Path) -> dict[str, str]:
    return parse_lenient(path.read_text(encoding="utf-8"))[0]


SCORED = {
    "scoring": "par2-bootstrap",
    "par2_mean": 463.0,
    "median": 463.0,
    "median_low": 365.0,
    "median_high": 561.0,
    "iqr": 98.0,
    "success_rate": 1.0,
    "rate_low": 0.3423802275,
    "rate_high": 1.0,
    "par_applied": True,
    "par_factor": 2,
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20261015,
}
SUMMARY = {"force_calls": 463, "force_calls_min": 365, "force_calls_median": 463.0, "force_calls_max": 561, "nfailed": 0}


def test_the_dat_names_the_template_the_counter_the_method_and_who_ran_it(tmp_path):
    template = load_template(ROOT, MINIMIZE)
    path = tmp_path / "out" / "entry.dat"
    _write_dat(path, template, SUMMARY, {"maxstep": 0.1}, starts=2, partial=False, host="bench-host", scored=SCORED)
    fields = _fields(path)
    assert fields["force_calls"] == "463"
    assert fields["force_calls_median"] == "463"
    assert fields["nfailed"] == "0"
    assert fields["starts"] == "2"
    assert fields["template"] == MINIMIZE
    assert fields["problem"] == "minimization/lj38"
    assert fields["counter"] == "harness"
    assert fields["reducer"] == "capped"
    assert fields["host"] == "bench-host"
    assert fields["contributor"] == "Rohit Goswami"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["date"])
    assert fields["par2_mean"] == "463"
    assert fields["par2_median_low"] == "365"
    assert fields["par_applied"] == "true"
    assert fields["bootstrap_seed"] == "20261015"
    assert "partial" not in fields
    assert "ranked" not in fields
    assert "transition_states" not in fields
    method = json.loads(fields["method"])
    assert method["parameters"] == {"maxstep": 0.1}
    assert method["counter"] == "harness attach_ase_counter"
    assert method["stop"]["max_force_calls"] == 10000
    assert method["potential"]["kind"] == "lennard-jones"
    assert "Hessian index" in fields["note"]


def test_a_partial_unranked_dat_says_so_and_a_dat_without_scores_has_no_score_lines(tmp_path):
    template = dict(load_template(ROOT, MINIMIZE), ranked=False)
    path = tmp_path / "entry.dat"
    _write_dat(path, template, SUMMARY, {}, starts=2, partial=True, host="bench-host")
    fields = _fields(path)
    assert fields["partial"] == "true"
    assert fields["ranked"] == "false"
    assert "par2_mean" not in fields
    assert "scoring" not in fields


def test_a_band_note_says_the_hessian_is_not_applied_and_the_kept_pairs_are_counted_for_the_two_ts_problem(tmp_path):
    template = load_template(ROOT, "ase/de-lj38-2ts-default")
    path = tmp_path / "entry.dat"
    _write_dat(path, template, {"force_calls": 10, "nfailed": 3}, {}, starts=8, partial=False, host="bench-host")
    fields = _fields(path)
    assert "Hessian and Kabsch RMSD are not applied" in fields["note"]
    assert fields["transition_states"] == "5"


@pytest.fixture(scope="module")
def lj38_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("lj38-run")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("BENCH_HOST_LABEL", "bench-host")
        summary = run_template(ROOT, MINIMIZE, output=base / "out" / "entry.dat", limit=2, cache=base / "geometry")
    rows = [
        json.loads(line)
        for line in (base / "out" / "entry.starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return base, summary, rows


def test_a_limited_run_relaxes_the_first_clusters_and_marks_the_result_partial(lj38_run):
    base, summary, rows = lj38_run
    assert [row["start"] for row in rows] == ["cluster_0000.con", "cluster_0001.con"]
    for row in rows:
        assert row["success"] is True
        assert row["failure"] is None
        assert row["hessian_index"] == 0
        assert row["force_norm"] <= 0.01
        assert 0 < row["force_calls"] < 10000
        assert row["verify_calls"] >= 2 * 3 * 38
        assert row["schema"] == "optbench.start/v2"
    assert summary["starts"] == 2
    assert summary["template"] == MINIMIZE
    assert summary["problem"] == "minimization/lj38"
    assert summary["entry"] == "ase-3.29.0-lj38-bfgs-default"
    assert summary["output"] == str(base / "out" / "entry.dat")
    assert summary["nfailed"] == 0
    calls = [row["force_calls"] for row in rows]
    assert summary["force_calls_min"] == min(calls)
    assert summary["force_calls_max"] == max(calls)
    assert summary["force_calls"] == round(sum(calls) / 2)
    fields = _fields(base / "out" / "entry.dat")
    assert fields["partial"] == "true"
    assert fields["starts"] == "2"
    assert fields["host"] == "bench-host"
    assert fields["force_calls"] == str(summary["force_calls"])
    assert fields["success_rate"] == "1"


def test_the_first_row_carries_the_parameters_the_optimizer_actually_ran_with(lj38_run):
    _base, _summary, rows = lj38_run
    assert rows[0]["parameters"]["maxstep"] == 0.1
    assert "parameters" not in rows[1]
    method = json.loads(_fields(lj38_run[0] / "out" / "entry.dat")["method"])
    assert method["parameters"]["maxstep"] == 0.1
    assert method["parameters"]["alpha"] == 70.0


def test_a_resumed_run_keeps_the_rows_it_has_and_runs_only_the_starts_it_lacks(lj38_run, tmp_path, monkeypatch):
    base, _summary, rows = lj38_run
    output = tmp_path / "again" / "entry.dat"
    output.parent.mkdir()
    (output.parent / "entry.starts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    ran = []
    real = ase_sella._run_start

    def watched(payload):
        ran.append(payload["start"])
        return real(payload)

    monkeypatch.setattr(ase_sella, "_run_start", watched)
    summary = run_template(ROOT, MINIMIZE, output=output, limit=3, resume=True, cache=base / "geometry")
    assert ran == ["cluster_0002.con"]
    kept = [json.loads(line) for line in (output.parent / "entry.starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["start"] for row in kept] == ["cluster_0000.con", "cluster_0001.con", "cluster_0002.con"]
    assert kept[0]["force_calls"] == rows[0]["force_calls"]
    assert kept[1]["force_calls"] == rows[1]["force_calls"]
    assert summary["starts"] == 3


def test_a_run_that_is_not_resumed_starts_over_and_gives_the_same_counts(lj38_run, tmp_path):
    base, _summary, rows = lj38_run
    output = tmp_path / "fresh" / "entry.dat"
    output.parent.mkdir()
    stale = output.parent / "entry.starts.jsonl"
    stale.write_text(json.dumps({"start": "cluster_0000.con", "force_calls": 1}) + "\n", encoding="utf-8")
    run_template(ROOT, MINIMIZE, output=output, limit=2, cache=base / "geometry")
    again = [json.loads(line) for line in stale.read_text(encoding="utf-8").splitlines()]
    assert [row["force_calls"] for row in again] == [row["force_calls"] for row in rows]
    assert [row["force_norm"] for row in again] == [row["force_norm"] for row in rows]


def test_workers_in_other_processes_give_the_same_rows_in_the_order_of_the_starts(lj38_run, tmp_path):
    base, _summary, rows = lj38_run
    output = tmp_path / "parallel" / "entry.dat"
    run_template(ROOT, MINIMIZE, output=output, limit=2, workers=2, cache=base / "geometry")
    parallel = [
        json.loads(line)
        for line in (output.parent / "entry.starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in parallel] == [row["start"] for row in rows]
    assert [row["force_calls"] for row in parallel] == [row["force_calls"] for row in rows]
    assert [row["energy"] for row in parallel] == pytest.approx([row["energy"] for row in rows], rel=1e-12)


def test_one_start_run_alone_matches_its_row_in_the_campaign(lj38_run):
    base, _summary, rows = lj38_run
    record = run_point(ROOT, MINIMIZE, "cluster_0001.con", 0, cache=base / "geometry")
    assert record["start"] == "cluster_0001.con"
    assert record["template"] == MINIMIZE
    assert record["problem"] == "minimization/lj38"
    assert record["force_calls"] == rows[1]["force_calls"]
    assert record["energy"] == pytest.approx(rows[1]["energy"], rel=1e-12)
    assert record["wall_seconds"] >= 0.0
    assert record["parameters"]["maxstep"] == 0.1


def test_a_start_the_problem_does_not_have_is_a_key_error_naming_it(lj38_run):
    base, _summary, _rows = lj38_run
    with pytest.raises(KeyError, match=r"ase/lj38-bfgs-default has no start 'cluster_9999.con' seed 0"):
        run_point(ROOT, MINIMIZE, "cluster_9999.con", 0, cache=base / "geometry")
    with pytest.raises(KeyError, match="seed 1"):
        run_point(ROOT, MINIMIZE, "cluster_0000.con", 1, cache=base / "geometry")


def test_a_search_that_varies_the_seed_is_found_by_its_seed_whatever_start_is_named(tmp_path, monkeypatch):
    seen = []

    def stub(payload):
        seen.append(payload)
        return {"start": payload["start"], "seed": payload["seed"], "force_calls": 1, "success": True}

    monkeypatch.setattr(ase_sella, "_run_start", stub)
    record = run_point(ROOT, "ase/dimer-pt-island-default", "anything", 3, cache=tmp_path / "geometry")
    assert record["start"] == "seed-0003"
    assert record["template"] == "ase/dimer-pt-island-default"
    assert record["problem"] == "saddle-search/pt-island"
    (payload,) = seen
    assert payload["seed"] == 3
    assert payload["task"] == "dimer"
    assert payload["snapshot"] is True
    assert len(payload["paths"]) == 1
    with pytest.raises(KeyError, match="seed 16"):
        run_point(ROOT, "ase/dimer-pt-island-default", "anything", 16, cache=tmp_path / "geometry")


def test_a_template_of_another_engine_is_not_an_ase_campaign(tmp_path):
    with pytest.raises(ValueError, match="lammpsneb/ci-fire is not an ASE or Sella template"):
        run_template(ROOT, "lammpsneb/ci-fire", output=tmp_path / "x.dat", cache=tmp_path / "geometry")


def test_a_template_whose_archive_holds_no_starts_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ase_sella, "discover", lambda material, template: [])
    with pytest.raises(FileNotFoundError, match=f"{MINIMIZE} found no geometries"):
        run_template(ROOT, MINIMIZE, output=tmp_path / "x.dat", cache=tmp_path / "geometry")


def test_an_errored_start_is_reported_with_its_error_and_counts_as_a_failure(tmp_path, monkeypatch, capsys):
    def stub(payload):
        return {
            "schema": "optbench.start/v2",
            "start": payload["start"],
            "seed": 0,
            "force_calls": 0,
            "verify_calls": 0,
            "counter": "harness",
            "comparable": True,
            "success": False,
            "failed": True,
            "failure": "crash",
            "reasons": ["crash", "force_norm"],
            "error": "RuntimeError: boom",
            "force_norm": None,
        }

    monkeypatch.setattr(ase_sella, "_run_start", stub)
    summary = run_template(ROOT, MINIMIZE, output=tmp_path / "x.dat", limit=2, cache=tmp_path / "geometry")
    assert summary["nfailed"] == 2
    assert summary["force_calls"] is None
    printed = capsys.readouterr().out
    assert f"error {MINIMIZE} cluster_0000.con: RuntimeError: boom" in printed
    assert f"error {MINIMIZE} cluster_0001.con: RuntimeError: boom" in printed
    assert f"{MINIMIZE} 2/2" in printed
    fields = _fields(tmp_path / "x.dat")
    assert fields["nfailed"] == "2"
    assert "force_calls" not in fields


def _finished(start: str, calls: int, success: bool, failure=None) -> dict:
    return {
        "start": start,
        "force_calls": calls,
        "success": success,
        "failure": failure,
        "reasons": [] if success else [failure],
        "force_norm": 0.005 if success else 0.4,
    }


def test_finished_rows_are_written_with_the_defaults_a_start_row_needs(tmp_path, labelled):
    rows = [_finished("cluster_0000.con", 120, True), _finished("cluster_0001.con", 10000, False, "cap")]
    summary = write_rows(ROOT, MINIMIZE, rows, tmp_path / "out" / "entry.dat")
    written = [
        json.loads(line)
        for line in (tmp_path / "out" / "entry.starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["failed"] for row in written] == [False, True]
    assert all(row["schema"] == "optbench.start/v2" for row in written)
    assert all(row["counter"] == "harness" and row["comparable"] is True for row in written)
    assert summary["starts"] == 2
    assert summary["nfailed"] == 1
    assert summary["force_calls"] == 120
    assert summary["entry"] == "ase-3.29.0-lj38-bfgs-default"
    assert summary["par2_mean"] == pytest.approx((120 + 2 * 10000) / 2)
    fields = _fields(tmp_path / "out" / "entry.dat")
    assert fields["host"] == "bench-host"
    assert "partial" not in fields
    assert fields["starts"] == "2"
    assert fields["nfailed"] == "1"


def test_a_row_that_already_names_its_parameters_decides_what_the_dat_reports(tmp_path):
    rows = [dict(_finished("cluster_0000.con", 120, True), parameters={"maxstep": 0.05, "alpha": 70.0})]
    write_rows(ROOT, MINIMIZE, rows, tmp_path / "entry.dat")
    method = json.loads(_fields(tmp_path / "entry.dat")["method"])
    assert method["parameters"] == {"alpha": 70.0, "maxstep": 0.05}


def test_starts_already_on_disk_are_read_by_name_and_a_missing_file_is_no_starts(tmp_path):
    assert _load_rows(tmp_path / "absent.jsonl") == {}
    path = tmp_path / "starts.jsonl"
    path.write_text(
        json.dumps({"start": "a", "force_calls": 1}) + "\n\n" + json.dumps({"start": "b", "force_calls": 2}) + "\n",
        encoding="utf-8",
    )
    assert {name: row["force_calls"] for name, row in _load_rows(path).items()} == {"a": 1, "b": 2}


def test_every_ranked_template_runs_lightest_problem_first_and_unranked_ones_wait_for_a_request(tmp_path, monkeypatch):
    ran = []

    def stub(root, template_id, **kwargs):
        ran.append((template_id, kwargs["output"].name, kwargs["workers"], kwargs["limit"], kwargs["resume"]))
        return {"template": template_id}

    monkeypatch.setattr(ase_sella, "run_template", stub)
    summaries = run_many(ROOT, tmp_path / "out", workers=3, limit=2, resume=True)
    ids = [item[0] for item in ran]
    assert len(summaries) == len(ids) > 20
    assert "ase/dimer-pt-island-default" not in ids
    assert "sella/pt-island-search-default" not in ids
    assert all(item[2:] == (3, 2, True) for item in ran)
    assert ids.index("ase/morse-bfgs-default") < ids.index("ase/lj38-bfgs-default")
    assert ids.index("ase/lj38-bfgs-default") < ids.index("ase/lj38-gpmin-default")
    assert ids.index("ase/lj38-bfgs-default") < ids.index("ase/de-pt-island-default")
    assert dict((item[0], item[1]) for item in ran)["ase/lj38-bfgs-default"] == "ase-3.29.0-lj38-bfgs-default.dat"

    ran.clear()
    run_many(ROOT, tmp_path / "out", ranked_only=False, match="pt-island")
    ids = [item[0] for item in ran]
    assert "ase/dimer-pt-island-default" in ids
    assert "sella/pt-island-search-default" in ids
    assert all("pt-island" in name for name in ids)

    ran.clear()
    run_many(ROOT, tmp_path / "out", match="lj38-lbfgs-default")
    assert [item[0] for item in ran] == ["ase/lj38-lbfgs-default"]


def _campaign_dat(directory: Path, stem: str, **fields) -> Path:
    base = {
        "force_calls": "463",
        "force_calls_min": "365",
        "force_calls_median": "462.5",
        "force_calls_max": "561",
        "nfailed": "0",
        "starts": "50",
        "code": "ASE",
        "algorithm": "BFGS",
        "code_version": "3.29.0",
        "commit": "f27c0005ae6a67ea419f996e728668865bfc1f86",
        "problem": "minimization/lj38",
        "method": json.dumps({"parameters": {"maxstep": 0.1}}),
        "note": "kept",
    }
    base.update(fields)
    text = "".join(f"{key} {value}\n" for key, value in base.items() if value is not None)
    path = directory / f"{stem}.dat"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_problem_record_lists_the_complete_runs_of_that_problem_with_numeric_metrics(tmp_path, labelled):
    root = tmp_path / "root"
    campaign = root / "results" / "campaign"
    campaign.mkdir(parents=True)
    _campaign_dat(campaign, "ase-3.29.0-lj38-bfgs-default")
    _campaign_dat(campaign, "ase-3.29.0-lj38-lbfgs-default", algorithm="LBFGS", force_calls="400", method=None, nfailed="2")
    _campaign_dat(campaign, "ase-3.29.0-lj38-partial", partial="true")
    _campaign_dat(campaign, "ase-3.29.0-morse-bfgs-default", problem="minimization/morse-bulk")
    destination = tmp_path / "record.json"
    write_problem_record(root, campaign, destination, "minimization/lj38", "complete")
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["schema"] == "optbench.result/v1"
    assert record["problem"] == "minimization/lj38"
    assert record["status"] == "complete"
    assert record["host"] == "bench-host"
    assert record["submitter"] == "Rohit Goswami"
    assert [run["entry"] for run in record["runs"]] == ["ase-3.29.0-lj38-bfgs-default", "ase-3.29.0-lj38-lbfgs-default"]
    first, second = record["runs"]
    assert first["metrics"] == {
        "force_calls": 463,
        "force_calls_min": 365,
        "force_calls_max": 561,
        "force_calls_median": 462.5,
        "nfailed": 0,
    }
    assert isinstance(first["metrics"]["force_calls"], int)
    assert first["method"] == {"parameters": {"maxstep": 0.1}}
    assert first["starts"] == 50
    assert first["starts_file"] == "results/campaign/ase-3.29.0-lj38-bfgs-default.starts.jsonl"
    assert first["notes"] == "kept"
    assert first["present_source"] is True
    assert second["method"] == {}
    assert second["nfailed"] == 2
    assert second["algorithm"] == "LBFGS"
