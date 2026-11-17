"""What a template and a problem have to agree on before a campaign starts.

The stop is the contract. A template may not stop earlier than the problem
says, may not use another force norm, and has to carry a force-call cap from
somewhere. The same checks decide which problem a template may run on, which
files a start is staged with, and which archive a start is cut from. The
fixtures are the shipped LJ38 problem and the shipped templates, changed one
number at a time.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import io
import json
import tarfile
from pathlib import Path

import pytest

from optbench.campaign import (
    _check_stop,
    _template_inputs,
    benchmark_text,
    client_count,
    client_jobs,
    client_record,
    client_result,
    format_ini,
    geometry_archive,
    iter_geometries,
    list_templates,
    load_template,
    parse_client_dat,
    require_limit,
    resolve_problem,
    run,
)

ROOT = Path(__file__).resolve().parents[1]


def _lj38() -> dict:
    return resolve_problem(ROOT, "minimization/lj38")


def _ase_template() -> dict:
    return copy.deepcopy(load_template(ROOT, "ase/lj38-bfgs-default"))


def _client_template() -> dict:
    return copy.deepcopy(load_template(ROOT, "eon-client/lbfgs-m20"))


def test_the_shipped_lj38_minimizers_take_the_problem_cap_as_their_cap():
    assert _check_stop(_lj38(), _ase_template()) == 10000
    assert _check_stop(_lj38(), _client_template()) == 10000


def test_a_template_that_stops_at_another_force_norm_than_the_problem_is_refused():
    template = _ase_template()
    template["stop"]["force_norm"] = 0.011
    with pytest.raises(ValueError, match="template force_norm is not the problem force norm"):
        _check_stop(_lj38(), template)
    template["stop"]["force_norm"] = 0.01 + 1e-13
    assert _check_stop(_lj38(), template) == 10000


def test_a_template_that_stops_at_another_gradient_norm_than_the_problem_is_refused():
    problem = {"id": "toy/gradient", "convergence": {"gradient_norm": 0.01}}
    with pytest.raises(ValueError, match="template gradient_norm is not the problem gradient norm"):
        _check_stop(problem, {"id": "toy/t", "stop": {"gradient_norm": 0.02}})
    assert _check_stop(problem, {"id": "toy/t", "stop": {"gradient_norm": 0.01}}) is None


def test_a_client_that_converges_at_another_force_than_the_problem_is_refused():
    template = _client_template()
    template["ini"]["Optimizer"]["converged_force"] = 0.02
    with pytest.raises(ValueError, match="template converged_force is not the problem force norm"):
        _check_stop(_lj38(), template)


def test_a_template_budget_under_the_problem_cap_is_refused_and_one_over_it_is_not_used():
    template = _ase_template()
    template["stop"]["max_force_calls"] = 9999
    with pytest.raises(ValueError, match="template max_force_calls is below the problem force-call cap"):
        _check_stop(_lj38(), template)
    template["stop"]["max_force_calls"] = 20000
    assert _check_stop(_lj38(), template) == 10000


def test_a_minimizer_iteration_limit_cannot_sit_under_the_force_budget_but_a_band_limit_can():
    template = _client_template()
    template["ini"]["Optimizer"]["max_iterations"] = 9999
    with pytest.raises(ValueError, match="template max_iterations is below the problem force-call cap"):
        _check_stop(_lj38(), template)

    band = resolve_problem(ROOT, "double-ended-saddle-search/lj38_2ts_separation")
    shipped = load_template(ROOT, "eon/develop-de-2ts")
    assert shipped["ini"]["Optimizer"]["max_iterations"] == 2000
    assert _check_stop(band, shipped) == band["convergence"]["max_force_calls"]


def test_the_cap_comes_from_the_problem_first_then_the_template_in_a_fixed_order():
    assert _check_stop(_lj38(), {"id": "toy/t", "failure_cap": 5}) == 10000

    problem = {"id": "toy/open", "suite": "toy"}
    full = {"id": "toy/t", "failure_cap": 7, "stop": {"max_force_calls": 9, "max_steps": 8}, "ini": {"Optimizer": {"max_iterations": 11}}}
    assert _check_stop(problem, full) == 7
    del full["failure_cap"]
    assert _check_stop(problem, full) == 9
    del full["stop"]["max_force_calls"]
    assert _check_stop(problem, full) == 8
    del full["stop"]
    assert _check_stop(problem, full) == 11
    assert _check_stop(problem, {"id": "toy/t"}) is None


@pytest.mark.parametrize(
    ("token", "count"),
    [("120", 120), ("120.0", 120), ("0", 0), ("nan", None), ("inf", None), ("-3", None), ("12e", None), ("", None)],
)
def test_a_force_call_field_is_a_count_only_when_it_is_a_finite_non_negative_number(token, count):
    assert client_count(token) == count


def test_a_client_result_file_gives_the_last_good_count_and_the_reason_it_stopped():
    text = "\n".join(
        [
            "120 total_force_calls",
            "0 termination_reason",
            "7",
            "garbage line here",
            "130 total_force_calls",
            "nan total_force_calls",
        ]
    )
    assert parse_client_dat(text) == (130, "0")
    assert parse_client_dat("nan total_force_calls\n") == (None, None)
    assert parse_client_dat("3 termination_reason\n") == (None, "3")
    assert parse_client_dat("") == (None, None)


def test_a_run_directory_with_no_result_has_no_count_and_no_reason(tmp_path):
    assert client_result(tmp_path) == {"force_calls": None, "reason": None}
    (tmp_path / "results.dat").write_text("55 total_force_calls\n0 termination_reason\n", encoding="utf-8")
    assert client_result(tmp_path) == {"force_calls": 55, "reason": "0"}


def test_a_bad_ending_counts_at_the_cap_and_keeps_what_the_client_wrote():
    good = client_record({"force_calls": 40, "reason": "0"}, 100)
    assert (good["force_calls"], good["success"], good["failure"]) == (40.0, True, None)
    unfinished = client_record({"force_calls": 40, "reason": "3"}, 100)
    assert (unfinished["force_calls"], unfinished["success"], unfinished["failure"]) == (100.0, False, "cap")
    assert (unfinished["client_calls"], unfinished["reason"]) == (40, "3")
    spent = client_record({"force_calls": 100, "reason": "0"}, 100)
    assert (spent["success"], spent["failure"]) == (False, "cap")
    lost = client_record({"force_calls": None, "reason": None}, 100)
    assert (lost["force_calls"], lost["success"], lost["failure"], lost["client_calls"]) == (100.0, False, "crash", None)


def test_a_template_is_found_by_name_by_path_and_refused_without_an_id_and_an_engine(tmp_path):
    by_name = load_template(ROOT, "ase/lj38-bfgs-default")
    assert by_name["id"] == "ase/lj38-bfgs-default"
    assert by_name["_path"] == ROOT / "templates" / "ase" / "lj38-bfgs-default.toml"

    own = tmp_path / "own.toml"
    own.write_text('id = "toy/own"\nengine = "toy"\n', encoding="utf-8")
    assert load_template(ROOT, str(own))["_path"] == own

    with pytest.raises(FileNotFoundError, match="no template toy/missing"):
        load_template(tmp_path, "toy/missing")
    own.write_text('id = "toy/own"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="needs id and engine"):
        load_template(tmp_path, str(own))
    own.write_text('id = "toy/own"\nengine = "eon-client"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="needs ini"):
        load_template(tmp_path, str(own))


def test_a_problem_is_found_by_name_or_by_path_and_an_unknown_one_is_named():
    by_name = resolve_problem(ROOT, "minimization/lj38")
    by_path = resolve_problem(ROOT, str(ROOT / "problems" / "minimization" / "lj38.toml"))
    assert by_name["id"] == by_path["id"] == "minimization/lj38"
    with pytest.raises(FileNotFoundError, match="no problem minimization/nothing"):
        resolve_problem(ROOT, "minimization/nothing")


def _manifest(root: Path, rows: list[object]) -> None:
    path = root / "inputs" / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((json.dumps(row) if row else "") + "\n" for row in rows), encoding="utf-8")


def test_a_problem_archive_is_the_stored_file_the_manifest_lists_for_its_path(tmp_path):
    _manifest(
        tmp_path,
        [
            {"path": "website/toy/other.tgz", "stored": "inputs/sha256/aa"},
            None,
            {"path": "website/toy/clusters.tgz", "stored": "inputs/sha256/bb"},
        ],
    )
    problem = {"id": "toy/min", "input": {"archive_path": "website/toy/clusters.tgz"}}
    assert geometry_archive(tmp_path, problem) == tmp_path / "inputs" / "sha256" / "bb"
    with pytest.raises(FileNotFoundError, match="website/toy/gone.tgz is not in the geometry manifest"):
        geometry_archive(tmp_path, {"id": "toy/min", "input": {"archive_path": "website/toy/gone.tgz"}})
    with pytest.raises(ValueError, match="toy/min has no input.archive_path"):
        geometry_archive(tmp_path, {"id": "toy/min"})


def test_a_directory_of_geometries_is_read_sorted_and_a_lone_file_is_one_start(tmp_path):
    for name in ("b.con", "a.con", "c.xyz"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    assert [path.name for path in iter_geometries(tmp_path)] == ["a.con", "b.con"]
    assert iter_geometries(tmp_path / "b.con") == [tmp_path / "b.con"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no .con files in"):
        iter_geometries(empty)


def test_an_archive_is_not_a_directory_of_geometries(tmp_path):
    packed = tmp_path / "clusters.tgz"
    with tarfile.open(packed, "w:gz") as handle:
        info = tarfile.TarInfo("a.con")
        info.size = 2
        handle.addfile(info, io.BytesIO(b"x\n"))
    with pytest.raises(ValueError, match="pass a directory of .con files"):
        iter_geometries(packed)


def test_staged_inputs_are_named_by_file_unless_a_table_names_the_target(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "table.dat").write_text("1\n", encoding="utf-8")
    (tmp_path / "data" / "other.dat").write_text("2\n", encoding="utf-8")
    template = {"inputs": ["data/table.dat", {"source": "data/other.dat", "target": "sub/renamed.dat"}, {"source": "data/other.dat"}]}
    staged = _template_inputs(tmp_path, template)
    assert staged == (
        (str(tmp_path / "data" / "table.dat"), "table.dat"),
        (str(tmp_path / "data" / "other.dat"), str(Path("sub") / "renamed.dat")),
        (str(tmp_path / "data" / "other.dat"), "other.dat"),
    )
    assert _template_inputs(tmp_path, {}) == ()
    absolute = tmp_path / "data" / "table.dat"
    assert _template_inputs(tmp_path, {"inputs": [str(absolute)]}) == ((str(absolute), "table.dat"),)


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ("data/table.dat", ValueError, "template inputs must be a list"),
        ([{"target": "x.dat"}], ValueError, "template input entries need source"),
        ([7], ValueError, "template input entries must be strings or tables"),
        (["data/gone.dat"], FileNotFoundError, "template input data/gone.dat is not a file"),
        ([{"source": "data/table.dat", "target": "../escape.dat"}], ValueError, "must stay within the run directory"),
    ],
)
def test_a_staged_input_that_is_malformed_missing_or_leaves_the_run_directory_is_refused(tmp_path, inputs, error, message):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "table.dat").write_text("1\n", encoding="utf-8")
    with pytest.raises(error, match=message):
        _template_inputs(tmp_path, {"inputs": inputs})


def test_a_staged_input_may_not_be_written_to_an_absolute_target(tmp_path):
    (tmp_path / "table.dat").write_text("1\n", encoding="utf-8")
    template = {"inputs": [{"source": "table.dat", "target": str(tmp_path / "escape.dat")}]}
    with pytest.raises(ValueError, match="must stay within the run directory"):
        _template_inputs(tmp_path, template)


def test_ini_sections_come_in_the_client_order_with_the_rest_sorted_and_booleans_lower_case():
    text = format_ini(
        {
            "Zeta": {"flag": False},
            "Optimizer": {"converged_force": 0.01, "lbfgs_auto_scale": True},
            "Alpha": {"name": "x"},
            "Main": {"job": "minimization"},
            "Potential": {"potential": "ljcluster"},
        }
    )
    assert text == (
        "[Main]\njob = minimization\n\n"
        "[Potential]\npotential = ljcluster\n\n"
        "[Optimizer]\nconverged_force = 0.01\nlbfgs_auto_scale = true\n\n"
        "[Alpha]\nname = x\n\n"
        "[Zeta]\nflag = false\n"
    )


def test_the_benchmark_text_lists_the_scores_it_has_then_what_names_the_run():
    summary = {"force_calls": 70.5, "force_calls_min": 40, "nfailed": 2, "force_calls_median": None, "ignored": 1}
    template = {"id": "toy/t", "problem": "toy/p", "code": "toy", "code_version": "1.0", "commit": "abc", "algorithm": "LBFGS", "reducer": "capped", "kernel": "client"}
    assert benchmark_text(summary, template) == (
        "force_calls 70.5\nforce_calls_min 40\nnfailed 2\n"
        "code toy\ncode_version 1.0\ncommit abc\nalgorithm LBFGS\n"
        "template toy/t\nproblem toy/p\nreducer capped\nkernel client\n"
    )
    assert benchmark_text({}, {"id": "toy/t"}) == "template toy/t\n"


def _archive(root: Path, files: dict[str, str]) -> None:
    stored = root / "inputs" / "sha256" / "cc"
    stored.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(stored, "w:gz") as handle:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    _manifest(root, [{"path": "website/toy/clusters.tgz", "stored": "inputs/sha256/cc"}])


def test_a_client_job_is_cut_from_the_problem_archive_into_the_output_directory(tmp_path):
    _archive(tmp_path, {"clusters/b.con": "b\n", "a.con": "a\n", "notes.txt": "x\n"})
    problem = {"id": "toy/min", "input": {"archive_path": "website/toy/clusters.tgz"}, "convergence": {"max_force_calls": 50}}
    template = {"id": "eon-client/toy", "engine": "eon-client", "ini": {"Main": {"job": "minimization"}}}
    output = tmp_path / "out" / "bench.dat"
    jobs, cap, cons = client_jobs(tmp_path, problem, template, Path("client-binary"), geometries=None, output=output, work=None)
    unpacked = output.parent / "geometries"
    assert cap == 50
    assert cons == [unpacked / "a.con", unpacked / "clusters" / "b.con"]
    scratch = output.parent / "work" / "eon-client-toy"
    assert jobs == [
        ("client-binary", "[Main]\njob = minimization\n", str(unpacked / "a.con"), str(scratch / "a"), ()),
        ("client-binary", "[Main]\njob = minimization\n", str(unpacked / "clusters" / "b.con"), str(scratch / "b"), ()),
    ]


def test_a_directory_of_geometries_replaces_the_archive_and_a_given_work_directory_holds_the_runs(tmp_path):
    geometries = tmp_path / "given"
    geometries.mkdir()
    (geometries / "only.con").write_text("x\n", encoding="utf-8")
    problem = {"id": "toy/min", "convergence": {"max_force_calls": 50}}
    template = {"id": "eon-client/toy", "engine": "eon-client", "ini": {"Main": {"job": "minimization"}}}
    work = tmp_path / "scratch"
    jobs, _cap, cons = client_jobs(tmp_path, problem, template, Path("c"), geometries=geometries, output=tmp_path / "o.dat", work=work)
    assert cons == [geometries / "only.con"]
    assert jobs[0][3] == str(work / "only")


def test_a_client_job_needs_a_cap_and_an_archive_that_holds_geometries(tmp_path):
    template = {"id": "eon-client/toy", "engine": "eon-client", "ini": {"Main": {"job": "minimization"}}}
    with pytest.raises(ValueError, match="toy/open has no force-call cap"):
        client_jobs(tmp_path, {"id": "toy/open"}, template, Path("c"), geometries=None, output=tmp_path / "o.dat", work=None)

    _archive(tmp_path, {"notes.txt": "x\n"})
    problem = {"id": "toy/min", "input": {"archive_path": "website/toy/clusters.tgz"}, "convergence": {"max_force_calls": 50}}
    with pytest.raises(FileNotFoundError, match="has no .con files"):
        client_jobs(tmp_path, problem, template, Path("c"), geometries=None, output=tmp_path / "o.dat", work=None)


def test_a_negative_limit_is_refused_and_zero_keeps_every_start():
    assert require_limit(0) == 0
    assert require_limit(3) == 3
    with pytest.raises(ValueError, match="limit must be zero or positive"):
        require_limit(-1)


GUARDED = {
    "airss": "optbench.runners.airss_go",
    "ase": "optbench.ase_sella",
    "geodesic-ase": "optbench.ase_sella",
    "geometric": "optbench.runners.geometric_min",
    "gmin": "optbench.runners.gmin_min",
    "gmin-bh": "optbench.runners.gmin_go",
    "n2p2": "optbench.runners.n2p2_ml",
    "noise": "optbench.runners.noise_min",
    "ogolem": "optbench.runners.ogolem_go",
    "optking": "optbench.runners.optking_ts",
    "pele-basinhopping": "optbench.runners.pele_bh",
    "pele-lbfgs": "optbench.runners.pele_min",
    "pyamff": "optbench.runners.pyamff_ml",
    "rgmin": "optbench.runners.rgmin_min",
    "rgsaddle": "optbench.rgsaddle_minmode",
    "sella": "optbench.ase_sella",
    "softsaddle": "optbench.softsaddle_run",
    "vcsqnm": "optbench.runners.vcsqnm_min",
}


def _dispatch_key(template: dict) -> str:
    engine = template["engine"]
    if engine == "geodesic":
        return f"geodesic-{template.get('backend')}"
    if engine == "pele":
        method = str((template.get("pele") or {}).get("method") or "")
        return "pele-basinhopping" if method == "basinhopping" else "pele-lbfgs"
    return engine


def _template_for(key: str) -> dict:
    for template in sorted(list_templates(ROOT), key=lambda row: row["id"]):
        if template.get("problem") and _dispatch_key(template) == key:
            return template
    pytest.skip(f"no template with a problem runs on {key}")


def _other_problem(template: dict) -> str:
    return "minimization/morse-bulk" if template["problem"] != "minimization/morse-bulk" else "minimization/lj38"


@pytest.mark.parametrize("key", sorted(GUARDED))
def test_a_template_will_not_run_on_a_problem_it_does_not_belong_to(key, tmp_path):
    template = _template_for(key)
    with pytest.raises(ValueError) as caught:
        run(ROOT, _other_problem(template), template["id"], output=tmp_path / "out.dat")
    assert str(caught.value) == f"{template['id']} belongs to {template['problem']}"
    assert not (tmp_path / "out.dat").exists()


@pytest.mark.parametrize("key", sorted(GUARDED))
def test_the_dispatcher_calls_each_runner_with_what_the_runner_accepts(key, monkeypatch, tmp_path):
    template = _template_for(key)
    module = importlib.import_module(GUARDED[key])
    real = module.run_template
    seen = {}

    def recorder(*args, **kwargs):
        seen["bound"] = inspect.signature(real).bind(*args, **kwargs)
        return {"ran": key}

    monkeypatch.setattr(module, "run_template", recorder)
    output = tmp_path / "out.dat"
    result = run(
        ROOT,
        template["problem"],
        template["id"],
        output=output,
        workers=3,
        work=tmp_path / "work",
        limit=2,
        resume=True,
        geometries=tmp_path / "geometries",
    )
    assert result == {"ran": key}
    arguments = seen["bound"].arguments
    assert arguments["output"] == output
    assert arguments["workers"] == 3
    assert arguments["limit"] == 2


def test_a_client_template_needs_a_client_and_a_written_barrier_rate_is_not_run_by_itself(tmp_path):
    with pytest.raises(ValueError, match="eon-client run needs a client"):
        run(ROOT, "minimization/lj38", "eon-client/lbfgs-m20", output=tmp_path / "out.dat")
    rate = next(
        template
        for template in list_templates(ROOT)
        if template["engine"] == "eon" and template.get("task") == "htst-written"
    )
    with pytest.raises(RuntimeError, match=f"{rate['id']} is written by {rate['paired']}"):
        run(ROOT, rate["problem"], rate["id"], output=tmp_path / "out.dat")


def test_an_engine_nothing_runs_is_named(tmp_path):
    templates = tmp_path / "templates" / "toy"
    templates.mkdir(parents=True)
    (templates / "odd.toml").write_text('id = "toy/odd"\nengine = "nonesuch"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown engine nonesuch"):
        run(tmp_path, "minimization/lj38", "toy/odd", output=tmp_path / "out.dat")
