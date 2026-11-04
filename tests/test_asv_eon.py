import importlib
import json
import math
from pathlib import Path

import pytest

from optbench import asv_eon
from optbench.campaign import load_template

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"eon-svn/": "eon_svn_counter", "eon-client/": "eon_client_counter", "eon/": "eon_main"}
TRACKS = (
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)


def _cells(prefix: str) -> list[dict]:
    from benchmarks.bench_lj38 import TEMPLATES

    skip = frozenset(template for _name, template, _limit in TEMPLATES) if prefix == "eon-client/" else frozenset()
    return asv_eon.cells(prefix, skip)


def _classes() -> dict:
    found = {}
    for module in MODULES.values():
        for value in vars(importlib.import_module(f"benchmarks.bench_{module}")).values():
            if isinstance(value, type) and issubclass(value, asv_eon.ClientSearch) and value is not asv_eon.ClientSearch:
                found[value.template_id] = value
    return found


def _campaign_starts() -> dict[str, list[str]]:
    found = {}
    for dat in sorted((ROOT / "results" / "campaign").glob("*.dat")):
        lines = dat.read_text(encoding="utf-8").splitlines()
        template = next((line.split(" ", 1)[1] for line in lines if line.startswith("template ")), None)
        starts = dat.with_suffix(".starts")
        if template and starts.is_file():
            rows = [json.loads(line) for line in starts.read_text(encoding="utf-8").splitlines() if line.strip()]
            found[template] = [Path(row["geometry"]).name.removesuffix(".con") for row in rows]
    return found


def test_every_client_counted_board_row_is_a_class():
    from benchmarks.bench_lj38 import TEMPLATES

    board = json.loads((ROOT / "results" / "board" / "index.json").read_text(encoding="utf-8"))
    rows = []

    def visit(value):
        if isinstance(value, dict):
            if "source" in value and "template" in value:
                rows.append(value)
                return
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(board)
    have = {(benchmark.template_id, benchmark.problem_id) for benchmark in _classes().values()}
    lj38 = {template for _name, template, _limit in TEMPLATES}
    cells = {(cell["template_id"], cell["problem_id"]) for prefix in MODULES for cell in _cells(prefix)}
    wanted = set()
    for row in rows:
        if "campaign" not in row["source"] or row["asv_commit"] or not row["template"]:
            continue
        if (row["template"], row["problem"]) in cells:
            wanted.add((row["template"], row["problem"]))
    assert wanted <= have
    assert {
        ("eon-svn/cg", "minimization/lj38"),
        ("eon-svn/morse-sd", "minimization/morse-bulk"),
        ("eon-client/morse-lbfgs", "minimization/morse-bulk"),
        ("eon/lj38-lbfgs-m100", "minimization/lj38"),
        ("eon/de-pt-lbfgs-m100", "double-ended-saddle-search/pt-island"),
        ("eon/ts-lj38-dimer", "tsopt/LJ38"),
        ("eon/aladatom-akmc", "accelerated-md/aladatom"),
        ("eon/standard-aladatom-akmc", "accelerated-md/standard_aladatom"),
    } <= wanted
    assert not {template for template, _problem in have} & lj38


def test_classes_have_the_campaign_shape():
    classes = _classes()
    assert classes
    for prefix in MODULES:
        for cell in _cells(prefix):
            benchmark = classes[cell["template_id"]]
            assert benchmark.__name__.isidentifier()
            assert benchmark.param_names == ["seed", "start"]
            assert benchmark.params == [["0"], asv_eon.start_names(cell)[:1]]
            assert benchmark.timeout == math.inf
            assert benchmark.cap == asv_eon.cap_of(cell)
            template = load_template(ROOT, cell["template_id"])
            assert asv_eon.client_of(template) is not None
            assert template.get("problem") in (None, cell["problem_id"])
            for name in TRACKS:
                assert getattr(benchmark, name).unit


def test_starts_are_the_committed_campaign_starts():
    committed = _campaign_starts()
    checked = 0
    for prefix in MODULES:
        for cell in _cells(prefix):
            if cell["template_id"] in committed:
                assert asv_eon.start_names(cell) == committed[cell["template_id"]]
                checked += 1
    assert checked >= 8


def test_full_grid_is_every_campaign_start(monkeypatch):
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    for prefix in MODULES:
        for cell in _cells(prefix):
            assert asv_eon.bind(cell).params[1] == asv_eon.start_names(cell)


def test_main_prefix_is_the_main_commit_only():
    templates = {cell["template_id"] for cell in asv_eon.cells("eon/")}
    assert not any(template.startswith(("eon/develop-", "eon/500350d9-")) for template in templates)
    for template in templates:
        assert load_template(ROOT, template)["commit"] == asv_eon.clients()["eon/"]["commit"]


def _stub(monkeypatch, tmp_path: Path, calls: list):
    from optbench import eon_harness

    monkeypatch.setenv("OPTBENCH_ASV_CACHE", str(tmp_path))
    monkeypatch.setattr(eon_harness, "client_file", lambda spec: "/stub/eonclient")
    monkeypatch.setattr(eon_harness, "check_client", lambda spec, label: "/stub/eonclient")

    def search(cell, client, start, work):
        calls.append((cell["template_id"], start))
        return {"force_calls": 23.0, "success": False, "failure": "cap"}

    monkeypatch.setattr(asv_eon, "search", search)


@pytest.mark.parametrize("prefix", sorted(MODULES))
def test_every_track_has_a_value_in_a_fresh_process(monkeypatch, tmp_path, prefix):
    calls: list = []
    _stub(monkeypatch, tmp_path, calls)
    cell = asv_eon.cells(prefix)[0]
    benchmark = asv_eon.bind(cell)
    start = benchmark.params[1][0]
    expected = {
        "track_force_calls": 23.0,
        "track_success": 0.0,
        "track_failure_cap": 1.0,
        "track_failure_wrong_stationary": 0.0,
        "track_failure_divergence": 0.0,
        "track_failure_crash": 0.0,
    }
    for name in TRACKS:
        fresh = benchmark()
        fresh.setup("0", start)
        value = getattr(fresh, name)("0", start)
        assert isinstance(value, float) and not math.isnan(value)
        assert value == expected[name]
    assert calls == [(cell["template_id"], start)]
    timed = benchmark()
    timed.setup("0", start)
    timed.time_search("0", start)
    assert len(calls) == 2
    for name in TRACKS:
        fresh = benchmark()
        fresh.setup("0", start)
        assert getattr(fresh, name)("0", start) == expected[name]
    assert len(calls) == 2


def test_a_catalogue_start_is_the_campaign_repetition(monkeypatch, tmp_path):
    from optbench import eon_amd

    cell = next(cell for cell in asv_eon.cells("eon/") if cell["template_id"] == "eon/aladatom-akmc")
    assert cell["path"] == "amd"
    assert asv_eon.start_names(cell) == ["rep-1"]
    seen = {}

    def repetition(root, planned, client, work, workers, repetition, *, on_pass=None):
        seen.update(workers=workers, repetition=repetition, passes=planned["passes"])
        return {
            "repetition": repetition,
            "seed": 20261018 + repetition - 1,
            "force_calls": 108435,
            "reached": True,
            "rate": 855329109.89,
            "confidence": 0.9523809523809523,
            "repeats": 21,
            "total_searches": 128,
        }

    monkeypatch.setattr(eon_amd, "run_repetition", repetition)
    record = asv_eon.search(cell, "/stub/eonclient", "rep-1", tmp_path)
    committed = json.loads(
        (ROOT / "results" / "campaign" / "eon-3.6.0-aladatom-akmc.starts.jsonl").read_text(encoding="utf-8")
    )
    template = load_template(ROOT, "eon/aladatom-akmc")
    assert seen == {
        "workers": template["ini"]["Communicator"]["num_jobs"],
        "repetition": 1,
        "passes": template["pass_limit"],
    }
    for key in ("start", "geometry", "force_calls", "success", "failure", "seed", "counter", "template", "problem"):
        assert record[key] == committed[key]
    with pytest.raises(KeyError):
        asv_eon.search(cell, "/stub/eonclient", "rep-2", tmp_path)


def test_a_missing_client_is_a_skip(monkeypatch):
    cell = asv_eon.cells("eon/")[0]
    monkeypatch.setenv("EON_MAIN_CLIENT", "/nonexistent/eonclient")
    monkeypatch.setattr(asv_eon.Path, "home", classmethod(lambda cls: Path("/nonexistent")))
    with pytest.raises(NotImplementedError, match="EON_MAIN_CLIENT"):
        asv_eon.bind(cell)().setup("0", "x")
