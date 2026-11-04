import importlib
import json
import math
from pathlib import Path

import pytest

from benchmarks import pending
from optbench import asv_engines
from optbench.campaign import load_template

ROOT = Path(__file__).resolve().parents[1]
TRACKS = (
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)


def _classes(engine: str) -> dict:
    module = importlib.import_module(f"benchmarks.bench_{engine}")
    return {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, type) and issubclass(value, asv_engines.EngineSearch)
    }


def test_every_engine_cell_is_a_class_with_the_campaign_shape():
    for engine in pending.ENGINE_CLASSES:
        classes = _classes(engine)
        cells = asv_engines.cells(ROOT, engine)
        assert sorted(classes) == sorted(cell["class_name"] for cell in cells)
        for cell in cells:
            assert cell["class_name"].isidentifier()
            benchmark = classes[cell["class_name"]]
            assert benchmark.param_names == ["seed", "start"]
            assert benchmark.params == [["0"], asv_engines.start_names(ROOT, cell["problem_id"])[:1]]
            assert benchmark.template_id == cell["template_id"]
            assert benchmark.problem_id == cell["problem_id"]
            assert benchmark.timeout == math.inf
            template = load_template(ROOT, cell["template_id"])
            assert template.get("problem") in (None, cell["problem_id"])
            assert benchmark.cap == asv_engines.problem_cap(ROOT, cell["problem_id"])
            for name in TRACKS:
                assert getattr(benchmark, name).unit


def test_full_grid_is_every_campaign_start(monkeypatch):
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    for cell in asv_engines.cells(ROOT):
        benchmark = asv_engines.bind(cell)
        assert benchmark.params[1] == asv_engines.start_names(ROOT, cell["problem_id"])


def test_every_campaign_only_board_row_of_an_engine_is_a_class():
    board = json.loads((ROOT / "results" / "board" / "index.json").read_text(encoding="utf-8"))
    rows = [
        row
        for problem in board["problems"].values()
        for key in ("ranked", "short", "withheld")
        for row in problem[key]
    ]
    engines = {cell["engine"] for cell in asv_engines.cells(ROOT)}
    have = {(cell["template_id"], cell["problem_id"]) for cell in asv_engines.cells(ROOT)}
    wanted = set()
    for row in rows:
        if "campaign" not in row["source"] or row["asv_commit"] or not row["template"]:
            continue
        try:
            template = load_template(ROOT, row["template"])
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if template.get("engine") in engines and row["problem"] != asv_engines.WALK_PROBLEM:
            wanted.add((row["template"], row["problem"]))
    assert wanted
    assert wanted <= have


def test_walk_and_refusing_templates_have_no_cell():
    found = {(cell["template_id"], cell["problem_id"]) for cell in asv_engines.cells(ROOT)}
    assert not any(problem == asv_engines.WALK_PROBLEM for _template, problem in found)
    assert not any(template in {"optim/unhooked", "pykmc/lattice"} for template, _problem in found)


def _stub(monkeypatch, tmp_path: Path, calls: list) -> dict:
    from optbench import engines
    from optbench.engines import common

    monkeypatch.setenv("OPTBENCH_ASV_CACHE", str(tmp_path))
    cell = asv_engines.cells(ROOT, "lammps")[0]
    record = {"start": "x", "force_calls": 17, "success": False, "failure": "divergence"}

    class Stub:
        @staticmethod
        def available():
            return None

        @staticmethod
        def plan(root, problem_name, template_name, work):
            return {"jobs": [{"name": "cluster_0000"}], "execute": None}

    def run_start(jobs, execute, start):
        calls.append(start)
        return dict(record, start=start)

    monkeypatch.setitem(engines.ENGINES, cell["engine"], Stub)
    monkeypatch.setattr(common, "run_start", run_start)
    return cell


def test_every_track_has_a_value_in_a_fresh_process(monkeypatch, tmp_path):
    calls: list = []
    cell = _stub(monkeypatch, tmp_path, calls)
    benchmark = asv_engines.bind(cell)
    expected = {
        "track_force_calls": 17.0,
        "track_success": 0.0,
        "track_failure_cap": 0.0,
        "track_failure_wrong_stationary": 0.0,
        "track_failure_divergence": 1.0,
        "track_failure_crash": 0.0,
    }
    for name in TRACKS:
        fresh = benchmark()
        fresh.setup("0", "cluster_0000")
        value = getattr(fresh, name)("0", "cluster_0000")
        assert isinstance(value, float) and not math.isnan(value)
        assert value == expected[name]
    assert calls == ["cluster_0000"]
    timed = benchmark()
    timed.setup("0", "cluster_0000")
    timed.time_search("0", "cluster_0000")
    assert calls == ["cluster_0000", "cluster_0000"]
    for name in TRACKS:
        fresh = benchmark()
        fresh.setup("0", "cluster_0000")
        assert getattr(fresh, name)("0", "cluster_0000") == expected[name]
    assert len(calls) == 2


def test_a_missing_code_is_a_skip(monkeypatch, tmp_path):
    from optbench import engines

    cell = asv_engines.cells(ROOT, "lammps")[0]

    class Missing:
        @staticmethod
        def available():
            return "not built"

    monkeypatch.setitem(engines.ENGINES, cell["engine"], Missing)
    with pytest.raises(NotImplementedError, match="not built"):
        asv_engines.bind(cell)().setup("0", "cluster_0000")
