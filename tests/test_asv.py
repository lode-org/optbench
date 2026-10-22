import json
import os
import stat
from pathlib import Path

import pytest

from benchmarks import bench_rgsaddle
from benchmarks.bench_lj38 import Lj38EonLbfgsM100, selected_starts, start_names
from benchmarks.bench_rgsaddle import RgsaddleLj38DimerLbfgs, RgsaddlePtDimerLbfgs
from benchmarks.pending import code_ids, problem_ids
from benchmarks.quadratic import QuadraticDisplacement, QuadraticDimer
from optbench.asv_score import statistics_from_asv
from optbench.asv_uv import Uv
from optbench.campaign import run_client
from optbench.samples import sample_stats
from optbench.statistics import DISPLACEMENT_SAMPLES, SEED

ROOT = Path(__file__).resolve().parents[1]


def test_asv_uses_the_uv_environment():
    conf = json.loads((ROOT / "asv.conf.json").read_text(encoding="utf-8"))
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert conf["environment_type"] == "uv-pip"
    assert conf["plugins"] == [".optbench.asv_uv"]
    assert conf["pythons"] == ["3.12"]
    assert conf["results_dir"] == "results/asv"
    assert "uv pip install" in conf["install_command"][0]
    assert "asv==0.6.6" in text
    assert "asv-runner==0.3.1" in text
    assert "uv==0.13.1" in (ROOT / "optbench" / "asv_uv.py").read_text(encoding="utf-8")
    assert "env/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert Uv.tool_name == "uv-pip"
    assert Uv.matches("3.12") is True
    assert Uv.matches("not-a-version") is False
    workflow = (ROOT / ".github" / "workflows" / "asv.yml").read_text(encoding="utf-8")
    assert "HaoZeke/asv-perch@v1" in workflow
    assert "results/asv" in workflow
    assert 'regression-threshold: "10"' in workflow
    assert "asv-spyglass-args: --factor 10" in workflow


def test_rgsaddle_grid_is_seed_then_one_start_until_full(monkeypatch):
    classes = [
        value
        for value in vars(bench_rgsaddle).values()
        if isinstance(value, type) and getattr(value, "template_id", "").startswith("rgsaddle/")
    ]
    assert len(classes) == 22
    for benchmark in classes:
        assert benchmark.param_names == ["seed", "start"]
        assert benchmark.params[0] == ["0"]
        assert len(benchmark.params[1]) == 1
        assert benchmark.repeat == 1
        assert benchmark.rounds == 1
        assert benchmark.number == 1
        assert benchmark.sample_time == 0
        assert benchmark.warmup_time == 0
        assert benchmark.cap >= 1000
        assert benchmark.track_force_calls.unit == "force calls"
        assert benchmark.track_success.unit == "fraction"
        assert benchmark.track_failure_cap.unit == "count"
        assert benchmark.track_failure_wrong_stationary.unit == "count"
        assert benchmark.track_failure_divergence.unit == "count"
        assert benchmark.track_failure_crash.unit == "count"
        assert issubclass(benchmark, bench_rgsaddle._TrackedSearch)
    assert RgsaddleLj38DimerLbfgs.params[1] == ["0000"]
    assert RgsaddleLj38DimerLbfgs.cap == 10000
    assert RgsaddlePtDimerLbfgs.params[1] == ["initial_0"]
    assert len(bench_rgsaddle.lj38_start_names()) == 200
    assert len(bench_rgsaddle.pt_start_names()) == 49
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert bench_rgsaddle.selected_lj38() == bench_rgsaddle.lj38_start_names()
    assert bench_rgsaddle.selected_pt() == bench_rgsaddle.pt_start_names()
    monkeypatch.delenv("RGSADDLE_LIB", raising=False)
    with pytest.raises(RuntimeError):
        RgsaddleLj38DimerLbfgs().setup("0", "0000")


def test_each_lj38_start_is_one_asv_parameter(monkeypatch):
    names = start_names()
    assert len(names) == 1000
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    assert Lj38EonLbfgsM100.params[0] == ["0"]
    assert Lj38EonLbfgsM100.params[1] == [names[0]]
    assert Lj38EonLbfgsM100.param_names == ["seed", "start"]
    assert Lj38EonLbfgsM100.repeat == 1
    assert Lj38EonLbfgsM100.track_force_calls.unit == "force calls"
    assert selected_starts() == names[:1]
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert selected_starts() == names


def test_quadratic_grid_uses_the_sample_seeds():
    assert QuadraticDisplacement.param_names == ["seed", "start"]
    assert QuadraticDisplacement.params[0] == [str(SEED + index) for index in range(DISPLACEMENT_SAMPLES)]
    assert QuadraticDisplacement.params[1] == ["0"]
    assert QuadraticDisplacement.track_success.unit == "fraction"
    assert QuadraticDisplacement.track_failure_cap.unit == "count"
    record = QuadraticDisplacement().measure(str(SEED), "0")
    assert record["force_calls"] == 139
    assert record["success"] is True
    dimer = QuadraticDimer().measure(str(SEED), "0")
    assert dimer["success"] is True
    assert dimer["hessian_index"] == 1


def test_n2p2_benchmarks_select_one_start():
    from benchmarks.bench_n2p2 import N2p2C2h4Kalman, N2p2EmtPt38Kalman

    assert N2p2EmtPt38Kalman.params == [["20261015"], ["0"]]
    assert N2p2EmtPt38Kalman.template_id == "n2p2/emt-pt38-kalman"
    assert N2p2EmtPt38Kalman.problem_id == "mlopt/emt-pt38"
    assert N2p2EmtPt38Kalman.cap == 10000
    assert N2p2C2h4Kalman.params == [["20261015"], ["0"]]
    assert N2p2C2h4Kalman.template_id == "n2p2/c2h4-kalman"
    assert N2p2C2h4Kalman.problem_id == "mlopt/C2H4"
    assert N2p2C2h4Kalman.cap == 10000
    assert N2p2EmtPt38Kalman.spec["order"] == "n2p2"
    assert N2p2C2h4Kalman.spec["order"] == "n2p2"
    from benchmarks.bench_n2p2 import N2p2C2h4KalmanForce, N2p2EmtPt38KalmanForce

    assert N2p2EmtPt38KalmanForce.spec["order"] == "n2p2"
    assert N2p2EmtPt38KalmanForce.spec["loss"] == "force"
    assert N2p2EmtPt38KalmanForce.cap == 30000
    assert N2p2C2h4KalmanForce.spec["order"] == "n2p2"
    assert N2p2C2h4KalmanForce.cap == 30000
    assert not hasattr(__import__("benchmarks.bench_n2p2", fromlist=["x"]), "N2p2EmtPt38KalmanRng")


def test_optking_grid_is_seed_then_the_first_start():
    from benchmarks import bench_optking

    expected = {
        "OptkingLj38RsIrfo": ("optking/lj38-rs-irfo", "tsopt/LJ38", "0000", 10000),
        "OptkingLj38PRfo": ("optking/lj38-p-rfo", "tsopt/LJ38", "0000", 10000),
        "OptkingPtRsIrfo": ("optking/pt-rs-irfo", "tsopt/pt-island", "initial_0", 10000),
        "OptkingPtPRfo": ("optking/pt-p-rfo", "tsopt/pt-island", "initial_0", 10000),
    }
    for name, (template_id, problem_id, start, cap) in expected.items():
        benchmark = getattr(bench_optking, name)
        assert benchmark.param_names == ["seed", "start"]
        assert benchmark.params == [["0"], [start]]
        assert benchmark.template_id == template_id
        assert benchmark.problem_id == problem_id
        assert benchmark.cap == cap
        assert benchmark.timeout == 14400
        assert benchmark.repeat == 1
        assert benchmark.track_force_calls.unit == "force calls"


def test_pending_grid_is_every_problem_and_code():
    from benchmarks.pending import Pending

    assert Pending.param_names == ["problem", "code", "seed", "start"]
    assert len(Pending.params[0]) == len(problem_ids())
    assert len(problem_ids()) >= 25
    assert code_ids() == [
        "eon",
        "eon-main",
        "eon-develop",
        "eon-develop-500350d9",
        "eon-develop-f6817b53",
        "eon-svn",
        "rgmin",
        "rgsaddle",
        "rgsaddle-main",
        "geodesic",
        "ase",
        "sella",
        "gmin",
        "gmin-bh",
        "pele",
        "optking",
        "artn",
        "tsase",
        "pykmc",
        "lammps",
        "dlfind",
        "compass",
        "pysisyphus",
        "optim",
        "n2p2",
        "airss",
        "vcsqnm",
        "softsaddle",
        "lammpsneb",
        "sella-0.1.1",
    ]
    assert Pending.params[1] == code_ids()
    assert Pending.track_force_calls.unit == "force calls"


def test_statistics_refuses_to_invent_a_document(tmp_path: Path):
    with pytest.raises(SystemExit):
        statistics_from_asv(ROOT, tmp_path)


def test_one_client_subprocess_returns_its_force_calls(tmp_path: Path):
    con = tmp_path / "start.con"
    con.write_text("pos\n", encoding="utf-8")
    client = tmp_path / "eonclient"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "assert Path('pos.con').is_file()\n"
        "assert 'lbfgs_memory' in Path('config.ini').read_text()\n"
        "Path('results.dat').write_text('0 termination_reason\\n181 total_force_calls\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    calls = run_client(str(client), "lbfgs_memory = 100\n", str(con), str(tmp_path / "work"))
    assert calls == 181


def test_sample_stats_are_the_distribution_not_one_rounded_mean():
    stats = sample_stats([10, 20, 30, 40])
    assert stats["n"] == 4
    assert stats["mean"] == 25
    assert stats["min"] == 10
    assert stats["max"] == 40
    assert stats["q25"] == 17.5
    assert stats["q75"] == 32.5
    assert stats["stdev"] > 0
