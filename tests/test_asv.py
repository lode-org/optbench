import json
import os
import re
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
    comment_workflow = (ROOT / ".github" / "workflows" / "asv-comment.yml").read_text(
        encoding="utf-8"
    )
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    uses = re.findall(r"uses: (\S+)", workflow)
    assert uses == [
        "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
        "astral-sh/setup-uv@d0cc045d04ccac9d8b7881df0226f9e82c39688e",
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
    ]
    comment_uses = re.findall(r"uses: (\S+)", comment_workflow)
    assert comment_uses == [
        "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
    ]
    assert all(
        re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref)
        for ref in uses + comment_uses
    )
    assert 'version: "0.13.1"' in workflow
    assert 'python-version: "3.12.3"' in workflow
    assert "uv python install 3.12.3" in workflow
    assert "if: always() && github.event_name == 'pull_request'" in workflow
    assert "pull-requests: write" not in workflow
    assert "permissions:\n  contents: read\n" in workflow
    assert "workflow_run:" in comment_workflow
    assert (
        "permissions:\n  actions: read\n  contents: read\n  pull-requests: write\n"
        in comment_workflow
    )
    assert "metadata-file" not in comment_workflow
    assert "python3 ci/asv_comment.py" in comment_workflow
    assert "workflow_run.head_sha" not in comment_workflow
    assert "pull_requests[0]" not in comment_workflow
    assert "github.event.pull_request." not in comment_workflow
    assert "if: github.event.workflow_run.event == 'pull_request'" in comment_workflow
    assert workflow.count("if: always()") >= 2
    ci_conf = json.loads((ROOT / "asv.ci.conf.json").read_text(encoding="utf-8"))
    memray_conf = json.loads((ROOT / "asv.memray.conf.json").read_text(encoding="utf-8"))
    assert ci_conf["pythons"] == ["3.12.3"]
    assert memray_conf["pythons"] == ["3.12.3"]
    assert "snakemake --cores 1 asv_compare" in workflow
    assert "path: results/asv-ci/compare.txt" in workflow
    assert "--quick" not in workflow
    assert "--skip-existing" not in workflow
    assert "factor 10" not in workflow
    assert "--interleave-rounds" in snakefile
    assert "--quick" not in snakefile
    assert "--skip-existing" not in snakefile
    assert "asv.ci.conf.json" in snakefile
    rules = {}
    matches = list(re.finditer(r"^rule (\w+):", snakefile, re.M))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(snakefile)
        rules[match.group(1)] = snakefile[match.start() : end]
    measure = rules["asv_measure"]
    compare = rules["asv_compare"]
    assert 'report="results/asv-ci/compare.txt"' in measure
    assert 'decision="results/asv-ci/gate.json"' in measure
    assert "output:" not in compare
    assert "compare.txt" not in compare
    assert "rules.asv_measure.output.report" in compare
    assert "rules.asv_measure.output.decision" in compare
    assert "{input.decision}" in compare
    assert 'HASHFILE:results/asv-ci/commits.txt' in measure
    assert "--record-samples" in measure
    assert "asv continuous" not in snakefile
    assert "test $? -le 1" not in snakefile
    assert 'test "$status" -eq 1' in measure
    assert "test -s {output.report}" in measure
    assert "test -s {output.decision}" in measure
    assert 'results_dir": "results/asv-ci/data"' in (ROOT / "asv.ci.conf.json").read_text(
        encoding="utf-8"
    )


def _fenced_blocks(text: str) -> list[list[str]]:
    blocks = []
    block = None
    for line in text.splitlines():
        if line.startswith("```"):
            if block is None:
                block = []
            else:
                blocks.append(block)
                block = None
        elif block is not None:
            block.append(line.strip())
    return blocks


def test_asv_commands_name_the_committed_machine():
    records = sorted((ROOT / "results" / "asv").glob("*/machine.json"))
    assert len(records) == 1, records
    name = records[0].parent.name
    assert json.loads(records[0].read_text(encoding="utf-8"))["machine"] == name
    launch = "uv run python -m optbench.asv_launch"
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    checked = 0
    for path in documents:
        for block in _fenced_blocks(path.read_text(encoding="utf-8")):
            machine = [
                line
                for line in block
                if line.startswith("uv run asv machine") or line.startswith(f"{launch} machine")
            ]
            runs = [
                line
                for line in block
                if line.startswith("uv run asv run") or line.startswith(f"{launch} run")
            ]
            if machine:
                assert machine == [f"{launch} machine --yes --machine {name}"], path
            for line in runs:
                tokens = line.split()
                assert "--machine" in tokens, (path, line)
                assert tokens[tokens.index("--machine") + 1] == name, (path, line)
            checked += len(machine) + len(runs)
    assert checked >= 6
    steps = (ROOT / "docs" / "asv.md").read_text(encoding="utf-8")
    assert f"7. `{launch} machine --yes --machine {name}` once on the machine." in steps
    assert f"The committed machine is `{name}`." in steps
    assert f"8. `{launch} run --machine {name} <commit>^! -b YourClass`." in steps


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
    from benchmarks import pending

    assert len(problem_ids()) >= 25
    assert code_ids() == [
        "eon",
        "eon-main",
        "eon-develop",
        "eon-develop-500350d9",
        "eon-develop-6d9d629a",
        "eon-develop-f6817b53",
        "eon-svn",
        "rgmin",
        "rgsaddle",
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
        "gpridge",
    ]
    covered = pending.covered()
    grid = set()
    classes = [value for name, value in vars(pending).items() if name.startswith("Pending")]
    assert classes
    for benchmark in classes:
        assert benchmark.__name__.isidentifier()
        assert benchmark.param_names == ["problem", "code", "seed", "start"]
        problems, codes, seeds, starts = benchmark.params
        assert len(codes) == 1 and seeds == ["0"] and starts == ["0"]
        assert benchmark.track_force_calls.unit == "force calls"
        cells = {(problem, codes[0]) for problem in problems}
        assert not cells & covered
        assert not cells & grid
        grid |= cells
    assert grid | covered == {(problem, code) for problem in problem_ids() for code in code_ids()}


def test_template_modules_name_the_registry_code_of_their_templates():
    import importlib
    import tomllib

    from benchmarks import pending
    from optbench.campaign import load_template

    names = {
        row["id"]: row["name"]
        for row in tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))["code"]
    }
    for code, module in pending.TEMPLATE_MODULES:
        templates = [
            getattr(value, "template_id", None) or getattr(value, "template_name", None)
            for value in vars(importlib.import_module(f"benchmarks.bench_{module}")).values()
            if isinstance(value, type)
        ]
        templates = [template for template in templates if template]
        assert templates
        for template in templates:
            assert load_template(ROOT, template)["code"] == names[code]
        assert {(problem, code) for problem in pending.template_problems(module)} <= pending.covered()


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
