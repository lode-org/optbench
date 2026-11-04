"""Bridge and grid checks for the r2549 harness."""

import os
from pathlib import Path

from optbench.extpot import format_extpot
from optbench.potserver import PotServer

ROOT = Path(__file__).resolve().parents[1]


def test_c_bridge_matches_the_python_reply(tmp_path, monkeypatch):
    from optbench.eon_harness import bridge_binary

    def evaluate(positions, numbers, box):
        del numbers, box
        energy = float(sum(positions))
        forces = tuple(-float(value) for value in positions)
        return energy, forces

    server = PotServer(evaluate).start()
    monkeypatch.setenv("OPTBENCH_POTSERVER", server.endpoint())
    monkeypatch.chdir(tmp_path)
    request = "\n".join(
        [
            "10 0 0",
            "0 10 0",
            "0 0 10",
            "1 0.0 0.0 0.0",
            "1 1.5 0.0 0.0",
        ]
    )
    (tmp_path / "from_eon_to_extpot").write_text(request + "\n", encoding="utf-8")
    import subprocess

    proc = subprocess.run([str(bridge_binary())], cwd=tmp_path, check=False, text=True, capture_output=True)
    server.stop()
    assert proc.returncode == 0, proc.stderr
    reply = (tmp_path / "from_extpot_to_eon").read_text(encoding="utf-8")
    expected = format_extpot(1.5, (0.0, 0.0, 0.0, -1.5, 0.0, 0.0))
    got = [[float(part) for part in line.split()] for line in reply.splitlines() if line.strip()]
    want = [[float(part) for part in line.split()] for line in expected.splitlines() if line.strip()]
    assert got == want


def test_svn_default_grid_is_one_start():
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    from benchmarks.bench_eon_lane import EonSvnTsLj38DimerLjcut

    assert EonSvnTsLj38DimerLjcut.params[0] == [1]
    assert EonSvnTsLj38DimerLjcut.params[1] == ["0000"]
    assert EonSvnTsLj38DimerLjcut.template_name == "eon-svn/ts-lj38-dimer-ljcut"


def test_f6817b53_classes_are_the_harness_templates_of_that_build():
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    from benchmarks import bench_eon_f6817b53
    from optbench.eon_harness import _jobs

    classes = {
        value.template_name: value
        for value in vars(bench_eon_f6817b53).values()
        if isinstance(value, type) and getattr(value, "template_name", None)
    }
    assert sorted(classes) == sorted(template["id"] for template in _jobs("f6817b53"))
    assert "eon/develop-f6817b53-de-lj38-cap" in classes
    for benchmark in classes.values():
        assert len(benchmark.params[0]) == 1 and len(benchmark.params[1]) == 1


def test_500350d9_templates_run_on_the_500350d9_build():
    from benchmarks import bench_eon_500350d9, bench_eon_client
    from optbench.campaign import load_template
    from optbench.eon_harness import _family_of

    def names(module):
        return {
            value.template_name
            for value in vars(module).values()
            if isinstance(value, type) and getattr(value, "template_name", None)
        }

    built = names(bench_eon_500350d9)
    assert {
        "eon/500350d9-neb-pt-cg",
        "eon/500350d9-de-lj38-ci-swdneb",
        "eon/develop-500350d9-de-2ts",
        "eon-client/de-2ts-500350d9-dimer",
    } <= built
    assert not built & names(bench_eon_client)
    for template in built:
        assert load_template(ROOT, template)["commit"].startswith("500350d9")
        assert _family_of(load_template(ROOT, template)) == "500350d9"
    for template in names(bench_eon_client):
        assert load_template(ROOT, template)["commit"].startswith("4d8b34ba")


def test_client_default_grid_is_one_start():
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    from benchmarks.bench_eon_client import EonClientNebPtLbfgs

    assert len(EonClientNebPtLbfgs.params[1]) == 1
    assert EonClientNebPtLbfgs.template_name == "eon-client/neb-pt-lbfgs"
