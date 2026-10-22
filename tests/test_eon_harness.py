"""Bridge and grid checks for the r2549 harness."""

import os
from pathlib import Path

from optbench.extpot import format_extpot
from optbench.potserver import PotServer


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


def test_client_default_grid_is_one_start():
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    from benchmarks.bench_eon_client import EonClientNebPtLbfgs

    assert len(EonClientNebPtLbfgs.params[1]) == 1
    assert EonClientNebPtLbfgs.template_name == "eon-client/neb-pt-lbfgs"


def test_client_htst_pt_uses_the_problem_force_call_cap():
    from benchmarks.bench_eon_client import EonClientHtstPtDimerLbfgs, EonClientHtstPtLanczosLbfgs

    assert EonClientHtstPtDimerLbfgs.cap == 20000000
    assert EonClientHtstPtLanczosLbfgs.cap == 20000000
