"""BLAS pools inside a benchmark that asv runs, with and without the launcher's pin."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from optbench import blas_threads

ROOT = Path(__file__).resolve().parents[1]

# The module sets the variables at import, as benchmarks/bench_geodesic.py does.
PROBE = '''
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from optbench.blas_threads import numpy_and_scipy


def _threads(owner):
    return float(max(pool["threads"] for pool in numpy_and_scipy() if pool["owner"] == owner))


class Probe:
    def track_numpy_threads(self):
        return _threads("numpy")

    def track_scipy_threads(self):
        return _threads("scipy")
'''


def _owners() -> set[str]:
    return {pool["owner"] for pool in blas_threads.numpy_and_scipy()}


def _asv_threads(tmp_path: Path, launcher: list[str]) -> dict[str, float]:
    """Run the probe suite through ``asv run`` and read its two tracks."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("asv run needs this checkout's git history")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "__init__.py").write_text("", encoding="utf-8")
    (suite / "bench_probe.py").write_text(PROBE, encoding="utf-8")
    results = tmp_path / "results"
    config = tmp_path / "asv.conf.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "project": "optbench",
                "repo": str(ROOT),
                "environment_type": "existing",
                "benchmark_dir": str(suite),
                "results_dir": str(results),
                "html_dir": str(tmp_path / "html"),
                "branches": ["HEAD"],
            }
        ),
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items() if key not in blas_threads.PINNED}
    home = tmp_path / "home"
    home.mkdir()
    fields = dict.fromkeys(("os", "arch", "cpu", "num_cpu", "ram"), "")
    (home / ".asv-machine.json").write_text(
        json.dumps({"probe": {"machine": "probe", **fields}, "version": 1}), encoding="utf-8"
    )
    env["HOME"] = str(home)
    done = subprocess.run(
        [
            *launcher,
            "run",
            "--config",
            str(config),
            "-E",
            f"existing:{sys.executable}",
            "--set-commit-hash",
            head.stdout.strip(),
            "--machine",
            "probe",
            "--launch-method",
            "forkserver",
            "--show-stderr",
            "--bench",
            "Probe",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    (path,) = [path for path in (results / "probe").glob("*.json") if path.name != "machine.json"]
    document = json.loads(path.read_text(encoding="utf-8"))
    column = document["result_columns"].index("result")
    return {
        owner: document["results"][f"bench_probe.Probe.track_{owner}_threads"][column][0]
        for owner in ("numpy", "scipy")
    }


def test_numpy_blas_is_one_thread_inside_a_launched_benchmark(tmp_path: Path):
    if not {"numpy", "scipy"} <= _owners():
        pytest.skip("needs the OpenBLAS that the numpy and scipy wheels bundle")
    threads = _asv_threads(tmp_path, [sys.executable, "-m", "optbench.asv_launch"])
    assert threads == {"numpy": 1.0, "scipy": 1.0}


def test_bare_asv_leaves_numpy_at_the_library_default(tmp_path: Path):
    """asv-runner 0.3.1 loads numpy before the benchmark sets the variables."""
    if not {"numpy", "scipy"} <= _owners():
        pytest.skip("needs the OpenBLAS that the numpy and scipy wheels bundle")
    if (os.cpu_count() or 1) < 2:
        pytest.skip("one core: the default pool is one thread too")
    threads = _asv_threads(tmp_path, [sys.executable, "-m", "asv"])
    assert threads["numpy"] > 1.0
    assert threads["scipy"] == 1.0


def test_asv_workflow_job_sets_the_pinned_variables():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "asv.yml").read_text(encoding="utf-8"))
    assert workflow["jobs"]["bench"]["env"] == blas_threads.PINNED


def test_pin_replaces_values_already_set():
    environ = {"OPENBLAS_NUM_THREADS": "8", "PATH": "/bin"}
    blas_threads.pin(environ)
    assert environ["PATH"] == "/bin"
    assert {name: environ[name] for name in blas_threads.PINNED} == blas_threads.PINNED
