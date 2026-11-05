"""BLAS thread counts and the launcher's record of them."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optbench import blas_threads
from optbench.asv_launch import interpreter


def _bare_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in blas_threads.VARIABLES}


def test_pools_report_the_library_count_not_the_variable():
    source = Path(blas_threads.__file__).read_text(encoding="utf-8")
    env = dict(_bare_environment(), OPENBLAS_NUM_THREADS="2")
    done = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, check=True)
    pools = json.loads(done.stdout)
    if not pools:
        pytest.skip("no OpenBLAS found in /proc/self/maps")
    assert {pool["threads"] for pool in pools} == {min(2, os.cpu_count() or 1)}
    assert all(pool["version"].startswith("OpenBLAS ") for pool in pools)


def test_launcher_writes_the_variables_pools_and_arguments(tmp_path: Path):
    target = tmp_path / "record.json"
    env = dict(_bare_environment(), OPTBENCH_ASV_RECORD=str(target))
    done = subprocess.run(
        [sys.executable, "-m", "optbench.asv_launch", "--version"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    record = json.loads(target.read_text(encoding="utf-8"))
    assert set(record) == {"variables", "blas", "asv"}
    assert set(record["variables"]) == set(blas_threads.VARIABLES)
    assert record["asv"] == ["--version"]
    assert all(set(pool) == {"owner", "library", "version", "threads"} for pool in record["blas"])
    assert all(pool["threads"] >= 1 for pool in record["blas"])
    assert "optbench.asv_launch {" in done.stderr


def test_geodesic_sidecar_records_the_pools_its_search_ran_with(monkeypatch):
    import optbench.rgsaddle_band as band
    from benchmarks import bench_geodesic

    found = {"start": "product_0", "force_calls": 12, "success": True, "failure": None}
    monkeypatch.setattr(band, "search_named", lambda *args: dict(found))
    record = bench_geodesic.PtIslandGeodesicCiNeb().measure("0", "product_0")
    assert record["force_calls"] == 12.0
    assert record["blas"] == blas_threads.loaded()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["run", "-E", "existing:envs/py/bin/python", "HEAD^!"], "envs/py/bin/python"),
        (["run", "--environment=existing:envs/py/bin/python"], "envs/py/bin/python"),
        (["run", "-Eexisting:envs/py/bin/python"], "envs/py/bin/python"),
        (["run", "-E", "uv-pip", "HEAD^!"], sys.executable),
        (["run", "HEAD^!"], sys.executable),
    ],
)
def test_probe_uses_the_existing_environment_python(args, expected):
    assert interpreter(args) == expected
