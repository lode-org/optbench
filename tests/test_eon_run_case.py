"""One harness start, run against a client that behaves as each outcome says.

The stub client reads its instructions from a ``[Stub]`` section of the
generated ``config.ini``. It makes the requested number of force calls
through the real C bridge, so the harness counts them itself, and then
writes the files a client leaves behind. What these tests check is the
decision the harness makes from the count, the reason, and the geometry.
"""

from __future__ import annotations

import shutil
import signal
import sys
from pathlib import Path

import pytest

from optbench import eon_harness

STUB = '''
import configparser
import os
import shlex
import shutil
import sys
import time

config = configparser.ConfigParser()
config.read("config.ini")
bridge = shlex.quote(config["Potential"]["ext_pot_path"])
for _ in range(config.getint("Stub", "calls", fallback=0)):
    with open("from_eon_to_extpot", "w") as request:
        request.write("10 0 0\\n0 10 0\\n0 0 10\\n1 0.0 0.0 0.0\\n1 1.5 0.0 0.0\\n")
    if os.system(f"{bridge}; exit $?") != 0:
        sys.exit(1)
    with open("from_extpot_to_eon") as reply:
        float(reply.readline())
time.sleep(config.getfloat("Stub", "sleep", fallback=0.0))
for name, option in (("results.dat", "results"), ("earr.dat", "earr")):
    if config.has_option("Stub", option):
        with open(name, "w") as out:
            out.write(config.get("Stub", option).replace("|", "\\n") + "\\n")
if config.has_option("Stub", "saddle"):
    shutil.copy(config.get("Stub", "saddle"), "saddle.con")
sys.exit(config.getint("Stub", "status", fallback=0))
'''

SADDLE = "eon-svn/ts-lj38-dimer-ljcut"
BASIN = "eon-svn/bh-lj38"
MINIMA = "eon-svn/mh-lj38"
CONVERGED = "0 termination_reason|{calls} total_force_calls"


def _con(positions, fixed=()) -> str:
    head = [
        "stationary point under test",
        "0 0 0",
        "30.000000 30.000000 30.000000",
        "90.000000 90.000000 90.000000",
        "0 0",
        "0 0 0",
        "1",
        str(len(positions)),
        "39.948000",
        "Ar",
        "Coordinates of Component 1",
    ]
    atoms = [
        f"{x:.10f} {y:.10f} {z:.10f} {int(index in fixed)} {index}"
        for index, (x, y, z) in enumerate(positions)
    ]
    return "\n".join([*head, *atoms, ""])


def _pinned_pair(tmp_path: Path, half_gap: float, drift: float = 0.0) -> str:
    """Two pinned atoms with one free atom between them, drifted off the axis."""
    path = tmp_path / "saddle-geometry.con"
    positions = [(-half_gap, 0.0, 0.0), (0.0, drift, 0.0), (half_gap, 0.0, 0.0)]
    path.write_text(_con(positions, fixed=(0, 2)), encoding="utf-8")
    return str(path)


@pytest.fixture
def run(tmp_path, monkeypatch):
    if shutil.which("cc") is None:
        pytest.skip("cc is not installed")
    if not Path("/proc/self/status").is_file():
        pytest.skip("the bridge finds the client through /proc")
    client = tmp_path / "stub-eonclient"
    client.write_text(f"#!{sys.executable}\n{STUB}", encoding="utf-8")
    client.chmod(0o755)
    loaded = eon_harness.load_template
    stub: dict = {}

    def patched(root, name):
        template = dict(loaded(root, name))
        template["ini"] = {**template["ini"], "Stub": dict(stub)}
        return template

    monkeypatch.setattr(eon_harness, "load_template", patched)
    monkeypatch.setattr(eon_harness, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(eon_harness, "_client", lambda family: str(client))

    def start(template: str, begin: str = "0000", *, budget: int = 0, **settings) -> dict:
        stub.clear()
        stub.update(settings)
        return eon_harness.run_case(template, 1, begin, budget=budget)

    return start


def test_a_client_that_stops_on_an_index_one_saddle_is_a_success(run, tmp_path):
    record = run(SADDLE, calls=3, results=CONVERGED.format(calls=3), saddle=_pinned_pair(tmp_path, 1.5))
    assert record["success"] is True
    assert record["failure"] is None
    assert record["force_calls"] == 3.0
    assert record["client_calls"] == 3
    assert record["client_reason"] == "0"
    assert record["hessian_index"] == 1
    assert record["force_norm"] == pytest.approx(0.0, abs=1e-9)
    assert record["counter"] == "harness"
    assert record["kernel"] == "lj40"


def test_a_saddle_within_the_converged_force_is_accepted_and_beyond_it_is_a_divergence(run, tmp_path):
    inside = run(SADDLE, calls=3, results=CONVERGED.format(calls=3), saddle=_pinned_pair(tmp_path, 1.5, 1e-4))
    assert inside["success"] is True
    assert 0.0 < inside["force_norm"] < 0.001
    outside = run(SADDLE, "0001", calls=3, results=CONVERGED.format(calls=3), saddle=_pinned_pair(tmp_path, 1.5, 0.01))
    assert outside["success"] is False
    assert outside["failure"] == "divergence"
    assert outside["force_norm"] > 0.001
    assert outside["hessian_index"] == 1


@pytest.mark.parametrize(("half_gap", "index"), [(1.2, 0), (1.0, 2)])
def test_a_stationary_point_of_the_wrong_index_is_not_the_saddle_the_client_claims(run, tmp_path, half_gap, index):
    record = run(SADDLE, calls=3, results=CONVERGED.format(calls=3), saddle=_pinned_pair(tmp_path, half_gap))
    assert record["success"] is False
    assert record["failure"] == "wrong_stationary"
    assert record["hessian_index"] == index
    assert record["force_norm"] == pytest.approx(0.0, abs=1e-9)


def test_a_large_residual_force_is_a_divergence_even_at_the_wrong_index(run, tmp_path):
    record = run(SADDLE, calls=3, results=CONVERGED.format(calls=3), saddle=_pinned_pair(tmp_path, 1.2, 0.05))
    assert record["failure"] == "divergence"
    assert record["hessian_index"] == 0


def test_a_converged_client_that_leaves_no_saddle_file_is_a_crash(run):
    record = run(SADDLE, calls=3, results=CONVERGED.format(calls=3))
    assert record["success"] is False
    assert record["failure"] == "crash"
    assert "hessian_index" not in record


def test_a_client_that_gives_a_nonzero_reason_has_diverged_and_is_not_checked_further(run, tmp_path):
    record = run(SADDLE, calls=3, results="2 termination_reason|3 total_force_calls", saddle=_pinned_pair(tmp_path, 1.5))
    assert record["success"] is False
    assert record["failure"] == "divergence"
    assert record["client_reason"] == "2"
    assert "hessian_index" not in record


def test_a_client_that_dies_without_a_reason_is_a_crash(run):
    record = run(SADDLE, calls=3, status=3)
    assert record["success"] is False
    assert record["failure"] == "crash"
    assert record["force_calls"] == 3.0
    assert record["client_reason"] is None
    assert record["client_calls"] is None


def test_a_client_that_reports_calls_the_harness_never_saw_is_a_crash(run, tmp_path):
    record = run(SADDLE, calls=0, results=CONVERGED.format(calls=7), saddle=_pinned_pair(tmp_path, 1.5))
    assert record["force_calls"] == 0.0
    assert record["client_calls"] == 7
    assert record["success"] is False
    assert record["failure"] == "crash"


def test_a_start_that_was_run_is_read_back_and_not_run_again(run, tmp_path):
    first = run(SADDLE, calls=3, results=CONVERGED.format(calls=3), saddle=_pinned_pair(tmp_path, 1.5))
    again = run(SADDLE, calls=3, status=9)
    assert again == first
    other = run(SADDLE, "0001", calls=3, status=9)
    assert other["failure"] == "crash"


def test_a_start_that_outlives_its_budget_is_a_timeout_that_is_not_kept(run, tmp_path):
    previous = signal.getsignal(signal.SIGALRM)
    try:
        record = run(SADDLE, calls=2, sleep=30.0, budget=2)
    finally:
        signal.signal(signal.SIGALRM, previous)
    assert record["timed_out"] is True
    assert record["success"] is False
    assert record["failure"] == "crash"
    assert record["force_calls"] == 2.0
    assert record["seconds"] == 2.0
    assert record["uncached"] is True
    fresh = run(SADDLE, calls=3, results=CONVERGED.format(calls=3), saddle=_pinned_pair(tmp_path, 1.5))
    assert fresh["success"] is True


@pytest.mark.parametrize(
    ("results", "status", "outcome"),
    [
        ("-173.95 minimum_energy", 0, (True, None)),
        ("-150.00 minimum_energy", 0, (False, "wrong_stationary")),
        ("-173.927427 minimum_energy", 0, (False, "wrong_stationary")),
        ("0 other_field", 0, (False, "crash")),
        ("-173.95 minimum_energy", 3, (False, "crash")),
    ],
)
def test_a_basin_hopping_run_succeeds_only_below_the_stop_energy(run, results, status, outcome):
    record = run(BASIN, "0", calls=3, results=results, status=status)
    assert (record["success"], record["failure"]) == outcome
    assert record["force_calls"] == 3.0
    if "minimum_energy" in results:
        assert record["minimum_energy"] == float(results.split()[0])
    else:
        assert record["minimum_energy"] is None


@pytest.mark.parametrize(
    ("earr", "status", "outcome", "lowest"),
    [
        ("0 -150.0|1 -173.95|2 -160.0", 0, (True, None), -173.95),
        ("0 -150.0|1 -160.0", 0, (False, "wrong_stationary"), -160.0),
        ("0 -173.95", 3, (False, "crash"), -173.95),
    ],
)
def test_a_minima_hopping_run_succeeds_when_its_lowest_energy_is_under_the_target(run, earr, status, outcome, lowest):
    record = run(MINIMA, "0", calls=3, earr=earr, status=status)
    assert (record["success"], record["failure"]) == outcome
    assert record["minimum_energy"] == lowest


def test_a_minima_hopping_run_with_no_energy_file_is_a_crash(run):
    record = run(MINIMA, "0", calls=3)
    assert record["success"] is False
    assert record["failure"] == "crash"
    assert record["minimum_energy"] is None
