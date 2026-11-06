"""Bridge and grid checks for the r2549 harness."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from optbench.extpot import format_extpot
from optbench.potserver import PotServer, remote_count, remote_reset

# Two finished calls are the fewest that tell a tally that adds up across
# bridge runs from one that restarts at every run.
STUB_CAP = 2
CAP_FIELDS = ("client_calls_at_cap", "client_stop_at_cap")

STUB_CLIENT = '''
"""A client that speaks ext_pot the way eonclient does.

Each force call writes from_eon_to_extpot, runs ext_pot_path from
config.ini and reads from_extpot_to_eon. The count goes up after the
reply is read, as every eOn family counts after ExtPot::force returns,
and is rewritten to OPTBENCH_STUB_CALLS. [Stub] calls, when set, is the
number of calls after which the client stops on its own. [Stub]
exchange true runs the bridge as 3.5.0 does, in extpot_<pid>_0 with
EON_EXTPOT_RUN_DIR naming this directory. Otherwise it runs here, as
r2549 does.
"""
import configparser
import os
import shlex
import sys

config = configparser.ConfigParser()
config.read("config.ini")
bridge = shlex.quote(config["Potential"]["ext_pot_path"])
limit = config.getint("Stub", "calls", fallback=None)
here = os.getcwd()
if config.getboolean("Stub", "exchange", fallback=False):
    exchange = os.path.join(here, f"extpot_{os.getpid()}_0")
    os.mkdir(exchange)
    command = f"cd {shlex.quote(exchange)} && export EON_EXTPOT_RUN_DIR={shlex.quote(here)} && {bridge}"
else:
    exchange = here
    command = bridge
calls = 0
while limit is None or calls < limit:
    with open(os.path.join(exchange, "from_eon_to_extpot"), "w") as request:
        request.write("10 0 0\\n0 10 0\\n0 0 10\\n1 0.0 0.0 0.0\\n1 1.5 0.0 0.0\\n")
    # The trailing exit keeps sh between this process and the bridge, as
    # system() does, even under a shell that would exec a last command.
    if os.system(f"{command}; exit $?") != 0:
        sys.exit(1)
    with open(os.path.join(exchange, "from_extpot_to_eon")) as reply:
        float(reply.readline())
    calls += 1
    with open(os.environ["OPTBENCH_STUB_CALLS"], "w") as own:
        own.write(f"{calls}\\n")
with open("results.dat", "w") as results:
    results.write(f"0 termination_reason\\n{calls} total_force_calls\\n")
'''


def _sum_kernel(positions, numbers=None, box=None):
    del numbers, box
    energy = float(sum(positions))
    forces = tuple(-float(value) for value in positions)
    return energy, forces


def _require_bridge():
    if shutil.which("cc") is None:
        pytest.skip("cc is not installed")
    if not Path("/proc/self/status").is_file():
        pytest.skip("the bridge finds the client through /proc")


def _stub_client(directory: Path) -> Path:
    script = directory / "stub-eonclient"
    script.write_text(f"#!{sys.executable}\n{STUB_CLIENT}", encoding="utf-8")
    script.chmod(0o755)
    return script


def _refusal(cap: int) -> str:
    return f"ForceCallCap: force-call cap {cap}"

ROOT = Path(__file__).resolve().parents[1]


def test_c_bridge_matches_the_python_reply(tmp_path, monkeypatch):
    from optbench.eon_harness import bridge_binary

    server = PotServer(_sum_kernel).start()
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

    proc = subprocess.run([str(bridge_binary())], cwd=tmp_path, check=False, text=True, capture_output=True)
    server.stop()
    assert proc.returncode == 0, proc.stderr
    reply = (tmp_path / "from_extpot_to_eon").read_text(encoding="utf-8")
    expected = format_extpot(1.5, (0.0, 0.0, 0.0, -1.5, 0.0, 0.0))
    got = [[float(part) for part in line.split()] for line in reply.splitlines() if line.strip()]
    want = [[float(part) for part in line.split()] for line in expected.splitlines() if line.strip()]
    assert got == want


@pytest.mark.parametrize("exchange", [False, True], ids=["r2549-directory", "350-exchange-directory"])
def test_bridge_reports_each_client_count_before_the_cap_kill(tmp_path, monkeypatch, exchange):
    """Lowest-mode attempts reuse one directory, one client after another."""
    from optbench.eon_harness import bridge_binary
    from optbench.eon_jobs import _lowest_once

    _require_bridge()
    client = str(_stub_client(tmp_path))
    own = tmp_path / "own-calls"
    con = tmp_path / "start.con"
    con.write_text("unused by the stub\n", encoding="utf-8")
    run = tmp_path / "run"
    ini = f"[Potential]\npotential = ext_pot\next_pot_path = {bridge_binary()}\n[Stub]\nexchange = {exchange}\n"
    server = PotServer(_sum_kernel, cap=STUB_CAP).start()
    monkeypatch.setenv("OPTBENCH_POTSERVER", server.endpoint())
    monkeypatch.setenv("OPTBENCH_STUB_CALLS", str(own))
    try:
        capped = _lowest_once(client, ini, str(run), str(con), "")
        assert remote_count(server.host, server.port)[0] == STUB_CAP
        assert int(own.read_text(encoding="utf-8")) == STUB_CAP
        assert capped["cap"] == {"client_calls_at_cap": STUB_CAP, "client_stop_at_cap": _refusal(STUB_CAP)}
        assert capped["raw"] is None and capped["reason"] is None
        assert (run / "stderr.txt").read_text(encoding="utf-8") == "optbench ext_pot: error\n"

        remote_reset(server.host, server.port)
        finished = _lowest_once(client, ini + "calls = 1\n", str(run), str(con), "")
        assert finished["cap"] == {}
        assert finished["raw"] == 1 and finished["reason"] == "0"

        remote_reset(server.host, server.port)
        again = _lowest_once(client, ini, str(run), str(con), "")
        assert again["cap"]["client_calls_at_cap"] == STUB_CAP
    finally:
        server.stop()


def _capped_run(monkeypatch, tmp_path, template_name: str, start: str, cap: int, client: str, extra=None) -> dict:
    from optbench import eon_harness

    loaded = eon_harness.load_template

    def capped(root, name):
        template = dict(loaded(root, name))
        template["failure_cap"] = cap
        template["ini"] = {**template["ini"], **(extra or {})}
        return template

    monkeypatch.setattr(eon_harness, "load_template", capped)
    # The cache key leaves out the cap.
    monkeypatch.setattr(eon_harness, "CACHE", tmp_path / "cache" / str(cap))
    monkeypatch.setattr(eon_harness, "_client", lambda family: client)
    record = eon_harness.run_case(template_name, 1, start, budget=0)
    return eon_harness._start_row(1, start, record)


def _scored(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in CAP_FIELDS}


def _assert_score_ignores_cap_fields(row: dict, cap: int) -> None:
    from optbench.score import normalize_start, score_v2

    with_fields = score_v2([normalize_start(row, None)], cap)
    without = score_v2([normalize_start(_scored(row), None)], cap)
    assert with_fields == without
    assert with_fields["n_cap"] == 1 and with_fields["successes"] == 0


def test_cap_row_adds_client_fields_and_keeps_the_scored_ones(tmp_path, monkeypatch):
    _require_bridge()
    own = tmp_path / "own-calls"
    monkeypatch.setenv("OPTBENCH_STUB_CALLS", str(own))
    client = str(_stub_client(tmp_path))
    exchange = {"Stub": {"exchange": True}}
    row = _capped_run(monkeypatch, tmp_path, "eon-client/ts-lj38-dimer", "0000", STUB_CAP, client, exchange)
    assert row["client_calls_at_cap"] == int(own.read_text(encoding="utf-8")) == STUB_CAP
    assert row["client_stop_at_cap"] == _refusal(STUB_CAP)
    scored = _scored(row)
    del scored["seconds"]
    assert scored == {
        "call_unit": "total",
        "count_key": "harness",
        "counter": "harness",
        "failure": "cap",
        "force_calls": STUB_CAP,
        "geometry": "0000",
        "raw": STUB_CAP,
        "seed": 1,
        "success": False,
    }
    _assert_score_ignores_cap_fields(row, STUB_CAP)


def test_svn_public_row_keeps_the_cap_fields():
    from optbench.eon_svn import _public_row

    record = {"start": "0000", "seed": "1", "force_calls": STUB_CAP, "success": False, "failure": "cap"}
    plain = _public_row(record)
    row = _public_row({**record, "client_calls_at_cap": STUB_CAP, "client_stop_at_cap": _refusal(STUB_CAP)})
    assert row["client_calls_at_cap"] == STUB_CAP
    assert row["client_stop_at_cap"] == _refusal(STUB_CAP)
    assert _scored(row) == plain


def test_eonclient_350_cap_failure_records_the_client_count(tmp_path, monkeypatch):
    """EONCLIENT_350 names a 3.5.0 eonclient. The start is run once without
    a lowered cap and then with half of the calls it took, so the refusal
    falls inside the search."""
    from optbench import eon_harness

    _require_bridge()
    binary = os.environ.get("EONCLIENT_350", "")
    if not binary or not os.access(binary, os.X_OK):
        pytest.skip("EONCLIENT_350 does not name an eonclient")
    try:
        client = eon_harness._client("client")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    template_name = "eon-client/ts-lj38-dimer"
    full = eon_harness.load_template(eon_harness.ROOT, template_name)["failure_cap"]
    uncapped = _capped_run(monkeypatch, tmp_path, template_name, "0000", full, client)
    assert uncapped["success"] and not set(CAP_FIELDS) & set(uncapped)
    assert uncapped["client_calls"] == uncapped["force_calls"]
    cap = uncapped["force_calls"] // 2
    row = _capped_run(monkeypatch, tmp_path, template_name, "0000", cap, client)
    assert row["failure"] == "cap" and row["success"] is False
    assert row["force_calls"] == row["raw"] == cap
    assert row["client_calls_at_cap"] == cap
    assert row["client_stop_at_cap"] == _refusal(cap)
    assert "client_calls" not in row and "client_reason" not in row
    _assert_score_ignores_cap_fields(row, cap)


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
