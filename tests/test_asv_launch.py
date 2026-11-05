"""The asv launcher's machine entry and machine file."""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optbench import asv_launch
from optbench.asv_launch import MACHINE_FIELDS, machine_entry, machine_fields, stored_entry, subcommand, write_machine

ROOT = Path(__file__).resolve().parents[1]

WORKER = """
import json
import sys

from asv.machine import Machine
from optbench.asv_launch import own_machine_file, write_machine

role, index, loops = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
failures = []
for step in range(loops):
    try:
        if role == "write":
            write_machine({"machine": f"w{index}-{step}"})
            continue
        name = f"m{index % 4}"
        with own_machine_file(["run", "--machine", name]):
            loaded = Machine.load(machine_name=name, interactive=False)
        if loaded.machine != name:
            failures.append(f"loaded {loaded.machine} for {name}")
    except BaseException as exc:
        failures.append(type(exc).__name__ + ": " + (str(exc).splitlines() or [""])[0])
print(json.dumps(failures))
"""


def _machine_command(launcher: list[str], home: Path, *args: str) -> dict:
    home.mkdir(exist_ok=True)
    env = dict(os.environ, HOME=str(home))
    subprocess.run([*launcher, "machine", *args], env=env, capture_output=True, text=True, check=True, timeout=120)
    return json.loads((home / ".asv-machine.json").read_text(encoding="utf-8"))


def test_launcher_machine_command_stores_every_field(tmp_path: Path):
    from asv.machine import Machine

    stored = _machine_command([sys.executable, "-m", "optbench.asv_launch"], tmp_path / "home", "--yes", "--machine", "repro")
    detected = Machine.get_defaults()
    assert stored["version"] == 1
    assert stored["repro"] == {**detected, "machine": "repro"}


def test_asv_machine_command_drops_the_fields_it_detects(tmp_path: Path):
    """asv 0.6.6 keeps only the fields that differ from its defaults."""
    stored = _machine_command([sys.executable, "-m", "asv"], tmp_path / "home", "--yes", "--machine", "repro")
    assert stored["repro"] == {"machine": "repro"}


def test_given_then_stored_then_detected(tmp_path: Path):
    from asv.machine import Machine

    path = tmp_path / ".asv-machine.json"
    path.write_text(
        json.dumps({"repro": {"machine": "repro", "cpu": "named by hand"}, "other": {"machine": "other"}, "version": 1}),
        encoding="utf-8",
    )
    entry = write_machine({"machine": "repro", "ram": "4GB"}, path)
    detected = Machine.get_defaults()
    assert entry == {**detected, "machine": "repro", "cpu": "named by hand", "ram": "4GB"}
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["repro"] == entry
    assert saved["other"] == {"machine": "other"}
    assert sorted(path.parent.iterdir()) == [path, tmp_path / ".asv-machine.json.lock"]


def test_machine_entry_has_every_asv_field():
    from asv.machine import Machine

    assert tuple(name for name, _description in Machine.fields) == MACHINE_FIELDS
    assert set(machine_entry(None, {"machine": "x"})) == set(MACHINE_FIELDS)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], None),
        (["--yes"], {}),
        (["--machine", "repro"], {"machine": "repro"}),
        (["--yes", "--machine=repro", "--os", "Linux"], {"machine": "repro", "os": "Linux"}),
    ],
)
def test_machine_fields_leave_the_bare_prompt_to_asv(args, expected):
    assert machine_fields(args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], None),
        (["run", "-m", "repro"], "run"),
        (["--config", "asv.conf.json", "-v", "run"], "run"),
        (["--config=asv.conf.json", "machine", "--yes"], "machine"),
    ],
)
def test_subcommand_skips_global_options(args, expected):
    assert subcommand(args) == expected


def test_stored_entry_is_the_one_asv_loads(tmp_path: Path):
    from asv import util
    from asv.machine import Machine, MachineCollection

    host = Machine.get_unique_machine_name()
    cases = [
        ({"repro": {"machine": "repro", "cpu": "a"}, "other": {"machine": "other"}}, "repro"),
        ({host: {"machine": host, "cpu": "b"}, "other": {"machine": "other"}}, None),
        ({"only": {"machine": "only", "cpu": "c"}}, None),
    ]
    path = tmp_path / "machines.json"
    for entries, name in cases:
        machines = {**entries, "version": 1}
        path.write_text(json.dumps(machines), encoding="utf-8")
        found_name, found = stored_entry(machines, name)
        assert found == MachineCollection.load(name or host, _path=str(path))
        assert entries[found_name] == found
    machines = {"a": {"machine": "a"}, "b": {"machine": "b"}, "version": 1}
    path.write_text(json.dumps(machines), encoding="utf-8")
    with pytest.raises(util.UserError):
        MachineCollection.load("missing", _path=str(path))
    with pytest.raises(SystemExit, match="machine --yes --machine missing"):
        stored_entry(machines, "missing")


def test_each_command_gets_the_machine_file_it_needs(tmp_path: Path, monkeypatch):
    """Readers get a private copy of one entry, writers the shared file under the lock."""
    from asv.machine import MachineCollection

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPTBENCH_ASV_RECORD", raising=False)
    monkeypatch.setattr(asv_launch.blas_threads, "pin", lambda environ=None: None)
    monkeypatch.setattr(asv_launch, "probe", lambda python: [])
    shared = tmp_path / ".asv-machine.json"
    machines = {"repro": {"machine": "repro", "cpu": "a"}, "other": {"machine": "other"}, "version": 1}
    shared.write_text(json.dumps(machines), encoding="utf-8")
    seen = {}

    def fake_asv(args):
        with open(tmp_path / ".asv-machine.json.lock", "a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                lock = "held"
            else:
                lock = "free"
                fcntl.flock(handle, fcntl.LOCK_UN)
        path = Path(MachineCollection.get_machine_file_path())
        seen[args[0]] = (lock, path == shared, json.loads(path.read_text(encoding="utf-8")))

    monkeypatch.setattr(asv_launch, "run_asv", fake_asv)
    for args in (["run", "--machine", "repro"], ["update"], ["machine"], ["publish"]):
        asv_launch.main(args)
    assert seen == {
        "run": ("free", False, {"repro": machine_entry(machines["repro"], {"machine": "repro"}), "version": 1}),
        "update": ("held", True, machines),
        "machine": ("held", True, machines),
        "publish": ("free", True, machines),
    }
    assert Path(MachineCollection.get_machine_file_path()) == shared


def test_concurrent_launched_runs_do_not_race_on_the_machine_file(tmp_path: Path):
    """Loads that start together, as several launched runs do, while the launcher writes."""
    from asv.machine import Machine

    home = tmp_path / "home"
    home.mkdir()
    shared = home / ".asv-machine.json"
    fields = Machine.get_defaults()
    seeded = {**{f"m{index}": {**fields, "machine": f"m{index}"} for index in range(4)}, "version": 1}
    shared.write_text(json.dumps(seeded), encoding="utf-8")
    env = dict(os.environ, HOME=str(home))
    roles = [("load", index, 100) for index in range(6)] + [("write", 0, 50)]
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", WORKER, role, str(index), str(loops)], env=env, stdout=subprocess.PIPE, text=True
        )
        for role, index, loops in roles
    ]
    failures = []
    for worker in workers:
        out, _err = worker.communicate(timeout=300)
        assert worker.returncode == 0
        failures += json.loads(out.splitlines()[-1])
    assert failures == []
    final = json.loads(shared.read_text(encoding="utf-8"))
    assert {key: final[key] for key in seeded} == seeded
    assert sorted(key for key in final if key.startswith("w")) == sorted(f"w0-{step}" for step in range(50))


def test_a_run_fills_the_fields_a_stored_entry_lacks(tmp_path: Path, monkeypatch):
    from asv.machine import Machine, MachineCollection

    monkeypatch.setenv("HOME", str(tmp_path))
    shared = tmp_path / ".asv-machine.json"
    shared.write_text(json.dumps({"repro": {"machine": "repro", "cpu": "named by hand"}, "version": 1}), encoding="utf-8")
    with asv_launch.own_machine_file(["run", "--machine", "repro"]):
        loaded = Machine.load(machine_name="repro", interactive=False)
        private = json.loads(Path(MachineCollection.get_machine_file_path()).read_text(encoding="utf-8"))
    assert private["repro"] == {**Machine.get_defaults(), "machine": "repro", "cpu": "named by hand"}
    assert {field: getattr(loaded, field) for field in MACHINE_FIELDS} == private["repro"]
    assert json.loads(shared.read_text(encoding="utf-8"))["repro"] == {"machine": "repro", "cpu": "named by hand"}


@pytest.mark.parametrize("writer", ["optbench.asv_launch", "asv"])
def test_saved_run_records_the_whole_machine(tmp_path: Path, writer: str):
    """``asv run`` copies the entry into ``results/<machine>/machine.json``.

    asv writes its private copy of the machine file, not the shared one.
    With ``asv`` as the writer the shared entry is just the name.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("asv run needs this checkout's git history")
    home = tmp_path / "home"
    _machine_command([sys.executable, "-m", writer], home, "--yes", "--machine", "repro")
    shared = home / ".asv-machine.json"
    before = (shared.read_bytes(), shared.stat().st_mtime_ns)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "__init__.py").write_text("", encoding="utf-8")
    (suite / "bench_one.py").write_text("def track_one():\n    return 1.0\n", encoding="utf-8")
    config = tmp_path / "asv.conf.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "project": "optbench",
                "repo": str(ROOT),
                "environment_type": "existing",
                "benchmark_dir": str(suite),
                "results_dir": str(tmp_path / "results"),
                "html_dir": str(tmp_path / "html"),
                "branches": ["HEAD"],
            }
        ),
        encoding="utf-8",
    )
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "optbench.asv_launch",
            "run",
            "--config",
            str(config),
            "-E",
            f"existing:{sys.executable}",
            "--set-commit-hash",
            head.stdout.strip(),
            "--machine",
            "repro",
            "--bench",
            "track_one",
        ],
        env=dict(os.environ, HOME=str(home)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    recorded = json.loads((tmp_path / "results" / "repro" / "machine.json").read_text(encoding="utf-8"))
    assert set(MACHINE_FIELDS) <= set(recorded)
    result = next(path for path in (tmp_path / "results" / "repro").glob("*.json") if path.name != "machine.json")
    params = json.loads(result.read_text(encoding="utf-8"))["params"]
    assert {field: params[field] for field in ("cpu", "os", "num_cpu")} == {
        field: recorded[field] for field in ("cpu", "os", "num_cpu")
    }
    assert (shared.read_bytes(), shared.stat().st_mtime_ns) == before
