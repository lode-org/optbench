"""The asv launcher's machine entry."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optbench.asv_launch import MACHINE_FIELDS, machine_entry, machine_fields, write_machine

ROOT = Path(__file__).resolve().parents[1]


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


def test_saved_run_records_the_whole_machine(tmp_path: Path):
    """``asv run`` copies the entry into ``results/<machine>/machine.json``."""
    if not (ROOT / ".git").exists():
        pytest.skip("asv run needs this checkout's git history")
    home = tmp_path / "home"
    _machine_command([sys.executable, "-m", "optbench.asv_launch"], home, "--yes", "--machine", "repro")
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
