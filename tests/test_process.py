"""An engine process that is still running must not leave its children."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from optbench.process import wait_group


def _parent(path: Path) -> None:
    path.write_text(
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "\"import os, time; print(os.getpid(), flush=True); time.sleep(60)\"])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _grandchild(tmp_path: Path) -> tuple[subprocess.Popen, int]:
    script = tmp_path / "parent.py"
    _parent(script)
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    return proc, int(line)


def test_an_interrupt_kills_the_child_process(tmp_path: Path, monkeypatch) -> None:
    proc, grandchild = _grandchild(tmp_path)

    def interrupt(self, timeout=None):
        del self, timeout
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess.Popen, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        wait_group(proc, 30)
    deadline = time.monotonic() + 2
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _alive(grandchild)
    assert proc.poll() is not None


def test_a_timeout_kills_the_child_process(tmp_path: Path) -> None:
    proc, grandchild = _grandchild(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        wait_group(proc, 0.2)
    deadline = time.monotonic() + 2
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _alive(grandchild)
    assert proc.poll() is not None
