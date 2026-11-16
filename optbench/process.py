"""Wait for one engine process and stop the session it leads.

A search binary can start another process, such as a potential hook or an
MPI rank. ``Popen.kill`` reaches only the direct child. The child is
started in its own session, and that session is killed if it is still
running when the wait ends or is interrupted.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time


def wait_group(proc: subprocess.Popen, timeout: float | None) -> int:
    """The child's exit code. A timeout or an interrupt kills its session first."""
    try:
        return int(proc.wait(timeout=timeout))
    finally:
        stop_group(proc)


def stop_group(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and the processes in its session, then reap ``proc``."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 5
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
