"""A killed worker must not leave the directory it created."""

import multiprocessing
import os
import time
from pathlib import Path

from optbench.scratch import _remove_owned, directory, mkdtemp, session


def _hold(flag: str) -> None:
    path = mkdtemp("worker-")
    Path(flag).write_text(path)
    time.sleep(60)


def test_a_killed_worker_leaves_nothing(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("keep")
    flag = tmp_path / "flag"
    context = multiprocessing.get_context("spawn")
    with session() as root:
        proc = context.Process(target=_hold, args=(str(flag),))
        proc.start()
        deadline = time.monotonic() + 30
        while not flag.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert flag.is_file()
        made = Path(flag.read_text(encoding="utf-8"))
        assert made.is_dir()
        assert root in made.parents
        proc.terminate()
        proc.join(5)
        assert proc.exitcode is not None
        assert not proc.is_alive()
        assert made.is_dir()
    assert not root.exists()
    assert not made.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert "OPTBENCH_RUN_TMP" not in os.environ


def test_a_directory_left_under_the_run_parent_is_removed() -> None:
    with session() as root:
        left = Path(mkdtemp("left-"))
        assert left.parent == root
        assert left.is_dir()
    assert not root.exists()
    assert not left.exists()


def test_a_foreign_directory_is_not_a_run_directory(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "marker"
    marker.write_text("keep")
    monkeypatch.setenv("OPTBENCH_RUN_TMP", str(tmp_path))
    assert directory() is None
    _remove_owned(tmp_path)
    assert marker.read_text(encoding="utf-8") == "keep"
    made = Path(mkdtemp("aside-"))
    try:
        assert tmp_path not in made.parents
    finally:
        os.rmdir(made)
