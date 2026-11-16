"""A directory the main process owns for files a worker creates.

A worker stopped with SIGTERM does not run the ``finally`` block that
would remove its own directory. The worker creates that directory under
the one this process made, and this process removes that parent at
close. A directory this process did not create is left alone.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_ENV = "OPTBENCH_RUN_TMP"
_PREFIX = "optbench-run-"

_root: Path | None = None
_owner: int | None = None
_depth: int = 0


def directory() -> Path | None:
    """The parent a worker should create directories under, if one is open."""
    if _owner == os.getpid() and _root is not None:
        return _root
    raw = os.environ.get(_ENV)
    if not raw:
        return None
    path = Path(raw)
    if _is_run_dir(path):
        return path
    return None


def mkdtemp(prefix: str) -> str:
    """A new directory under the run parent, or in the system temp dir."""
    parent = directory()
    if parent is None:
        return tempfile.mkdtemp(prefix=prefix)
    return tempfile.mkdtemp(prefix=prefix, dir=parent)


def temporary_directory(prefix: str) -> tempfile.TemporaryDirectory[str]:
    """Like ``tempfile.TemporaryDirectory``, under the run parent when one is open."""
    parent = directory()
    if parent is None:
        return tempfile.TemporaryDirectory(prefix=prefix)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=parent)


@contextmanager
def session() -> Iterator[Path]:
    """Create the run parent, and remove it when this process is done with it.

    A worker that inherited the parent does not remove it. Nested sessions
    in the owning process share one directory.
    """
    _claim()
    try:
        if _root is None:
            raise RuntimeError("the run directory was not created")
        yield _root
    finally:
        _release()


def _claim() -> None:
    global _root, _owner, _depth
    if _owner == os.getpid() and _root is not None:
        _depth += 1
        return
    inherited = os.environ.get(_ENV)
    if inherited and _is_run_dir(Path(inherited)):
        _root = Path(inherited)
        _owner = None
        return
    base = Path(tempfile.gettempdir())
    path = Path(tempfile.mkdtemp(prefix=_PREFIX, dir=base))
    _root = path
    _owner = os.getpid()
    _depth = 1
    os.environ[_ENV] = str(path)


def _release() -> None:
    global _root, _owner, _depth
    if _owner != os.getpid() or _root is None:
        return
    _depth -= 1
    if _depth > 0:
        return
    path = _root
    _root = None
    _owner = None
    _depth = 0
    if os.environ.get(_ENV) == str(path):
        os.environ.pop(_ENV, None)
    _remove_owned(path)


def _is_run_dir(path: Path) -> bool:
    name = path.name
    if not name.startswith(_PREFIX) or name != path.name:
        return False
    try:
        base = Path(tempfile.gettempdir()).resolve()
        parent = path.parent.resolve()
    except OSError:
        return False
    if parent != base:
        return False
    try:
        return path.is_dir() and not path.is_symlink()
    except OSError:
        return False


def _remove_owned(path: Path) -> None:
    """Remove ``path`` only when it is a run directory this module created."""
    if not _is_run_dir(path):
        return
    shutil.rmtree(path, ignore_errors=True)
