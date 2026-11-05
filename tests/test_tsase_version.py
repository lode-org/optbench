import subprocess
from pathlib import Path

import pytest

from optbench.engines.tsase_search import checkout_version, require_checkout

SETUP = """#!/usr/bin/env python
from numpy.distutils.core import setup, Extension

setup(
    name='tsase',
    version='1.0',
    packages=[],
)
"""


def _checkout(root: Path) -> str:
    (root / "tsase" / "neb").mkdir(parents=True)
    (root / "setup.py").write_text(SETUP, encoding="utf-8")
    (root / "tsase" / "neb" / "ssneb.py").write_text("class ssneb:\n    pass\n", encoding="utf-8")
    git = ["git", "-C", str(root), "-c", "user.name=TSASE Test", "-c", "user.email=tsase@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "tsase"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def test_the_version_comes_from_the_checkout_without_running_it(tmp_path: Path, monkeypatch):
    head = _checkout(tmp_path)
    monkeypatch.setenv("TSASE_SRC", str(tmp_path))
    assert checkout_version() == {"commit": head, "version": "1.0", "modified": ""}
    template = {"id": "tsase/dimer-pt", "commit": head, "code_version": "1.0"}
    assert require_checkout(template)["commit"] == head

    with pytest.raises(RuntimeError, match="names " + "a" * 40):
        require_checkout({**template, "commit": "a" * 40})
    with pytest.raises(RuntimeError, match="setup.py version is 1.0, tsase/dimer-pt names 1.1"):
        require_checkout({**template, "code_version": "1.1"})

    ssneb = tmp_path / "tsase" / "neb" / "ssneb.py"
    ssneb.write_text(ssneb.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    assert checkout_version()["modified"] == "tsase/neb/ssneb.py"
    with pytest.raises(RuntimeError, match="local changes to tsase/neb/ssneb.py"):
        require_checkout(template)


def test_a_directory_that_is_not_a_checkout_has_no_version(tmp_path: Path):
    assert checkout_version(tmp_path) == {"commit": "", "version": "", "modified": ""}
