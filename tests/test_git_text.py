"""One git process serves the same blobs ``git show`` would."""

import subprocess
from pathlib import Path

from optbench.board import _git_text

ROOT = Path(__file__).resolve().parents[1]


def test_git_text_matches_show() -> None:
    spec = "HEAD:README.md"
    shown = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", spec],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert _git_text(ROOT, spec) == shown
    assert _git_text(ROOT, "HEAD:templates/does-not-exist.toml") == ""
