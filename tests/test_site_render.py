"""The committed site is what optbench.site.render writes."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stat() -> str:
    return subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True).strip() or "(clean)"


def _status() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def test_site_render_leaves_no_diff():
    before_stat = _stat()
    before = subprocess.check_output(["git", "diff"], cwd=ROOT)
    before_status = _status()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path\nfrom optbench.site import render\nrender(Path('.'))",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    after = subprocess.check_output(["git", "diff"], cwd=ROOT)
    after_status = _status()
    if after == before and after_status == before_status:
        return
    after_stat = _stat()
    raise AssertionError(
        f"render changed the worktree\nbefore:\n{before_stat}\nafter:\n{after_stat}\n"
        f"status before:\n{before_status}\nstatus after:\n{after_status}"
    )
