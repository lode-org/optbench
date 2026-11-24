"""The committed site is what optbench.site.render writes."""

import math
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


def test_published_board_json_drops_last_bit_noise():
    from optbench.board import dumps_board

    base = 1649.199999999999
    noisy = base
    for _ in range(8):
        noisy = math.nextafter(noisy, math.inf)
    same = dumps_board({"median_high": base, "par2_mean": 4603.075, "b": 1, "a": "kept"})
    moved = dumps_board({"a": "kept", "b": 1, "par2_mean": 4603.075, "median_high": noisy})
    assert same == moved
    assert same.index('\n  "a"') < same.index('\n  "b"')
    assert dumps_board({"par2_mean": 4603.075}) != dumps_board({"par2_mean": 4610.075})
    assert dumps_board({"x": -0.0}) == dumps_board({"x": 0.0})
    assert "-0.0" not in dumps_board({"x": -0.0})
