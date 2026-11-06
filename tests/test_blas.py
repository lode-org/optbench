"""An ASE start runs every loaded OpenBLAS on one thread.

asv's runner imports NumPy before any benchmark, so NumPy's OpenBLAS has
read its thread count before ``optbench.ase_sella`` sets
``OPENBLAS_NUM_THREADS``. The subprocess here imports NumPy and SciPy
first in the same way, with every thread variable removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SCRIPT = """
import json
import numpy
import scipy.linalg
from optbench import blas
first = blas.threads()
from optbench import ase_sella
imported = blas.threads()
row = ase_sella._run_start({"start": "0000", "task": "none", "raw_cap": 1, "tol": 0.01})
print(json.dumps({"first": first, "imported": imported, "started": blas.threads(), "error": row["error"]}))
"""


def test_an_ase_start_pins_openblas_loaded_before_ase_sella():
    env = {key: value for key, value in os.environ.items() if key not in THREAD_VARIABLES}
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    seen = json.loads(done.stdout)
    if not seen["first"]:
        pytest.skip("no OpenBLAS in /proc/self/maps")
    assert seen["error"] == "ValueError: none"
    assert seen["imported"] == seen["first"]
    if len(os.sched_getaffinity(0)) > 1:
        assert max(seen["first"].values()) > 1
    assert seen["started"] == dict.fromkeys(seen["first"], 1)
