"""GMIN MYMYLBFGS through ``libgminlbfgs.so``.

The shared library is ``source/mymylbfgs.F90`` from the Wales tarball.
``POTENTIAL`` calls back into Python. The gradient passed in is the
gradient GMIN steps on. RMS is the root-mean-square of that gradient,
``sqrt(sum(g^2) / n)``, with a floor of ``1e-100``. The problem stop is
the Euclidean norm, so the RMS tolerance is that norm divided by
``sqrt(n)``.

``hold`` is 2 when the harness stop is met and 1 otherwise. The step
uses the gradient either way. A ``ForceCallCap`` returns status 1, which
sets ``FTEST`` and leaves that call uncounted.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
from collections.abc import Callable, Sequence
from ctypes import POINTER, c_double, c_int, c_void_p
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap
from optbench.wales import snapshot

DEFAULTS = {
    "m": 4,
    "maxbfgs": 0.4,
    "dguess": 0.1,
    "maxerise": 1.0e-10,
    "maxefall": -1.0e300,
}

_EVAL = ctypes.CFUNCTYPE(c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_int))
_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_STATE: dict = {"evaluate": None, "ready": None, "gtol": None, "error": None, "capped": False}


class LbfgsResult:
    """One quench. ``capped`` is a callback that refused the next search call."""

    def __init__(self, coords: np.ndarray, energy: float, mflag: bool, itdone: int, info: int, capped: bool):
        self.coords = coords
        self.energy = float(energy)
        self.mflag = bool(mflag)
        self.itdone = int(itdone)
        self.info = int(info)
        self.capped = bool(capped)


def library_path() -> Path | None:
    """The shared library, or ``None`` when this machine has not built it."""
    override = os.environ.get("OPTBENCH_GMIN_LBFGS")
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.append(Path(__file__).resolve().parents[1] / "codes" / "gmin" / "libgminlbfgs.so")
    candidates.append(Path("/opt/gmin/lib/libgminlbfgs.so"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def available() -> bool:
    return library_path() is not None


def _load() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = library_path()
    if path is None:
        raise FileNotFoundError(
            "libgminlbfgs.so is not installed. codes/gmin/build.sh writes /opt/gmin/lib/libgminlbfgs.so."
        )
    lib = ctypes.CDLL(str(path))
    if not hasattr(lib, "gmin_tarball_sha256"):
        raise RuntimeError(f"{path} has no gmin_tarball_sha256. Rebuild it with codes/gmin/build.sh.")
    lib.gmin_revision.argtypes = []
    lib.gmin_revision.restype = ctypes.c_char_p
    lib.gmin_tarball_sha256.argtypes = []
    lib.gmin_tarball_sha256.restype = ctypes.c_char_p
    lib.gmin_set_eval.argtypes = [c_void_p]
    lib.gmin_set_eval.restype = None
    lib.gmin_lbfgs_run.argtypes = [
        c_int,
        POINTER(c_double),
        c_int,
        c_double,
        c_int,
        c_double,
        c_double,
        c_double,
        c_double,
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_int),
        POINTER(c_int),
    ]
    lib.gmin_lbfgs_run.restype = None
    _LIBRARY = lib
    return lib


def revision() -> str:
    raw = _load().gmin_revision()
    if not raw:
        raise RuntimeError("gmin_revision returned an empty string")
    return raw.decode("ascii")


def build() -> dict:
    """``wales_sha256`` and ``wales_svnrev`` of the loaded library.

    ``codes/gmin/build.sh`` compiles the tarball sha256 and the tree's
    ``SVNREV`` into the library, and ``driver.f90`` returns them.
    """
    raw = _load().gmin_tarball_sha256()
    return snapshot(raw.decode("ascii") if raw else "", revision())


@_EVAL
def _callback(n, x_ptr, energy_ptr, grad_ptr, hold_ptr):
    try:
        coords = np.ctypeslib.as_array(x_ptr, shape=(n,)).copy()
        evaluate = _STATE["evaluate"]
        energy, gradient = evaluate(coords)
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if grad.size != n:
            raise ValueError(f"gradient length {grad.size} is not {n}")
        energy_ptr[0] = float(energy)
        np.ctypeslib.as_array(grad_ptr, shape=(n,))[:] = grad
        ready = _STATE["ready"]
        gtol = _STATE["gtol"]
        if ready is not None:
            met = bool(ready())
        else:
            met = float(np.linalg.norm(grad)) <= float(gtol)
        hold_ptr[0] = 2 if met else 1
        return 0
    except ForceCallCap:
        _STATE["capped"] = True
        return 1
    except Exception as exc:
        _STATE["error"] = exc
        return 2


def minimize(
    coords: Sequence[float],
    evaluate: Callable[[Sequence[float]], tuple[float, Sequence[float]]],
    *,
    gtol: float,
    cap: int,
    m: int = DEFAULTS["m"],
    maxbfgs: float = DEFAULTS["maxbfgs"],
    dguess: float = DEFAULTS["dguess"],
    maxerise: float = DEFAULTS["maxerise"],
    maxefall: float = DEFAULTS["maxefall"],
    ready: Callable[[], bool] | None = None,
) -> LbfgsResult:
    """Run one quench. ``gtol`` is the Euclidean norm of the gradient GMIN sees.

    ``ready`` replaces that test for the return. The noise driver passes
    the energy-window stop. ``cap`` is ``ITMAX`` and the harness budget.
    """
    lib = _load()
    x = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1).copy()
    n = int(x.size)
    if n <= 0 or n % 3 != 0:
        raise ValueError(f"GMIN LBFGS needs a positive multiple of 3 coordinates, got {n}")
    if int(m) <= 0 or int(cap) <= 0:
        raise ValueError("m and cap must be positive")
    if float(gtol) <= 0.0:
        raise ValueError("gtol must be positive")
    eps = float(gtol) / math.sqrt(n)
    mflag = c_int(0)
    energy = c_double(0.0)
    itdone = c_int(0)
    info = c_int(0)
    with _LOCK:
        _STATE["evaluate"] = evaluate
        _STATE["ready"] = ready
        _STATE["gtol"] = float(gtol)
        _STATE["error"] = None
        _STATE["capped"] = False
        lib.gmin_set_eval(ctypes.cast(_callback, c_void_p))
        lib.gmin_lbfgs_run(
            c_int(n),
            x.ctypes.data_as(POINTER(c_double)),
            c_int(int(m)),
            c_double(eps),
            c_int(int(cap)),
            c_double(float(maxbfgs)),
            c_double(float(dguess)),
            c_double(float(maxerise)),
            c_double(float(maxefall)),
            ctypes.byref(mflag),
            ctypes.byref(energy),
            ctypes.byref(itdone),
            ctypes.byref(info),
        )
        error = _STATE["error"]
        capped = bool(_STATE["capped"])
        _STATE["evaluate"] = None
        _STATE["ready"] = None
    if error is not None:
        raise RuntimeError(f"GMIN potential failed: {type(error).__name__}: {error}") from error
    return LbfgsResult(x, float(energy.value), mflag.value == 1, int(itdone.value), int(info.value), capped)
