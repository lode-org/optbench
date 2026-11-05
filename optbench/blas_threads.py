"""BLAS thread variables and the thread counts BLAS actually uses.

OpenBLAS reads ``OPENBLAS_NUM_THREADS``, then ``OMP_NUM_THREADS``, once,
when the library is loaded. The numpy and scipy wheels each bundle their
own OpenBLAS, so their two pools differ when one of them loads before the
variables are set. The asv benchmark server imports numpy (asv-runner
loads pympler, which imports numpy) before any benchmark module. A
module-level ``os.environ`` setting then reaches scipy's pool and not
numpy's. ``optbench.asv_launch`` sets ``PINNED`` in the process that
starts asv, which is the environment asv copies into its server.

The pool sizes come from the loaded libraries, so they are what ran, not
what was asked for. They are read from ``/proc/self/maps``, so the lists
are empty on a system without it.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# One thread per process is what the committed campaign and asv rows ran with.
PINNED = dict.fromkeys(VARIABLES, "1")

_THREADS = (
    "scipy_openblas_get_num_threads64_",
    "scipy_openblas_get_num_threads",
    "openblas_get_num_threads64_",
    "openblas_get_num_threads",
)
_CONFIG = (
    "scipy_openblas_get_config64_",
    "scipy_openblas_get_config",
    "openblas_get_config64_",
    "openblas_get_config",
)


def pin(environ: dict | None = None) -> dict:
    """Set every ``PINNED`` variable, replacing values already there."""
    target = os.environ if environ is None else environ
    target.update(PINNED)
    return target


def variables() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in VARIABLES}


def _mapped() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    found: list[str] = []
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if "openblas" in os.path.basename(path).lower() and path not in found and os.path.isfile(path):
            found.append(path)
    return found


def _owner(path: str) -> str:
    """The package whose wheel bundled the library, or ``system``."""
    parent = Path(path).parent
    if parent.name.endswith(".libs"):
        return parent.name[: -len(".libs")]
    if parent.name == ".dylibs":
        return parent.parent.name
    return "system"


def _first(library: ctypes.CDLL, names: tuple[str, ...], restype):
    for name in names:
        if hasattr(library, name):
            function = getattr(library, name)
            function.restype = restype
            function.argtypes = []
            return function()
    return None


def loaded() -> list[dict]:
    """Each OpenBLAS in this process, with its owner, file, version, and threads."""
    pools = []
    for path in _mapped():
        library = ctypes.CDLL(path)
        threads = _first(library, _THREADS, ctypes.c_int)
        if threads is None:
            continue
        config = _first(library, _CONFIG, ctypes.c_char_p)
        pools.append(
            {
                "owner": _owner(path),
                "library": os.path.basename(path),
                "version": " ".join(config.decode("utf-8", "replace").split()) if config else None,
                "threads": int(threads),
            }
        )
    return sorted(pools, key=lambda pool: (pool["owner"], pool["library"]))


def numpy_and_scipy() -> list[dict]:
    """The pools after numpy and scipy.linalg are imported.

    Importing them maps both bundled libraries, so neither pool is missed
    because nothing has used it yet.
    """
    import numpy  # noqa: F401

    try:
        import scipy.linalg  # noqa: F401
    except ImportError:
        pass
    return loaded()


def report() -> dict:
    """The variables and the pools of this process, for provenance."""
    return {"variables": variables(), "blas": loaded()}


if __name__ == "__main__":
    import json
    import sys

    # The launcher passes this source to ``python -c``, so the interpreter it
    # checks needs numpy and not optbench.
    if "--asv-order" in sys.argv[1:]:
        try:
            import asv_runner.benchmarks  # noqa: F401
        except ImportError:
            pass
    print(json.dumps(numpy_and_scipy()))
