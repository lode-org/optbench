"""Run every OpenBLAS loaded in this process on one thread.

OpenBLAS reads ``OPENBLAS_NUM_THREADS`` once, when it loads. asv's
runner imports NumPy before it imports any benchmark, so under asv the
variable that ``ase_sella`` sets comes too late for NumPy's OpenBLAS,
which then starts one thread per CPU. A threaded BLAS call splits its
sums by thread, so a search would depend on the host's CPU count. One
thread is a count every host has, and the committed ASE rows were
measured with it.

The libraries are found in ``/proc/self/maps``; where that file does not
exist nothing is pinned.
"""

from __future__ import annotations

import ctypes

SETTERS = (
    "scipy_openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads",
    "openblas_set_num_threads64_",
    "openblas_set_num_threads",
)
GETTERS = tuple(name.replace("_set_", "_get_") for name in SETTERS)


def loaded_openblas() -> list[str]:
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            fields = [line.split(maxsplit=5) for line in maps]
    except OSError:
        return []
    paths = {field[5].strip() for field in fields if len(field) == 6}
    return sorted(path for path in paths if "openblas" in path.rsplit("/", 1)[-1].lower())


def _symbol(path: str, names: tuple[str, ...]):
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return None
    for name in names:
        function = getattr(library, name, None)
        if function is not None:
            return function
    return None


def threads() -> dict[str, int]:
    counts = {}
    for path in loaded_openblas():
        getter = _symbol(path, GETTERS)
        if getter is not None:
            getter.restype = ctypes.c_int
            counts[path] = int(getter())
    return counts


def pin_one_thread() -> dict[str, int]:
    for path in loaded_openblas():
        setter = _symbol(path, SETTERS)
        if setter is not None:
            setter.argtypes = [ctypes.c_int]
            setter(1)
    return threads()
