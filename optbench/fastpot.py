"""Native pair kernels and the eonclient ``ext_pot`` bridge.

The C loops are the same arithmetic as the Python kernels, in the same
order. ``OPTBENCH_FASTPOT=0`` keeps the Python loops. A machine without
``gcc`` keeps them too. The bridge only forwards ``from_eon_to_extpot``
to the potential server; the server still owns the counter.
"""

from __future__ import annotations

import array
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

NATIVE = Path(__file__).resolve().parent / "native"
_KERNEL_SRC = NATIVE / "kernels.c"
_BRIDGE_SRC = NATIVE / "extpot_bridge.c"
_CFLAGS = [
    "-std=c11",
    "-O3",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fno-builtin",
    "-Wall",
    "-Wextra",
]


def enabled() -> bool:
    return os.environ.get("OPTBENCH_FASTPOT") != "0"


def _stamp(source: Path) -> str:
    data = source.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _build_dir() -> Path:
    root = Path(os.environ.get("OPTBENCH_FASTPOT_DIR", Path(tempfile.gettempdir()) / "optbench-fastpot"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _link(command: list[str], dest: Path, **run) -> subprocess.CompletedProcess:
    """Run ``command -o <scratch>`` and move a clean build onto ``dest``.

    Each call writes into its own directory beside ``dest``, so concurrent
    builders never share a file and ``dest`` only ever holds a complete
    build. ``run`` goes to ``subprocess.run``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dest.name}.", dir=dest.parent) as scratch:
        built = Path(scratch) / "out"
        result = subprocess.run([*command, "-o", str(built)], **run)
        if result.returncode == 0:
            built.chmod(built.stat().st_mode | 0o755)
            os.replace(built, dest)
    return result


def _compile(source: Path, dest: Path, extra: list[str]) -> bool:
    gcc = shutil.which("gcc")
    if gcc is None or not source.is_file():
        return False
    result = _link(
        [gcc, *_CFLAGS, *extra, str(source), "-lm"],
        dest,
        check=False,
        capture_output=True,
        text=True,
    )
    log = dest.with_suffix(dest.suffix + ".log")
    if result.returncode != 0:
        log.write_text(result.stderr, encoding="utf-8")
        return False
    log.unlink(missing_ok=True)
    return True


_KERNELS: ctypes.CDLL | None | bool = None


def kernels() -> ctypes.CDLL | None:
    """The shared pair library, or ``None`` when it cannot be built."""
    global _KERNELS
    if not enabled():
        return None
    if _KERNELS is False:
        return None
    if isinstance(_KERNELS, ctypes.CDLL):
        return _KERNELS
    dest = _build_dir() / f"kernels-{_stamp(_KERNEL_SRC)}.so"
    if not dest.is_file() and not _compile(_KERNEL_SRC, dest, ["-fPIC", "-shared"]):
        _KERNELS = False
        return None
    lib = ctypes.CDLL(str(dest))
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.optbench_morse.argtypes = [
        pointer,
        ctypes.c_int,
        pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        pointer,
        pointer,
    ]
    lib.optbench_morse.restype = ctypes.c_int
    lib.optbench_lj_cluster.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        pointer,
        pointer,
    ]
    lib.optbench_lj_cluster.restype = ctypes.c_int
    lib.optbench_lj_cut.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        pointer,
        pointer,
    ]
    lib.optbench_lj_cut.restype = ctypes.c_int
    _KERNELS = lib
    return lib


def bridge_executable() -> Path | None:
    """Path to the native ``ext_pot`` bridge, or ``None``."""
    if not enabled():
        return None
    dest = _build_dir() / f"extpot-bridge-{_stamp(_BRIDGE_SRC)}"
    if dest.is_file():
        return dest
    if not _compile(_BRIDGE_SRC, dest, []):
        return None
    return dest


def _pointer(buf: array.array):
    address, _length = buf.buffer_info()
    return ctypes.cast(address, ctypes.POINTER(ctypes.c_double))


def _positions(values) -> tuple[array.array, int]:
    count = len(values)
    if count % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    return array.array("d", (float(value) for value in values)), count // 3


def _forces(count: int) -> array.array:
    if count == 0:
        return array.array("d")
    return array.array("d", [0.0]) * count


def morse_forces(positions, lengths, well: float, width: float, equilibrium: float, cutoff: float, shift: float):
    """Energy and forces, or ``None`` when the native kernel is unavailable."""
    lib = kernels()
    if lib is None:
        return None
    pos, atoms = _positions(positions)
    out = _forces(atoms * 3)
    energy = ctypes.c_double()
    length_buf = None
    length_ptr = None
    if lengths is not None:
        length_buf = array.array("d", (float(value) for value in lengths))
        if len(length_buf) != 3:
            raise ValueError("morse lengths must be three floats")
        length_ptr = _pointer(length_buf)
    code = lib.optbench_morse(
        _pointer(pos) if atoms else None,
        atoms,
        length_ptr,
        float(well),
        float(width),
        float(equilibrium),
        float(cutoff),
        float(shift),
        ctypes.byref(energy),
        _pointer(out) if atoms else None,
    )
    if code != 0:
        return None
    return float(energy.value), tuple(out)


def lj_cluster_forces(positions, u0: float, psi: float, cutoff: float, cutoff_u: float):
    lib = kernels()
    if lib is None:
        return None
    pos, atoms = _positions(positions)
    out = _forces(atoms * 3)
    energy = ctypes.c_double()
    code = lib.optbench_lj_cluster(
        _pointer(pos) if atoms else None,
        atoms,
        float(u0),
        float(psi),
        float(cutoff),
        float(cutoff_u),
        ctypes.byref(energy),
        _pointer(out) if atoms else None,
    )
    if code != 0:
        return None
    return float(energy.value), tuple(out)


def lj_cut_forces(positions, epsilon: float, sigma: float, cutoff: float):
    lib = kernels()
    if lib is None:
        return None
    pos, atoms = _positions(positions)
    out = _forces(atoms * 3)
    energy = ctypes.c_double()
    code = lib.optbench_lj_cut(
        _pointer(pos) if atoms else None,
        atoms,
        float(epsilon),
        float(sigma),
        float(cutoff),
        ctypes.byref(energy),
        _pointer(out) if atoms else None,
    )
    if code != 0:
        return None
    return float(energy.value), tuple(out)
