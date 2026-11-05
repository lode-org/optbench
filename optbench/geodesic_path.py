"""Geodesic initial path for a band.

The path is geodesic-interpolate at the commit in ``codes/registry.toml``.
Smoothing uses the command-line defaults of that package: scaling 1.7,
distance cutoff 3, friction 0.01, tolerance 0.002, 15 iterations, and a
sweep when the system has more than 35 atoms. The smoother is geometric.
It does not evaluate the spec potential, and its steps are not force calls.

Endpoints stay on the reactant and the minimum-image product. A slab
with fixed atoms is rotated back onto those fixed atoms, then the fixed
atoms are put back on the reactant. A free cluster gets the same rigid
repair, spread along the path, so both endpoints match. The archive
writes Lennard-Jones atoms as ``A`` and platinum as ``Pt``. Neither
symbol is in the initialiser radius table, so those atoms use the
library fallback radius 1.5. Coordinates stay in the problem's units.
The nudge inside the bisection is seeded with 0 for every pair.

``OPTBENCH_GEODESIC_CACHE`` keeps each built path. Its key includes
numpy's and scipy's BLAS library, version, and thread count, because the
path changes in its last bits with the thread setup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from .pes import _mic

COMMIT = "476aa041f562d37d5fb4121b86817a52d75e4c5d"
VERSION = "1.0.0"
SCALING = 1.7
DIST_CUTOFF = 3.0
FRICTION = 1e-2
TOL = 2e-3
MAX_ITER = 15
MICRO_ITER = 20
SEED = 0


def symbols_for(kind: str, n_atoms: int) -> list[str]:
    """Element labels the initialiser uses for one potential."""
    if n_atoms < 1:
        raise ValueError("a path needs at least one atom")
    if kind in {"lj", "lennard-jones"}:
        return ["A"] * n_atoms
    if kind in {"morse_pt", "morse"}:
        return ["Pt"] * n_atoms
    raise ValueError(f"no geodesic symbols for {kind}")


def geodesic_free(
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    n_interior: int,
    box: np.ndarray | None,
    symbols: list[str],
    *,
    seed: int = SEED,
    info: dict | None = None,
) -> np.ndarray:
    """Free coordinates along the geodesic, shape ``(n_interior + 2, n_free * 3)``."""
    frames = geodesic_positions(
        reactant,
        product,
        symbols,
        n_interior,
        box,
        free=free,
        seed=seed,
        info=info,
    )
    mask = np.asarray(free, dtype=bool)
    return np.vstack([frame[mask].reshape(-1) for frame in frames])


def geodesic_positions(
    reactant: np.ndarray,
    product: np.ndarray,
    symbols: list[str],
    n_interior: int,
    box: np.ndarray | None,
    *,
    free: np.ndarray | None = None,
    seed: int = SEED,
    info: dict | None = None,
) -> np.ndarray:
    """Cartesian images. The two ends are the reactant and the minimum-image product."""
    if n_interior < 1:
        raise ValueError("a band needs at least one interior image")
    reactant = np.array(reactant, dtype=np.float64, copy=True)
    product = np.array(product, dtype=np.float64, copy=True)
    if reactant.shape != product.shape or reactant.ndim != 2 or reactant.shape[1] != 3:
        raise ValueError("endpoints must be (n_atoms, 3) and the same shape")
    if len(symbols) != reactant.shape[0]:
        raise ValueError("symbol count does not match the atoms")
    mask = None if free is None else np.asarray(free, dtype=bool)
    if mask is not None and mask.shape != (reactant.shape[0],):
        raise ValueError("free mask does not match the atoms")
    target = _minimum_image(reactant, product, box, mask)
    setup = _blas_setup() if _cache_dir() is not None else None
    cached = _read_cache(reactant, target, symbols, n_interior, box, mask, seed, setup)
    if cached is not None:
        if info is not None:
            info["seconds"] = 0.0
            info["cached"] = True
        return cached
    started = time.perf_counter()
    path = _interpolate(reactant, target, symbols, n_interior, seed)
    anchored = mask is not None and int(np.count_nonzero(~mask)) >= 3
    if anchored:
        restored, product_rmsd = _anchor_fixed(path, reactant, target, mask)
    else:
        restored, product_rmsd = _land_endpoints(path, reactant, target)
    restored = _snap_ends(restored, reactant, target, mask)
    elapsed = time.perf_counter() - started
    _write_cache(restored, reactant, target, symbols, n_interior, box, mask, seed, setup)
    if info is not None:
        info["seconds"] = elapsed
        info["cached"] = False
        info["sweep"] = reactant.shape[0] > 35
        info["product_rmsd"] = product_rmsd
    return restored


def _minimum_image(
    reactant: np.ndarray,
    product: np.ndarray,
    box: np.ndarray | None,
    free: np.ndarray | None,
) -> np.ndarray:
    lengths = None if box is None else np.asarray(box, dtype=np.float64).reshape(-1)
    if lengths is not None and lengths.size != 3:
        raise ValueError("box must be three lengths")
    target = reactant + _mic(product - reactant, lengths)
    if free is not None:
        target[~free] = reactant[~free]
    return target


def _interpolate(
    reactant: np.ndarray,
    product: np.ndarray,
    symbols: list[str],
    n_interior: int,
    seed: int,
) -> np.ndarray:
    try:
        from geodesic_interpolate.geodesic import Geodesic
        from geodesic_interpolate.interpolation import redistribute
    except ImportError as exc:
        raise ImportError(
            "geodesic-interpolate is not installed. "
            f"Build commit {COMMIT} from https://github.com/virtualzx-nad/geodesic-interpolate."
        ) from exc
    loggers = [
        logging.getLogger(name)
        for name in (
            "geodesic_interpolate",
            "geodesic_interpolate.interpolation",
            "geodesic_interpolate.geodesic",
            "geodesic_interpolate.coord_utils",
        )
    ]
    saved = [(item, item.disabled) for item in loggers]
    for item in loggers:
        item.disabled = True
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        raw = redistribute(list(symbols), [reactant.copy(), product.copy()], n_interior + 2, tol=TOL * 5)
        smoother = Geodesic(
            list(symbols),
            raw,
            SCALING,
            threshold=DIST_CUTOFF,
            friction=FRICTION,
            log_level=logging.DEBUG,
        )
        if len(symbols) > 35:
            smoother.sweep(tol=TOL, max_iter=MAX_ITER, micro_iter=MICRO_ITER)
        else:
            smoother.smooth(tol=TOL, max_iter=MAX_ITER)
        path = np.asarray(smoother.path, dtype=np.float64)
    finally:
        np.random.set_state(state)
        for item, disabled in saved:
            item.disabled = disabled
    if path.shape != (n_interior + 2, reactant.shape[0], 3):
        raise RuntimeError(f"geodesic path shape {path.shape} is not {(n_interior + 2, reactant.shape[0], 3)}")
    return path


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid map ``mobile @ rot + t`` onto ``reference``."""
    center_m = mobile.mean(axis=0)
    center_r = reference.mean(axis=0)
    cov = (mobile - center_m).T @ (reference - center_r)
    left, _, right = np.linalg.svd(cov)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        left = left.copy()
        left[:, -1] *= -1.0
    rot = left @ right
    shift = center_r - center_m @ rot
    return rot, shift


def _fractional_rotation(rot: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return np.eye(3)
    if fraction >= 1.0:
        return rot
    from scipy.spatial.transform import Rotation

    vector = Rotation.from_matrix(rot).as_rotvec()
    return Rotation.from_rotvec(vector * float(fraction)).as_matrix()


def _land_endpoints(path: np.ndarray, reactant: np.ndarray, product: np.ndarray) -> tuple[np.ndarray, float]:
    """Put a free-cluster path back between the original endpoints.

    Each image stays rigid. The residual rotation that the initialiser
    applied to the product is spread along the band.
    """
    rot0, shift0 = _kabsch(path[0], reactant)
    restored = np.stack([frame @ rot0 + shift0 for frame in path])
    rot, shift = _kabsch(restored[-1], product)
    origin = reactant.mean(axis=0)
    drift = shift - origin + origin @ rot
    count = len(restored)
    out = np.empty_like(restored)
    for index, frame in enumerate(restored):
        fraction = index / (count - 1)
        turn = _fractional_rotation(rot, fraction)
        out[index] = (frame - origin) @ turn + origin + fraction * drift
    product_rmsd = float(np.sqrt(np.mean((out[-1] - product) ** 2)))
    return out, product_rmsd


def _anchor_fixed(
    path: np.ndarray,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Rotate every image so its fixed atoms sit on the reactant slab."""
    fixed = ~free
    out = np.empty_like(path)
    for index, frame in enumerate(path):
        rot, shift = _kabsch(frame[fixed], reactant[fixed])
        moved = frame @ rot + shift
        moved[fixed] = reactant[fixed]
        out[index] = moved
    product_rmsd = float(np.sqrt(np.mean((out[-1][free] - product[free]) ** 2)))
    return out, product_rmsd


def _snap_ends(
    path: np.ndarray,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray | None,
) -> np.ndarray:
    out = np.array(path, dtype=np.float64, copy=True)
    out[0] = reactant
    out[-1] = product
    if free is not None:
        fixed = ~free
        out[:, fixed] = reactant[fixed]
    return out


def _cache_dir() -> Path | None:
    raw = os.environ.get("OPTBENCH_GEODESIC_CACHE", "/tmp/optbench-geodesic-cache")
    if raw in {"", "0", "off"}:
        return None
    return Path(raw)


def _blas_setup() -> str | None:
    """numpy's and scipy's BLAS pools: library, version, and threads.

    The path changes in its last bits with the thread counts, so a cached
    path is reused only under the pools that built it. ``None`` means the
    pools could not be read, and then the cache is not used.
    """
    from .blas_threads import numpy_and_scipy

    pools = numpy_and_scipy()
    owners = {pool["owner"] for pool in pools}
    if not ({"numpy", "scipy"} <= owners or "system" in owners):
        return None
    return json.dumps(pools, sort_keys=True)


def _cache_key(
    reactant: np.ndarray,
    product: np.ndarray,
    symbols: list[str],
    n_interior: int,
    box: np.ndarray | None,
    free: np.ndarray | None,
    seed: int,
    setup: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"geodesic-path-v2")
    digest.update(COMMIT.encode())
    digest.update(setup.encode())
    digest.update(str(n_interior).encode())
    digest.update(str(int(seed)).encode())
    digest.update(",".join(symbols).encode())
    digest.update(np.ascontiguousarray(reactant, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(product, dtype=np.float64).tobytes())
    if box is None:
        digest.update(b"no-box")
    else:
        digest.update(np.ascontiguousarray(box, dtype=np.float64).tobytes())
    if free is None:
        digest.update(b"all-free")
    else:
        digest.update(np.ascontiguousarray(free, dtype=bool).tobytes())
    try:
        import geodesic_interpolate.geodesic as geodesic_module

        digest.update(Path(geodesic_module.__file__).read_bytes())
    except (ImportError, OSError):
        digest.update(b"missing")
    return digest.hexdigest()


def _read_cache(
    reactant: np.ndarray,
    product: np.ndarray,
    symbols: list[str],
    n_interior: int,
    box: np.ndarray | None,
    free: np.ndarray | None,
    seed: int,
    setup: str | None,
) -> np.ndarray | None:
    directory = _cache_dir()
    if directory is None or setup is None:
        return None
    path = directory / (_cache_key(reactant, product, symbols, n_interior, box, free, seed, setup) + ".npy")
    if not path.is_file():
        return None
    found = np.load(path)
    if found.shape != (n_interior + 2, reactant.shape[0], 3):
        return None
    return found


def _write_cache(
    frames: np.ndarray,
    reactant: np.ndarray,
    product: np.ndarray,
    symbols: list[str],
    n_interior: int,
    box: np.ndarray | None,
    free: np.ndarray | None,
    seed: int,
    setup: str | None,
) -> None:
    directory = _cache_dir()
    if directory is None or setup is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (_cache_key(reactant, product, symbols, n_interior, box, free, seed, setup) + ".npy")
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, np.asarray(frames, dtype=np.float64))
    temporary.replace(path)
