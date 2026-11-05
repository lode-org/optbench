"""LAMMPS climbing-image NEB on the harness pair potentials.

The binary is the source build named by ``LAMMPS_BIN`` or
``/tmp/lammps/install/bin/lmp``. Lennard-Jones is ``pair_style lj/cut
40.0`` with no shift. Morse platinum is ``pair_style morse 9.5`` with
``pair_modify shift yes``. The slab neighbor skin is zero: the cell is
only slightly longer than twice the cutoff.

``fix neb`` uses Kspring 1.0 and the parallel style recorded on the
template. Climbing is the second stage of the ``neb`` command. The
timestep is the unit default, 0.005 for ``units lj`` and 0.001 for
``units metal``. ``min_modify dmax`` is the problem ``max_move``. FIRE's template
leaves one fifth of the cap unused. A velocity reset evaluates the
pair style again, and that compute has to stay inside the cap.

A force call is one pair compute. Each stage logs ``neval`` and omits
the setup compute. A stop other than ``max iterations`` computes the
forces once more, and that compute is also absent from ``neval``. The
start total sums both stages on every replica, including the
endpoints. The harness then judges the climbing replica: free-atom
gradient at the problem norm, and Hessian index 1. Finite-difference
probes are ``verify_calls`` and are not part of the search cost.

A search without ``work`` runs in a temporary directory that is removed
on return. ``OPTBENCH_LAMMPS_KEEP=1`` or ``cleanup=False`` keeps it and
prints its path on stderr.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .failures import classify_failure
from .pes import evaluate, load_pairs
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance, write_starts

COMMIT = "8de817dd79bfe4525d5d39246a212d833e6dee07"
VERSION = "30 Sep 2026"
CODE_ID = "lammpsneb"
_FD_STEP = 1e-5
_LJ_PAD = 100.0
_MORSE_CUTOFF = 9.5
_TIMESTEP = {"lj": 0.005, "morse_pt": 0.001}
_CLIMBING = re.compile(r"Climbing replica = (\d+)")
_DEFAULT_BIN = Path("/tmp/lammps/install/bin/lmp")


def lammps_executable() -> Path | None:
    """The pinned LAMMPS binary, or None when this machine has no build."""
    env = os.environ.get("LAMMPS_BIN")
    if env:
        path = Path(env)
        return path if path.is_file() else None
    if _DEFAULT_BIN.is_file():
        return _DEFAULT_BIN
    found = shutil.which("lmp")
    return Path(found) if found else None


def iteration_budget(n_images: int, cap: int, reserve: float = 0.0) -> tuple[int, int]:
    """Largest equal N1, N2 whose pair-compute bound stays under ``cap``.

    The bound allows one setup and one early-stop compute on each of
    the two stages, on every replica: ``n_images * (N1 + N2 + 4)``.
    ``reserve`` shrinks that bound. FIRE spends it on the extra pair
    compute that a velocity reset adds inside ``neval``.
    """
    if n_images < 2:
        raise ValueError("NEB needs at least two replicas")
    if not 0.0 <= float(reserve) < 1.0:
        raise ValueError(f"call reserve {reserve} is outside [0, 1)")
    usable = int((int(cap) - 1) * (1.0 - float(reserve)))
    room = usable // int(n_images) - 4
    if room < 2:
        raise ValueError(f"cap {cap} cannot fund two stages on {n_images} replicas")
    n1 = room // 2
    n2 = room - n1
    if n_images * (n1 + n2 + 4) > usable:
        raise ValueError("iteration budget meets the force-call cap")
    return n1, n2


def minimization_blocks(text: str) -> list[tuple[str, int]]:
    """``(stopping criterion, neval)`` for each minimization stats block."""
    blocks: list[tuple[str, int]] = []
    stop: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Stopping criterion = "):
            stop = line.split("=", 1)[1].strip()
        elif line.startswith("Iterations, force evaluations = "):
            if stop is None:
                raise ValueError("force evaluations appear before a stopping criterion")
            neval = int(line.split("=", 1)[1].split()[1])
            blocks.append((stop, neval))
            stop = None
    return blocks


def calls_from_blocks(blocks: list[tuple[str, int]]) -> int:
    """Pair computes implied by minimization stats.

    Setup is one compute. ``neval`` counts the iterates. A stop other
    than max iterations adds the final compute that ``Min::run`` makes
    after the loop.
    """
    total = 0
    for stop, neval in blocks:
        extra = 0 if stop == "max iterations" else 1
        total += 1 + int(neval) + extra
    return total


def minimization_calls(text: str) -> int:
    """Pair computes recorded in one replica log."""
    return calls_from_blocks(minimization_blocks(text))


def _registry(root: Path) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == CODE_ID:
            return row
    raise ValueError("codes/registry.toml has no lammpsneb entry")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the lammpsneb registry")
    if template.get("engine") != "lammps-neb":
        raise ValueError(f"{template['id']} engine is {template.get('engine')}")
    band = template["band"]
    if not bool(band["climbing"]):
        raise ValueError(f"{template['id']} is not a climbing-image run")
    style = str(band["min_style"])
    if style not in {"quickmin", "fire"}:
        raise ValueError(f"{template['id']} min_style {style}")
    parallel = str(band["parallel"])
    if parallel not in {"neigh", "ideal", "equal"}:
        raise ValueError(f"{template['id']} parallel spring {parallel}")
    problem_band = problem["band"]
    convergence = problem["convergence"]
    kind = str(problem_band["potential"])
    if kind not in _TIMESTEP:
        raise ValueError(f"{problem['id']} potential {kind}")
    n_interior = int(problem_band["interior_images"])
    return {
        "potential": kind,
        "n_interior": n_interior,
        "n_images": n_interior + 2,
        "max_move": float(problem_band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": problem["geometry"]["kind"],
        "climbing": True,
        "min_style": style,
        "spring_k": float(band["spring_k"]),
        "parallel": parallel,
        "call_reserve": float(band.get("call_reserve", 0.0)),
        "timestep": _TIMESTEP[kind],
        "units": "lj" if kind == "lj" else "metal",
    }


def _fmt(value: float) -> str:
    return format(float(value), ".16g")


def _box_bounds(kind: str, reactant: np.ndarray, product: np.ndarray, box: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if kind == "lj":
        stacked = np.vstack([np.asarray(reactant, dtype=np.float64), np.asarray(product, dtype=np.float64)])
        lo = stacked.min(axis=0) - _LJ_PAD
        hi = stacked.max(axis=0) + _LJ_PAD
        return lo, hi
    if box is None:
        raise ValueError("morse_pt needs the orthorhombic cell")
    lengths = np.asarray(box, dtype=np.float64)
    half = float(np.min(lengths)) / 2.0
    if _MORSE_CUTOFF >= half:
        raise ValueError(f"Morse cutoff {_MORSE_CUTOFF} does not fit the cell {lengths.tolist()}")
    return np.zeros(3, dtype=np.float64), lengths


def _write_data(path: Path, coords: np.ndarray, free: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
    xyz = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(free, dtype=bool)
    n_types = 1 if bool(np.all(mask)) else 2
    lines = [
        "LAMMPS data",
        "",
        f"{xyz.shape[0]} atoms",
        f"{n_types} atom types",
        "",
        f"{_fmt(lo[0])} {_fmt(hi[0])} xlo xhi",
        f"{_fmt(lo[1])} {_fmt(hi[1])} ylo yhi",
        f"{_fmt(lo[2])} {_fmt(hi[2])} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for type_id in range(1, n_types + 1):
        lines.append(f"{type_id} 1.0")
    lines.extend(["", "Atoms", ""])
    for index, (x, y, z) in enumerate(xyz, start=1):
        type_id = 1 if mask[index - 1] else 2
        lines.append(f"{index} {type_id} {_fmt(x)} {_fmt(y)} {_fmt(z)}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_final(path: Path, product: np.ndarray, free: np.ndarray) -> None:
    xyz = np.asarray(product, dtype=np.float64)
    mask = np.asarray(free, dtype=bool)
    rows = [str(int(np.count_nonzero(mask)))]
    for index, (x, y, z) in enumerate(xyz, start=1):
        if not mask[index - 1]:
            continue
        rows.append(f"{index} {_fmt(x)} {_fmt(y)} {_fmt(z)}")
    rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


def _pair_lines(kind: str) -> list[str]:
    if kind == "lj":
        return [
            "pair_style lj/cut 40.0",
            "pair_coeff * * 1.0 1.0",
            "pair_modify shift no",
            "neighbor 0.3 bin",
        ]
    if kind == "morse_pt":
        return [
            "pair_style morse 9.5",
            "pair_coeff * * 0.7102 1.6047 2.897",
            "pair_modify shift yes",
            "neighbor 0.0 multi",
            "comm_modify mode multi",
        ]
    raise ValueError(f"unknown potential {kind}")


def render_neb_input(settings: dict, *, n1: int, n2: int, frozen: bool) -> str:
    """The NEB input script. ``write_dump`` follows the ``neb`` command."""
    kind = settings["potential"]
    units = settings["units"]
    boundary = "f f f" if kind == "lj" else "p p p"
    n_images = int(settings["n_images"])
    lines = [
        f"units {units}",
        "atom_style atomic",
        "atom_modify map array",
        f"boundary {boundary}",
        "read_data start.data",
        *_pair_lines(kind),
        "group neb type 1",
    ]
    if frozen:
        lines.append("group frozen type 2")
    lines.extend(
        [
            "thermo_style custom step pe",
            "thermo_modify norm no",
            "thermo 100000",
            f"min_style {settings['min_style']}",
            f"min_modify dmax {_fmt(settings['max_move'])} norm two",
            f"fix neb1 neb neb {_fmt(settings['spring_k'])} parallel {settings['parallel']}",
        ]
    )
    if frozen:
        lines.append("fix freeze frozen setforce 0.0 0.0 0.0")
    lines.extend(
        [
            "partition yes 1 fix hold neb setforce 0.0 0.0 0.0",
            f"partition yes {n_images} fix hold neb setforce 0.0 0.0 0.0",
            f"neb 0.0 {_fmt(settings['force_norm'])} {int(n1)} {int(n2)} 1 final final.neb",
            "variable rep equal part",
            'write_dump all custom final.${rep}.dump id type xu yu zu modify format line "%d %d %.17g %.17g %.17g"',
            "",
        ]
    )
    return "\n".join(lines)


def _read_dump(path: Path, columns: tuple[str, ...]) -> dict[int, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms: dict[int, np.ndarray] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("ITEM: ATOMS"):
            header = line.split()[2:]
            index += 1
            while index < len(lines) and not lines[index].startswith("ITEM:"):
                parts = lines[index].split()
                row = {name: parts[pos] for pos, name in enumerate(header)}
                atoms[int(row["id"])] = np.array([float(row[name]) for name in columns], dtype=np.float64)
                index += 1
            continue
        index += 1
    return atoms


def _coords_from_dump(path: Path, n_atoms: int) -> np.ndarray:
    atoms = _read_dump(path, ("xu", "yu", "zu"))
    if len(atoms) != n_atoms or set(atoms) != set(range(1, n_atoms + 1)):
        raise ValueError(f"{path.name} has {len(atoms)} atoms, the data file has {n_atoms}")
    return np.vstack([atoms[ident] for ident in range(1, n_atoms + 1)])


def _last_pe(text: str) -> float:
    value = None
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) != 2:
            continue
        try:
            int(parts[0])
            value = float(parts[1])
        except ValueError:
            continue
    if value is None:
        raise ValueError("LAMMPS log has no potential energy")
    return value


def _error_line(text: str) -> str:
    for raw in text.splitlines():
        if "ERROR:" in raw or "Lost atoms" in raw:
            return raw.strip()
    return ""


def _mpirun() -> Path:
    found = shutil.which("mpirun")
    if not found:
        raise FileNotFoundError("mpirun is not on PATH")
    return Path(found)


def _run_process(cmd: list[str], work: Path, timeout: float) -> int | None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    with (work / "stderr.txt").open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=work,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return int(proc.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=60)
            return None


def _launch_neb(binary: Path, work: Path, n_images: int, timeout: float) -> int | None:
    cmd = [
        str(_mpirun()),
        "--oversubscribe",
        "--bind-to",
        "none",
        "-np",
        str(n_images),
        str(binary),
        "-partition",
        f"{n_images}x1",
        "-in",
        "in.neb",
        "-log",
        "log.lammps",
        "-screen",
        "none",
    ]
    return _run_process(cmd, work, timeout)


def probe_pair(
    kind: str,
    coords: np.ndarray,
    box: np.ndarray | None,
    *,
    binary: Path | None = None,
    timeout: float = 120.0,
) -> tuple[float, np.ndarray]:
    """Energy and ``dE/dx`` from one LAMMPS ``run 0`` on ``coords``."""
    executable = binary or lammps_executable()
    if executable is None:
        raise FileNotFoundError("LAMMPS binary is not available")
    xyz = np.asarray(coords, dtype=np.float64)
    free = np.ones(xyz.shape[0], dtype=bool)
    lo, hi = _box_bounds(kind, xyz, xyz, box)
    with tempfile.TemporaryDirectory(prefix="lammps-probe-") as folder:
        work = Path(folder)
        _write_data(work / "start.data", xyz, free, lo, hi)
        units = "lj" if kind == "lj" else "metal"
        boundary = "f f f" if kind == "lj" else "p p p"
        script = "\n".join(
            [
                f"units {units}",
                "atom_style atomic",
                "atom_modify map array",
                f"boundary {boundary}",
                "read_data start.data",
                *_pair_lines(kind),
                "thermo_style custom step pe",
                "thermo_modify norm no format float %.16g",
                "thermo 1",
                "run 0",
                'write_dump all custom forces.dump id type fx fy fz modify format line "%d %d %.17g %.17g %.17g"',
                "",
            ]
        )
        (work / "in.probe").write_text(script, encoding="utf-8")
        rc = _run_process(
            [str(executable), "-in", "in.probe", "-log", "log.lammps", "-screen", "none"],
            work,
            timeout,
        )
        log = (work / "log.lammps").read_text(encoding="utf-8", errors="replace")
        if rc != 0:
            raise RuntimeError(_error_line(log) or f"LAMMPS probe exited {rc}")
        energy = _last_pe(log)
        atoms = _read_dump(work / "forces.dump", ("fx", "fy", "fz"))
        if set(atoms) != set(range(1, xyz.shape[0] + 1)):
            raise RuntimeError("LAMMPS force dump does not list every atom")
        force = np.vstack([atoms[ident] for ident in range(1, xyz.shape[0] + 1)])
    return energy, -force


def _blank(
    name: str,
    *,
    calls: int = 0,
    failure: str = "crash",
    reasons: list[str] | None = None,
    error: str = "",
    force: float | None = None,
    energy: float | None = None,
    hessian_index: int | None = None,
    verify_calls: int = 0,
    climbing: int | None = None,
    n1: int = 0,
    n2: int = 0,
    n_images: int = 0,
    returncode: int | None = None,
) -> dict:
    reason_list = list(reasons or [failure])
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": failure,
        "reasons": reason_list,
        "energy": energy,
        "force_norm": force,
        "hessian_index": hessian_index,
        "converged": False,
        "climbing_replica": climbing,
        "n1": int(n1),
        "n2": int(n2),
        "n_images": int(n_images),
        "returncode": returncode,
        "error": error,
    }


def _measured(kind: str, coords: np.ndarray, free: np.ndarray, box: np.ndarray | None) -> tuple[float, float]:
    mask = np.asarray(free, dtype=bool)
    energy, grad = evaluate(kind, np.asarray(coords, dtype=np.float64), None if kind == "lj" else box)
    force = float(np.linalg.norm(np.asarray(grad, dtype=np.float64)[mask]))
    return float(energy), force


def _hessian_index(kind: str, coords: np.ndarray, free: np.ndarray, box: np.ndarray | None) -> tuple[int, int]:
    mask = np.asarray(free, dtype=bool)
    full = np.asarray(coords, dtype=np.float64)
    point = full[mask].reshape(-1)
    project = bool(np.all(mask) and int(np.count_nonzero(mask)) >= 2 and kind == "lj")
    hessian = numerical_hessian(
        lambda flat, template=full, active=mask: _free_grad(kind, flat, template, active, box),
        point.tolist(),
        step=_FD_STEP,
    )
    index = vibrational_index(
        np.asarray(hessian, dtype=np.float64),
        point.reshape((-1, 3)),
        project,
    )
    return int(index), 2 * int(point.size)


def _free_grad(kind: str, flat, template: np.ndarray, free: np.ndarray, box: np.ndarray | None) -> list[float]:
    full = np.array(template, dtype=np.float64, copy=True)
    full[free] = np.asarray(flat, dtype=np.float64).reshape((-1, 3))
    _energy, grad = evaluate(kind, full, None if kind == "lj" else box)
    return grad[free].reshape(-1).tolist()


def search_geometry(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    settings: dict,
    n1: int,
    n2: int,
    binary: Path | None = None,
    work: Path | None = None,
    cleanup: bool = True,
    timeout: float = 3600.0,
) -> dict:
    """One climbing-image NEB. ``n1`` and ``n2`` are the two stage lengths."""
    executable = binary or lammps_executable()
    if executable is None:
        raise FileNotFoundError("LAMMPS binary is not available")
    start = np.asarray(reactant, dtype=np.float64)
    end = np.asarray(product, dtype=np.float64)
    mask = np.asarray(free, dtype=bool)
    if start.shape != end.shape or mask.shape != (start.shape[0],):
        raise ValueError(f"{name} endpoints do not match")
    if int(np.count_nonzero(mask)) < 1:
        raise ValueError(f"{name} has no free atoms")
    owned: Path | None = None
    if work is None:
        # TemporaryDirectory deletes itself when collected, after the return.
        owned = Path(tempfile.mkdtemp(prefix="lammpsneb-"))
        work = owned
    else:
        work.mkdir(parents=True, exist_ok=True)
    n_images = int(settings["n_images"])
    try:
        lo, hi = _box_bounds(settings["potential"], start, end, box)
        frozen = not bool(np.all(mask))
        if frozen:
            shift = float(np.max(np.abs(end[~mask] - start[~mask])))
            if shift > 1e-4:
                raise ValueError(f"{name} moves a fixed atom by {shift}")
        _write_data(work / "start.data", start, mask, lo, hi)
        _write_final(work / "final.neb", end, mask)
        (work / "in.neb").write_text(
            render_neb_input(settings, n1=n1, n2=n2, frozen=frozen),
            encoding="utf-8",
        )
        returncode = _launch_neb(executable, work, n_images, timeout)
        calls, blocks, missing = _logged_calls(work, n_images)
        log_text = ""
        universe = work / "log.lammps"
        if universe.is_file():
            log_text = universe.read_text(encoding="utf-8", errors="replace")
        stderr = work / "stderr.txt"
        if stderr.is_file():
            log_text += "\n" + stderr.read_text(encoding="utf-8", errors="replace")
        error = _error_line(log_text)
        if returncode is None:
            error = error or "timeout"
        climbed = [int(value) for value in _CLIMBING.findall(log_text)]
        replica = climbed[-1] if climbed else None
        complete = (
            returncode == 0
            and not missing
            and blocks == 2 * n_images
            and replica is not None
            and 1 <= replica <= n_images
        )
        coords = None
        if complete and replica is not None:
            dump = work / f"final.{replica - 1}.dump"
            try:
                coords = _coords_from_dump(dump, start.shape[0])
            except (OSError, ValueError) as exc:
                error = error or str(exc)
                complete = False
        return _finish(
            name,
            settings,
            coords,
            mask,
            box,
            calls=calls,
            complete=complete and coords is not None,
            error=error,
            replica=replica,
            n1=n1,
            n2=n2,
            returncode=returncode,
        )
    finally:
        if owned is not None:
            if cleanup and os.environ.get("OPTBENCH_LAMMPS_KEEP") != "1":
                shutil.rmtree(owned, ignore_errors=True)
            else:
                print(f"{name}: LAMMPS work directory kept at {owned}", file=sys.stderr)


def _logged_calls(work: Path, n_images: int) -> tuple[int, int, bool]:
    total = 0
    blocks = 0
    missing = False
    for index in range(n_images):
        path = work / f"log.lammps.{index}"
        if not path.is_file():
            missing = True
            continue
        found = minimization_blocks(path.read_text(encoding="utf-8", errors="replace"))
        total += calls_from_blocks(found)
        blocks += len(found)
    return total, blocks, missing


def _finish(
    name: str,
    settings: dict,
    coords: np.ndarray | None,
    free: np.ndarray,
    box: np.ndarray | None,
    *,
    calls: int,
    complete: bool,
    error: str,
    replica: int | None,
    n1: int,
    n2: int,
    returncode: int | None,
) -> dict:
    cap = int(settings["cap"])
    force_norm = float(settings["force_norm"])
    reasons: list[str] = []
    energy = None
    force = None
    hessian_index = None
    verify_calls = 0
    if not complete or coords is None:
        reasons.append("crash")
    if calls >= cap:
        reasons.append("cap")
    if complete and coords is not None:
        energy, force = _measured(settings["potential"], coords, free, box)
        if not np.isfinite(force) or force > force_norm:
            reasons.append("force_norm")
        elif "cap" not in reasons and "crash" not in reasons:
            hessian_index, verify_calls = _hessian_index(settings["potential"], coords, free, box)
            if hessian_index != 1:
                reasons.append("hessian_index")
    elif "crash" in reasons:
        reasons.append("force_norm")
    failure = classify_failure(reasons)
    record = _blank(
        name,
        calls=calls,
        failure=failure or "crash",
        reasons=reasons,
        error=error,
        force=None if force is None or not np.isfinite(force) else float(force),
        energy=None if energy is None or not np.isfinite(energy) else float(energy),
        hessian_index=hessian_index,
        verify_calls=verify_calls,
        climbing=replica,
        n1=n1,
        n2=n2,
        n_images=int(settings["n_images"]),
        returncode=returncode,
    )
    record["success"] = failure is None
    record["failure"] = failure
    record["reasons"] = reasons
    record["converged"] = failure is None
    return record


_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    key = (str(Path(root).resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        loaded = load_pairs(geometry_archive(root, problem), settings["geometry"], settings["pairs"])
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def _execute(pair, settings: dict, *, n1: int | None = None, n2: int | None = None, work: Path | None = None) -> dict:
    stages = iteration_budget(settings["n_images"], settings["cap"], settings.get("call_reserve", 0.0))
    first = stages[0] if n1 is None else int(n1)
    second = stages[1] if n2 is None else int(n2)
    try:
        return search_geometry(
            name=pair.name,
            reactant=np.asarray(pair.reactant, dtype=np.float64),
            product=np.asarray(pair.product, dtype=np.float64),
            free=np.asarray(pair.free, dtype=bool),
            box=None if pair.box is None else np.asarray(pair.box, dtype=np.float64),
            settings=settings,
            n1=first,
            n2=second,
            work=work,
        )
    except Exception as exc:
        return _blank(pair.name, error=f"{type(exc).__name__}: {exc}", n_images=settings["n_images"], n1=first, n2=second)


def search_named(
    root: Path,
    problem_name: str,
    template_name: str,
    start: str,
    seed: str = "0",
    *,
    n1: int | None = None,
    n2: int | None = None,
    work: Path | None = None,
) -> dict:
    """One archived pair. The seed is recorded. The band does not draw random numbers."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(pairs[start], settings, n1=n1, n2=n2, work=work)
    record["seed"] = str(seed)
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, problem: dict) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness pair computes summed across replicas",
        "# force_calls are the band total and are not divided by the image count",
        "# the mean is round-half-even over successes",
        f"problem {problem['id']}",
    ]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        images = int(settings["n_interior"])
        if images > 0:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
            lines.append("call_unit total")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row.get('hessian_index') == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append(f"images {settings['n_interior']}")
    return "\n".join(lines) + "\n"


def _success_mean(records: list[dict]) -> dict:
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary = {
        "successes": len(successes),
        "nfailed": len(records) - len(successes),
        "n": len(records),
        "transition_states": sum(1 for row in records if row.get("hessian_index") == 1),
    }
    if successes:
        summary["force_calls"] = int(round(statistics.fmean(successes)))
        summary["force_calls_min"] = min(successes)
        summary["force_calls_max"] = max(successes)
        summary["force_calls_median"] = statistics.median(successes)
    return summary


def write_outputs(output: Path, records: list[dict], template: dict, settings: dict, problem: dict) -> dict:
    """Write ``benchmark.dat``, the per-start file, and the v2 score."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(records, template, settings, problem), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    n1, n2 = iteration_budget(settings["n_images"], settings["cap"], settings.get("call_reserve", 0.0))
    parameters = dict(settings)
    parameters.update({"n1": n1, "n2": n2, "nevery": 1, "counter": "harness", "seed": "0"})
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score_v2(
            [
                {
                    "force_calls": row["force_calls"],
                    "success": row["success"],
                    "failure": row.get("failure"),
                    "reasons": row.get("reasons") or [],
                }
                for row in records
            ],
            settings["cap"],
        ),
        "success_mean": _success_mean(records),
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    success_only = document["success_mean"]
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": success_only.get("force_calls"),
        "nfailed": success_only["nfailed"],
        "starts": success_only["n"],
        "successes": success_only["successes"],
        "output": str(output),
    }


def publish_sidecars(root: Path, directory: Path, output_dir: Path) -> list[dict]:
    """Reduce a complete sidecar tree into one dat per problem and template."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    paths = sorted(Path(directory).rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no sidecar records in {directory}")
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault((row["problem"], row["template"]), []).append(row)
    summaries = []
    commit = _registry(root)["commit"]
    for problem_name, template_name in sorted(grouped):
        records = grouped[(problem_name, template_name)]
        problem = resolve_problem(root, problem_name)
        template = load_template(root, template_name)
        settings = _settings(template, problem, commit)
        order = list(cached_pairs(root, problem, settings))
        found = {row["start"] for row in records}
        if found != set(order) or len(records) != len(order):
            raise ValueError(
                f"{template_name} on {problem_name} has {len(records)} records, the problem has {len(order)} pairs"
            )
        rank = {name: index for index, name in enumerate(order)}
        records.sort(key=lambda row: rank[row["start"]])
        slug = f"lammps-{VERSION.replace(' ', '')}-{problem['id'].replace('/', '-')}-{template['id'].split('/', 1)[1]}"
        summaries.append(write_outputs(output_dir / f"{slug}.dat", records, template, settings, problem))
    return summaries


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Run every pair in one problem and write the dat, starts, and v2 score.

    Replicas share the machine through MPI, so the starts run one at a time.
    ``resume`` keeps a start whose sidecar is already on disk.
    """
    del workers
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem, settings).values())
    if limit > 0:
        pairs = pairs[:limit]
    parts = output.with_suffix(".parts")
    records = []
    done: dict[str, dict] = {}
    if resume and parts.is_dir():
        for path in sorted(parts.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            done[str(row["start"])] = row
    parts.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pairs, start=1):
        if pair.name in done:
            records.append(done[pair.name])
            continue
        record = _execute(pair, settings)
        record["seed"] = "0"
        record["problem"] = problem["id"]
        record["template"] = template["id"]
        record["code"] = template["code"]
        record["code_version"] = template["code_version"]
        record["commit"] = template["commit"]
        record["algorithm"] = template["algorithm"]
        record["tuning"] = template["tuning"]
        path = parts / f"{pair.name}.json"
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        records.append(record)
        print(
            f"{index}/{len(pairs)} {template['id']} {pair.name} "
            f"calls={record['force_calls']} success={record['success']} failure={record['failure']}",
            flush=True,
        )
    return write_outputs(output, records, template, settings, problem)
