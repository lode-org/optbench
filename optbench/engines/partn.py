"""pARTn through ``libartn.so`` and ``next_displ``.

The plugin and this standalone loop are the same library. Each step
evaluates the harness potential, passes the force (the negative
gradient), and applies the returned displacement. ``nevalf`` is recorded
and is not the score. The product of a pair chooses the initial push.
The search does not build a path to that product.

``engine_units`` is ``lammps/metal`` on Morse platinum and ``lammps/lj``
on Lennard-Jones, so a push length is an angstrom or a sigma. k-ART is
not this library.
"""

from __future__ import annotations

import os
import sys
import tempfile
from ctypes import POINTER, byref, c_char_p
from pathlib import Path

import numpy as np

from .common import (
    cached_pairs,
    initial_push,
    judge_mode,
    judge_saddle,
    lattice,
    pair_seed,
    prepared,
    run_jobs,
    write_campaign,
)
from .geometries import load_starts, point_settings

ENGINE = "partn"
CODE = "artn"
_SEED_BASE = 20261015


def library_path() -> Path:
    env = os.environ.get("ARTN_LIB")
    if env:
        return Path(env)
    return Path("/tmp/src/artn/lib/libartn.so")


def interface_dir() -> Path:
    env = os.environ.get("ARTN_INTERFACE")
    if env:
        return Path(env)
    return Path("/tmp/src/artn/interface")


def available() -> str | None:
    library = library_path()
    if not library.is_file():
        return f"libartn.so is not at {library}. Set ARTN_LIB to the built library."
    interface = interface_dir() / "pypARTn.py"
    if not interface.is_file():
        return f"pypARTn.py is not at {interface}. Set ARTN_INTERFACE to the interface directory."
    return None


def _units(kind: str) -> str:
    if kind == "morse_pt":
        return "lammps/metal"
    if kind == "lj":
        return "lammps/lj"
    raise ValueError(f"{kind} has no pARTn engine_units")


def _section(template: dict) -> dict:
    section = dict(template.get("partn") or {})
    section.setdefault("push_step_size", 0.1)
    section.setdefault("eigen_step_size", 0.15)
    section.setdefault("ninit", 3)
    section.setdefault("lanczos_max_size", 16)
    section.setdefault("lanczos_disp", None)
    section.setdefault("etot_diff_limit", None)
    section.setdefault("lpush_final", False)
    section.setdefault("converge_property", "norm")
    return section


def _job(pair, settings: dict, template: dict, seed: str) -> dict:
    section = _section(template)
    label = f"{seed}:{pair.name}"
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64).tolist(),
        "product": np.asarray(pair.product, dtype=np.float64).tolist(),
        "free": np.asarray(pair.free, dtype=bool).tolist(),
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64).reshape(-1).tolist(),
        "kind": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
        "push_step_size": float(section["push_step_size"]),
        "eigen_step_size": float(section["eigen_step_size"]),
        "ninit": int(section["ninit"]),
        "lanczos_max_size": int(section["lanczos_max_size"]),
        "lanczos_disp": section.get("lanczos_disp"),
        "etot_diff_limit": section.get("etot_diff_limit"),
        "lpush_final": bool(section["lpush_final"]),
        "converge_property": str(section["converge_property"]),
        "fire_dt": _fire_dt(section, settings["potential"]),
        "task": "saddle",
        "reference": None,
        "overlap": 0.99,
        "seed": str(seed),
        "zseed": pair_seed(label, _SEED_BASE),
        "shlib": str(library_path()),
        "interface": str(interface_dir()),
        "commit": template["commit"],
    }


def _fire_dt(section: dict, kind: str) -> float | None:
    """FIRE ``dt_init`` in engine units.

    The library default is 20, which is an atomic unit. On ``lammps/lj``
    that step is large enough to leave the cluster. Lennard-Jones uses
    0.05 unless the template sets ``fire_dt_init``. Morse keeps the default.
    """
    if "fire_dt_init" in section and section["fire_dt_init"] is not None:
        return float(section["fire_dt_init"])
    if kind == "lj":
        return 0.05
    return None


def _mode_overlap(art, reference, free: np.ndarray) -> float | None:
    """Absolute overlap of the pARTn eigenvector with the free-atom reference."""
    if reference is None:
        return None
    try:
        raw = art.get_runparam("eigenvec")
    except Exception:
        return None
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    nat = int(len(free))
    if values.size != nat * 3:
        return None
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    mask = np.asarray(free, dtype=bool)
    best = 0.0
    for shaped in (values.reshape(nat, 3), values.reshape(3, nat).T):
        piece = shaped[mask].reshape(-1)
        scale = float(np.linalg.norm(piece))
        if scale < 1e-14 or piece.size != target.size:
            continue
        best = max(best, abs(float(np.dot(piece / scale, target / np.linalg.norm(target)))))
    return best


def _gitinfo(art) -> str:
    """Read ``artn_gitinfo``. ``pypARTn.gitinfo`` passes the wrong pointer depth."""
    library = art.lib
    library.artn_gitinfo.restype = None
    library.artn_gitinfo.argtypes = [POINTER(c_char_p)]
    holder = c_char_p()
    library.artn_gitinfo(byref(holder))
    if not holder.value:
        return ""
    return holder.value.decode()


def _set(art, name: str, value) -> None:
    try:
        art.set(name, value)
    except Exception as exc:
        raise RuntimeError(f"pARTn set {name} failed: {exc}") from exc


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    interface = job["interface"]
    if interface not in sys.path:
        sys.path.insert(0, interface)
    import pypARTn

    from optbench.counter import ForceCallCap, HarnessCounter
    from optbench.pes import evaluate

    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = None if job.get("product") is None else np.asarray(job["product"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    task = str(job.get("task") or "saddle")
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    counter = HarnessCounter(cache=True, cap=cap)
    pos = reactant.copy()
    energy = None
    gradient = None
    error = ""
    lconv = False
    overlap = 0.0
    has_sad = None
    has_error = None
    artn_nevalf = None
    art = None
    previous = Path.cwd()
    try:
        version = None
        with tempfile.TemporaryDirectory(prefix="partn-") as folder:
            os.chdir(folder)
            art = pypARTn.artn(engine="other", shlib=job["shlib"])
            version = art.version()
            if version != (1, 1, 0):
                raise RuntimeError(f"libartn version is {version}, the pin is 1.1.0")
            gitinfo = _gitinfo(art)
            token = str(job["commit"])[:7]
            if token not in gitinfo:
                raise RuntimeError(f"libartn gitinfo {gitinfo!r} does not contain {token}")
            nat = int(pos.shape[0])
            _set(art, "engine_units", _units(kind))
            _set(art, "verbose", 0)
            _set(art, "forc_thr", float(job["force_norm"]))
            _set(art, "converge_property", job["converge_property"])
            _set(art, "ninit", int(job["ninit"]))
            _set(art, "push_step_size", float(job["push_step_size"]))
            _set(art, "eigen_step_size", float(job["eigen_step_size"]))
            _set(art, "lanczos_max_size", int(job["lanczos_max_size"]))
            if job.get("lanczos_disp") is not None:
                _set(art, "lanczos_disp", float(job["lanczos_disp"]))
            if job.get("etot_diff_limit") is not None:
                _set(art, "etot_diff_limit", float(job["etot_diff_limit"]))
            _set(art, "lpush_final", bool(job["lpush_final"]))
            _set(art, "struc_format_out", "none")
            _set(art, "nevalf_max", int(cap))
            _set(art, "zseed", int(job["zseed"]))
            if job.get("fire_dt") is not None:
                art.fire_set("dt_init", float(job["fire_dt"]))
            if product is None:
                push = np.zeros_like(reactant)
                push[int(np.flatnonzero(free)[0]), 0] = float(job["push_step_size"])
            else:
                push = initial_push(reactant, product, free, float(job["push_step_size"]))
            _set(art, "push_init", np.ascontiguousarray(push, dtype=np.float64))
            typ = np.ones(nat, dtype=np.int32)
            if_pos = np.ones((nat, 3), dtype=np.int32)
            if_pos[~free] = 0
            cell = np.ascontiguousarray(lattice(box, pos), dtype=np.float64)
            lanczos_code = int(art.get_runparam("LANC")) if task == "mode" else None
            stalled = 0
            for _step in range(cap + 2):
                if counter.calls >= cap:
                    break
                key = tuple(float(value) for value in pos.reshape(-1))
                fresh, cached = counter.charge(key)
                if fresh:
                    energy, gradient = evaluate(kind, pos, box)
                    gradient = np.asarray(gradient, dtype=np.float64)
                    counter.remember(key, (float(energy), gradient))
                else:
                    energy, gradient = cached
                    gradient = np.asarray(gradient, dtype=np.float64)
                force = np.ascontiguousarray(-gradient, dtype=np.float64)
                dr, lconv = art.next_displ(
                    nat,
                    float(energy),
                    force,
                    typ,
                    np.ascontiguousarray(pos, dtype=np.float64),
                    cell,
                    if_pos,
                )
                if task == "mode":
                    found = _mode_overlap(art, job.get("reference"), free)
                    if found is not None:
                        overlap = max(overlap, found)
                    # The Lanczos vector is the mode. The next phase pushes
                    # away from this geometry, which is not the mode score.
                    if overlap >= float(job.get("overlap", 0.99)) or int(art.disp_code()) != lanczos_code:
                        lconv = overlap >= float(job.get("overlap", 0.99))
                        break
                if lconv:
                    break
                step = np.array(dr, dtype=np.float64, copy=True)
                if not np.all(np.isfinite(step)):
                    raise RuntimeError("pARTn returned a non-finite displacement")
                if float(np.linalg.norm(step)) < 1e-14:
                    stalled += 1
                else:
                    stalled = 0
                if stalled >= 3:
                    raise RuntimeError("pARTn displacement stayed at zero")
                pos = pos + step
                pos[~free] = reactant[~free]
            if counter.calls < cap and np.all(np.isfinite(pos)):
                key = tuple(float(value) for value in pos.reshape(-1))
                fresh, cached = counter.charge(key)
                if fresh:
                    energy, gradient = evaluate(kind, pos, box)
                    gradient = np.asarray(gradient, dtype=np.float64)
                    counter.remember(key, (float(energy), gradient))
                else:
                    energy, gradient = cached
                    gradient = np.asarray(gradient, dtype=np.float64)
            try:
                has_error = bool(art.extract("has_error"))
                has_sad = bool(art.extract("has_sad"))
                artn_nevalf = int(art.extract("nevalf"))
            except Exception:
                pass
            if has_error:
                ierr, message = art.get_error()
                raise RuntimeError(f"pARTn error {ierr}: {message}")
    except ForceCallCap as exc:
        error = f"force-call cap: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
        if art is not None:
            try:
                art.destroy()
            except Exception:
                pass
    prefix: list[str] = []
    if error:
        if "force-call cap" in error or "ForceCallCap" in error:
            prefix.append("cap")
        else:
            prefix.append("crash")
    if task == "mode":
        record = judge_mode(
            job["name"],
            int(counter.calls),
            cap,
            overlap,
            float(job.get("overlap", 0.99)),
            error,
        )
        record["seed"] = job["seed"]
        record["zseed"] = int(job["zseed"])
        record["lconv"] = bool(lconv)
        record["artn_has_sad"] = has_sad
        record["artn_nevalf"] = artn_nevalf
        return record
    record = judge_saddle(
        job["name"],
        pos if error == "" or "force-call cap" in error else pos,
        None if energy is None else float(energy),
        None if gradient is None else np.asarray(gradient, dtype=np.float64),
        counter=counter,
        cap=cap,
        force_norm=float(job["force_norm"]),
        kind=kind,
        free=free,
        box=box,
        prefix=prefix,
        keep_mode=bool(job.get("keep_mode")),
    )
    if job.get("keep_positions"):
        record["positions"] = np.asarray(pos, dtype=np.float64).reshape(-1).tolist()
    if error:
        record["error"] = error
    record["seed"] = job["seed"]
    record["zseed"] = int(job["zseed"])
    record["lconv"] = bool(lconv)
    record["artn_has_sad"] = has_sad
    record["artn_nevalf"] = artn_nevalf
    return record


def search_named(root: Path, problem_name: str, template_name: str, start: str, seed: str = "0") -> dict:
    """One pair. ``seed`` enters the pARTn ``zseed`` together with the pair name."""
    reason = available()
    if reason:
        raise RuntimeError(reason)
    problem, template, settings = prepared(root, problem_name, template_name, CODE)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(pairs[start], settings, template, seed))
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def _point_job(start: dict, settings: dict, template: dict) -> dict:
    section = _section(template)
    task = "mode" if settings.get("task") == "mode" else "saddle"
    label = f"0:{start['name']}"
    return {
        "name": start["name"],
        "reactant": start["positions"],
        "product": None,
        "free": [not bool(flag) for flag in start["fixed"]],
        "box": start["box"],
        "kind": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
        "push_step_size": float(section["push_step_size"]),
        "eigen_step_size": float(section["eigen_step_size"]),
        "ninit": int(section["ninit"]),
        "lanczos_max_size": int(section["lanczos_max_size"]),
        "lanczos_disp": section.get("lanczos_disp"),
        "etot_diff_limit": section.get("etot_diff_limit"),
        "lpush_final": bool(section["lpush_final"]),
        "converge_property": str(section["converge_property"]),
        "fire_dt": _fire_dt(section, settings["potential"]),
        "task": task,
        "reference": start.get("reference"),
        "overlap": float(settings["overlap"]),
        "seed": "0",
        "zseed": pair_seed(label, _SEED_BASE),
        "shlib": str(library_path()),
        "interface": str(interface_dir()),
        "commit": template["commit"],
    }


def run_points(root: Path, problem: dict, template: dict, settings: dict, *, output: Path, workers: int) -> dict:
    """One ARTn search from each sample geometry."""
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    jobs = [_point_job(start, settings, template) for start in starts]
    records = run_jobs(jobs, _execute, workers)
    section = _section(template)
    parameters = {
        **{key: value for key, value in settings.items() if key != "task"},
        **section,
        "task": settings.get("task"),
        "engine_units": "lammps/metal for morse_pt, lammps/lj for lj",
        "counter": "harness",
        "seed": "0",
        "zseed_base": _SEED_BASE,
        "fire_dt_init": _fire_dt(section, settings["potential"]),
    }
    if settings.get("task") == "mode":
        comment = (
            "# counter harness. The search stops when the pARTn eigenvector "
            "overlaps the reference mode by the problem threshold.\n"
            "# The displacement that would leave that geometry is not applied after the overlap is met.\n"
            "# nevalf is the pARTn tally and is not this count."
        )
    else:
        comment = (
            "# counter harness. The start is the sample geometry, not a reactant/product pair.\n"
            "# ninit initial pushes use push_init along x of the first free atom.\n"
            "# A success is under the cap, at the force stop, and Hessian index 1.\n"
            "# nevalf is the pARTn tally and is not this count."
        )
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )


def _minimize_free(kind: str, positions: np.ndarray, free: np.ndarray, box, force_tol: float, budget: int):
    """L-BFGS on the free atoms. The returned call count is the harness count."""
    from optbench.pes import evaluate
    from optbench.rgsaddle_minmode import HarnessSurface, _lbfgs_minimize

    def call(point):
        full = np.array(positions, dtype=np.float64, copy=True)
        full[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
        energy, grad = evaluate(kind, full, box)
        return float(energy), np.asarray(grad, dtype=np.float64)[free].reshape(-1)

    surface = HarnessSurface(call)
    point, ok = _lbfgs_minimize(
        surface,
        positions[free].reshape(-1).copy(),
        force_tol,
        budget,
    )
    energy, _gradient = surface.energy_forces(point)
    out = np.array(positions, dtype=np.float64, copy=True)
    out[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    return out, bool(ok), int(surface.calls), float(energy)


def run_catalog(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    """Displace the Pt island until the goal count of connected index-1 saddles.

    Each search is one pARTn run from a kick off the minimized reactant.
    The harness count for that search is the pARTn count plus the two
    endpoint minimizations. The repetition total also includes the
    initial minimization. ``workers`` is unused: the next kick waits on
    the saddles already stored.
    """
    del workers
    import statistics

    from optbench.connection import ProcessTable, per_atom_max, periodic_lengths, tolerances
    from optbench.connection import connected as endpoints_connected
    from optbench.failures import classify_failure
    from optbench.v2 import write_starts

    settings = point_settings(problem)
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    search = dict(template.get("search") or {})
    repetitions = int(search.get("repetitions", 1))
    magnitude = float(search.get("displace_magnitude", 0.1))
    distance_difference, energy_difference = tolerances(problem)
    max_searches = int(search.get("max_searches", 20000))
    budget = int(search.get("search_budget", 10000))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    starts = load_starts(root, problem, output.parent / "work" / template["id"].replace("/", "-"))
    base = starts[0]
    positions = np.asarray(base["positions"], dtype=np.float64)
    free = np.asarray([not bool(flag) for flag in base["fixed"]], dtype=bool)
    box = None if base["box"] is None else np.asarray(base["box"], dtype=np.float64)
    lengths = periodic_lengths(box, (problem.get("potential") or {}).get("pbc"))
    comments = []
    records = []
    failed_reps = 0
    for rep in range(1, repetitions + 1):
        minimum, min_ok, minimize_calls, min_energy = _minimize_free(
            settings["potential"], positions, free, box, settings["force_norm"], 5000
        )
        rng = np.random.default_rng(20261015 + rep)
        found = ProcessTable(lengths, distance_difference, energy_difference)
        connected = 0
        searches = 0
        search_calls = 0
        while connected < goal and searches < max_searches:
            searches += 1
            if searches == 1 or searches % 25 == 0:
                print(
                    f"{template['id']} rep {rep} searches {searches} connected {connected}",
                    flush=True,
                )
            atom = int(listed[int(rng.integers(0, len(listed)))])
            if not bool(free[atom]):
                raise RuntimeError(f"displace atom {atom} is not free")
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            kicked = np.array(minimum, copy=True)
            kicked[atom] = kicked[atom] + magnitude * direction
            job = _point_job(
                {
                    "name": f"rep{rep}-search{searches}",
                    "positions": kicked.tolist(),
                    "fixed": (~free).tolist(),
                    "box": base["box"],
                    "reference": None,
                },
                {**settings, "cap": budget, "task": "saddle", "overlap": 0.99},
                template,
            )
            job["keep_mode"] = True
            job["keep_positions"] = True
            job["ninit"] = int(_section(template)["ninit"])
            record = _execute(job)
            climb = int(record["force_calls"])
            saddle = None if record.get("positions") is None else np.asarray(record["positions"], dtype=np.float64).reshape(-1, 3)
            record["reasons"] = list(record.get("reasons") or [])
            if record["success"] and saddle is not None and record.get("lowest_mode"):
                axis = np.asarray(record["lowest_mode"], dtype=np.float64)
                axis = axis / np.linalg.norm(axis)
                saddle_free = saddle[free].reshape(-1)
                plus, _plus_ok, plus_calls, _plus_energy = _minimize_free(
                    settings["potential"],
                    _placed(saddle, free, saddle_free + 0.1 * axis),
                    free,
                    box,
                    settings["force_norm"],
                    2000,
                )
                minus, _minus_ok, minus_calls, _minus_energy = _minimize_free(
                    settings["potential"],
                    _placed(saddle, free, saddle_free - 0.1 * axis),
                    free,
                    box,
                    settings["force_norm"],
                    2000,
                )
                links = endpoints_connected(minimum[free], plus[free], minus[free], lengths, distance_difference)
                barrier = float(record["energy"] - min_energy)
                novel = links and not found.repeat(saddle[free], barrier)
                record["barrier"] = barrier
                record["per_atom_max_plus"] = per_atom_max(plus[free], minimum[free], lengths)
                record["per_atom_max_minus"] = per_atom_max(minus[free], minimum[free], lengths)
                record["endpoint_calls"] = int(plus_calls + minus_calls)
                record["force_calls"] = climb + int(plus_calls + minus_calls)
                record["reasons"] = []
                if novel:
                    found.add(saddle[free], barrier)
                if links and novel and 0.0 < barrier < barrier_max:
                    record["success"] = True
                    record["failure"] = None
                    connected += 1
                else:
                    if not links:
                        record["reasons"].append("not_connected")
                    elif not novel:
                        record["reasons"].append("duplicate")
                    if barrier <= 0.0 or barrier >= barrier_max:
                        record["reasons"].append("barrier")
                    record["success"] = False
                    record["failure"] = classify_failure(record["reasons"])
            search_calls += int(record["force_calls"])
            record.pop("positions", None)
            record.pop("lowest_mode", None)
            records.append(record)
        total_calls = int(minimize_calls + search_calls)
        if connected < goal:
            failed_reps += 1
        fraction = connected / searches if searches else 0.0
        comments.append(
            f"# rep {rep} force_calls {total_calls} unique_saddles {len(found)} "
            f"fraction_connected {fraction} total_searches {searches} "
            f"connected {connected} minimize_calls {minimize_calls} "
            f"minimum_converged {int(min_ok)} seed {20261015 + rep}"
        )
    totals = [int(line.split()[line.split().index("force_calls") + 1]) for line in comments]
    text = "\n".join(comments) + "\n"
    if totals:
        text += f"force_calls {statistics.fmean(totals)}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
    text += (
        f"code {template.get('code', CODE)}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    return {
        "force_calls": statistics.fmean(totals) if totals else None,
        "nfailed": failed_reps,
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
    }


def _placed(saddle: np.ndarray, free: np.ndarray, free_flat: np.ndarray) -> np.ndarray:
    out = np.array(saddle, dtype=np.float64, copy=True)
    out[free] = np.asarray(free_flat, dtype=np.float64).reshape(-1, 3)
    return out


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    if problem.get("suite") in {"tsopt", "lowest-eigenvector"}:
        settings = point_settings(problem)
        if settings["cap"] is None:
            raise ValueError(f"{problem['id']} has no force-call cap")
        settings["task"] = "mode" if problem["suite"] == "lowest-eigenvector" else "saddle"
        if problem["suite"] == "tsopt":
            # The geometry is already a transition-state guess. Skip the
            # blind push and let Lanczos follow the lowest mode.
            template = dict(template)
            partn = dict(template.get("partn") or {})
            partn.setdefault("ninit", 0)
            template["partn"] = partn
        return run_points(root, problem, template, settings, output=output, workers=workers)
    if problem["id"] == "saddle-search/pt-island":
        return run_catalog(root, problem, template, output=output, workers=workers)
    problem, template, settings = prepared(root, problem_name, template_name, CODE)
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_job(pair, settings, template, "0") for pair in pairs]
    records = run_jobs(jobs, _execute, workers)
    for record in records:
        record["problem"] = problem["id"]
        record["template"] = template["id"]
    section = _section(template)
    parameters = {
        **settings,
        **section,
        "engine_units": "lammps/metal for morse_pt, lammps/lj for lj",
        "push": "one free atom from the reactant toward the product",
        "counter": "harness",
        "seed": "0",
        "zseed_base": _SEED_BASE,
    }
    comment = (
        "# counter harness. One force call is one potential evaluation in the next_displ loop.\n"
        "# nevalf is the pARTn tally and is not this count.\n"
        "# The product sets the initial push. This search does not build a band to the product.\n"
        "# A success is under the cap, at the force stop, and Hessian index 1."
    )
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )
