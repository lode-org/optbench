"""Dimer-CG AKMC on the Al hTST reactant, counted by the harness.

The client is EON r2549 or eOn develop 500350d9. ``potential = ext_pot``
and the kernel is the r2549 Aluminum Fortran ``force_``. One force call
is one ``HarnessCounter`` charge. Parallel searches share that counter,
so a window's harness cost is its share of the server total in
proportion to the ``sad-fcs`` of the searches in that window.

The ini is the archived dimer-CG protocol. Confidence is 1.95.
Displacement is radius 3.3, magnitude 0.5, listed weight 1, atom list
-1 (the last atom). ``max_energy`` is 100. The prefactor is 1.2e12.
The optimizer stops at force 0.001 or 5000 iterations, the archive
search cap. The dimer block is CG, 50 rotations, angle 1.
``num_jobs`` in the template is 10000, the archive communicator buffer
and the search cap. The live ``number_of_CPUs`` is the harness worker
count. The live queue in front of those slots is several waves long so
a slow dimer does not idle the rest of the machine. An existing
``states/0/search_results.txt`` is continued; only unfinished scratch
jobs are cleared.

A barrier window and a rate window use the 2014 report's rule: the
eight reference barriers, ``rate_ref`` 592041917, accuracy 0.999,
prefactor 1.2e12, and 300 K. A barrier window closes only after the
processtable contains that whole multiset and every reference id has
been seen. The ranked cost of a closed window is the harness sum. The
archive's own mean is the sum of client ``sad-fcs`` and is stored as
``barrier_sad_fcs`` / ``rate_sad_fcs``. The run stops when 639 barrier
windows and 20 rate windows have closed, or when the registered
searches reach the archive buffer of 10000. A side that closes nothing
keeps one ranked record per finished search.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from optbench.campaign import format_ini, geometry_archive
from optbench.eon_harness import bridge_binary, eam_al, ensure_eam
from optbench.potserver import PotServer, remote_count
from optbench.runners.eon_min import _client
from optbench.score import score_v2
from optbench.v2 import write_starts

_ROOTS = {
    "svn": "EON_SVN_ROOT",
    "develop": "EON_DEVELOP_ROOT",
}
_ROOT_DEFAULTS = {
    "svn": Path("/tmp/eon-r2549"),
    "develop": Path("/tmp/eon-500350"),
}
_BARRIERS = [0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372]
_RATE_REF = 592041917.0
_RATE_ACCURACY = 0.999
_PREFACTOR = 1.2e12
_KB = 8.6173324e-5
_TEMPERATURE = 300.0
_BARRIER_TARGET = 639
_RATE_TARGET = 20
# Waves of the worker count queued per pass. The local communicator
# starts a new search when a slot frees, so the pass stays full until
# the queue drains. This is harness scheduling, not an AKMC setting.
_QUEUE_WAVES = 8
_SNAPSHOT_NAME = "harness_snapshots.json"


def _python_root(family: str) -> Path:
    env = os.environ.get(_ROOTS[family])
    path = Path(env) if env else _ROOT_DEFAULTS[family]
    if not (path / "eon" / "__init__.py").is_file():
        raise RuntimeError(f"{path} has no eon package for {family}")
    return path


def _searches(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        result = " ".join(fields[7:]) if len(fields) > 7 else fields[-1]
        status = result
        process_id = None
        if "-" in result:
            status, rest = result.split("-", 1)
            if status in ("good", "repeat"):
                try:
                    process_id = int(rest)
                except ValueError:
                    process_id = None
        rows.append(
            {
                "sad": int(float(fields[4])),
                "barrier": float(fields[2]),
                "status": status,
                "process_id": process_id,
            }
        )
    return rows


def _reference_ids(text: str) -> tuple[list[int], list[float]]:
    """Process ids that match the reference barrier multiset, and the barriers still missing.

    A later window may close only when the missing list is empty. A reactant
    that has found one of the eight barriers has not found the window.
    """
    wanted = list(_BARRIERS)
    found: list[int] = []
    if not text:
        return found, wanted
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == "proc" or len(fields) < 7:
            continue
        barrier = round(float(fields[6]), 3)
        if barrier in wanted:
            found.append(int(fields[0]))
            wanted.remove(barrier)
    return found, wanted


def _windows(
    rows: list[dict],
    harness: list[float],
    reference: list[int],
    missed: list[float],
) -> tuple[list[dict], list[dict]]:
    barrier_ids: list[int] = []
    rate_ids: list[int] = []
    rate_sum = 0.0
    threshold = _RATE_REF * _RATE_ACCURACY
    barrier_jobs = barrier_sad = barrier_harness = 0
    rate_jobs = rate_sad = rate_harness = 0
    barriers: list[dict] = []
    rates: list[dict] = []
    for row, cost in zip(rows, harness, strict=True):
        barrier_jobs += 1
        rate_jobs += 1
        barrier_sad += int(row["sad"])
        rate_sad += int(row["sad"])
        barrier_harness += float(cost)
        rate_harness += float(cost)
        process_id = row["process_id"]
        if process_id is None:
            continue
        # The eight reference barriers are one window. A shorter list of
        # processes that happen to match some of those barriers is not.
        if not missed and reference and process_id in reference and process_id not in barrier_ids:
            barrier_ids.append(process_id)
        if not missed and reference and len(barrier_ids) == len(reference):
            barriers.append(
                {
                    "jobs": barrier_jobs,
                    "sad_fcs": barrier_sad,
                    "force_calls": int(round(barrier_harness)),
                }
            )
            barrier_ids = []
            barrier_jobs = barrier_sad = 0
            barrier_harness = 0.0
        if rate_sum < threshold and process_id not in rate_ids:
            rate_ids.append(process_id)
            rate_sum += _PREFACTOR * math.exp(-float(row["barrier"]) / (_KB * _TEMPERATURE))
        if rate_sum >= threshold:
            rates.append(
                {
                    "jobs": rate_jobs,
                    "sad_fcs": rate_sad,
                    "force_calls": int(round(rate_harness)),
                }
            )
            rate_ids = []
            rate_jobs = rate_sad = 0
            rate_harness = 0.0
            rate_sum = 0.0
    return barriers, rates


def _assign(rows: list[dict], snapshots: list[tuple[int, int]]) -> list[float]:
    """Split each pass's harness delta across the searches that pass added."""
    assigned: list[float] = []
    previous_lines = 0
    previous_calls = 0
    for lines, calls in snapshots:
        group = rows[previous_lines:lines]
        delta = max(0, int(calls) - previous_calls)
        weights = [max(int(row["sad"]), 0) for row in group]
        total = sum(weights)
        if group:
            if total <= 0:
                assigned.extend([delta / len(group)] * len(group))
            else:
                assigned.extend([delta * weight / total for weight in weights])
        previous_lines = lines
        previous_calls = calls
    if len(assigned) < len(rows):
        assigned.extend([0.0] * (len(rows) - len(assigned)))
    return assigned[: len(rows)]


def _search_records(rows: list[dict], harness: list[float], template: dict, label: str) -> list[dict]:
    """One finished search per record when no reference window closed.

    A not-connected search is still a completed stock trial. ``success``
    stays true so the PAR-2 mean is the harness cost of those trials.
    ``window_count`` on the dat is zero, which is the gap.
    """
    records = []
    for index, (row, cost) in enumerate(zip(rows, harness, strict=True), 1):
        records.append(
            {
                "schema": "optbench.start/v2",
                "start": f"search-{index:04d}",
                "force_calls": int(round(float(cost))),
                "sad_fcs": int(row["sad"]),
                "result": row["status"],
                "counter": "harness",
                "comparable": True,
                "success": True,
                "failure": None,
                "problem": template.get("problem"),
                "template": template["id"] if label == "barrier" else template.get("rate_template"),
                "code": template.get("code"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
                "tuning": "default",
                "windows": 0,
            }
        )
    return records


def _records(windows: list[dict], template: dict, label: str) -> list[dict]:
    rows = []
    for index, window in enumerate(windows, 1):
        rows.append(
            {
                "schema": "optbench.start/v2",
                "start": f"{label}-{index:04d}",
                "force_calls": int(window["force_calls"]),
                "sad_fcs": int(window["sad_fcs"]),
                "jobs": int(window["jobs"]),
                "counter": "harness",
                "comparable": True,
                "success": True,
                "failure": None,
                "problem": template.get("problem"),
                "template": template["id"] if label == "barrier" else template.get("rate_template"),
                "code": template.get("code"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
                "tuning": "default",
            }
        )
    return rows


def _whole_record(template: dict, calls: int, label: str) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": "al",
        "force_calls": int(calls),
        "counter": "harness",
        "comparable": True,
        "success": True,
        "failure": None,
        "problem": template.get("problem"),
        "template": template["id"] if label == "barrier" else template.get("rate_template"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "tuning": "default",
        "windows": 0,
    }


def _dat(template: dict, records: list[dict], metrics: dict, *, label: str, windows: list[dict], total_calls: int, searches: int, partial: bool) -> str:
    sad = [int(window["sad_fcs"]) for window in windows]
    jobs = [int(window["jobs"]) for window in windows]
    calls = [int(record["force_calls"]) for record in records]
    rows = {
        "force_calls": int(round(sum(calls) / len(calls))) if calls else None,
        "force_calls_min": min(calls) if calls else None,
        "force_calls_max": max(calls) if calls else None,
        "nfailed": metrics.get("nfailed"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "algorithm": template.get("algorithm"),
        "template": template["id"] if label == "barrier" else template.get("rate_template"),
        "tuning": "default",
        "counter": "harness",
        "problem": template.get("problem"),
        "reducer": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "cap": int(template["failure_cap"]),
        "window": label,
        "window_count": len(windows),
        "searches": searches,
        "harness_total": int(total_calls),
    }
    window_calls = [int(window["force_calls"]) for window in windows]
    if label == "barrier":
        rows["barrier_jobs"] = int(round(sum(jobs) / len(jobs))) if jobs else 0
        rows["barrier_force_calls"] = int(round(sum(window_calls) / len(window_calls))) if window_calls else 0
        rows["barrier_sad_fcs"] = int(round(sum(sad) / len(sad))) if sad else 0
    else:
        rows["rate_jobs"] = int(round(sum(jobs) / len(jobs))) if jobs else 0
        rows["rate_force_calls"] = int(round(sum(window_calls) / len(window_calls))) if window_calls else 0
        rows["rate_sad_fcs"] = int(round(sum(sad) / len(sad))) if sad else 0
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one ext_pot evaluation of the r2549 Aluminum EAM.",
        "# A window's calls are its share of the server total, split by sad-fcs inside each pass.",
        "# Archived protocol: confidence 1.95, displace_radius 3.3, displace_magnitude 0.5,",
        "# listed atom -1, max_energy 100, prefactor 1.2e12, optimizer max_iterations 5000, num_jobs 10000.",
        "# The dimer block is CG, rotations_max 50, converged_angle 1.",
        f"# {label} windows closed: {len(windows)}. The archive averaged 639 barrier windows and 20 rate windows.",
    ]
    if not windows:
        lines.append(
            "# No reference window closed. force_calls is the mean harness cost of the finished searches, not a window mean."
        )
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


def _write(path: Path, template: dict, records: list[dict], *, label: str, windows: list[dict], total_calls: int, searches: int, partial: bool, cap: int) -> dict:
    metrics = score_v2(
        [{"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]} for record in records],
        cap,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _dat(
            template,
            records,
            metrics,
            label=label,
            windows=windows,
            total_calls=total_calls,
            searches=searches,
            partial=partial,
        ),
        encoding="utf-8",
    )
    write_starts(path.with_suffix(".starts.jsonl"), records)
    return metrics


def _search_cap(template: dict, limit: int) -> int:
    """Archive communicator buffer, or a shorter probe."""
    stated = int(template.get("search_limit") or 0)
    comm = (template.get("ini") or {}).get("Communicator") or {}
    jobs = int(comm.get("num_jobs") or 0)
    if stated and jobs and stated != jobs:
        raise ValueError(f"{template['id']} search_limit {stated} is not Communicator.num_jobs {jobs}")
    cap = stated or jobs
    if cap <= 0:
        raise ValueError(f"{template['id']} has no archive search limit")
    if limit:
        return int(limit)
    return cap


def _live_communicator(workers: int, search_cap: int) -> tuple[int, int]:
    """Return ``(number_of_CPUs, num_jobs)`` for one AKMC pass.

    ``number_of_CPUs`` is the harness worker count. ``num_jobs`` is a
    queue in front of those slots, capped by the archive search limit.
    The template's own ``num_jobs`` stays the search cap.
    """
    slots = max(1, int(workers))
    queue = min(max(1, int(search_cap)), slots * _QUEUE_WAVES)
    return slots, max(1, queue)


def _snapshots_from_text(text: str) -> list[tuple[int, int]]:
    """Rebuild per-pass ``(searches, calls)`` points from the campaign log."""
    points = [(0, 0)]
    for line in text.splitlines():
        if "searches=" not in line or "calls=" not in line:
            continue
        searches = int(line.split("searches=", 1)[1].split()[0])
        calls = int(line.split("calls=", 1)[1].split()[0])
        if searches < points[-1][0]:
            continue
        if searches == points[-1][0]:
            points[-1] = (searches, max(points[-1][1], calls))
            continue
        points.append((searches, calls))
    return points


def _snapshot_path(dest: Path) -> Path:
    return dest / _SNAPSHOT_NAME


def _load_snapshots(dest: Path) -> list[tuple[int, int]]:
    path = _snapshot_path(dest)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        points = [(int(item[0]), int(item[1])) for item in data]
        if not points or points[0] != (0, 0):
            points.insert(0, (0, 0))
        return points
    log = Path(str(dest) + ".log")
    if log.is_file():
        return _snapshots_from_text(log.read_text(encoding="utf-8", errors="replace"))
    return [(0, 0)]


def _save_snapshots(dest: Path, snapshots: list[tuple[int, int]]) -> None:
    _snapshot_path(dest).write_text(json.dumps(snapshots) + "\n", encoding="utf-8")


def _can_resume(dest: Path) -> bool:
    """True when a previous AKMC pass already registered searches."""
    return (dest / "states" / "0" / "search_results.txt").is_file() and (dest / "info.txt").is_file()


def _clear_inflight(dest: Path) -> None:
    """Drop scratch from a killed batch. Registered state stays."""
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete"):
        path = dest / relative
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _prepare_server(template: dict, client: str, seed: int, workers: int, wrapper: Path, search_cap: int) -> str:
    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["job"] = "akmc"
    main["random_seed"] = int(seed)
    dimer = sections.setdefault("Dimer", {})
    for key in ("rotations_max", "converged_angle", "opt_method"):
        if key not in dimer:
            raise ValueError(f"{template['id']} is missing the dimer algorithm key {key}")
    optimizer = sections.get("Optimizer") or {}
    if int(optimizer.get("max_iterations") or 0) != 5000:
        raise ValueError(f"{template['id']} optimizer max_iterations is not the archive cap 5000")
    saddle = sections.get("Saddle Search") or {}
    for key in ("displace_radius", "displace_magnitude", "max_energy", "displace_listed_atom_weight", "displace_atom_list"):
        if key not in saddle:
            raise ValueError(f"{template['id']} is missing the archived displacement key {key}")
    akmc = sections.get("AKMC") or {}
    if "confidence" not in akmc:
        raise ValueError(f"{template['id']} is missing AKMC confidence")
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    comm = sections.setdefault("Communicator", {})
    comm["type"] = "local"
    comm["client_path"] = client
    slots, queue = _live_communicator(workers, search_cap)
    comm["number_of_CPUs"] = slots
    # The archive buffer is the search cap. The local communicator runs
    # the queue before it returns, keeping ``slots`` clients busy.
    comm["num_jobs"] = queue
    return format_ini(sections)


def _flush_pending(akmc_mod, config) -> None:
    """Register jobs the pass just finished.

    ``akmc`` registers the previous pass and then submits the next one.
    The local communicator runs that batch before returning, so a second
    ``register_results`` is what puts those rows in ``search_results.txt``.
    """
    import inspect

    from eon import explorer

    kT = config.main_temperature / 11604.5
    getter = akmc_mod.get_akmc_metadata
    if "config" in inspect.signature(getter).parameters:
        meta = getter(config)
        states = akmc_mod.get_statelist(kT, config)
    else:
        meta = getter()
        states = akmc_mod.get_statelist(kT)
    start_state_num, _time, previous_state_num, _first, _previous_temperature = meta
    current = states.get_state(start_state_num)
    if int(previous_state_num) == -1:
        previous = current
    else:
        previous = states.get_state(previous_state_num)
    factory = explorer.get_minmodexplorer
    if "config" in inspect.signature(factory).parameters:
        state_explorer = factory(config)(states, previous, current, superbasin=None, config=config)
    else:
        state_explorer = factory()(states, previous, current, superbasin=None)
    state_explorer.register_results()


def _load_akmc(family: str):
    root = _python_root(family)
    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if name == "eon" or name.startswith("eon."):
            del sys.modules[name]
    import inspect

    from eon import akmc as akmc_mod
    from eon.config import ConfigClass
    from eon.config import config as shared_config

    parameters = inspect.signature(akmc_mod.get_akmc_metadata).parameters
    if "config" in parameters:
        config = ConfigClass()
    else:
        config = shared_config
        config.init_done = False
    return akmc_mod, config


def _publish(
    output: Path,
    template: dict,
    rows: list[dict],
    harness: list[float],
    barriers: list[dict],
    rates: list[dict],
    *,
    total_calls: int,
    partial: bool,
    cap: int,
) -> dict:
    """Write the barrier and rate campaign files from the searches so far."""
    barrier_records = (
        _records(barriers, template, "barrier")
        or _search_records(rows, harness, template, "barrier")
        or [_whole_record(template, int(total_calls), "barrier")]
    )
    rate_template = dict(template)
    rate_template["id"] = template.get("rate_template") or template["id"]
    rate_records = (
        _records(rates, template, "rate")
        or _search_records(rows, harness, rate_template, "rate")
        or [_whole_record(template, int(total_calls), "rate")]
    )
    barrier_metrics = _write(
        output,
        template,
        barrier_records,
        label="barrier",
        windows=barriers,
        total_calls=int(total_calls),
        searches=len(rows),
        partial=partial,
        cap=cap,
    )
    if template.get("rate_template"):
        rate_output = output.with_name(str(template.get("rate_stem") or "htst-rate") + ".dat")
        _write(
            rate_output,
            rate_template,
            rate_records,
            label="rate",
            windows=rates,
            total_calls=int(total_calls),
            searches=len(rows),
            partial=partial,
            cap=cap,
        )
    return barrier_metrics


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
) -> dict:
    del geometries
    output = Path(output).resolve()
    family = str(template.get("client_family") or "")
    if family not in _ROOTS:
        raise ValueError(f"{template['id']} needs client_family svn or develop")
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    client = _client(family)
    wrapper = bridge_binary()
    cap = int(template["failure_cap"])
    search_cap = _search_cap(template, limit)
    partial = bool(limit)
    seed = int((template.get("ini") or {}).get("Main", {}).get("random_seed", 42))
    dest = (work or output.parent / "work" / template["id"].replace("/", "-")).resolve()
    resuming = _can_resume(dest)
    if dest.exists() and not resuming:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
        (dest / relative).mkdir(parents=True, exist_ok=True)
    if resuming:
        # Keep states/, info.txt, prng.pkl, and jobs.tbl. A killed batch
        # leaves incomplete clients in scratch; registering those would
        # invent searches that did not finish.
        _clear_inflight(dest)
    else:
        geometry = geometry_archive(root, {"id": problem["id"], "input": {"archive_path": template["geometry"]}})
        shutil.copy(geometry, dest / "pos.con")
    (dest / "config.ini").write_text(
        _prepare_server(template, client, seed, workers, wrapper, search_cap),
        encoding="utf-8",
    )
    server = PotServer(eam_al, cap=cap).start()
    # Each client evaluates the Fortran Aluminum EAM in its own bridge
    # process and only reports the call. The counter stays on this server.
    os.environ["OPTBENCH_EAM_LIB"] = str(ensure_eam())
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    previous = Path.cwd()
    snapshots = _load_snapshots(dest) if resuming else [(0, 0)]
    if resuming:
        held = dest / "states" / "0" / "search_results.txt"
        held_rows = _searches(held.read_text(encoding="utf-8", errors="replace"))
        snapshots = [point for point in snapshots if point[0] <= len(held_rows)] or [(0, 0)]
        if held_rows and snapshots[-1][0] < len(held_rows):
            snapshots.append((len(held_rows), snapshots[-1][1]))
        _save_snapshots(dest, snapshots)
        print(
            f"{template['id']} resume searches={len(held_rows)} calls={snapshots[-1][1]} workers={workers}",
            flush=True,
        )
    # The new counter starts at zero. Earlier passes stay on this offset.
    base_calls = snapshots[-1][1]
    error = ""
    started = time.monotonic()
    try:
        os.chdir(dest)
        akmc_mod, config = _load_akmc(family)
        config.init()
        from eon import communicator

        if hasattr(communicator.get_communicator, "comm"):
            del communicator.get_communicator.comm
        index = 0
        stalls = 0
        faults = 0
        while True:
            index += 1
            try:
                akmc_mod.akmc(config)
                _flush_pending(akmc_mod, config)
            except SystemExit as exc:
                error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                faults += 1
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                print(f"{template['id']} pass {index} fault {faults}: {error.splitlines()[0]}", flush=True)
                if hasattr(communicator.get_communicator, "comm"):
                    del communicator.get_communicator.comm
                if faults >= 5:
                    break
                continue
            faults = 0
            results = dest / "states" / "0" / "search_results.txt"
            text = results.read_text(encoding="utf-8", errors="replace") if results.is_file() else ""
            rows = _searches(text)
            live, _verify = remote_count(server.host, server.port)
            calls = base_calls + int(live)
            # A client that dies with SIGFPE, or a communicator that drops
            # one batch, leaves the reactant search count unchanged. That is
            # not the archive stop. Retry unless it keeps happening.
            if snapshots and rows and len(rows) == snapshots[-1][0]:
                stalls += 1
                print(
                    f"{template['id']} pass {index} no new searches at {len(rows)} (stall {stalls})",
                    flush=True,
                )
                if hasattr(communicator.get_communicator, "comm"):
                    del communicator.get_communicator.comm
                if stalls >= 5 or len(rows) >= search_cap:
                    if len(rows) < search_cap:
                        error = error or f"no new searches at {len(rows)}"
                    break
                continue
            stalls = 0
            snapshots.append((len(rows), int(calls)))
            _save_snapshots(dest, snapshots)
            table = dest / "states" / "0" / "processtable"
            reference, missed = _reference_ids(
                table.read_text(encoding="utf-8", errors="replace") if table.is_file() else ""
            )
            harness = _assign(rows, snapshots)
            barriers, rates = _windows(rows, harness, reference, missed)
            print(
                f"{template['id']} pass {index} searches={len(rows)} "
                f"barrier_windows={len(barriers)} rate_windows={len(rates)} calls={calls}",
                flush=True,
            )
            if index % 20 == 0:
                _publish(
                    output,
                    template,
                    rows,
                    harness,
                    barriers,
                    rates,
                    total_calls=int(calls),
                    partial=partial,
                    cap=cap,
                )
            if len(barriers) >= _BARRIER_TARGET and len(rates) >= _RATE_TARGET:
                break
            if len(rows) >= search_cap:
                break
    finally:
        os.chdir(previous)
        try:
            live, _verify = remote_count(server.host, server.port)
            calls = base_calls + int(live)
        except Exception:
            calls = snapshots[-1][1] if snapshots else 0
        server.stop()
        os.environ.pop("OPTBENCH_POTSERVER", None)
        os.environ.pop("OPTBENCH_EAM_LIB", None)
    results = dest / "states" / "0" / "search_results.txt"
    text = results.read_text(encoding="utf-8", errors="replace") if results.is_file() else ""
    rows = _searches(text)
    table = dest / "states" / "0" / "processtable"
    reference, missed = _reference_ids(table.read_text(encoding="utf-8", errors="replace") if table.is_file() else "")
    if len(snapshots) == 1 or snapshots[-1][0] != len(rows):
        snapshots.append((len(rows), int(calls)))
    harness = _assign(rows, snapshots)
    barriers, rates = _windows(rows, harness, reference, missed)
    barrier_metrics = _publish(
        output,
        template,
        rows,
        harness,
        barriers,
        rates,
        total_calls=int(calls),
        partial=partial,
        cap=cap,
    )
    if barriers:
        starts = len(barriers)
    elif rows:
        starts = len(rows)
    else:
        starts = 1
    document = {
        "schema": "optbench.htst/stock",
        "error": error,
        "seconds": time.monotonic() - started,
        "searches": len(rows),
        "reference_ids": reference,
        "harness_total": int(calls),
        "barrier_windows": barriers,
        "rate_windows": rates,
        "workers": int(workers),
    }
    output.with_suffix(".windows.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        f"{template['id']} searches={len(rows)} barrier_windows={len(barriers)} "
        f"rate_windows={len(rates)} calls={calls} error={error.splitlines()[:1]}",
        flush=True,
    )
    return {
        "starts": starts,
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": barrier_metrics["par2_mean"],
        "nfailed": barrier_metrics["nfailed"],
        "partial": partial,
        "barrier_windows": len(barriers),
        "rate_windows": len(rates),
        "searches": len(rows),
        "harness_total": int(calls),
        "rate_output": (
            str(output.with_name(str(template.get("rate_stem") or "htst-rate") + ".dat"))
            if template.get("rate_template")
            else ""
        ),
    }
