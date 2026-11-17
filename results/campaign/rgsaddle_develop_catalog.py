"""Pt-island catalog on rgsaddle develop, plus the eOn classic comparator.

Seed 20261016. Each row stops at 27 connected saddles or 20000 searches.
One process uses one core and one thread. Checkpoints are catalog.json
and searches.jsonl under build/rgsaddle/<method>/ so a row can resume.

The csv command writes results/campaign/rgsaddle-develop-179e90f2-pt-island.csv
from the search ledgers, the campaign files, and
results/campaign/rgsaddle-develop-179e90f2-pt-island-hardware.json.
Nothing in that file is typed by hand. Method rows keep the force-call
budget in ``budget`` and the cap failure count in ``cap_failures``.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHA = "179e90f29a3b2cbd4ea5804a5c4c132ab19acba4"
SEED = 20261016
GOAL = 27
MAX_SEARCHES = 20000
BARRIER_EV = 1.5
CSV_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-179e90f2-pt-island.csv"
HARDWARE_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-179e90f2-pt-island-hardware.json"
TIMING_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-179e90f2-pt-island-timing.json"
COMMANDS_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-179e90f2-pt-island-commands.json"
PROVENANCE_PATH = ROOT / "results" / "campaign" / "rgsaddle-develop-179e90f2-eon-provenance.json"

RGSADDLE_ROWS = {
    "dimer-fire": "rgsaddle/develop-saddle-dimer-fire",
    "dimer-lbfgs": "rgsaddle/develop-saddle-dimer-lbfgs",
    "dimer-rotskip": "rgsaddle/develop-saddle-dimer-rotskip",
    "lanczos": "rgsaddle/develop-saddle-lanczos",
}
THREAD_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _single_thread(cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})
    for name in THREAD_VARS:
        os.environ[name] = "1"


def _checkpoint_dir(method: str) -> Path:
    path = ROOT / "build" / "rgsaddle" / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_path(method: str) -> Path:
    if method == "eon-classic":
        return ROOT / "results" / "campaign" / "eon-3.5.0-saddle-pt-classic-develop-protocol.dat"
    return ROOT / "results" / "campaign" / f"rgsaddle-develop-179e90f2-saddle-{method}.dat"


def classify_search(record: dict, step_ceiling: int | None) -> str:
    """One class per search.

    Divergence is only a non-finite point or a step past the ceiling.
    When more than one reason is present the class is the first of
    duplicate, over_barrier, disconnected, cap, and hessian_index.
    eOn ``Saddle Search Terminated High Energy`` is ``high_energy``.
    Anything left is wrong_stationary.
    """
    if record.get("success"):
        return "connected"
    reasons = {str(item) for item in (record.get("reasons") or [])}
    step_max = int(record.get("step_calls_max") or 0)
    barrier = record.get("barrier")
    non_finite = "divergence" in reasons
    if barrier is not None:
        try:
            non_finite = non_finite or not (float(barrier) == float(barrier))
        except (TypeError, ValueError):
            non_finite = True
    if non_finite or (step_ceiling is not None and step_max > step_ceiling):
        return "divergence"
    if "duplicate" in reasons:
        return "duplicate"
    if "barrier" in reasons or "over_barrier" in reasons:
        return "over_barrier"
    if barrier is not None:
        try:
            if float(barrier) >= BARRIER_EV and "not_connected" not in reasons:
                return "over_barrier"
        except (TypeError, ValueError):
            return "divergence"
    if "not_connected" in reasons:
        return "disconnected"
    if "cap" in reasons:
        return "cap"
    if "hessian_index" in reasons:
        return "hessian_index"
    if str(record.get("result") or "") == "Saddle Search Terminated High Energy":
        return "high_energy"
    return "wrong_stationary"


def _overlap(record: dict) -> dict[str, int]:
    reasons = {str(item) for item in (record.get("reasons") or [])}
    barrier = record.get("barrier")
    high = False
    if barrier is not None:
        try:
            high = float(barrier) >= BARRIER_EV
        except (TypeError, ValueError):
            high = False
    return {
        "disconnected": int("not_connected" in reasons),
        "over_barrier": int("barrier" in reasons or "over_barrier" in reasons or high),
        "duplicate": int("duplicate" in reasons),
    }


def tally(records: list[dict], step_ceiling: int | None) -> dict[str, int]:
    counts = {
        "searches": len(records),
        "connected": 0,
        "disconnected": 0,
        "over_barrier": 0,
        "duplicate": 0,
        "wrong_stationary": 0,
        "divergence": 0,
        "cap": 0,
        "hessian_index": 0,
        "high_energy": 0,
        "overlap_disconnected": 0,
        "overlap_over_barrier": 0,
        "overlap_duplicate": 0,
    }
    for record in records:
        counts[classify_search(record, step_ceiling)] += 1
        if record.get("success"):
            continue
        for key, value in _overlap(record).items():
            counts[f"overlap_{key}"] += value
    return counts


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def step_ceiling(method: str) -> int | None:
    if method not in RGSADDLE_ROWS:
        return None
    from optbench.campaign import load_template
    from optbench.rgsaddle_minmode import _step_ceiling

    template = load_template(ROOT, RGSADDLE_ROWS[method])
    return int(_step_ceiling(template["ini"]["SaddleSearch"]))


def _ledger(method: str) -> list[dict]:
    finished = _output_path(method).with_suffix(".starts.jsonl")
    if finished.is_file() and finished.stat().st_size:
        return _load_jsonl(finished)
    return _load_jsonl(_checkpoint_dir(method) / "searches.jsonl")


def progress_text() -> str:
    lines = [
        f"rgsaddle develop {SHA}",
        "ancestor ef1fd6463763be811a52ec33bae1f21870a600d4",
        "includes 116b92e83fdbd5ec24fcfe500e9de2663d1a35b2 retain rotation state across translations",
        f"seed {SEED}",
        f"stop {GOAL} connected or {MAX_SEARCHES} searches",
        "exclusive class order: connected, divergence, duplicate, over_barrier, disconnected, cap, hessian_index, high_energy, wrong_stationary",
        "divergence: non-finite point or a step past the ceiling",
        f"over_barrier: barrier at least {BARRIER_EV} eV",
    ]
    for method in (*RGSADDLE_ROWS, "eon-classic"):
        counts = tally(_ledger(method), step_ceiling(method))
        lines.append(
            "method {method} searches {searches} connected {connected} "
            "disconnected {disconnected} over_barrier {over_barrier} "
            "duplicate {duplicate} cap {cap} hessian_index {hessian_index} "
            "high_energy {high_energy} wrong_stationary {wrong_stationary} "
            "divergence {divergence} "
            "overlap_disconnected {overlap_disconnected} "
            "overlap_over_barrier {overlap_over_barrier} "
            "overlap_duplicate {overlap_duplicate}".format(method=method, **counts)
        )
    return "\n".join(lines) + "\n"


def _write_command(method: str, cpu: int) -> None:
    head = ""
    src = os.environ.get("RGSADDLE_SRC")
    if src:
        import subprocess

        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    payload = {
        "method": method,
        "argv": sys.argv,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "requested_cpu": cpu,
        "threads": {name: os.environ.get(name, "") for name in THREAD_VARS},
        "rgsaddle_head": head,
        "rgsaddle_lib": os.environ.get("RGSADDLE_LIB", ""),
    }
    (_checkpoint_dir(method) / "command.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_rgsaddle(method: str, cpu: int) -> None:
    _single_thread(cpu)
    os.environ["OPTBENCH_RGSADDLE_CHECKPOINT_DIR"] = str(_checkpoint_dir(method))
    _write_command(method, cpu)
    from optbench.campaign import load_template, resolve_problem
    from optbench.rgsaddle_minmode import run_template

    template = load_template(ROOT, RGSADDLE_ROWS[method])
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    output = _output_path(method)
    started = time.time()
    (_checkpoint_dir(method) / "timing.json").write_text(
        json.dumps({"method": method, "started_unix": started}) + "\n", encoding="utf-8"
    )
    run_template(ROOT, problem, template, output=output, workers=1, repetition=1)
    finished = time.time()
    (_checkpoint_dir(method) / "timing.json").write_text(
        json.dumps(
            {
                "method": method,
                "started_unix": started,
                "finished_unix": finished,
                "wall_seconds": finished - started,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _eon_records(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 7:
            continue
        if len(rows) >= MAX_SEARCHES:
            break
        try:
            barrier = float(fields[2])
            sad = int(fields[4])
            mins = int(fields[5])
        except ValueError:
            continue
        token = " ".join(fields[7:])
        reasons: list[str] = []
        success = False
        finite = barrier == barrier
        if not finite:
            reasons = ["divergence"]
        elif token.startswith("repeat"):
            reasons = ["duplicate"]
            if barrier >= BARRIER_EV:
                reasons.append("barrier")
        elif token.startswith("good") and barrier < BARRIER_EV:
            success = True
        elif token.startswith("good"):
            reasons = ["barrier"]
        elif "Not Connected" in token:
            reasons = ["not_connected"]
            if barrier >= BARRIER_EV:
                reasons.append("barrier")
        else:
            reasons = ["force_norm"]
        rows.append(
            {
                "success": success,
                "reasons": reasons,
                "barrier": barrier,
                "result": token,
                "force_calls": sad + mins,
            }
        )
        if sum(1 for row in rows if row["success"]) >= GOAL:
            break
    return rows


def _client_version(client: Path) -> str:
    import subprocess

    out = subprocess.check_output([str(client), "--version"], text=True, stderr=subprocess.STDOUT)
    return " ".join(out.split())


def _write_eon_ledger(folder: Path, records: list[dict]) -> None:
    ledger = folder / "searches.jsonl"
    with ledger.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def run_eon(cpu: int, client: Path) -> None:
    """Classic dimer, one core, seed 20261016, stop at 27 or 20000 searches.

    ``client`` is a local prefix of conda-forge ``eon`` 3.5.0 build
    ``np2py312h3e7f63d_0``. ``eonclient --version`` prints the feedstock
    commit. The public source pin is ``rgsaddle-develop-179e90f2-eon-provenance.json``.
    """
    import inspect
    import logging
    import traceback

    _single_thread(cpu)
    folder = _checkpoint_dir("eon-classic")
    _write_command("eon-classic", cpu)
    version = _client_version(client)
    from optbench.campaign import load_template
    from optbench.eon_akmc import _write_config, prepare_client_env
    from optbench.eon_parse import saddle_progress

    template = copy.deepcopy(load_template(ROOT, "eon/develop-saddle-pt-dimer"))
    template["ini"]["Communicator"]["num_jobs"] = 1
    template["ini"]["Communicator"]["number_of_CPUs"] = 1
    template["ini"]["Main"]["random_seed"] = SEED
    work = folder / "work"
    fresh = not (work / "states").is_dir()
    if fresh:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        pos = ROOT / "website" / "saddle-search" / "pt-island" / "pt-island.con"
        if not pos.is_file():
            from optbench.campaign import geometry_archive

            pos = geometry_archive(
                ROOT,
                {"id": "saddle", "input": {"archive_path": "website/saddle-search/pt-island/pt-island.con"}},
            )
        shutil.copy(pos, work / "pos.con")
        _write_config(template, work / "config.ini", client, SEED, 1)
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
        (work / relative).mkdir(parents=True, exist_ok=True)
    prepare_client_env(client)
    for name in ("akmc", "eon", "state", "communicator"):
        logging.getLogger(name).setLevel(logging.WARNING)
    started = time.time()
    timing_path = folder / "timing.json"
    if not timing_path.is_file():
        timing_path.write_text(
            json.dumps({"method": "eon-classic", "started_unix": started, "client_version": version}) + "\n",
            encoding="utf-8",
        )
    else:
        started = float(json.loads(timing_path.read_text(encoding="utf-8")).get("started_unix", started))
    previous = Path.cwd()
    last_error = ""
    records: list[dict] = []
    try:
        os.chdir(work)
        from eon import akmc as akmc_mod
        from eon.config import ConfigClass
        from eon.config import config as shared_config

        parameters = inspect.signature(akmc_mod.get_akmc_metadata).parameters
        if "config" in parameters:
            config = ConfigClass()
        else:
            config = shared_config
            config.init_done = False
        config.init()
        from eon import communicator

        if hasattr(communicator.get_communicator, "comm"):
            del communicator.get_communicator.comm
        for index in range(MAX_SEARCHES):
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                last_error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                break
            results = work / "states" / "0" / "search_results.txt"
            if not results.is_file():
                continue
            text = results.read_text(encoding="utf-8", errors="replace")
            records = _eon_records(text)
            _write_eon_ledger(folder, records)
            progress = saddle_progress(text)
            connected = sum(1 for record in records if record["success"])
            if (index + 1) % 25 == 0 or connected >= GOAL or len(records) >= MAX_SEARCHES:
                print(
                    f"eon-classic searches {len(records)} connected {connected} "
                    f"qualifying {progress['qualifying']}",
                    flush=True,
                )
            if connected >= GOAL or len(records) >= MAX_SEARCHES or progress["reached"]:
                break
    finally:
        os.chdir(previous)
    results = work / "states" / "0" / "search_results.txt"
    if results.is_file():
        records = _eon_records(results.read_text(encoding="utf-8", errors="replace"))
        _write_eon_ledger(folder, records)
    finished = time.time()
    timing_path.write_text(
        json.dumps(
            {
                "method": "eon-classic",
                "started_unix": started,
                "finished_unix": finished,
                "wall_seconds": finished - started,
                "client_version": version,
                "error": last_error,
                "ledger_searches": len(records),
                "ledger_connected": sum(1 for record in records if record["success"]),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_eon_dat(records, finished - started, version, last_error)


def _write_eon_dat(records: list[dict], wall: float, version: str, error: str) -> None:
    output = _output_path("eon-classic")
    connected = sum(1 for record in records if record["success"])
    calls = [int(record["force_calls"]) for record in records]
    stopped = connected >= GOAL or len(records) >= MAX_SEARCHES
    lines = [
        f"# rep 1 force_calls {sum(calls)} total_searches {len(records)} connected {connected} seed {SEED}",
        "code eOn",
        f"code_version {version}",
        "algorithm Dimer",
        "dimer_improved false",
        "template eon/develop-saddle-pt-dimer",
        "problem saddle-search/pt-island",
        "counter client-search-results",
        "count_keys sad-fcs+mins-fcs",
        f"seed {SEED}",
        "communicator_jobs 1",
        "repetitions 1",
        f"total_searches {len(records)}",
        f"connected {connected}",
        f"goal {GOAL}",
        f"nfailed {0 if stopped else 1}",
        f"wall_seconds {wall:.6f}",
    ]
    if error:
        lines.append(f"error {error.splitlines()[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    starts = output.with_suffix(".starts.jsonl")
    with starts.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _hardware_rows() -> list[dict[str, str]]:
    recorded = json.loads(HARDWARE_PATH.read_text(encoding="utf-8"))
    keys = (
        "model_name",
        "cpu_cores",
        "cpus_online",
        "mem_total_kb",
        "affinity_policy",
        "thread_vars",
    )
    return [{"row_kind": "hardware", "key": key, "value": str(recorded[key])} for key in keys]


def _command_rows() -> list[dict[str, str]]:
    stored = json.loads(COMMANDS_PATH.read_text(encoding="utf-8")) if COMMANDS_PATH.is_file() else {}
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")) if PROVENANCE_PATH.is_file() else {}
    rows = []
    for method in (*RGSADDLE_ROWS, "eon-classic"):
        payload = stored.get(method)
        if payload is None:
            path = _checkpoint_dir(method) / "command.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "row_kind": "command",
            "method": method,
            "argv": " ".join(payload.get("argv") or []) if isinstance(payload.get("argv"), list) else str(payload.get("argv") or ""),
            "cpu_affinity": ",".join(str(cpu) for cpu in payload.get("cpu_affinity") or []),
            "threads": " ".join(f"{key}={value}" for key, value in sorted((payload.get("threads") or {}).items())),
            "rgsaddle_head": payload.get("rgsaddle_head", ""),
        }
        if method == "eon-classic":
            row["argv_origin"] = str(provenance.get("argv_origin") or "")
        rows.append(row)
    return rows


def _protocol_budget(method: str, fields: dict[str, str]) -> str:
    """Force-call budget on the dat.

    The eOn classic dat has no cap line. Its template uses the same
    saddle and optimizer iteration limits as the dimer rows, so the
    bound is climb steps times the step ceiling plus both endpoint
    minimizations.
    """
    recorded = fields.get("cap", "")
    if recorded or method != "eon-classic":
        return recorded
    from optbench.campaign import load_template
    from optbench.rgsaddle_minmode import _step_ceiling

    template = load_template(ROOT, fields.get("template") or "eon/develop-saddle-pt-dimer")
    ini = template["ini"]
    saddle = dict(ini["Saddle Search"])
    dimer = ini.get("Dimer") or {}
    if "max_rotations" not in saddle and dimer.get("rotations_max") not in (None, ""):
        saddle["max_rotations"] = dimer["rotations_max"]
    ceiling = _step_ceiling(saddle)
    climb = int(saddle["max_iterations"]) * ceiling
    return str(climb + 2 * int(ini["Optimizer"]["max_iterations"]))


def _method_rows() -> list[dict[str, str]]:
    from optbench.score import score_v2

    rows = []
    for method in (*RGSADDLE_ROWS, "eon-classic"):
        records = _ledger(method)
        counts = tally(records, step_ceiling(method))
        timing = {}
        if TIMING_PATH.is_file():
            timing = json.loads(TIMING_PATH.read_text(encoding="utf-8")).get(method) or {}
        dat = _output_path(method)
        fields = {}
        if dat.is_file():
            for line in dat.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                key, _, value = line.partition(" ")
                fields[key] = value
        scored = ""
        if records and method != "eon-classic":
            cap = int(fields.get("cap") or 0) or None
            metrics = score_v2(
                [
                    {
                        "force_calls": int(record["force_calls"]),
                        "success": bool(record["success"]),
                        "failure": record.get("failure"),
                    }
                    for record in records
                ],
                cap,
            )
            scored = f"{metrics['par2_mean']}"
            counts["par2_nfailed"] = metrics["nfailed"]
        row = {
            "row_kind": "method",
            "method": method,
            "template": fields.get("template", RGSADDLE_ROWS.get(method, "")),
            "commit": fields.get("commit", SHA if method != "eon-classic" else ""),
            "code_version": fields.get("code_version", ""),
            "client_version": timing.get("client_version", ""),
            "seed": fields.get("seed", str(SEED)),
            "wall_seconds": "" if timing.get("wall_seconds") is None else f"{float(timing['wall_seconds']):.6f}",
            "par2_mean": scored or fields.get("par2_mean", ""),
            "force_calls": fields.get("force_calls", "")
            or (str(sum(int(record["force_calls"]) for record in records)) if records else ""),
            "budget": _protocol_budget(method, fields),
            "goal": fields.get("goal", str(GOAL)),
            "cap_failures": str(counts["cap"]),
        }
        for key, value in counts.items():
            if key == "cap":
                continue
            row[key] = str(value)
        rows.append(row)
    return rows


def write_csv() -> Path:
    hardware = _hardware_rows()
    commands = _command_rows()
    methods = _method_rows()
    columns: list[str] = []
    for row in hardware + commands + methods:
        for key in row:
            if key not in columns:
                columns.append(key)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in hardware + commands + methods:
            writer.writerow(row)
    return CSV_PATH


def main(argv: list[str]) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: rgsaddle_develop_catalog.py run <method> <cpu> | eon <cpu> <client> | progress <path> | csv",
            file=sys.stderr,
        )
        raise SystemExit(2)
    command = argv[0]
    if command == "run":
        method, cpu = argv[1], int(argv[2])
        if method not in RGSADDLE_ROWS:
            raise SystemExit(f"unknown method {method}")
        run_rgsaddle(method, cpu)
        return
    if command == "eon":
        run_eon(int(argv[1]), Path(argv[2]))
        return
    if command == "progress":
        Path(argv[1]).write_text(progress_text(), encoding="utf-8")
        return
    if command == "csv":
        print(write_csv())
        return
    raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
