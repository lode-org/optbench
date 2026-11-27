"""Head-to-head rgsaddle develop runs on Pt-island.

Each arm runs on a pinned core with one thread. Timing is segment-based:
every invocation appends a segment, and the total is the sum of the
completed segments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import socket
import time
from pathlib import Path

from optbench.campaign import load_template, resolve_problem
from optbench.rgsaddle_minmode import run_template

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261016
SHAS = {
    "old": "179e90f29a3b2cbd4ea5804a5c4c132ab19acba4",
    "new": "aa6dd9371d6a63103010a6829cb62c74a39c7ff1",
}
METHODS = ("dimer-fire", "dimer-lbfgs", "dimer-rotskip", "lanczos")
THREAD_VARS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _load_catalog_module():
    path = ROOT / "results" / "campaign" / "rgsaddle_develop_catalog.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_develop_catalog", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog = _load_catalog_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_thread(cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})
    for name in THREAD_VARS:
        os.environ[name] = "1"


def _checkpoint_dir(method: str, arm: str) -> Path:
    path = ROOT / "build" / "rgsaddle-h2h" / method / arm
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_token(token: str) -> str:
    candidate = Path(token)
    if candidate.is_absolute():
        return candidate.name
    return token


def _sanitize_argv(argv: list[str]) -> list[str]:
    return [_sanitize_token(item) for item in argv]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _merge_wall_comparable(stored: dict, extra: dict | None) -> None:
    if not extra:
        return
    merged = dict(extra)
    if stored.get("wall_comparable") is False:
        merged["wall_comparable"] = False
    stored.update(merged)


def _timing_start(method: str, arm: str, extra: dict | None = None) -> float:
    now = time.time()
    folder = _checkpoint_dir(method, arm)
    timing_path = folder / "timing.json"
    segments = folder / "timing.segments.jsonl"
    payload = _read_json(timing_path)
    payload.setdefault("started_unix", now)
    payload.setdefault("host", socket.gethostname())
    _merge_wall_comparable(payload, extra)
    timing_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with segments.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "segment_started_unix": now,
                    "argv": _sanitize_argv(os.sys.argv),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                },
                sort_keys=True,
            )
            + "\n"
        )
    return now


def _timing_finish(method: str, arm: str, started: float, extra: dict | None = None) -> None:
    finished = time.time()
    elapsed = max(0.0, finished - started)
    folder = _checkpoint_dir(method, arm)
    timing_path = folder / "timing.json"
    segments = folder / "timing.segments.jsonl"
    payload = _read_json(timing_path)
    payload["finished_unix"] = finished
    payload["wall_seconds_total"] = float(payload.get("wall_seconds_total") or 0.0) + elapsed
    _merge_wall_comparable(payload, extra)
    timing_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with segments.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "segment_finished_unix": finished,
                    "segment_wall_seconds": elapsed,
                    "argv": _sanitize_argv(os.sys.argv),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _output_path(method: str, arm: str) -> Path:
    if method == "eon-classic":
        return ROOT / "results" / "campaign" / f"eon-3.5.0-saddle-pt-classic-h2h-{arm}.dat"
    sha8 = SHAS[arm][:8]
    return ROOT / "results" / "campaign" / f"rgsaddle-develop-{sha8}-h2h-saddle-{method}.dat"


def _write_command(method: str, arm: str, cpu: int, extra: dict) -> None:
    payload = {
        "method": method,
        "arm": arm,
        "seed": SEED,
        "argv": _sanitize_argv(os.sys.argv),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "requested_cpu": cpu,
        "threads": {name: os.environ.get(name, "") for name in THREAD_VARS},
    }
    payload.update(extra)
    (_checkpoint_dir(method, arm) / "command.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_rgsaddle(method: str, arm: str, cpu: int, lib: Path, src: Path | None) -> None:
    _single_thread(cpu)
    os.environ["RGSADDLE_LIB"] = str(lib)
    if src is not None:
        os.environ["RGSADDLE_SRC"] = str(src)
    os.environ["OPTBENCH_RGSADDLE_CHECKPOINT_DIR"] = str(_checkpoint_dir(method, arm))
    _write_command(
        method,
        arm,
        cpu,
        {
            "rgsaddle_head": SHAS[arm],
            "rgsaddle_lib_sha256": _sha256(lib),
        },
    )
    template = copy.deepcopy(load_template(ROOT, catalog.RGSADDLE_ROWS[method]))
    template["commit"] = SHAS[arm]
    template["code_version"] = f"(develop {SHAS[arm][:8]})"
    template["seed"] = str(SEED)
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    output = _output_path(method, arm)
    started = _timing_start(method, arm, {"method": method, "arm": arm, "commit": SHAS[arm], "wall_comparable": True})
    run_template(ROOT, problem, template, output=output, workers=1, repetition=1)
    _timing_finish(method, arm, started, {"method": method, "arm": arm, "commit": SHAS[arm], "wall_comparable": True})


def run_eon(arm: str, cpu: int, client: Path, provenance_facts: dict | None = None) -> None:
    _single_thread(cpu)
    method = "eon-classic"
    os.environ["OPTBENCH_RGSADDLE_CHECKPOINT_DIR"] = str(_checkpoint_dir(method, arm))
    _write_command(
        method,
        arm,
        cpu,
        {
            "eon_client": "eonclient",
            "eon_client_sha256": _sha256(client),
            "eon_arm_commit": SHAS[arm],
        },
    )
    output = _output_path(method, arm)
    checkpoint = _checkpoint_dir(method, arm)
    starts = output.with_suffix(".starts.jsonl")
    # Reuse the existing runner with arm-local paths.
    saved_output = catalog._output_path
    saved_checkpoint = catalog._checkpoint_dir
    saved_provenance = catalog.PROVENANCE_PATH
    try:
        catalog._output_path = lambda _method: output  # type: ignore[assignment]
        catalog._checkpoint_dir = lambda _method: checkpoint  # type: ignore[assignment]
        catalog.PROVENANCE_PATH = checkpoint / "eon-provenance.json"
        catalog.run_eon(cpu, client, provenance_facts)
    finally:
        catalog._output_path = saved_output  # type: ignore[assignment]
        catalog._checkpoint_dir = saved_checkpoint  # type: ignore[assignment]
        catalog.PROVENANCE_PATH = saved_provenance
    timing_path = checkpoint / "timing.json"
    payload = _read_json(timing_path)
    payload.setdefault("method", method)
    payload.setdefault("arm", arm)
    if payload.get("wall_comparable") is not False:
        payload["wall_comparable"] = False
    timing_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    if starts.is_file():
        # catalog.run_eon already wrote the starts ledger; keep a deterministic newline.
        starts.write_text(starts.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")


def sample_system(interval: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        now = time.time()
        load1, load5, load15 = os.getloadavg()
        stat = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        payload = {
            "unix": now,
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "cpu_total_jiffies": int(sum(int(value) for value in stat[1:])),
            "cpu_steal_jiffies": int(stat[8]) if len(stat) > 8 else 0,
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    rgs = sub.add_parser("run-rgsaddle")
    rgs.add_argument("method", choices=list(METHODS))
    rgs.add_argument("arm", choices=["old", "new"])
    rgs.add_argument("cpu", type=int)
    rgs.add_argument("lib", type=Path)
    rgs.add_argument("--src", type=Path, default=None)

    eon = sub.add_parser("run-eon")
    eon.add_argument("arm", choices=["old", "new"])
    eon.add_argument("cpu", type=int)
    eon.add_argument("client", type=Path)
    eon.add_argument("facts", type=Path)

    sample = sub.add_parser("sample-system")
    sample.add_argument("--interval", type=int, default=60)
    sample.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "rgsaddle-h2h" / "system-samples.jsonl",
    )

    args = parser.parse_args()
    if args.command == "run-rgsaddle":
        run_rgsaddle(args.method, args.arm, args.cpu, args.lib, args.src)
    elif args.command == "run-eon":
        run_eon(args.arm, args.cpu, args.client, catalog._load_provenance_facts(args.facts))
    elif args.command == "sample-system":
        sample_system(args.interval, args.output)


if __name__ == "__main__":
    main()
