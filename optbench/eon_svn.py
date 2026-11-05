"""Write harness campaign files for the eOn SVN r2549 templates.

The start list is the campaign grid on ``benchmarks.bench_eon_svn``.
A file with fewer starts than that grid sets ``partial true``.
"""

from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.failures import FAILURES
from optbench.score import score_v2

ROOT = Path(__file__).resolve().parents[1]


def stem_for(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return f"eon-svn-r2549-{slug}"


def _spec_map() -> dict[str, dict]:
    from benchmarks.bench_eon_svn import SPECS

    return {spec["template"]: spec for spec in SPECS}


def _job(spec: tuple) -> dict:
    template_name, kind, seed, start, passes, workers = spec
    from benchmarks.bench_eon_svn import run_case

    try:
        return run_case(template_name, kind, int(seed), str(start), passes=int(passes), workers=int(workers))
    except Exception as exc:
        return {
            "force_calls": 0.0,
            "success": False,
            "failure": "crash",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": str(seed),
            "start": str(start),
            "seconds": 0.0,
            "counter": "harness",
            "template": template_name,
        }


def _key(row: dict) -> tuple[str, str]:
    return str(row.get("seed")), str(row.get("start"))


def _load_done(path: Path) -> dict[tuple[str, str], dict]:
    found = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("force_calls") is None or "success" not in row:
            continue
        found[_key(row)] = row
    return found


def _public_row(row: dict) -> dict:
    kept = {
        "schema": "optbench.start/v2",
        "start": str(row.get("start")),
        "seed": str(row.get("seed")),
        "force_calls": int(row["force_calls"]) if float(row["force_calls"]).is_integer() else float(row["force_calls"]),
        "counter": "harness",
        "comparable": True,
        "success": bool(row["success"]),
        "template": row.get("template", ""),
        "problem": row.get("problem", ""),
        "code": row.get("code", "Eon svn"),
        "code_version": row.get("code_version", "r2549"),
        "commit": row.get("commit", "r2549"),
        "tuning": row.get("tuning", "default"),
        "seconds": float(row.get("seconds") or 0.0),
        "call_unit": "total",
    }
    if not kept["success"]:
        kept["failure"] = row.get("failure") or "crash"
    for name in (
        "client_calls",
        "client_reason",
        "hessian_index",
        "overlap",
        "k",
        "extrema",
        "fraction_connected",
        "passes",
        "qualifying",
        "total_searches",
        "unique_saddles",
        "error",
    ):
        if row.get(name) not in (None, ""):
            kept[name] = row[name]
    return kept


def _dat(template: dict, rows: list[dict], *, partial: bool, cap: int) -> str:
    records = []
    for row in rows:
        item = {"force_calls": row["force_calls"], "success": row["success"]}
        if not row["success"]:
            item["failure"] = row.get("failure") or "crash"
        records.append(item)
    summary = score_v2(records, cap)
    calls = [float(row["force_calls"]) for row in rows]
    lines = [
        f"# eOn SVN r2549. HarnessCounter through ext_pot. Kernel {template['ini']['Potential']['potential']}.",
        f"# {template.get('note', '').splitlines()[0] if template.get('note') else ''}".rstrip(),
        "counter harness",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template.get('algorithm', '')}",
        f"tuning {template.get('tuning', 'default')}",
        f"template {template['id']}",
        f"problem {template['problem']}",
        "reducer harness",
        f"cap {cap}",
        f"n {summary['n']}",
        f"par2_mean {summary['par2_mean']}",
        f"nfailed {summary['nfailed']}",
        f"mean_calls {sum(calls) / len(calls)}",
        f"median_calls {statistics.median(calls)}",
        f"min_calls {min(calls)}",
        f"max_calls {max(calls)}",
    ]
    for name in FAILURES:
        lines.append(f"n_{name} {summary[f'n_{name}']}")
    if partial:
        lines.append("partial true")
    return "\n".join(line for line in lines if not line.endswith(" ")) + "\n"


def run_template(template_id: str, *, workers: int = 1, limit: int = 0) -> dict:
    from optbench.campaign import load_template

    spec = _spec_map()[template_id]
    template = load_template(ROOT, template_id)
    if template["ini"]["Potential"]["potential"] == "morse_pt":
        from benchmarks.bench_eon_svn import _morse_lib

        _morse_lib()
    cap = int(template["failure_cap"])
    seeds = list(spec["campaign_seeds"])
    starts = list(spec["campaign_starts"])
    if limit:
        starts = starts[:limit]
        if spec["kind"] == "akmc":
            seeds = seeds[:limit]
    pairs = [(seed, start) for seed in seeds for start in starts]
    stem = stem_for(template_id)
    directory = ROOT / "results" / "campaign"
    directory.mkdir(parents=True, exist_ok=True)
    jsonl = directory / f"{stem}.starts.jsonl"
    done = _load_done(jsonl)
    pending = [pair for pair in pairs if pair not in done]
    passes = int(spec.get("campaign_passes") or 0)
    inside = int(spec.get("workers") or 1) if spec["kind"] == "akmc" else 1
    pool_workers = 1 if spec["kind"] == "akmc" else max(1, workers)
    print(
        f"{template_id}: {len(done)} done, {len(pending)} pending, pool {pool_workers}",
        flush=True,
    )
    if pending:
        jobs = [
            (template_id, spec["kind"], seed, start, passes, inside)
            for seed, start in pending
        ]
        finished = 0
        with jsonl.open("a", encoding="utf-8") as handle:
            if pool_workers == 1:
                for job in jobs:
                    row = _public_row(_job(job))
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    finished += 1
                    print(
                        f"  {finished}/{len(pending)} {row['start']} seed {row['seed']} "
                        f"calls {row['force_calls']} success {row['success']} {row.get('failure') or ''}",
                        flush=True,
                    )
            else:
                with ProcessPoolExecutor(max_workers=pool_workers) as pool:
                    futures = [pool.submit(_job, job) for job in jobs]
                    for future in as_completed(futures):
                        row = _public_row(future.result())
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
                        finished += 1
                        print(
                            f"  {finished}/{len(pending)} {row['start']} seed {row['seed']} "
                            f"calls {row['force_calls']} success {row['success']} {row.get('failure') or ''}",
                            flush=True,
                        )
    done = _load_done(jsonl)
    ordered = []
    missing = []
    for pair in pairs:
        row = done.get(pair)
        if row is None:
            missing.append(pair)
        else:
            ordered.append(row)
    partial = bool(missing) or len(ordered) < len(spec["campaign_seeds"]) * len(spec["campaign_starts"])
    if limit:
        partial = True
    if not ordered:
        raise RuntimeError(f"{template_id} produced no starts")
    text = _dat(template, ordered, partial=partial, cap=cap)
    (directory / f"{stem}.dat").write_text(text, encoding="utf-8")
    body = "\n".join(json.dumps(_public_row(row), sort_keys=True) for row in ordered) + "\n"
    jsonl.write_text(body, encoding="utf-8")
    print(text, flush=True)
    return {"template": template_id, "n": len(ordered), "partial": partial, "stem": stem}


def write_asv(commit: str) -> Path:
    """One asv result for every finished SVN campaign, keyed by ``commit``."""
    import time

    from asv.results import Results

    from benchmarks.bench_eon_svn import SPECS

    machine = json.loads((ROOT / "results" / "asv" / "xeon-8c" / "machine.json").read_text(encoding="utf-8"))
    params = {
        "arch": machine.get("arch", "x86_64"),
        "cpu": machine.get("cpu", ""),
        "machine": "xeon-8c",
        "num_cpu": str(machine.get("num_cpu", "8")),
        "os": machine.get("os", ""),
        "ram": str(machine.get("ram", "")),
        "python": "3.12",
    }
    result = Results(params, {}, commit, int(time.time() * 1000), "3.12", "uv-pip-py3.12", {})
    tracks = (
        "time_search",
        "track_force_calls",
        "track_success",
        "track_failure_cap",
        "track_failure_wrong_stationary",
        "track_failure_divergence",
        "track_failure_crash",
    )
    wrote = 0
    for spec in SPECS:
        stem = stem_for(spec["template"])
        path = ROOT / "results" / "campaign" / f"{stem}.starts.jsonl"
        dat = ROOT / "results" / "campaign" / f"{stem}.dat"
        if not path.is_file() or not dat.is_file():
            continue
        if "partial true" in dat.read_text(encoding="utf-8"):
            continue
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(str(row["seed"]), str(row["start"]))] = row
        combos = [(str(seed), str(start)) for seed in spec["campaign_seeds"] for start in spec["campaign_starts"]]
        if any(combo not in rows for combo in combos):
            continue
        ordered = [rows[combo] for combo in combos]
        param_lists = [[repr(str(seed)) for seed in spec["campaign_seeds"]], [repr(str(start)) for start in spec["campaign_starts"]]]
        columns = {
            "time_search": [float(row.get("seconds") or 0.0) for row in ordered],
            "track_force_calls": [float(row["force_calls"]) for row in ordered],
            "track_success": [1.0 if row["success"] else 0.0 for row in ordered],
            "track_failure_cap": [1.0 if row.get("failure") == "cap" else 0.0 for row in ordered],
            "track_failure_wrong_stationary": [1.0 if row.get("failure") == "wrong_stationary" else 0.0 for row in ordered],
            "track_failure_divergence": [1.0 if row.get("failure") == "divergence" else 0.0 for row in ordered],
            "track_failure_crash": [1.0 if row.get("failure") == "crash" else 0.0 for row in ordered],
        }
        for method in tracks:
            name = f"bench_eon_svn.{spec['cls']}.{method}"
            result._results[name] = columns[method]
            result._benchmark_params[name] = param_lists
            result._benchmark_version[name] = "r2549"
            result._started_at[name] = int(time.time() * 1000)
            result._duration[name] = float(sum(columns["time_search"]))
            result._samples[name] = None
            result._stats[name] = None
        wrote += 1
    if wrote == 0:
        raise RuntimeError("no finished eOn SVN campaign to store")
    result.save(str(ROOT / "results" / "asv"))
    dest = ROOT / "results" / "asv" / "xeon-8c" / result._filename
    print(f"wrote {dest} classes {wrote}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    workers = 8
    limit = 0
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if not args or args[0] in {"-h", "--help"}:
        raise SystemExit("usage: python -m optbench.eon_svn run TEMPLATE | run-all | write-asv COMMIT")
    command = args[0]
    if command == "run":
        run_template(args[1], workers=workers, limit=limit)
        return
    if command == "run-all":
        for template_id in _spec_map():
            run_template(template_id, workers=workers, limit=limit)
        return
    if command == "write-asv":
        write_asv(args[1])
        return
    raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
