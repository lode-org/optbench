"""Ranked hTST/Pt rows for the eOn client templates.

A start is one barrier window from ``archive/entries/hTST/Pt`` report.py.
The window closes when every reference barrier, rounded to 3 decimals,
has been seen once. Force calls are the harness tally for the searches
in that window. The cap is ``problems/hTST/Pt.toml``
``convergence.max_force_calls``.

The client is eOn develop at 4c1b814f. The potential is ``morse_shift``
through ``PotServer``. AKMC stays on the reactant: confidence 1.9 does
not take a KMC step before the windows close.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from .campaign import format_ini, geometry_archive, load_template, resolve_problem
from .eon_akmc import prepare_client_env
from .eon_harness import bridge_binary, kernel_for, stated_failure_cap
from .potserver import PotServer, remote_count
from .htst_windows import HTST, htst_dat, reduce_htst
from .v2 import collect_provenance, write_starts

ROOT = Path(__file__).resolve().parents[1]
DEVELOP_COMMIT = "4c1b814f62f1dcf5fcd0cf18ba0b6df3bcff7a28"
TEMPLATES = (
    "eon-client/htst-pt-dimer-lbfgs",
    "eon-client/htst-pt-lanczos-lbfgs",
)


def parse_searches(text: str) -> list[dict]:
    """Search lines from ``states/0/search_results.txt``.

    ``process_id`` is set for a ``good`` or ``repeat`` result. Other
    results stay in the window and add their force calls.
    """
    rows = []
    for line in text.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 7:
            continue
        result = fields[-1]
        process_id = None
        if "-" in result:
            kind, ident = result.split("-", 1)
            if kind in {"good", "repeat"}:
                process_id = int(ident)
        rows.append(
            {
                "barrier": float(fields[2]),
                "sad_fcs": int(fields[4]),
                "mins_fcs": int(fields[5]),
                "process_id": process_id,
            }
        )
    return rows


def assign_delta(rows: list[dict], delta: int) -> None:
    """Give ``delta`` harness calls to ``rows``. The shares sum to ``delta``."""
    if not rows:
        if delta:
            raise ValueError("harness calls arrived with no search line")
        return
    weights = [max(0, int(row["sad_fcs"]) + int(row["mins_fcs"])) for row in rows]
    total = sum(weights)
    if len(rows) == 1 or total <= 0:
        for row in rows[:-1]:
            row["force_calls"] = 0
        rows[-1]["force_calls"] = int(delta)
        return
    spent = 0
    for row, weight in zip(rows[:-1], weights[:-1], strict=True):
        share = int(delta * weight / total)
        row["force_calls"] = share
        spent += share
    rows[-1]["force_calls"] = int(delta) - spent


def _client_path() -> Path:
    env = os.environ.get("EON_DEVELOP_CLIENT") or os.environ.get("EONCLIENT")
    if env:
        return Path(env)
    return Path.home() / "opt" / "eon-develop" / "bin" / "eonclient"


def _require_develop(client: Path) -> tuple[str, str]:
    from .validate_counter import client_version

    version, commit, first = client_version(client)
    token = commit.strip()
    if not token or not DEVELOP_COMMIT.startswith(token):
        raise RuntimeError(f"{client} prints {first!r}, expected commit {DEVELOP_COMMIT}")
    return version, DEVELOP_COMMIT


def _load_ledger(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_pending(path: Path) -> int:
    if not path.is_file():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8"))["pending"])


def _save_pending(path: Path, pending: int) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"pending": int(pending)}) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_results(work: Path) -> list[dict]:
    results_path = work / "states" / "0" / "search_results.txt"
    if not results_path.is_file():
        return []
    return parse_searches(results_path.read_text(encoding="utf-8", errors="replace"))


def window_failure(calls: float, cap: int) -> str | None:
    """``cap`` when the window spent the force-call budget, else a success."""
    if calls >= cap:
        return "cap"
    return None


def _window_records(rows: list[dict], spec: dict, cap: int, target: int) -> list[dict]:
    summary = reduce_htst(rows, spec)
    calls = [int(value) for value in summary["barrier_call_list"][:target]]
    records = []
    for index, spent in enumerate(calls, start=1):
        failure = window_failure(float(spent), cap)
        records.append(
            {
                "force_calls": spent,
                "success": failure is None,
                "failure": failure,
                "start": f"barrier-{index}",
            }
        )
    return records


def _publish(template: dict, problem: dict, rows: list[dict], output: Path) -> dict:
    spec = HTST[problem["id"]]
    cap = stated_failure_cap(template)
    target = int(spec["barrier_cycles"])
    summary = reduce_htst(rows, spec)
    done = summary["barrier_cycles"] >= target
    records = _window_records(rows, spec, cap, target if done else summary["barrier_cycles"])
    comments = [
        "# counter harness. eOn AKMC on the Pt hTST reactant.",
        f"# Temperature {spec['temperature']:g} K, prefactor {spec['prefactor']:.6g}, from {spec['source']}.",
        "# A barrier window closes when every reference barrier, rounded to 3 decimals, has been seen once.",
        f"# The ranked sample is {target} barrier windows. PAR-2 uses the hTST/Pt force-call cap {cap}.",
        "# A rate window closes at 0.999 of rate_ref. A missing rate window leaves a finished barrier window in the sample.",
        "# Potential is shifted Morse, r2549 Morse::force, served by the harness.",
        f"# matched {summary['matched_barriers']} of {summary['reference_barriers']} reference barriers "
        f"after {summary['searches']} searches. barrier windows {summary['barrier_cycles']}. "
        f"rate windows {summary['rate_cycles']}.",
    ]
    if summary["unmatched_barriers"]:
        comments.append(
            "# unmatched reference barriers: " + " ".join(f"{value:.3f}" for value in summary["unmatched_barriers"])
        )
    text = htst_dat(
        summary,
        template=template,
        problem=problem,
        spec=spec,
        searches=len(rows),
        minimize_calls=0,
        unfinished=not done,
        comments=comments,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    starts = output.with_suffix(".starts.jsonl")
    score_path = output.with_suffix(".v2.json")
    if done and records:
        written = []
        for record in records:
            written.append(
                {
                    "schema": "optbench.start/v2",
                    "start": record["start"],
                    "force_calls": int(record["force_calls"]),
                    "counter": "harness",
                    "comparable": True,
                    "success": bool(record["success"]),
                    "failure": record["failure"],
                    "problem": problem["id"],
                    "template": template["id"],
                    "code": template.get("code", "eOn"),
                    "code_version": template.get("code_version", ""),
                    "commit": template.get("commit", ""),
                    "tuning": str(template.get("tuning") or ""),
                }
            )
        write_starts(starts, written)
        from .score import score_v2

        score = score_v2(
            [
                {"force_calls": row["force_calls"], "success": row["success"], "failure": row.get("failure")}
                for row in written
            ],
            cap,
        )
        score_path.write_text(
            json.dumps(
                {
                    "schema": "optbench.score/v2",
                    "problem": problem["id"],
                    "template": template["id"],
                    "counter": "harness",
                    "provenance": collect_provenance(
                        code=template.get("code", "eOn"),
                        code_version=template.get("code_version", ""),
                        commit=template.get("commit", ""),
                        parameters={"kernel": template.get("kernel"), "cap": cap, "windows": target},
                    ),
                    "score": score,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output.with_suffix(".starts").unlink(missing_ok=True)
    return {"done": done, "summary": summary, "records": records, "cap": cap}


def _prepare_work(template: dict, client: Path, work: Path) -> None:
    if (work / "config.ini").is_file():
        return
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    geometry = geometry_archive(ROOT, {"id": "htst", "input": {"archive_path": template["geometry"]}})
    shutil.copy(geometry, work / "pos.con")
    sections = json.loads(json.dumps(template["ini"]))
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(bridge_binary())}
    comm = sections.setdefault("Communicator", {})
    comm["client_path"] = str(client.resolve())
    comm["type"] = "local"
    comm["num_jobs"] = 1
    comm["number_of_CPUs"] = 1
    (work / "config.ini").write_text(format_ini(sections), encoding="utf-8")
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
        (work / relative).mkdir(parents=True, exist_ok=True)


def run_template(template_id: str, *, client: Path | None = None) -> dict:
    """Run until the problem's barrier-window count is stored."""
    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, str(template["problem"]))
    binary = Path(client) if client is not None else _client_path()
    if not binary.is_file():
        raise RuntimeError(f"develop eonclient is not at {binary}")
    version, commit = _require_develop(binary)
    template = dict(template)
    template["code_version"] = version
    template["commit"] = commit
    spec = HTST[str(problem["id"])]
    target = int(spec["barrier_cycles"])
    cap = stated_failure_cap(template)
    ceiling = int(os.environ.get("OPTBENCH_HTST_SEARCHES") or template.get("max_searches") or 100000)
    work = ROOT / "results" / "campaign" / "work" / template_id.replace("/", "-")
    _prepare_work(template, binary, work)
    ledger_path = work / "searches.jsonl"
    rows = _load_ledger(ledger_path)
    output = ROOT / "results" / "campaign" / f"{template_id.replace('/', '-')}.dat"
    evaluate = kernel_for(str(template["kernel"]))
    # One trailing search is still in flight when the 13th window closes.
    server = PotServer(evaluate, cap=cap * (target + 1)).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    prepare_client_env(binary)
    previous = Path.cwd()
    started = time.monotonic()
    pending_path = work / "pending.json"
    try:
        os.chdir(work)
        import inspect
        import logging

        for name in ("akmc", "eon", "state", "communicator", "explorer"):
            logging.getLogger(name).setLevel(logging.WARNING)

        from eon import akmc as akmc_mod
        from eon.config import ConfigClass
        from eon.config import config as shared_config

        parameters = inspect.signature(akmc_mod.akmc).parameters
        if "config" in parameters:
            config = ConfigClass()
        else:
            config = shared_config
            config.init_done = False
        try:
            config.init()
        except SystemExit as exc:
            raise RuntimeError(f"eon rejected config.ini (exit {exc.code})") from exc
        from eon import communicator

        if hasattr(communicator.get_communicator, "comm"):
            del communicator.get_communicator.comm
        # register_results runs at the start of the next akmc() call, so the
        # force calls of this call belong to the search line that appears next.
        pending = _load_pending(pending_path)
        parsed = _read_results(work)
        if len(parsed) < len(rows):
            raise RuntimeError(f"search_results shrank from {len(rows)} to {len(parsed)}")
        if len(parsed) > len(rows):
            fresh = parsed[len(rows) :]
            assign_delta(fresh, pending)
            _append_ledger(ledger_path, fresh)
            rows.extend(fresh)
            pending = 0
        _save_pending(pending_path, pending)
        stalled = 0
        while True:
            summary = reduce_htst(rows, spec)
            if summary["barrier_cycles"] >= target or len(rows) >= ceiling:
                break
            print(
                f"{template_id} searches {len(rows)} barrier_cycles {summary['barrier_cycles']} "
                f"matched {summary['matched_barriers']} calls {sum(int(row['force_calls']) for row in rows)}",
                flush=True,
            )
            before, _verify = remote_count(server.host, server.port)
            akmc_mod.akmc(config)
            os.chdir(work)
            after, _verify = remote_count(server.host, server.port)
            parsed = _read_results(work)
            if len(parsed) < len(rows):
                raise RuntimeError(f"search_results shrank from {len(rows)} to {len(parsed)}")
            fresh = parsed[len(rows) :]
            spent = int(after) - int(before)
            if fresh:
                assign_delta(fresh, pending)
                _append_ledger(ledger_path, fresh)
                rows.extend(fresh)
                pending = spent
                stalled = 0
            else:
                pending += spent
                if spent == 0:
                    stalled += 1
                    if stalled > 3:
                        raise RuntimeError("AKMC stopped making searches before the windows closed")
            _save_pending(pending_path, pending)
        config.comm_job_buffer_size = 0
        before, _verify = remote_count(server.host, server.port)
        akmc_mod.akmc(config)
        os.chdir(work)
        after, _verify = remote_count(server.host, server.port)
        parsed = _read_results(work)
        fresh = parsed[len(rows) :]
        if fresh:
            assign_delta(fresh, pending + int(after) - int(before))
            _append_ledger(ledger_path, fresh)
            rows.extend(fresh)
            _save_pending(pending_path, 0)
    finally:
        os.chdir(previous)
        server.stop()
        os.environ.pop("OPTBENCH_POTSERVER", None)
    summary = reduce_htst(rows, spec)
    if summary["barrier_cycles"] < target:
        raise RuntimeError(
            f"{template_id} closed {summary['barrier_cycles']} of {target} barrier windows "
            f"after {len(rows)} searches"
        )
    published = _publish(template, problem, rows, output)
    published["seconds"] = time.monotonic() - started
    published["template"] = template_id
    summary = published["summary"]
    print(
        f"{template_id} windows {summary['barrier_cycles']} "
        f"barrier_force_calls {summary['barrier_force_calls']} done {published['done']}",
        flush=True,
    )
    return published


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rank eOn develop hTST/Pt windows")
    parser.add_argument("template", nargs="*", default=list(TEMPLATES))
    parser.add_argument("--client", type=Path)
    args = parser.parse_args(argv)
    for template_id in args.template:
        run_template(template_id, client=args.client)


if __name__ == "__main__":
    main()
