"""Exact-setup reruns of the archived GMIN global-optimisation rows.

Each row copies ``archive/entries/globalopt/<problem>/<entry>/data`` and
prepares coordinates the way that entry's ``run.sh`` does. The binary is
the serial GMIN build: the archived data files do not contain the MPI
keyword, and the MPI build's non-MPI path faults in ``mc``. The potential
is GMIN's own, because ``OPTBENCH_POTPORT`` is unset. The archived score
is the ``Target hit after`` line: quenches are field 5 and function calls
are field 7, the same split as ``run_gmin.sh``.

``codes/gmin/build-bh.sh`` with ``GMIN_MPI=OFF`` builds the binary and
stamps it with its Wales snapshot. Each row's ``.dat`` opens with that
``wales_sha256`` and ``wales_svnrev``, and every start record carries
them. A stored start from another build is run again.

GMIN stops with a ``STOP`` message on stderr and exit status 0 when it
cannot read its data file. The record keeps that message as ``error``,
as it does the stderr of a nonzero exit.

``cpu_time`` is the 2022 machine and is recorded, not gated.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.campaign import geometry_archive
from optbench.wales import header, same_build

# Numbered starts that exist in the geometry tarball. lj98 and lj150
# run.sh loops to 200 and names a directory that is not in the tarball.
# The tarballs contain 100 numbered coords files, which is the problem sample.
ROWS: list[dict] = [
    {"problem": "lj38", "entry": "gmin-nosym", "kind": "xyz", "start": 0, "stop": 99, "metric": "force"},
    {"problem": "lj38", "entry": "gmin-csm", "kind": "xyz", "start": 0, "stop": 99, "metric": "force"},
    {"problem": "lj38", "entry": "gmin-symmetrise", "kind": "xyz", "start": 0, "stop": 99, "metric": "force"},
    {"problem": "lj75", "entry": "gmin", "kind": "coords", "start": 1, "stop": 200, "metric": "force"},
    {"problem": "lj98", "entry": "gmin-nosym", "kind": "coords", "start": 1, "stop": 100, "metric": "energy"},
    {"problem": "lj98", "entry": "gmin-symmetrise", "kind": "coords", "start": 1, "stop": 100, "metric": "energy"},
    {"problem": "lj150", "entry": "gmin-nosym", "kind": "coords", "start": 1, "stop": 100, "metric": "energy"},
    {"problem": "lj150", "entry": "gmin-symmetrise", "kind": "coords", "start": 1, "stop": 100, "metric": "energy"},
    {"problem": "tip4p10", "entry": "gmin", "kind": "aa", "start": 0, "stop": 9999, "metric": "force"},
    {"problem": "tip4p10", "entry": "gmin-tbp", "kind": "aa", "start": 0, "stop": 9999, "metric": "force"},
    {"problem": "tip4p15", "entry": "gmin", "kind": "aa", "start": 0, "stop": 499, "metric": "force"},
    {"problem": "tip4p15", "entry": "gmin-tbp", "kind": "aa", "start": 0, "stop": 499, "metric": "force"},
    {"problem": "tip4p20", "entry": "gmin", "kind": "aa", "start": 0, "stop": 99, "metric": "force"},
    {"problem": "tip4p20", "entry": "gmin-tbp", "kind": "aa", "start": 0, "stop": 99, "metric": "force"},
]

def serial_binary() -> Path:
    env = os.environ.get("OPTBENCH_GMIN_SERIAL", "")
    path = Path(env) if env else Path("/tmp/gmin-serial/GMIN")
    if not path.is_file():
        raise RuntimeError("set OPTBENCH_GMIN_SERIAL to the serial GMIN binary")
    return path


def parse_log(text: str) -> dict:
    """The last ``Target hit after`` line, as ``run_gmin.sh`` greps it."""
    ncalls = None
    quenches = None
    success = 0
    seconds = None
    for line in text.splitlines():
        if "Target hit after" in line:
            numbers = [int(part) for part in line.split() if part.isdigit()]
            if len(numbers) >= 2:
                quenches, ncalls = numbers[0], numbers[1]
                success = 1
        if "time elapsed" in line:
            parts = line.split()
            for part in parts:
                try:
                    seconds = float(part)
                    break
                except ValueError:
                    continue
    return {
        "ncalls": ncalls,
        "quenches": quenches,
        "success": success,
        "seconds": seconds,
    }


def _member(names: list[str], index: int, kind: str) -> str:
    if kind == "xyz":
        suffix = f"/{index}.xyz"
    elif kind == "coords":
        suffix = f"/coords.{index}"
    elif kind == "aa":
        suffix = f"/{index}.aa"
    else:
        raise ValueError(kind)
    found = [name for name in names if name.endswith(suffix)]
    if len(found) != 1:
        raise FileNotFoundError(f"index {index} kind {kind} matched {found}")
    return found[0]


def _coords_text(raw: bytes, kind: str) -> str:
    if kind != "xyz":
        text = raw.decode("utf-8", errors="replace")
        if not text.endswith("\n"):
            text += "\n"
        return text
    lines = []
    body = raw.decode("utf-8", errors="replace").splitlines()[2:]
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            continue
        coords = " ".join(parts[1:4]).replace("e", "D").replace("E", "D")
        lines.append(coords)
    if not lines:
        raise ValueError("xyz has no coordinates")
    return "\n".join(lines) + "\n"


def _problem_dict(root: Path, problem: str) -> dict:
    import tomllib

    path = root / "problems" / "globalopt" / f"{problem}.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def prepare_jobs(root: Path, row: dict, work: Path) -> list[dict]:
    problem = _problem_dict(root, row["problem"])
    archive = geometry_archive(root, problem)
    data = (root / "archive" / "entries" / "globalopt" / row["problem"] / row["entry"] / "data").read_text(
        encoding="utf-8"
    )
    with tarfile.open(archive) as packed:
        names = packed.getnames()
        payloads = []
        for index in range(int(row["start"]), int(row["stop"]) + 1):
            member = _member(names, index, row["kind"])
            extracted = packed.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member)
            payloads.append((index, _coords_text(extracted.read(), row["kind"])))
    jobs = []
    for index, coords in payloads:
        jobs.append(
            {
                "index": index,
                "work": str(work / f"{index:05d}"),
                "data": data,
                "coords": coords,
                "binary": str(serial_binary()),
                "resume": True,
            }
        )
    return jobs


def run_one(job: dict) -> dict:
    work = Path(job["work"])
    stored = work / "record.json"
    wales = job.get("wales") or {}
    if job.get("resume") and stored.is_file():
        record = json.loads(stored.read_text(encoding="utf-8"))
        if same_build(record, wales):
            return record
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "data").write_text(job["data"], encoding="utf-8")
    (work / "coords").write_text(job["coords"], encoding="utf-8")
    env = os.environ.copy()
    env.pop("OPTBENCH_POTPORT", None)
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [job["binary"]],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    log = proc.stdout
    output = work / "output"
    if output.is_file():
        log += "\n" + output.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log(log)
    record = {
        "index": int(job["index"]),
        "returncode": int(proc.returncode),
        "ncalls": parsed["ncalls"],
        "quenches": parsed["quenches"],
        "success": int(parsed["success"]) if proc.returncode == 0 else 0,
        "seconds": parsed["seconds"],
        **wales,
    }
    stopped = any(line.startswith("STOP") for line in (proc.stderr or "").splitlines())
    if proc.returncode != 0 or stopped:
        record["error"] = (proc.stderr or "")[-400:]
    stored.write_text(json.dumps(record), encoding="utf-8")
    for child in work.iterdir():
        if child.name != "record.json":
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    return record


def _round_mean(values: np.ndarray) -> int:
    return int(round(float(np.mean(values))))


def summarize(records: list[dict], metric: str) -> dict[str, str]:
    ordered = sorted(records, key=lambda row: int(row["index"]))
    success = [row for row in ordered if int(row["success"]) == 1 and row.get("ncalls") is not None]
    if not success:
        raise RuntimeError("no successful starts")
    calls = np.array([float(row["ncalls"]) for row in success], dtype=float)
    quenches = np.array([float(row["quenches"]) for row in success], dtype=float)
    seconds = np.array(
        [float(row["seconds"]) for row in success if row.get("seconds") is not None],
        dtype=float,
    )
    nfailed = sum(1 for row in ordered if int(row["success"]) != 1)
    fields = {"nfailed": str(nfailed), "n": str(len(ordered))}
    if metric == "force":
        fields.update(
            {
                "force_calls": f"{_round_mean(calls):.5e}",
                "force_calls_median": f"{int(round(float(np.median(calls)))):.5e}",
                "force_calls_min": f"{int(round(float(np.min(calls)))):.4e}",
                "force_calls_max": f"{int(round(float(np.max(calls)))):.4e}",
            }
        )
    else:
        fields.update(
            {
                "energy_calls_mean": f"{float(np.mean(calls)):.8e}",
                "energy_calls_stddev": f"{float(np.std(calls, ddof=1)):.8e}",
                "basin_hopping_steps_mean": f"{float(np.mean(quenches)):.6e}",
                "basin_hopping_steps_stddev": f"{float(np.std(quenches, ddof=1)):.6e}",
            }
        )
    if seconds.size:
        fields["cpu_time_mean"] = f"{float(np.mean(seconds)):.5e}"
        if seconds.size > 1:
            fields["cpu_time_stddev"] = f"{float(np.std(seconds, ddof=1)):.4e}"
    return fields


def write_dat(path: Path, fields: dict[str, str], row: dict, wales: dict) -> None:
    lines = [
        *header(wales),
        "# exact archive data file, serial GMIN, internal potential",
        f"# entry globalopt/{row['problem']}/{row['entry']}",
        "# ncalls is the Target hit after function-call field",
    ]
    for key, value in fields.items():
        lines.append(f"{key} {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_row(root: Path, row: dict, *, workers: int, work: Path, output: Path) -> dict[str, str]:
    from optbench.runners.gmin_go import binary_build

    wales = binary_build(serial_binary())
    jobs = [dict(job, wales=wales) for job in prepare_jobs(root, row, work)]
    if workers <= 1:
        records = []
        for done, job in enumerate(jobs, 1):
            records.append(run_one(job))
            print(f"{row['problem']}/{row['entry']} {done}/{len(jobs)}", flush=True)
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                print(f"{row['problem']}/{row['entry']} {done}/{total}", flush=True)
    fields = summarize(records, row["metric"])
    write_dat(output, fields, row, wales)
    starts = output.with_suffix(".starts.jsonl")
    starts.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in sorted(records, key=lambda item: item["index"])),
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    for key, value in fields.items():
        print(f"  {key} {value}", flush=True)
    return fields


def campaign_name(row: dict) -> str:
    entry = row["entry"].removeprefix("gmin-") if row["entry"] != "gmin" else ""
    suffix = f"-{entry}" if entry else ""
    return f"gmin-paper-{row['problem']}{suffix}"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rerun archived GMIN globalopt rows")
    parser.add_argument("--row", action="append", default=[], help="problem/entry, repeatable")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    selected = ROWS
    if args.row:
        wanted = set(args.row)
        selected = [row for row in ROWS if f"{row['problem']}/{row['entry']}" in wanted]
        if len(selected) != len(wanted):
            raise SystemExit(f"unknown row in {sorted(wanted)}")
    root = args.root
    for row in selected:
        name = campaign_name(row)
        run_row(
            root,
            row,
            workers=args.workers,
            work=Path("/tmp/gmin-paper") / name,
            output=root / "results" / "campaign" / f"{name}.dat",
        )


if __name__ == "__main__":
    main()
