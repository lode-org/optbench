"""Count failed loads and lost entries when asv processes share ~/.asv-machine.json.

Run: uv run python docs/asv-gaps/machine_race_trials.py [--trials N] [--workers N] [--loads N] [--merges N]

Each trial seeds a new machine file with four entries. WORKERS processes
each load one of them LOADS times, all starting together. A load is the
call ``asv run`` makes, ``Machine.load(machine_name=NAME, interactive=True)``,
with stdout off a console, as in a scripted run. The scenarios are:

- ``loads``: the loads alone, on the shared file.
- ``asv-merges``: the loads, while one more process adds MERGES entries
  as ``asv machine --yes --machine NAME`` does.
- ``launcher-merges``: the loads, while one more process adds MERGES
  entries with ``optbench.asv_launch.write_machine``, as
  ``python -m optbench.asv_launch machine --yes --machine NAME`` does.
- ``launched``: each load inside ``optbench.asv_launch.own_machine_file``,
  as a launched ``asv run`` loads, with the launcher merges.

The default workers, loads, and merges are the configuration whose counts
``docs/asv-gaps.md`` reports. A failed merge leaves its entry out, and a
merged entry can also be dropped by a load that read the file before the
merge and wrote it back after. The race depends on timing, so one trial
is one draw. With nine trials, the 2nd and 8th smallest counts bound the
median with probability 1 - 2 * (1 + 9) / 2**9 = 0.961. Nine is the
smallest count of trials for which that pair reaches 0.95.

The exit status is 1 when the bare scenarios fail no load and lose no
entry in any trial, so the gap has closed, or when a launched load
fails or a launched trial loses an entry.
"""

import argparse
import json
import multiprocessing
import os
import statistics
import sys
import tempfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

SCENARIOS = ("loads", "asv-merges", "launcher-merges", "launched")
SEEDED = 4


def _failure(exc: BaseException, home: str) -> str:
    message = (str(exc).replace(home, "~").splitlines() or [""])[0]
    return f"{type(exc).__name__}: {message[:72]}"


def _asv_merge(name: str) -> None:
    from asv.commands import make_argparser
    from asv.commands.machine import Machine as MachineCommand

    parser, _subparsers = make_argparser()
    MachineCommand.run(**vars(parser.parse_args(["machine", "--yes", "--machine", name])))


def _worker(scenario, home, index, count, barrier, queue):
    os.environ["HOME"] = home
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    from asv import console
    from asv.machine import Machine

    from optbench.asv_launch import own_machine_file, write_machine

    console.log.error = lambda *args, **kwargs: None
    failures = []
    barrier.wait()
    for step in range(count):
        try:
            if index < 0:
                name = f"w{step}"
                if scenario == "asv-merges":
                    _asv_merge(name)
                else:
                    write_machine({"machine": name})
                continue
            name = f"m{index % SEEDED}"
            scope = own_machine_file(["run", "--machine", name]) if scenario == "launched" else nullcontext()
            with scope:
                loaded = Machine.load(machine_name=name, interactive=True)
            if loaded.machine != name:
                failures.append(f"loaded {loaded.machine} for {name}")
        except BaseException as exc:  # noqa: BLE001
            failures.append(_failure(exc, home))
    queue.put((index, failures))


def trial(scenario: str, workers: int, loads: int, merges: int) -> dict:
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as home:
        os.environ["HOME"] = home
        from asv.machine import Machine

        path = Path(home) / ".asv-machine.json"
        fields = Machine.get_defaults()
        seeded = {f"m{index}": {**fields, "machine": f"m{index}"} for index in range(SEEDED)}
        path.write_text(json.dumps({**seeded, "version": 1}), encoding="utf-8")
        jobs = [(index, loads) for index in range(workers)]
        if scenario != "loads":
            jobs.append((-1, merges))
        barrier = context.Barrier(len(jobs))
        queue = context.Queue()
        processes = [
            context.Process(target=_worker, args=(scenario, home, index, count, barrier, queue)) for index, count in jobs
        ]
        for process in processes:
            process.start()
        results = dict(queue.get() for _ in processes)
        for process in processes:
            process.join()
        try:
            final = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            final = None
    loaded = [failure for index, failures in results.items() if index >= 0 for failure in failures]
    return {
        "failed_loads": len(loaded),
        "kinds": Counter(loaded),
        "failed_merges": len(results.get(-1, [])),
        "merge_kinds": Counter(results.get(-1, [])),
        "parses": final is not None,
        "kept": sum(1 for key in final or {} if key.startswith("w")),
        "seeded_unchanged": final is not None and all(final.get(key) == value for key, value in seeded.items()),
    }


def _summary(values: list[int]) -> str:
    ordered = sorted(values)
    text = f"median {statistics.median(ordered):g}, min {ordered[0]}, max {ordered[-1]}"
    if len(ordered) >= 9:
        text += f", 2nd to {len(ordered) - 1}th {ordered[1]} to {ordered[-2]}"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--loads", type=int, default=200)
    parser.add_argument("--merges", type=int, default=50)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = parser.parse_args()
    import asv

    print(f"asv {asv.__version__}, cpus {os.cpu_count()}, python {sys.version.split()[0]}")
    print(f"workers {args.workers}, loads {args.workers * args.loads}, merges {args.merges}, trials {args.trials}")
    closed = True
    launched_failed = False
    for scenario in args.scenario or SCENARIOS:
        merging = scenario != "loads"
        rows = [trial(scenario, args.workers, args.loads, args.merges) for _ in range(args.trials)]
        print(f"\n{scenario}")
        for number, row in enumerate(rows, 1):
            line = f"  trial {number}: failed loads {row['failed_loads']}"
            if merging:
                line += f", failed merges {row['failed_merges']}, merged entries kept {row['kept']} of {args.merges}"
            line += f", seeded entries unchanged {row['seeded_unchanged']}, file parses {row['parses']}"
            print(line)
        print(f"  failed loads of {args.workers * args.loads}: {_summary([row['failed_loads'] for row in rows])}")
        if merging:
            print(f"  failed merges of {args.merges}: {_summary([row['failed_merges'] for row in rows])}")
            dropped = [args.merges - row["failed_merges"] - row["kept"] for row in rows]
            print(f"  merged entries a later write dropped, of {args.merges}: {_summary(dropped)}")
            print(f"  merged entries missing at the end, of {args.merges}: {_summary([args.merges - row['kept'] for row in rows])}")
        print(f"  trials that left an unparseable machine file: {sum(not row['parses'] for row in rows)} of {len(rows)}")
        kinds = sum((row["kinds"] + row["merge_kinds"] for row in rows), Counter())
        for kind, count in kinds.most_common():
            print(f"  {count:6d}  {kind}")
        lost = any(merging and row["kept"] != args.merges or not row["seeded_unchanged"] for row in rows)
        failed = any(row["failed_loads"] or row["failed_merges"] for row in rows)
        if scenario == "launched":
            launched_failed = failed or lost
        elif failed or lost:
            closed = False
    if closed:
        print("\nthe bare scenarios no longer fail or lose entries")
    if launched_failed:
        print("\nlaunched loads failed or lost entries")
    return 1 if closed or launched_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
