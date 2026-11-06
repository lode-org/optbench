"""Draw the tutorial figures from the two tutorial runs.

Run from the repository root after the two ``optbench run`` commands on the
tutorial page:

    uv run --group campaign python docs/source/_static/tutorial/figures.py

``calls.svg`` is the fraction of starts finished against harness force
calls, read from the per-start files the runs wrote. ``convergence.svg``
replays one start with each template and records the force norm after
every optimizer step. The replay uses the same template, potential, stop,
and counter as ``optbench run``, and the script stops if its final count
differs from the per-start record.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.lj import LennardJones
from ase.io import read
import ase.optimize

from optbench.ase_sella import discover, materialize
from optbench.campaign import load_template, resolve_problem
from optbench.counter import attach_ase_counter

HERE = Path(__file__).resolve().parent
RUNS = {"LBFGS": "lbfgs", "FIRE": "fire"}
COLORS = {"LBFGS": "#1f5fbf", "FIRE": "#c2410c"}

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "svg.hashsalt": "optbench-tutorial",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_starts(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def trace(root: Path, template_id: str, start: str, cache: Path) -> list[tuple[int, float]]:
    template = load_template(root, template_id)
    problem = resolve_problem(root, template["problem"])
    material = materialize(root, template, problem, cache)
    paths = {name: found[0] for name, found, _seed in discover(material, template)}
    atoms = read(paths[start], format="eon")
    potential = template["potential"]
    atoms.pbc = potential["pbc"]
    calc = LennardJones(epsilon=potential["epsilon"], sigma=potential["sigma"], rc=potential["rc"])
    stop = template["stop"]
    attach_ase_counter(calc, cap=int(stop["max_force_calls"]))
    atoms.calc = calc
    spec = template["optimizer"]
    opt = getattr(ase.optimize, spec["name"])(atoms, logfile=None, **(spec.get("parameters") or {}))
    points = []
    for _ in opt.irun(fmax=0.0, steps=int(stop["max_steps"])):
        norm = float(np.linalg.norm(atoms.get_forces()))
        points.append((calc.optbench_counter["calls"], norm))
        if norm <= float(stop["force_norm"]):
            break
    return points


def draw_calls(runs: dict[str, list[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for label, rows in runs.items():
        calls = np.sort([row["force_calls"] for row in rows if row["success"]])
        fraction = np.arange(1, calls.size + 1) / len(rows)
        ax.step(calls, fraction, where="post", color=COLORS[label], label=f"{label}, {len(rows)} starts")
    ax.set_xlabel("harness force calls")
    ax.set_ylabel("fraction of starts finished")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, metadata={"Date": None})
    plt.close(fig)


def draw_convergence(traces: dict[str, list[tuple[int, float]]], start: str, tol: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for label, points in traces.items():
        calls, norms = zip(*points)
        ax.semilogy(calls, norms, color=COLORS[label], label=label)
    ax.axhline(tol, color="0.4", linestyle="--", linewidth=1)
    ax.annotate(f"stop {tol}", (0, tol), xytext=(2, 3), textcoords="offset points", color="0.3")
    ax.set_xlabel("harness force calls")
    ax.set_ylabel("force norm")
    ax.set_title(start, loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--runs", type=Path, default=Path("tutorial"))
    parser.add_argument("--start", default="cluster_0000.con")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    runs = {label: read_starts(args.runs / f"{stem}.starts.jsonl") for label, stem in RUNS.items()}
    draw_calls(runs, args.out / "calls.svg")

    traces = {}
    tol = None
    with tempfile.TemporaryDirectory() as cache:
        for label, stem in RUNS.items():
            template_id = f"ase/lj38-{stem}-default"
            points = trace(args.root, template_id, args.start, Path(cache))
            recorded = next(row for row in runs[label] if row["start"] == args.start)
            if points[-1][0] != recorded["force_calls"]:
                raise SystemExit(f"{label}: replay {points[-1][0]} calls, run recorded {recorded['force_calls']}")
            print(f"{label} {args.start} replay={points[-1][0]} recorded={recorded['force_calls']} final_norm={points[-1][1]:.6g}")
            traces[label] = points
            tol = float(load_template(args.root, template_id)["stop"]["force_norm"])
    draw_convergence(traces, args.start, tol, args.out / "convergence.svg")
    for label, rows in runs.items():
        calls = [row["force_calls"] for row in rows]
        print(f"{label} starts={len(rows)} success={sum(row['success'] for row in rows)} min={min(calls)} max={max(calls)}")


if __name__ == "__main__":
    main()
