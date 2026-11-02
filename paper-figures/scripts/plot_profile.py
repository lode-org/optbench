#!/usr/bin/env python3
"""Draw one Dolan–Moré family profile and write its SVG/PDF pair."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optbench.profile import performance_profile, profile_cost

from _plotting import PALETTE, load_figure, save_pair, style_axis


def _costs(rows: list[dict]) -> tuple[dict[str, list[float]], int]:
    keyed = []
    for row in rows:
        found: dict[str, float] = {}
        order: list[str] = []
        for record in row["records"]:
            start = str(record["start"])
            if start in found:
                continue
            found[start] = profile_cost(record)
            order.append(start)
        keyed.append((row, found, order))
    common = set.intersection(*(set(found) for _row, found, _order in keyed))
    if not common:
        return {}, 0
    order = [start for start in keyed[0][2] if start in common]
    usable = [
        start
        for start in order
        if any(math.isfinite(found[start]) for _row, found, _order in keyed)
    ]
    all_unsolved = len(order) - len(usable)
    return {
        str(row["code"]): [found[start] for start in usable]
        for row, found, _order in keyed
    }, all_unsolved


def _exact_taus(costs: dict[str, list[float]]) -> list[float]:
    if not costs:
        return []
    width = len(next(iter(costs.values())))
    ratios = {1.0}
    for index in range(width):
        finite = [values[index] for values in costs.values() if math.isfinite(values[index])]
        if not finite:
            continue
        best = min(finite)
        ratios.update(
            value / best
            for values in costs.values()
            if math.isfinite(value := values[index])
        )
    return sorted(ratios)


def draw(rows: list[dict], title: str):
    figure, axis = plt.subplots(figsize=(3.35, 2.45), layout="constrained")
    costs, all_unsolved = _costs(rows)
    taus = _exact_taus(costs)
    if not costs or not taus or not next(iter(costs.values()), []):
        axis.text(
            0.5,
            0.52,
            "No aligned start has a finite\nreference cost for this family.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        axis.set_title(title)
        return figure
    profile = performance_profile(costs, taus=taus)
    for index, (code, values) in enumerate(profile["rho"].items()):
        axis.step(
            profile["taus"],
            values,
            where="post",
            linewidth=1.15,
            color=PALETTE[index % len(PALETTE)],
            label=code,
        )
        axis.scatter(
            profile["taus"],
            values,
            s=7,
            color=PALETTE[index % len(PALETTE)],
            zorder=3,
        )
    if max(taus) > 1.0:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlim(left=1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Performance ratio τ")
    axis.set_ylabel("Fraction of aligned starts")
    axis.set_title(title)
    style_axis(axis, grid_axis="both")
    axis.legend(frameon=False, loc="lower right")
    note = "Dolan & Moré (2002)"
    if all_unsolved:
        note += f"; {all_unsolved} all-unsolved start{'s' if all_unsolved != 1 else ''} omitted"
    axis.text(0.0, -0.25, note, transform=axis.transAxes, fontsize=5.8)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    figure_info, rows = load_figure(
        args.data,
        args.manifest,
        "profile",
        args.figure,
    )
    title = f"{figure_info['problem_title']} — {figure_info['family_title']}"
    figure = draw(rows, title)
    save_pair(figure, args.svg, args.pdf)
    plt.close(figure)


if __name__ == "__main__":
    main()
