#!/usr/bin/env python3
"""Draw one per-start force-call distribution and write SVG/PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from _plotting import PALETTE, load_figure, save_pair, style_axis


def draw(rows: list[dict], title: str):
    height = max(2.2, 0.34 * len(rows) + 1.15)
    figure, axis = plt.subplots(figsize=(3.35, height), layout="constrained")
    positions = list(range(1, len(rows) + 1))
    values = [[float(record["force_calls"]) for record in row["records"]] for row in rows]
    boxes = axis.boxplot(
        values,
        positions=positions,
        orientation="horizontal",
        widths=0.44,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    for index, patch in enumerate(boxes["boxes"]):
        patch.set(facecolor=PALETTE[index % len(PALETTE)], alpha=0.16, linewidth=0.8)
    for median in boxes["medians"]:
        median.set(color="#111111", linewidth=1.0)
    for element in ("whiskers", "caps"):
        for artist in boxes[element]:
            artist.set(color="#666666", linewidth=0.7)
    failure_labeled = False
    for index, (position, row) in enumerate(zip(positions, rows)):
        color = PALETTE[index % len(PALETTE)]
        successes = [
            float(record["force_calls"])
            for record in row["records"]
            if record["success"]
        ]
        failures = [
            float(record["force_calls"])
            for record in row["records"]
            if not record["success"]
        ]
        if successes:
            axis.scatter(
                successes,
                [position] * len(successes),
                s=9,
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
        if failures:
            axis.scatter(
                failures,
                [position] * len(failures),
                s=17,
                marker="x",
                color=PALETTE[1],
                linewidth=0.9,
                zorder=4,
                label="Failure" if not failure_labeled else None,
            )
            failure_labeled = True
    axis.set_yticks(positions, [str(row["code"]) for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Force calls per start")
    axis.set_title(title)
    flat = [value for row in values for value in row]
    if flat and min(flat) > 0:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(ScalarFormatter())
    style_axis(axis)
    if any(not record["success"] for row in rows for record in row["records"]):
        axis.legend(frameon=False, loc="lower right")
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
        "distribution",
        args.figure,
    )
    figure = draw(rows, figure_info["problem_title"])
    save_pair(figure, args.svg, args.pdf)
    plt.close(figure)


if __name__ == "__main__":
    main()
