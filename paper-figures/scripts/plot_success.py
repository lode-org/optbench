#!/usr/bin/env python3
"""Draw one exact success-rate chart and write its SVG/PDF pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from _plotting import PALETTE, load_figure, save_pair, style_axis


def draw(rows: list[dict], title: str):
    height = max(2.0, 0.33 * len(rows) + 1.05)
    figure, axis = plt.subplots(figsize=(3.35, height), layout="constrained")
    positions = list(range(len(rows)))
    rates = []
    counts = []
    for row in rows:
        n = len(row["records"])
        solved = sum(1 for record in row["records"] if record["success"])
        rates.append(solved / n)
        counts.append((solved, n))
    bars = axis.barh(
        positions,
        rates,
        color=[PALETTE[index % len(PALETTE)] for index in positions],
        height=0.62,
    )
    for bar, rate, (solved, n) in zip(bars, rates, counts):
        axis.text(
            min(rate + 0.018, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{solved}/{n}",
            ha="left" if rate < 0.9 else "right",
            va="center",
            fontsize=6.3,
            color="#1a1a1a",
        )
    axis.set_yticks(positions, [str(row["code"]) for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_xlabel("Successful starts")
    axis.set_title(title)
    style_axis(axis)
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
        "success",
        args.figure,
    )
    figure = draw(rows, figure_info["problem_title"])
    save_pair(figure, args.svg, args.pdf)
    plt.close(figure)


if __name__ == "__main__":
    main()
