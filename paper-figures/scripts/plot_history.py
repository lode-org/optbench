#!/usr/bin/env python3
"""Draw one unsmoothed cost history and write its SVG/PDF pair."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from _plotting import PALETTE, load_figure, save_pair, style_axis


MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def _method(row: dict) -> str:
    return str(row.get("method") or row.get("label") or "recorded result")


def draw(rows: list[dict], title: str):
    points = sorted(
        {str(row["history_point"] or "recorded result") for row in rows},
        key=str.casefold,
    )
    methods = sorted({_method(row) for row in rows}, key=str.casefold)
    height = max(3.35, 3.0 + 0.13 * math.ceil(len(methods) / 2))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, height),
        sharex=True,
        layout="constrained",
    )
    by_method_point = {
        (_method(row), str(row["history_point"] or "recorded result")): row
        for row in rows
    }
    x = list(range(len(points)))
    for axis, (key, ylabel) in zip(
        axes,
        (("par2_mean", "PAR-2"), ("mean_calls", "Mean force calls")),
    ):
        all_values: list[float] = []
        for index, method in enumerate(methods):
            coordinates = [
                (position, float(row[key]))
                for position, point in enumerate(points)
                if (row := by_method_point.get((method, point))) is not None
                and row.get(key) is not None
            ]
            if not coordinates:
                continue
            positions, values = zip(*coordinates)
            all_values.extend(values)
            axis.plot(
                positions,
                values,
                color=PALETTE[index % len(PALETTE)],
                linewidth=0.9,
                marker=MARKERS[(index // len(PALETTE)) % len(MARKERS)],
                markersize=3.0,
            )
        axis.set_ylabel(ylabel)
        if all_values and min(all_values) > 0:
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(ScalarFormatter())
        if not all_values:
            axis.text(
                0.5,
                0.5,
                "Not recorded",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#555555",
            )
        style_axis(axis, grid_axis="y")
    axes[-1].set_xticks(x, points, rotation=35, ha="right")
    axes[-1].set_xlabel("Code version or commit")
    figure.suptitle(title, fontsize=8.5)
    handles = [
        Line2D(
            [],
            [],
            color=PALETTE[index % len(PALETTE)],
            marker=MARKERS[(index // len(PALETTE)) % len(MARKERS)],
            linewidth=0.9,
            markersize=3.0,
            label=method,
        )
        for index, method in enumerate(methods)
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
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
        "history",
        args.figure,
    )
    title = f"{figure_info['problem_title']} — {figure_info['code']}"
    figure = draw(rows, title)
    save_pair(figure, args.svg, args.pdf)
    plt.close(figure)


if __name__ == "__main__":
    main()
