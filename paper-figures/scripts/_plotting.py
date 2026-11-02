"""Shared deterministic I/O and styling for the paper figures."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "axes.edgecolor": "#4d4d4d",
        "axes.labelcolor": "#1a1a1a",
        "axes.linewidth": 0.7,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "figure.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "legend.fontsize": 6.2,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "optbench-paper-figures",
        "xtick.color": "#333333",
        "xtick.labelsize": 6.5,
        "ytick.color": "#333333",
        "ytick.labelsize": 6.5,
    }
)

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)


def load_figure(data_path: Path, manifest_path: Path, kind: str, stem: str) -> tuple[dict, list[dict]]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with gzip.open(data_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    matches = [
        figure
        for figure in manifest["figures"]
        if figure["type"] == kind and figure["stem"] == stem
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {kind} figure named {stem}, found {len(matches)}")
    figure = matches[0]
    problem = data["problems"][figure["problem"]]
    by_id = {row["id"]: row for row in problem["rows"]}
    return {**figure, "problem_title": problem["title"]}, [by_id[row_id] for row_id in figure["row_ids"]]


def short_label(row: dict) -> str:
    label = str(row.get("label") or row.get("code") or "")
    if len(label) <= 48:
        return label
    return label[:45].rstrip() + "…"


def style_axis(axis, *, grid_axis: str = "x") -> None:
    axis.set_axisbelow(True)
    axis.grid(True, axis=grid_axis, color="#d9d9d9", linewidth=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_pair(figure, svg_path: Path, pdf_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        svg_path,
        format="svg",
        metadata={"Creator": "OptBench"},
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(
        pdf_path,
        format="pdf",
        metadata={
            "Creator": "OptBench",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
