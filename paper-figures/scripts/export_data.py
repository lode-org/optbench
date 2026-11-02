#!/usr/bin/env python3
"""Export the committed board rows used by the paper and site figures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optbench.board import _geometry_key, build_board
from optbench.families import family_title


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "unknown"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _row_id(problem_id: str, row: dict) -> str:
    fields = (
        problem_id,
        str(row.get("code") or ""),
        str(row.get("code_version") or ""),
        str(row.get("commit") or ""),
        str(row.get("template") or ""),
        str(row.get("entry") or ""),
        str(row.get("path") or ""),
    )
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()[:16]


def _results_index_hash(root: Path) -> str:
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-s", "results"],
    )
    return hashlib.sha256(tracked).hexdigest()


def _record(record: dict) -> dict:
    return {
        "start": _geometry_key(record.get("start")),
        "force_calls": float(record["force_calls"]),
        "success": bool(record["success"]),
        "failure": str(record.get("failure") or ""),
    }


def _row(problem_id: str, row: dict) -> dict:
    records = [_record(record) for record in row.get("_records") or []]
    code = str(row.get("code") or "").strip() or "unknown"
    version = str(row.get("code_version") or "").strip()
    commit = str(row.get("commit") or "").strip()
    asv_commit = str(row.get("asv_commit") or "").strip()
    point = version or commit[:12] or asv_commit[:12] or str(row.get("entry") or "")
    return {
        "id": _row_id(problem_id, row),
        "code": code,
        "version": version,
        "commit": commit,
        "asv_commit": asv_commit,
        "history_point": point,
        "method": str(row.get("method") or ""),
        "label": str(row.get("label") or code),
        "family": str(row.get("family") or ""),
        "family_title": family_title(str(row.get("family") or "")),
        "origin": str(row.get("origin") or ""),
        "source": str(row.get("source") or ""),
        "path": str(row.get("path") or ""),
        "cap": row.get("cap"),
        "n": int(row.get("n") or 0),
        "par2_mean": row.get("par2_mean"),
        "mean_calls": row.get("mean_calls"),
        "success_rate": row.get("success_rate"),
        "records": records,
    }


def _figure(
    kind: str,
    stem: str,
    problem_id: str,
    *,
    row_ids: list[str],
    code: str = "",
    family: str = "",
    family_name: str = "",
) -> dict:
    return {
        "type": kind,
        "stem": stem,
        "problem": problem_id,
        "row_ids": row_ids,
        "code": code,
        "family": family,
        "family_title": family_name,
        "site_svg": f"docs/source/_static/paper/{kind}/{stem}.svg",
        "pdf": f"paper-figures/pdfs/{kind}/{stem}.pdf",
    }


def export(root: Path) -> tuple[dict, dict]:
    board = build_board(root)
    data_problems: dict[str, dict] = {}
    figures: list[dict] = []
    for problem_id, problem in board["problems"].items():
        rows = [_row(problem_id, row) for row in problem.get("_ranked_rows") or []]
        row_by_id = {row["id"]: row for row in rows}
        problem_slug = _slug(problem_id)

        by_code: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["par2_mean"] is not None or row["mean_calls"] is not None:
                by_code[row["code"]].append(row)
        for code in sorted(by_code, key=str.casefold):
            grouped = by_code[code]
            stem = f"{problem_slug}--{_slug(code)}-{_short_hash(code)}"
            figures.append(
                _figure(
                    "history",
                    stem,
                    problem_id,
                    row_ids=[row["id"] for row in grouped],
                    code=code,
                )
            )

        selected: list[dict] = []
        seen_codes: set[str] = set()
        for row in rows:
            if not row["records"] or row["code"] in seen_codes:
                continue
            seen_codes.add(row["code"])
            selected.append(row)

        if selected:
            figures.append(
                _figure(
                    "distribution",
                    problem_slug,
                    problem_id,
                    row_ids=[row["id"] for row in selected],
                )
            )
            figures.append(
                _figure(
                    "success",
                    problem_slug,
                    problem_id,
                    row_ids=[row["id"] for row in selected],
                )
            )
        selected_by_family: dict[str, list[dict]] = defaultdict(list)
        for row in selected:
            selected_by_family[row["family"]].append(row)
        for family in sorted(selected_by_family):
            grouped = selected_by_family[family]
            family_name = grouped[0]["family_title"]
            stem = f"{problem_slug}--{_slug(family or 'other')}"
            figures.append(
                _figure(
                    "profile",
                    stem,
                    problem_id,
                    row_ids=[row["id"] for row in grouped],
                    family=family,
                    family_name=family_name,
                )
            )

        data_problems[problem_id] = {
            "id": problem_id,
            "title": str(problem.get("title") or problem_id),
            "rows": rows,
            "selected_row_ids": [row["id"] for row in selected],
            "row_count": len(row_by_id),
        }

    figures.sort(key=lambda item: (item["type"], item["problem"], item["stem"]))
    data = {
        "schema": "optbench.paper-figures-data/v1",
        "results_index_sha256": _results_index_hash(root),
        "problems": data_problems,
    }
    manifest = {
        "schema": "optbench.paper-figures-manifest/v1",
        "results_index_sha256": data["results_index_sha256"],
        "selection": (
            "For distributions, success rates, and family profiles, use the "
            "lowest-ranked full row with per-start records for each code."
        ),
        "figures": figures,
    }
    return data, manifest


def _write_gzip_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed:
            packed.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "paper-figures" / "data" / "results.json.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "paper-figures" / "data" / "manifest.json",
    )
    args = parser.parse_args()
    data, manifest = export(args.root.resolve())
    _write_gzip_json(args.data, data)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
