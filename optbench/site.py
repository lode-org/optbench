"""Render the public site from the v2 board.

Prose stays in ``docs/orgmode``. This module writes the ranked tables,
the code page, and the labelled 2014 archive those pages include.
"""

from __future__ import annotations

import html
import tomllib
from pathlib import Path

from .check import load_oracle
from .dat import normalize
from .problems import load_all

SUITE_ORDER = (
    "minimization",
    "saddle-search",
    "double-ended-saddle-search",
    "globalopt",
    "lowest-eigenvector",
    "tsopt",
    "mlopt",
    "neb",
    "hTST",
    "accelerated-md",
)

SUITE_TITLE = {
    "minimization": "Minimization",
    "saddle-search": "Saddle search",
    "double-ended-saddle-search": "Double-ended saddle search",
    "globalopt": "Global optimization",
    "lowest-eigenvector": "Lowest eigenvector",
    "tsopt": "Transition state optimization",
    "mlopt": "Machine learning optimization",
    "neb": "Nudged elastic band",
    "hTST": "Harmonic transition state theory",
    "accelerated-md": "Accelerated molecular dynamics",
}

PRIMARY = (
    "force_calls",
    "mean_epoch",
    "mean_epochs",
    "avg_energy",
    "barrier_force_calls",
    "true_rate",
)


def suite_title(suite: str) -> str:
    return SUITE_TITLE.get(suite, suite)


def _float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _held(fields: dict[str, str]) -> bool:
    for key in ("hidden", "hiden", "hide"):
        if fields.get(key, "").lower() == "true":
            return True
    return False


def _primary(problem: dict) -> str | None:
    required = list(problem["metrics"]["required"])
    for key in PRIMARY:
        if key in required:
            return key
    return required[0] if required else None


def _bits(problem: dict) -> list[tuple[str, str]]:
    bits: list[tuple[str, str]] = []
    convergence = problem.get("convergence") or {}
    if "force_norm" in convergence:
        unit = convergence.get("force_unit", "")
        kind = "L2 norm" if str(convergence.get("norm", "")).lower() == "l2" else "norm"
        bits.append(("Stop", f"{kind} of the force at or below {convergence['force_norm']} {unit}".strip()))
    if "gradient_norm" in convergence:
        bits.append(("Stop", f"gradient norm below {convergence['gradient_norm']}"))
    if "max_force_calls" in convergence:
        bits.append(("Cap", f"{convergence['max_force_calls']} force calls"))
    if "energy_above_minimum" in convergence:
        bits.append(("Stop", f"within {convergence['energy_above_minimum']} of the known minimum"))
    if "max_epochs" in convergence:
        bits.append(("Cap", f"{convergence['max_epochs']} epochs"))
    goal = problem.get("goal") or {}
    if goal.get("connected_saddles"):
        barrier = goal.get("barrier_below_eV")
        text = f"{goal['connected_saddles']} connected saddles"
        if barrier is not None:
            text += f" below {barrier} eV"
        bits.append(("Goal", text))
    sample = problem.get("sample") or {}
    if sample.get("starts"):
        bits.append(("Starts", str(sample["starts"])))
    potential = problem.get("potential") or {}
    if potential.get("kind"):
        label = str(potential["kind"])
        if potential.get("name"):
            label = f"{label} ({potential['name']})"
        bits.append(("Potential", label))
    thermal = problem.get("thermal") or {}
    if thermal.get("temperature_K") is not None:
        bits.append(("Temperature", f"{thermal['temperature_K']} K"))
    reference = problem.get("reference") or {}
    if reference.get("true_rate") is not None:
        bits.append(("Reference rate", f"{float(reference['true_rate']):.6g} per second"))
    if sample.get("products"):
        bits.append(("Products", str(sample["products"])))
    if sample.get("repetitions"):
        bits.append(("Repetitions", str(sample["repetitions"])))
    scoring = problem.get("scoring") or {}
    archive = scoring.get("archive") or {}
    if archive.get("failure"):
        bits.append(("Archived rule", str(archive["failure"])))
    version2 = scoring.get("v2") or {}
    if version2.get("version"):
        bits.append(("v2", str(version2["version"])))
    if version2.get("cost"):
        bits.append(("v2 cost", str(version2["cost"])))
    if problem.get("remark"):
        bits.append(("Note", str(problem["remark"])))
    if problem.get("contract") == "partial":
        bits.append(("Contract", "Partly stated. The columns are the keys shared by every archived file."))
    return bits


def _rows_for(problem: dict, oracle: list[dict]) -> list[dict]:
    ident = problem["id"]
    rows = [row for row in oracle if f"{row['suite']}/{row['problem']}" == ident]
    key = _primary(problem)

    def sort_key(row: dict) -> tuple:
        canonical, _renamed = normalize(row["fields"])
        value = _float(canonical.get(key, "")) if key else None
        missing = value is None
        return (missing, value if value is not None else 0.0, row["entry"])

    return sorted(rows, key=sort_key)


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def _caption(primary: str | None) -> str:
    captions = {
        "force_calls": "Fewest force calls first. Failures are left out of this mean.",
        "mean_epoch": "Fewest epochs first. Failures are left out of this mean.",
        "mean_epochs": "Fewest epochs first. Failures are left out of this mean.",
        "avg_energy": "Lowest energy first.",
        "barrier_force_calls": "Fewest barrier force calls first.",
        "true_rate": "Lowest rate first.",
    }
    detail = captions.get(primary or "", "The columns are the archived keys.")
    return f"2014 archive. {detail}"


def _plot(points: list[dict], primary: str | None, caption: str | None = None) -> str:
    numbers = [point["value"] for point in points if point["value"] is not None and point["value"] > 0]
    if len(numbers) < 2:
        return ""
    scale = "linear"
    if numbers and min(numbers) > 0 and max(numbers) / min(numbers) >= 8:
        scale = "log"
    if any(point["value"] is not None and point["value"] <= 0 for point in points):
        scale = "linear"
    payload = {
        "metric": primary or "",
        "scale": scale,
        "points": [
            {
                "id": point["id"],
                "entry": point["entry"],
                "code": point["code"],
                "value": point["value"],
                "best": point["best"],
                "held": point["held"],
                "failed": bool(point.get("failed")),
            }
            for point in points
            if point["value"] is not None
        ],
    }
    import json

    data = json.dumps(payload, ensure_ascii=True)
    label = caption or (primary or "score").replace("_", " ")
    rows = len(payload["points"])
    return (
        '<figure class="bench-plot">'
        f"<figcaption>{html.escape(label)}. Fewer calls is to the left. Same order as the table.</figcaption>"
        f'<div class="bench-plot-mount" style="--bench-rows:{rows}"></div>'
        f'<script type="application/json" class="bench-plot-data">{data}</script>'
        "</figure>"
    )


def _code_label(fields: dict[str, str]) -> str:
    """Code column. Sella 0.1.1 stays on the 2014 table."""
    code = fields.get("code", "")
    version = fields.get("code_version", "")
    return f"{code} {version}".strip() if version else code


def campaign_problem(stem: str) -> str | None:
    """Map a results/campaign file name to a problem id."""
    if "morse" in stem:
        return "minimization/morse-bulk"
    if stem.startswith("eon-3.5.0-neb-pt"):
        return "neb/Pt-heptamer"
    if stem.startswith("eon-3.5.0-de"):
        return "double-ended-saddle-search/lj38"
    if stem.startswith("eon-3.5.0-saddle-pt"):
        return "saddle-search/pt-island"
    if stem == "sella-2.6.0-ts-lj38":
        return "tsopt/LJ38"
    if stem.startswith("rgsaddle-") and "loweig" in stem:
        return "lowest-eigenvector/lj38"
    if stem.startswith("rgsaddle-") and "saddle-pt" in stem:
        return "saddle-search/pt-island"
    if stem.startswith("rgsaddle-") and "ts-pt" in stem:
        return "tsopt/pt-island"
    if stem.startswith("rgsaddle-") and "ts-lj38" in stem:
        return "tsopt/LJ38"
    if stem.startswith(("pele-", "eon-client-", "eon-svn-", "rgmin-")):
        return "minimization/lj38"
    return None


def problem_html(
    problem: dict,
    oracle: list[dict],
    smoke_note: str = "",
    present: list[dict] | None = None,
    *,
    section_id: str | None = None,
    heading: str = "h2",
    heading_text: str | None = None,
    include_rule: bool = True,
) -> str:
    """2014 archive table for one problem. The v2 rank is a separate section."""
    canonical_rows = []
    for row in _rows_for(problem, oracle):
        fields, _renamed = normalize(row["fields"])
        canonical_rows.append((row, fields))
    metrics = list(problem["metrics"]["required"])
    primary = _primary(problem)
    best = None
    if primary:
        numbers = [
            value
            for value in (_float(fields.get(primary, "")) for _row, fields in canonical_rows)
            if value is not None
        ]
        best = min(numbers) if numbers else None

    ident = section_id or str(problem["problem"])
    title = heading_text or str(problem.get("title") or problem["problem"])
    lines = [f'<section class="bench bench-legacy-section" id="{html.escape(ident)}">']
    lines.append(f"<{heading}>{html.escape(title)}</{heading}>")
    if include_rule:
        bits = _bits(problem)
        if bits:
            lines.append('<dl class="bench-rule">')
            for label, text in bits:
                lines.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(text)}</dd>")
            lines.append("</dl>")
    if smoke_note:
        lines.append(f'<p class="bench-smoke">{html.escape(smoke_note)}</p>')
    lines.append('<details class="bench-details">')
    lines.append("<summary>2014 archive</summary>")

    prepared = []
    for rank, (row, fields) in enumerate(canonical_rows, 1):
        value = _float(fields.get(primary, "")) if primary else None
        held = _held(fields)
        code = _code_label(fields)
        row_id = _slug(f"{problem['id']}-{row['entry']}")
        prepared.append(
            {
                "id": row_id,
                "rank": rank,
                "entry": row["entry"],
                "label": f"{row['entry']} (held out of the original table)" if held else row["entry"],
                "code": code,
                "fields": fields,
                "value": value,
                "best": best is not None and value == best,
                "held": held,
                "current": bool(row.get("current")),
            }
        )
    lines.append(
        '<p class="bench-legacy">2014 archive. These rows stay in the published order. '
        "They are not a v2 score.</p>"
    )
    lines.append('<div class="bench-table"><table class="bench-legacy">')
    lines.append(f"<caption>{html.escape(_caption(primary))}</caption>")
    headers = ["", "Entry", "Code", *metrics, "Contributor", "Date"]
    lines.append("<thead><tr>")
    for index, header in enumerate(headers):
        kind = "num" if index == 0 or header in metrics else "text"
        lines.append(f'<th class="{kind}" scope="col">{html.escape(header.replace("_", " "))}</th>')
    lines.append("</tr></thead><tbody>")
    for point in prepared:
        classes = []
        if point["best"]:
            classes.append("best")
        if point["held"]:
            classes.append("held")
        if point.get("current"):
            classes.append("current")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        cells = [str(point["rank"]), point["label"], point["code"]]
        for metric in metrics:
            cells.append(point["fields"].get(metric, ""))
        cells.append(point["fields"].get("contributor", ""))
        cells.append(point["fields"].get("date", ""))
        lines.append(f'<tr id="{html.escape(point["id"])}"{class_attr}>')
        for index, cell in enumerate(cells):
            kind = "num" if index == 0 or headers[index] in metrics else "text"
            lines.append(f'<td class="{kind}">{html.escape(cell)}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    lines.append(_plot(prepared, primary, "2014 scores"))
    lines.append("</details>")
    lines.append("</section>")
    return "\n".join(lines)


def _raw(fragment: str) -> str:
    lines = ["#+begin_export rst", ".. raw:: html", ""]
    for line in fragment.splitlines():
        lines.append("   " + line)
    lines.append("#+end_export")
    lines.append("")
    return "\n".join(lines)


def _board_view(problem: dict, document: dict) -> dict:
    found = (document.get("problems") or {}).get(problem["id"])
    if found:
        return found
    return {
        "id": problem["id"],
        "title": str(problem.get("title") or problem["problem"]),
        "ranked": [],
        "history": [],
        "partial": [],
        "not_comparable": [],
    }


def _unparsed_by_problem(document: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in document.get("unparsed") or []:
        problem_id = str(row.get("problem") or "")
        if problem_id:
            grouped.setdefault(problem_id, []).append(row)
    return grouped


def _rule_html(problem: dict, smoke_note: str = "") -> str:
    lines: list[str] = []
    bits = _bits(problem)
    if bits:
        lines.append('<dl class="bench-rule">')
        for label, text in bits:
            lines.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(text)}</dd>")
        lines.append("</dl>")
    if smoke_note:
        lines.append(f'<p class="bench-smoke">{html.escape(smoke_note)}</p>')
    return "\n".join(lines)


def _legacy_section(problem: dict, oracle: list[dict]) -> str:
    return problem_html(
        problem,
        oracle,
        section_id=f"{problem['problem']}-legacy",
        heading="h3",
        heading_text="2014 archive",
        include_rule=False,
    )


def _sorted_problems(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda item: str(item.get("title") or item["problem"]).lower())


def _headlines_html(
    order: list[str],
    by_suite: dict[str, list[dict]],
    document: dict,
    pending: dict[str, list[dict]],
) -> str:
    from .board import gap_paragraph, headline_table

    lines = [
        '<section class="bench" id="headlines">',
        "<p>The first row of each problem is the best full harness run, ranked by PAR-2. "
        "The ranked table and the performance profile are on the problem page. "
        "The 2014 archive table is on that page.</p>",
    ]
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        lines.append(f'<h2 id="suite-{html.escape(suite)}">{html.escape(suite_title(suite))}</h2>')
        for problem in _sorted_problems(group):
            view = _board_view(problem, document)
            title = str(problem.get("title") or problem["problem"])
            href = f"generated/suites/{suite}.html#{problem['problem']}"
            lines.append(f"<h3>{html.escape(title)}</h3>")
            lines.append(headline_table(view))
            lead = gap_paragraph(view, pending.get(problem["id"]))
            if lead:
                lines.append(lead)
            lines.append(
                f'<p><a href="{html.escape(href)}">Ranked table, performance profile, and the 2014 archive</a>.</p>'
            )
    lines.append("</section>")
    return "\n".join(lines)


def _collection_html(
    order: list[str],
    by_suite: dict[str, list[dict]],
    document: dict,
    pending: dict[str, list[dict]],
    oracle: list[dict],
) -> str:
    from .board import gap_paragraph, headline_table

    lines = [
        '<section class="bench" id="collection">',
        "<p>Each problem leads with the v2 rank. "
        "The 2014 rows follow, in the published order.</p>",
    ]
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        lines.append(f'<h2 id="suite-{html.escape(suite)}">{html.escape(suite_title(suite))}</h2>')
        for problem in _sorted_problems(group):
            view = _board_view(problem, document)
            title = str(problem.get("title") or problem["problem"])
            href = f"generated/suites/{suite}.html#{problem['problem']}"
            lines.append(f"<h3>{html.escape(title)}</h3>")
            lines.append(headline_table(view))
            lead = gap_paragraph(view, pending.get(problem["id"]))
            if lead:
                lines.append(lead)
            lines.append(f'<p><a href="{html.escape(href)}">Full ranked table and performance profile</a>.</p>')
            lines.append(_legacy_section(problem, oracle))
    lines.append("</section>")
    return "\n".join(lines)


def _code_keys(pins: list[dict]) -> set[str]:
    keys: set[str] = set()
    for pin in pins:
        keys.add(str(pin.get("id") or "").lower())
        keys.add(str(pin.get("name") or "").lower())
    keys.discard("")
    return keys


def _matches_code(row: dict, keys: set[str]) -> bool:
    return str(row.get("code") or "").lower() in keys


def _codes_html(root: Path, document: dict) -> str:
    from .board import ranked_cells

    record = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    contributor = str(record.get("benchmark_contributor") or "")
    pins = [code for code in record.get("code") or [] if isinstance(code, dict)]
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for pin in pins:
        name = str(pin.get("name") or pin.get("id") or "")
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(pin)
    lines = [
        '<section class="bench" id="current-codes">',
        "<h2>Codes</h2>",
        f"<p>The benchmark contributor is {html.escape(contributor)}. "
        "A ranked row is a full harness run. "
        "The commit on that row is the commit that produced the score.</p>",
    ]
    score_fields = (
        "template",
        "version",
        "commit",
        "par2_mean",
        "mean_calls",
        "median_calls",
        "failures",
        "n",
    )
    for name in order:
        group = grouped[name]
        keys = _code_keys(group)
        lines.append(f"<h3>{html.escape(name)}</h3>")
        lines.append('<div class="bench-scroll"><table>')
        lines.append("<caption>Registry pins.</caption>")
        lines.append(
            "<thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>"
        )
        for pin in group:
            commit = str(pin.get("commit") or "")
            repo = str(pin.get("repository") or "")
            lines.append("<tr>")
            lines.append(f"<td>{html.escape(str(pin.get('role') or ''))}</td>")
            lines.append(f"<td>{html.escape(str(pin.get('ref') or ''))}</td>")
            lines.append(f"<td>{html.escape(commit[:12])}</td>")
            lines.append(f'<td><a href="{html.escape(repo)}">{html.escape(repo)}</a></td>')
            lines.append("</tr>")
        lines.append("</tbody></table></div>")
        best: list[tuple[dict, dict]] = []
        for problem in (document.get("problems") or {}).values():
            matched = [row for row in problem.get("ranked") or [] if _matches_code(row, keys)]
            if not matched:
                continue
            matched.sort(key=lambda row: (row.get("par2_mean") is None, float(row.get("par2_mean") or 0)))
            best.append((problem, matched[0]))
        best.sort(key=lambda item: item[0]["id"])
        if best:
            lines.append('<div class="bench-scroll"><table data-board-table="code-best">')
            lines.append("<caption>Best full harness run on each problem.</caption>")
            lines.append(
                "<thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th>"
                "<th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th>"
                "</tr></thead><tbody>"
            )
            for problem, row in best:
                cells = dict(ranked_cells(row, False))
                template = str(row.get("template") or row.get("entry") or "")
                lines.append(
                    "<tr "
                    f'data-problem="{html.escape(str(problem["id"]))}" '
                    f'data-template="{html.escape(template)}">'
                )
                lines.append(f"<td>{html.escape(str(problem.get('title') or problem['id']))}</td>")
                for field in score_fields:
                    lines.append(f'<td data-field="{field}">{html.escape(cells[field])}</td>')
                lines.append("</tr>")
            lines.append("</tbody></table></div>")
        else:
            lines.append("<p>No full ranked run.</p>")
        missing: list[str] = []
        for problem in (document.get("problems") or {}).values():
            partial = [row for row in problem.get("partial") or [] if _matches_code(row, keys)]
            off = [row for row in problem.get("not_comparable") or [] if _matches_code(row, keys)]
            if not partial and not off:
                continue
            reasons = []
            if off:
                reasons.append("not yet on the harness counter")
            if partial:
                reasons.append("partial")
            title = str(problem.get("title") or problem["id"])
            missing.append(f"<li>{html.escape(title)}: {html.escape(', '.join(reasons))}.</li>")
        for row in document.get("unparsed") or []:
            if not _matches_code(row, keys) or not row.get("problem"):
                continue
            label = str(row.get("template") or row.get("entry") or row.get("path") or "")
            note = str(row.get("note") or "no per-start record")
            missing.append(f"<li>{html.escape(label)}: {html.escape(note)}.</li>")
        if missing:
            lines.append("<p>No full ranked run</p><ul>")
            lines.extend(missing)
            lines.append("</ul>")
    lines.append("</section>")
    return "\n".join(lines)


def _smoke_note(root: Path) -> str:
    smoke_path = root / "softsaddle" / "entry.toml"
    if not smoke_path.is_file():
        return ""
    record = tomllib.loads(smoke_path.read_text(encoding="utf-8"))
    archived = record.get("archived") or {}
    smoke_run = record.get("smoke") or {}
    if not archived or not smoke_run:
        return ""
    return (
        f"A {smoke_run.get('searches')}-search SoftSaddle smoke is outside this rank. "
        f"The {archived.get('date')} {archived.get('entry')} entry records "
        f"{archived.get('force_calls')} force calls."
    )


def _suite_problem_link(problem: dict, templates: bool = False) -> str:
    ident = str(problem.get("id") or "")
    name = ident.split("/", 1)[1] if "/" in ident else ident
    anchor = f"{name}-templates" if templates else name
    return f"#{anchor}"


def render(root: Path) -> None:
    """Write the site tables from the v2 scorer."""
    from .board import _registry, plot_names_for, render_problem_section, summary_codes, summary_table, write_board

    root = Path(root)
    document = write_board(root, root / "results" / "board")
    problems = load_all(root / "problems")
    oracle = load_oracle(root / "oracle" / "entries.jsonl")
    smoke = _smoke_note(root)
    pending = _unparsed_by_problem(document)
    by_suite: dict[str, list[dict]] = {}
    for problem in problems.values():
        by_suite.setdefault(problem["suite"], []).append(problem)
    order = list(SUITE_ORDER) + sorted(set(by_suite) - set(SUITE_ORDER))

    include = root / "docs" / "include"
    include.mkdir(parents=True, exist_ok=True)
    (include / "codes.org").write_text(_raw(_codes_html(root, document)), encoding="utf-8")
    (include / "sota.org").write_text(
        _raw(_headlines_html(order, by_suite, document, pending)),
        encoding="utf-8",
    )
    (include / "collection.org").write_text(
        _raw(_collection_html(order, by_suite, document, pending, oracle)),
        encoding="utf-8",
    )
    for stale in ("featured.org", "terra-pass.org"):
        path = include / stale
        if path.exists():
            path.unlink()

    suite_dir = root / "docs" / "orgmode" / "generated" / "suites"
    suite_dir.mkdir(parents=True, exist_ok=True)
    codes = summary_codes(document, _registry(root))
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        sections: list[str] = []
        views = [_board_view(problem, document) for problem in _sorted_problems(group)]
        sections.append('<section class="bench" id="summary">')
        sections.append(summary_table(views, codes, _suite_problem_link))
        sections.append("</section>")
        for problem in sorted(group, key=lambda item: item["problem"].lower()):
            view = _board_view(problem, document)
            note = smoke if problem["id"] == "saddle-search/pt-island" else ""
            sections.append(
                render_problem_section(
                    view,
                    plot_names_for(view),
                    asset_prefix="../../_static/board",
                    preamble=_rule_html(problem, note),
                    section_id=str(problem["problem"]),
                    unparsed=pending.get(problem["id"]),
                )
            )
            sections.append(_legacy_section(problem, oracle))
        body = [
            f"#+TITLE: {suite_title(suite)}",
            "#+OPTIONS: toc:nil num:nil author:nil",
            "",
            _raw("".join(sections)).rstrip(),
            "",
        ]
        (suite_dir / f"{suite}.org").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    render(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()


