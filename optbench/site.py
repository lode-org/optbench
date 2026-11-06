"""Render the public site from the v2 board.

Prose stays in ``docs/orgmode``. This module writes the ranked tables
and one page per code.
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
        norm = str(convergence.get("norm", "")).lower()
        value = convergence["force_norm"]
        if norm == "l2":
            stop = f"L2 norm of the force at or below {value} {unit}".strip()
        elif norm in {"linf", "linfnorm"}:
            stop = (
                "largest Cartesian force component (the L-infinity norm) "
                f"at or below {value} {unit}"
            ).strip()
        else:
            stop = f"norm of the force at or below {value} {unit}".strip()
        basis = str(convergence.get("stop_basis") or "").strip()
        if basis:
            stop = f"{stop}. {basis}"
        bits.append(("Stop", stop))
    if "gradient_norm" in convergence:
        bits.append(("Stop", f"gradient norm below {convergence['gradient_norm']}"))
    if "max_force_calls" in convergence:
        cap = f"{convergence['max_force_calls']} force calls"
        basis = str(convergence.get("cap_basis") or "").strip()
        if basis:
            cap = f"{cap}. {basis}"
        bits.append(("Cap", cap))
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
        starts = str(sample["starts"])
        basis = str(sample.get("starts_basis") or "").strip()
        if basis:
            starts = f"{starts}. {basis}"
        bits.append(("Starts", starts))
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
    if problem.get("contract") in {"partial", "shared"}:
        bits.append(("Contract", "The columns are the keys shared by every archived file."))
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
        "force_calls": "Fewest force calls first. Failures are left out of this mean. The 2014 archive rows stay in this table.",
        "mean_epoch": "Fewest epochs first. Failures are left out of this mean.",
        "mean_epochs": "Fewest epochs first. Failures are left out of this mean.",
        "avg_energy": "Lowest energy first.",
        "barrier_force_calls": "Fewest barrier force calls first.",
        "true_rate": "Lowest rate first.",
    }
    return captions.get(primary or "", "The columns are the archived keys.")


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
    """Code and version, without a status word."""
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
    """Archived rows for one problem. The ranked tables are a separate section."""
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
    lines = [f'<section class="bench" id="{html.escape(ident)}">']
    lines.append(f"<{heading}>{html.escape(title)}</{heading}>")
    if include_rule:
        bits = _bits(problem)
        if bits:
            lines.append('<dl class="bench-rule">')
            for label, text in bits:
                lines.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(text)}</dd>")
            lines.append("</dl>")
    if smoke_note:
        lines.append(f'<p class="bench-aside">{html.escape(smoke_note)}</p>')
    lines.append('<div class="bench-archive">')

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
            }
        )
    lines.append('<div class="bench-table"><table class="bench-archive">')
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
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        cells = [str(point["rank"]), point["label"], point["code"]]
        for metric in metrics:
            cells.append(point["fields"].get(metric, ""))
        cells.append(point["fields"].get("contributor", ""))
        cells.append(point["fields"].get("date", ""))
        lines.append(f'<tr id="{html.escape(point["id"])}"{class_attr}>')
        for index, cell in enumerate(cells):
            kind = "num" if index == 0 or headers[index] in metrics else "text"
            label = headers[index].replace("_", " ") or "Rank"
            lines.append(
                f'<td class="{kind}" data-label="{html.escape(label)}">{html.escape(cell)}</td>'
            )
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    lines.append(_plot(prepared, primary, "Archived scores"))
    lines.append("</div>")
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
        "short": [],
        "withheld": [],
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
        lines.append(f'<p class="bench-aside">{html.escape(smoke_note)}</p>')
    return "\n".join(lines)


def _sorted_problems(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda item: str(item.get("title") or item["problem"]).lower())


def _by_slug(problems: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for problem in problems.values():
        grouped.setdefault(str(problem.get("problem") or ""), []).append(problem)
    return grouped


def _titled_view(problem: dict, document: dict, by_slug: dict[str, list[dict]]) -> dict:
    from .site_pages import human_title

    view = dict(_board_view(problem, document))
    view["title"] = human_title(problem, by_slug)
    return view


def _headlines_html(
    order: list[str],
    by_suite: dict[str, list[dict]],
    document: dict,
    pending: dict[str, list[dict]],
    by_slug: dict[str, list[dict]],
) -> str:
    from .board import gap_paragraph, headline_table

    lines = [
        '<section class="bench" id="headlines">',
        "<p>The first row of each problem is the lowest cost. "
        "The family tables, the overall table, and the performance profile are on the problem page.</p>",
    ]
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        lines.append(f'<h2 id="suite-{html.escape(suite)}">{html.escape(suite_title(suite))}</h2>')
        for problem in _sorted_problems(group):
            view = _titled_view(problem, document, by_slug)
            title = view["title"]
            href = f"generated/suites/{suite}/{problem['problem']}.html"
            lines.append(f"<h3>{html.escape(title)}</h3>")
            lines.append(headline_table(view))
            lead = gap_paragraph(view, pending.get(problem["id"]))
            if lead:
                lines.append(lead)
            lines.append(
                f'<p><a href="{html.escape(href)}">Family tables, overall table, and performance profile</a>.</p>'
            )
    lines.append("</section>")
    return "\n".join(lines)


def _collection_html(
    order: list[str],
    by_suite: dict[str, list[dict]],
    document: dict,
    pending: dict[str, list[dict]],
    oracle: list[dict],
    by_slug: dict[str, list[dict]],
) -> str:
    from .board import gap_paragraph, headline_table

    lines = [
        '<section class="bench" id="collection">',
        "<p>Each problem leads with its lowest-cost row. "
        "The full tables are on the problem page. Archived rows sit in those tables.</p>",
    ]
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        lines.append(f'<h2 id="suite-{html.escape(suite)}">{html.escape(suite_title(suite))}</h2>')
        for problem in _sorted_problems(group):
            view = _titled_view(problem, document, by_slug)
            title = view["title"]
            href = f"generated/suites/{suite}/{problem['problem']}.html"
            lines.append(f"<h3>{html.escape(title)}</h3>")
            lines.append(headline_table(view))
            lead = gap_paragraph(view, pending.get(problem["id"]))
            if lead:
                lines.append(lead)
            lines.append(f'<p><a href="{html.escape(href)}">Family tables, overall table, and performance profile</a>.</p>')
    lines.append("</section>")
    return "\n".join(lines)


def _registry_pins(root: Path) -> tuple[str, list[dict]]:
    record = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    contributor = str(record.get("benchmark_contributor") or "")
    pins = [pin for pin in record.get("code") or [] if isinstance(pin, dict)]
    return contributor, pins


def _problem_titles(root: Path) -> dict[str, str]:
    from .site_pages import human_title

    problems = load_all(root / "problems")
    by_slug = _by_slug(problems)
    return {problem["id"]: human_title(problem, by_slug) for problem in problems.values()}


def _code_page_html(name: str, pins: list[dict], document: dict, registry: list[dict], titles: dict[str, str]) -> str:
    from .board import INFERRED_RULE, _headers, board_rows, ranked_cells
    from .site_charts import cost_bars_svg
    from .site_pages import code_blurb, code_page_name, code_summaries, versions_html

    found: list[tuple[dict, dict]] = []
    for problem in (document.get("problems") or {}).values():
        for row in board_rows(problem):
            if code_page_name(row, registry) == name:
                found.append((problem, row))
    found.sort(
        key=lambda item: (
            str(item[0].get("id") or ""),
            item[1].get("rank_cost") is None,
            float(item[1].get("rank_cost") or 0),
            str(item[1].get("label") or ""),
        )
    )
    from .paper_embed import code_charts

    codes = []
    for _problem, row in found:
        code = str(row.get("code") or "").strip()
        if code and code not in codes:
            codes.append(code)
    lines = [
        f'<section class="bench" id="code-{html.escape(_slug(name))}">',
        f"<p>{html.escape(code_blurb(name, pins))}</p>",
        versions_html(pins),
    ]
    history = code_charts(Path(__file__).resolve().parents[1], codes, "../../_static/paper")
    if history:
        lines.append(history)
    seen: set[str] = set()
    charts: list[tuple[str, list[dict]]] = []
    for problem, _row in found:
        ident = str(problem.get("id") or "")
        if not ident or ident in seen:
            continue
        seen.add(ident)
        ranked = list(problem.get("ranked") or [])
        items = code_summaries(problem, ranked, registry)
        if not items:
            continue
        title = titles.get(ident) or str(problem.get("title") or ident)
        charts.append((title, items))
    if charts:
        lines.append("<h3>Best cost</h3>")
        lines.append(cost_bars_svg(charts))
    lines.append("<h3>Results</h3>")
    lines.append(
        "<p>Ranked rows for this code, one version and one method on each row. "
        "PAR-2 counts a finished run as the calls it used and a failed run as twice the cap. "
        "An archived row shows the average it published, "
        "and its PAR-2 applies that rule to the failures the entry records. "
        '<a href="../../methodology.html">Scoring rules</a>.</p>'
    )
    if any(row.get("failures_inferred") for _problem, row in found):
        lines.append(f'<p data-board-note="inferred">{html.escape(INFERRED_RULE)}</p>')
    if found:
        headers = _headers({"ranked": []})
        lines.append('<div class="bench-scroll"><table class="bench-sortable" data-board-table="code-rows">')
        lines.append("<caption>Every ranked row for this code. Select a column heading to sort.</caption>")
        lines.append(
            "<thead><tr><th>Suite</th><th>Problem</th>"
            + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
            + "</tr></thead><tbody>"
        )
        for problem, row in found:
            cells = dict(ranked_cells(row, False))
            ident = str(problem["id"])
            suite, _, slug = ident.partition("/")
            href = f"../suites/{suite}/{slug}.html"
            title = titles.get(ident) or str(problem.get("title") or ident)
            lines.append(
                "<tr "
                f'data-problem="{html.escape(ident)}" '
                f'data-label="{html.escape(str(row.get("label") or ""))}">'
            )
            lines.append(f'<td data-label="Suite">{html.escape(suite_title(suite))}</td>')
            lines.append(
                f'<td data-label="Problem"><a href="{html.escape(href)}">{html.escape(title)}</a></td>'
            )
            for header, (field, value) in zip(headers, cells.items()):
                lines.append(
                    f'<td data-label="{html.escape(header)}" data-field="{field}">{html.escape(value)}</td>'
                )
            lines.append("</tr>")
        lines.append("</tbody></table></div>")
    else:
        lines.append("<p>No ranked row.</p>")
    lines.append("</section>")
    return "\n".join(lines)


def _codes_index_html(contributor: str, order: list[str]) -> str:
    lines = [
        '<section class="bench" id="codes">',
        "<h2>Codes</h2>",
        f"<p>The benchmark contributor is {html.escape(contributor)}. "
        "Each code page lists the pinned versions and every ranked row for that code.</p>",
        '<div class="code-grid">',
    ]
    for name in order:
        slug = _slug(name)
        lines.append(
            f'<a class="code-card" href="generated/codes/{html.escape(slug)}.html">{html.escape(name)}</a>'
        )
    lines.append("</div></section>")
    return "\n".join(lines)


def _write_code_pages(root: Path, document: dict) -> None:
    from .board import board_rows
    from .site_pages import code_page_name, group_pins

    contributor, pins = _registry_pins(root)
    grouped = dict(group_pins(pins))
    order = [name for name, _pins in group_pins(pins)]
    titles = _problem_titles(root)
    for problem in (document.get("problems") or {}).values():
        for row in board_rows(problem):
            name = code_page_name(row, pins)
            if name not in grouped:
                grouped[name] = []
                order.append(name)
    code_dir = root / "docs" / "orgmode" / "generated" / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    keep: set[str] = set()
    toc: list[str] = []
    for name in order:
        slug = _slug(name)
        keep.add(f"{slug}.org")
        toc.append(f"   generated/codes/{slug}")
        page = [
            f"#+TITLE: {name}",
            "#+OPTIONS: toc:nil num:nil author:nil",
            "",
            _raw(_code_page_html(name, grouped[name], document, pins, titles)).rstrip(),
            "",
        ]
        (code_dir / f"{slug}.org").write_text("\n".join(page), encoding="utf-8")
    for path in code_dir.glob("*.org"):
        if path.name not in keep:
            path.unlink()
    include = root / "docs" / "include"
    include.mkdir(parents=True, exist_ok=True)
    (include / "codes.org").write_text(_raw(_codes_index_html(contributor, order)), encoding="utf-8")
    body = [
        "#+TITLE: Codes",
        "#+OPTIONS: toc:nil num:nil author:nil",
        "",
        "Each code has its own page.",
        "The name is the registry page name.",
        "A ranked row is one version and one stock method.",
        "Archived rows that spell the same code are on that page.",
        "",
        '#+INCLUDE: "../include/codes.org"',
        "",
        "#+begin_export rst",
        ".. toctree::",
        "   :hidden:",
        "   :maxdepth: 1",
        "",
        *toc,
        "#+end_export",
        "",
    ]
    (root / "docs" / "orgmode" / "codes.org").write_text("\n".join(body), encoding="utf-8")


def _smoke_note(root: Path) -> str:
    entry_path = root / "softsaddle" / "entry.toml"
    if not entry_path.is_file():
        return ""
    record = tomllib.loads(entry_path.read_text(encoding="utf-8"))
    archived = record.get("archived") or {}
    short_run = record.get("check") or {}
    if not archived or not short_run:
        return ""
    return (
        f"A {short_run.get('searches')}-search SoftSaddle batch stays off this rank. "
        f"The {archived.get('date')} {archived.get('entry')} entry records "
        f"{archived.get('force_calls')} force calls."
    )


def _suite_problem_link(problem: dict, templates: bool = False) -> str:
    ident = str(problem.get("id") or "")
    name = ident.split("/", 1)[1] if "/" in ident else ident
    anchor = f"{name}-templates" if templates else name
    return f"#{anchor}"


def _fact(value: str, note: str, notes: list[str]) -> str:
    shown = value if value else "—"
    mark = ""
    if note:
        notes.append(note)
        mark = f"<sup>{len(notes)}</sup>"
    return f"{html.escape(shown)}{mark}"


def _problem_leaderboard(problem: dict, ranked: list[dict], registry: list[dict], code_prefix: str) -> str:
    from .site_charts import cost_bars_svg
    from .site_pages import code_summaries, leaderboard_html

    items = []
    for item in code_summaries(problem, ranked, registry):
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(item["code"])).strip("-")
        items.append({**item, "group": str(problem["id"]), "code_href": f"{code_prefix}{slug}.html"})
    title = str(problem.get("title") or problem.get("id") or "")
    chart = cost_bars_svg([(title, items)]) if items else ""
    leaderboard = leaderboard_html(items, show_problem=False, bench="leader")
    if chart:
        return chart + "\n<h2>Leaderboard</h2>\n" + leaderboard
    return "<h2>Leaderboard</h2>\n" + leaderboard


def _problem_preamble(root: Path, problem: dict, ranked: list[dict], note: str = "") -> str:
    from .site_pages import atom_text, recorded_runs, system_text

    system = system_text(problem)
    atoms = atom_text(problem)
    runs, run_note = recorded_runs(root, problem, ranked)
    caps = [text for label, text in _bits(problem) if label == "Cap"]
    cap = ", ".join(caps)
    notes: list[str] = []
    system_note = "" if system else "The problem file does not state a system."
    atom_note = "" if atoms else "The problem file does not state an atom count."
    if not runs and not run_note:
        run_note = "The problem file does not state a run count."
    cap_note = "" if cap else "The problem file does not state a cap."
    cards = [
        ("System", _fact(system, system_note, notes), ""),
        ("Atoms", _fact(atoms, atom_note, notes), " num"),
        ("Runs", _fact(runs, run_note, notes), " num"),
        ("Cap", _fact(cap, cap_note, notes), ""),
    ]
    lines = ['<div class="stat-row">']
    for label, value, extra in cards:
        lines.append(f'<div class="stat"><span class="k">{label}</span><span class="v{extra}">{value}</span></div>')
    lines.append("</div>")
    for index, text in enumerate(notes, start=1):
        lines.append(f'<p class="bench-note"><sup>{index}</sup> {html.escape(text)}</p>')
    if note:
        lines.append(f'<p class="bench-aside">{html.escape(note)}</p>')
    convergence = problem.get("convergence") or {}
    sample = problem.get("sample") or {}
    for label, text in _bits(problem):
        if label == "Stop" and str(convergence.get("stop_basis") or "").strip():
            lines.append(f'<p class="bench-note">{html.escape(text)}</p>')
        if label == "Starts" and str(sample.get("starts_basis") or "").strip():
            lines.append(f'<p class="bench-note">{html.escape(text)}</p>')
    return "\n".join(lines)


def render(root: Path) -> None:
    """Write the site tables from the v2 scorer."""
    from .board import plot_names_for, render_problem_section, write_board
    from .site_charts import suite_bars_svg
    from .site_pages import (
        human_title,
        metric_html,
        plot_figure,
        restyle_problem_plots,
        success_note,
        suite_paragraph,
        suite_tables,
        write_suite_plot,
    )

    root = Path(root)
    document = write_board(root, root / "results" / "board", site=True)
    problems = load_all(root / "problems")
    by_slug = _by_slug(problems)
    smoke = _smoke_note(root)
    pending = _unparsed_by_problem(document)
    by_suite: dict[str, list[dict]] = {}
    for problem in problems.values():
        by_suite.setdefault(problem["suite"], []).append(problem)
    order = list(SUITE_ORDER) + sorted(set(by_suite) - set(SUITE_ORDER))

    include = root / "docs" / "include"
    include.mkdir(parents=True, exist_ok=True)
    _write_code_pages(root, document)
    (include / "sota.org").write_text(
        _raw(_headlines_html(order, by_suite, document, pending, by_slug)),
        encoding="utf-8",
    )
    (include / "collection.org").write_text(
        _raw(_collection_html(order, by_suite, document, pending, [], by_slug)),
        encoding="utf-8",
    )
    for path in [include / "featured.org", *include.glob("*-pass.org")]:
        if path.exists():
            path.unlink()

    suite_dir = root / "docs" / "orgmode" / "generated" / "suites"
    suite_dir.mkdir(parents=True, exist_ok=True)
    _contributor, pins = _registry_pins(root)
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        problem_dir = suite_dir / suite
        problem_dir.mkdir(parents=True, exist_ok=True)
        keep: set[str] = set()
        toc: list[str] = []
        ordered = sorted(group, key=lambda item: human_title(item, by_slug).lower())
        for problem in ordered:
            name = str(problem["problem"])
            keep.add(f"{name}.org")
            toc.append(f"   {suite}/{name}")
            title = human_title(problem, by_slug)
            view = _titled_view(problem, document, by_slug)
            note = smoke if problem["id"] == "saddle-search/pt-island" else ""
            ranked = list(view.get("ranked") or [])
            section = render_problem_section(
                view,
                plot_names_for(view),
                asset_prefix="../../../_static/board",
                preamble=_problem_preamble(root, problem, ranked, note),
                leaderboard=_problem_leaderboard(problem, ranked, pins, "../../codes/"),
                metric_line=metric_html(problem, _primary(problem), "../../../methodology.html"),
                success_note=success_note(problem),
                section_id=name,
                unparsed=pending.get(problem["id"]),
                show_heading=False,
            )
            page = [
                f"#+TITLE: {title}",
                "#+OPTIONS: toc:nil num:nil author:nil",
                "",
                _raw(section).rstrip(),
                "",
            ]
            (problem_dir / f"{name}.org").write_text("\n".join(page), encoding="utf-8")
        for path in problem_dir.glob("*.org"):
            if path.name not in keep:
                path.unlink()
        leaderboard, problems_html, series = suite_tables(ordered, document, pins, root, by_slug, suite)
        plot_dir = root / "docs" / "source" / "_static" / "board"
        plot_name = write_suite_plot(suite, series, plot_dir)
        chart = suite_bars_svg(series)
        codes = []
        for item in series:
            if item["code"] not in codes:
                codes.append(item["code"])
        parts = [
            '<section class="bench plate-suite">',
            suite_paragraph(suite_title(suite), ordered, root, by_slug),
            '<div class="stat-row stat-row-2">',
            f'<div class="stat"><span class="k">Problems</span><span class="v num">{len(ordered)}</span></div>',
            f'<div class="stat"><span class="k">Codes</span><span class="v num">{len(codes)}</span></div>',
            "</div>",
        ]
        if chart or plot_name:
            parts.append("<h2>Charts</h2>")
            parts.append('<div class="lead-charts">')
            if chart:
                parts.append(chart)
            if plot_name:
                parts.append(
                    '<p class="chart-files"><a href="'
                    + html.escape(f"../../_static/board/{plot_name}")
                    + '">SVG file</a></p>'
                )
                parts.append('<div class="bench-export">')
                parts.append(plot_figure("../../_static/board", plot_name, f"{suite_title(suite)} best ranked cost"))
                parts.append("</div>")
            parts.append("</div>")
        parts.append(leaderboard)
        parts.append(problems_html)
        parts.append("</section>")
        body = [
            f"#+TITLE: {suite_title(suite)}",
            "#+OPTIONS: toc:nil num:nil author:nil",
            "",
            _raw("\n".join(parts)).rstrip(),
            "",
            "#+begin_export rst",
            ".. toctree::",
            "   :hidden:",
            "   :maxdepth: 1",
            "",
            *toc,
            "#+end_export",
            "",
        ]
        (suite_dir / f"{suite}.org").write_text("\n".join(body), encoding="utf-8")
    titles = {problem["id"]: human_title(problem, by_slug) for problem in problems.values()}
    restyle_problem_plots(document, root / "docs" / "source" / "_static" / "board", titles)
    (include / "overview.org").write_text(_raw(_overview_html(order, by_suite, by_slug)), encoding="utf-8")


def _overview_html(
    order: list[str],
    by_suite: dict[str, list[dict]],
    by_slug: dict[str, list[dict]],
) -> str:
    from .site_pages import human_title

    cards: list[str] = []
    suite_count = 0
    problem_count = 0
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        suite_count += 1
        problem_count += len(group)
        names = [human_title(problem, by_slug) for problem in sorted(group, key=lambda item: human_title(item, by_slug).lower())]
        shown = ", ".join(names[:3])
        if len(names) > 3:
            shown += f", and {len(names) - 3} more"
        cards.append(
            f'<a class="suite-card" href="generated/suites/{html.escape(suite)}.html">'
            f'<span class="suite-count">{len(group)}</span>'
            f'<span class="suite-name">{html.escape(suite_title(suite))}</span>'
            f'<span class="suite-problems">{html.escape(shown)}</span>'
            "</a>"
        )
    lines = [
        '<section class="bench" id="suites" data-bench="suites">',
        '<div class="section-kicker">',
        "<h2>Suites</h2>",
        '<a class="board-link" href="board.html">Open the board</a>',
        "</div>",
        f'<p class="count-line">{suite_count} suites, {problem_count} problems.</p>',
        '<div class="suite-grid">',
        *cards,
        "</div>",
        '<p class="page-links"><a href="methodology.html">Scoring rules</a></p>',
        "</section>",
    ]
    return "\n".join(lines)


def main() -> None:
    render(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()


