"""Page text for the public site.

Names and numbers come from the problem files, the code registry, and the
ranked board. A slug stays in the URL. A human name is the registry page
or a title already stored on the problem.
"""

from __future__ import annotations

import html
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .families import coordinate_count

_KIND = {
    "lennard-jones": "Lennard-Jones",
    "lj": "Lennard-Jones",
    "binary-lennard-jones": "binary Lennard-Jones",
    "tip4p": "TIP4P",
    "morse": "Morse",
    "morse_pt": "Morse Pt",
    "eam": "EAM",
    "eam_al": "EAM Al",
    "cuh2": "CuH2",
}


def pin_page(pin: dict) -> str:
    """Registry name used as the code page title."""
    return str(pin.get("page") or pin.get("name") or pin.get("id") or "").strip()


def _alias_list(pin: dict) -> list[str]:
    raw = pin.get("aliases") or []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw if str(item).strip()]


def _alias_page(code: str, registry: list[dict]) -> str | None:
    """Longest exact alias or name. Case distinguishes eOn from Eon."""
    best: str | None = None
    best_len = -1
    for pin in registry:
        candidates = [str(pin.get("name") or "")] + _alias_list(pin)
        for candidate in candidates:
            if candidate and candidate == code and len(candidate) > best_len:
                best = pin_page(pin)
                best_len = len(candidate)
    return best


def _id_page(code: str, registry: list[dict]) -> str | None:
    folded = code.lower()
    for pin in registry:
        names = {str(pin.get("id") or "").lower(), str(pin.get("name") or "").lower()}
        names.discard("")
        if folded in names:
            return pin_page(pin)
    return None


def _infer_empty(row: dict, registry: list[dict]) -> str | None:
    """An archived file with no code key sometimes names the code in the entry."""
    entry = str(row.get("entry") or "").strip()
    template = str(row.get("template") or "").strip()
    token = ""
    if entry:
        token = entry.split("-", 1)[0].split("/", 1)[0]
    elif "/" in template:
        token = template.split("/", 1)[0]
    token = token.strip()
    if not token:
        return None
    return _id_page(token, registry)


def code_page_name(row: dict, registry: list[dict]) -> str:
    """Code page for one ranked row."""
    code = str(row.get("code") or "").strip()
    if code:
        named = _alias_page(code, registry)
        if named:
            return named
        identified = _id_page(code, registry)
        if identified:
            return identified
        return code
    inferred = _infer_empty(row, registry)
    return inferred or "unknown"


def group_pins(registry: list[dict]) -> list[tuple[str, list[dict]]]:
    """Pins that share a page name, in registry order."""
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for pin in registry:
        if not isinstance(pin, dict):
            continue
        name = pin_page(pin)
        if not name:
            continue
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(pin)
    return [(name, grouped[name]) for name in order]


def _bare_slug(title: str, slug: str) -> bool:
    if title.lower().replace("_", "-") != slug.lower().replace("_", "-"):
        return False
    if title == title.lower():
        return True
    return bool(re.fullmatch(r"lj\d+", slug, re.IGNORECASE))


def _kind_of(problem: dict) -> str:
    potential = problem.get("potential") or {}
    kind = str(potential.get("kind") or potential.get("name") or potential.get("evaluator") or "")
    if not kind:
        band = problem.get("band") or {}
        kind = str(band.get("potential") or "")
    return kind.strip()


def _size_token(problem: dict) -> str:
    remark = str(problem.get("remark") or "")
    found = re.search(r"\bN=(\d+)\b", remark)
    if found:
        return found.group(1)
    kind = _kind_of(problem)
    slug = str(problem.get("problem") or "")
    digits = re.search(r"(\d+)", slug)
    if digits and kind in {"lennard-jones", "lj", "tip4p", "binary-lennard-jones"}:
        return digits.group(1)
    if kind in {"lennard-jones", "lj", "morse", "morse_pt"}:
        coords = coordinate_count(problem)
        if coords and coords % 3 == 0:
            return str(coords // 3)
    return ""


METRIC_LABELS = {
    "force_calls": "Force calls",
    "force_calls_min": "Fewest force calls",
    "force_calls_max": "Most force calls",
    "force_calls_median": "Median force calls",
    "force_calls_mean_error": "Mean error of force calls",
    "nfailed": "Failed runs",
    "mean_epoch": "Mean epochs",
    "median_epoch": "Median epochs",
    "mean_epochs": "Mean epochs",
    "median_epochs": "Median epochs",
    "mean_energy_rmse": "Mean energy RMSE",
    "median_energy_rmse": "Median energy RMSE",
    "mean_force_rmse": "Mean force RMSE",
    "median_force_rmse": "Median force RMSE",
    "avg_energy": "Average energy",
    "min_energy": "Lowest energy",
    "max_energy": "Highest energy",
    "basin_hopping_steps_mean": "Mean basin-hopping steps",
    "basin_hopping_steps_stddev": "Basin-hopping step spread",
    "cpu_time_mean": "Mean CPU time",
    "cpu_time_stddev": "CPU time spread",
    "energy_calls_mean": "Mean energy calls",
    "energy_calls_stddev": "Energy-call spread",
    "barrier_force_calls": "Barrier force calls",
    "barrier_jobs": "Barrier jobs",
    "rate_force_calls": "Rate force calls",
    "rate_jobs": "Rate jobs",
    "fraction_connected": "Fraction connected",
    "total_searches": "Searches",
    "unique_saddles": "Unique saddles",
    "transition_states": "Transition states",
    "true_rate": "True rate",
}

PAR2_SENTENCE = (
    "PAR-2 is the mean cost of the runs. "
    "A run that finishes counts as the calls it used. "
    "A run that fails counts as twice the cap."
)

ARCHIVE_SENTENCE = (
    "An archived score is the average of the runs that finished under the cap. "
    "A run that hit the cap is a failure and is left out of that average."
)


def metric_label(key: str) -> str:
    """Public name for one recorded metric. The mapping lives here."""
    text = str(key or "").strip()
    if text in METRIC_LABELS:
        return METRIC_LABELS[text]
    parts = []
    for token in text.split("_"):
        if token.lower() == "rmse":
            parts.append("RMSE")
        elif token:
            parts.append(token[:1].upper() + token[1:])
    return " ".join(parts) or text


def profile_worth_showing(problem: dict) -> bool:
    """A profile with one curve is a flat line at 1 and is not shown."""
    profile = problem.get("profile") or {}
    curves = profile.get("rho") or {}
    return len(curves) >= 2


def _same_pt_island(problem: dict) -> bool:
    slug = str(problem.get("problem") or "")
    ident = str(problem.get("id") or "")
    if slug in {"pt-island", "Pt-heptamer", "Pt"}:
        return True
    if ident.endswith("/pt-island") or ident.endswith("/Pt-heptamer") or ident.endswith("/Pt"):
        return True
    band = str((problem.get("band") or {}).get("potential") or "")
    return _kind_of(problem) == "morse_pt" or band == "morse_pt"


def human_title(problem: dict, by_slug: dict[str, list[dict]] | None = None) -> str:
    """Title for a heading. A formula or a slug is replaced from the files."""
    title = str(problem.get("title") or problem.get("problem") or "").strip()
    slug = str(problem.get("problem") or "")
    if slug == "C2H4" or title == "C2H4":
        return "Ethene"
    if slug == "Al" and title == "Al":
        return "EAM aluminum"
    if _same_pt_island(problem) and (title in {"Pt", "Pt-heptamer", "pt-island"} or _bare_slug(title, slug)):
        return "Pt heptamer island"
    if title and not _bare_slug(title, slug):
        return title
    if by_slug:
        for other in by_slug.get(slug, []):
            other_title = str(other.get("title") or "").strip()
            if not other_title or _bare_slug(other_title, slug):
                continue
            if other.get("id") == problem.get("id"):
                continue
            blocked = ("global", "clusters", "noise", "double-ended")
            if any(word in other_title.lower() for word in blocked):
                continue
            return other_title
    kind = _kind_of(problem)
    label = _KIND.get(kind, "")
    size = _size_token(problem)
    composed = ""
    if label and size:
        composed = f"{label} {size}"
    elif label and kind == "tip4p":
        composed = label
    if composed and "2ts" in slug:
        composed = f"{composed} transition-state separation"
    return composed or title or slug


def system_text(problem: dict) -> str:
    kind = _kind_of(problem)
    remark = str(problem.get("remark") or "")
    sample = problem.get("sample") or {}
    source = str(sample.get("structure_source") or "").strip()
    if kind == "tip4p":
        found = re.search(r"TIP4P N=(\d+)", remark)
        if found:
            return f"TIP4P, N={found.group(1)}"
        return "TIP4P"
    if "Voter-Chen" in remark:
        return "Voter-Chen EAM"
    label = _KIND.get(kind, "")
    if label and source:
        return f"{label}. {source}"
    if label:
        return label
    if str(problem.get("problem") or "") == "C2H4":
        return "Ethene (C2H4)"
    if source:
        return source
    title = str(problem.get("title") or "").strip()
    if "Lennard-Jones" in title:
        return "Lennard-Jones"
    if title.lower().startswith("morse"):
        return "Morse"
    if title and not _bare_slug(title, str(problem.get("problem") or "")):
        return title
    return kind


def atom_text(problem: dict) -> str:
    remark = str(problem.get("remark") or "")
    found = re.search(r"(\d+)-atom", remark)
    if found:
        return found.group(1)
    if "same as the 250 K problem" in remark or _kind_of(problem) == "tip4p":
        return ""
    coords = coordinate_count(problem)
    if coords and coords % 3 == 0:
        return str(coords // 3)
    title = str(problem.get("title") or "")
    found = re.search(r"\bPt\s+(\d+)\b", title)
    if found:
        return found.group(1)
    return ""


def run_text(root: Path, problem: dict) -> str:
    sample = problem.get("sample") or {}
    initializations = sample.get("initializations")
    if initializations not in (None, "", 0, "0"):
        return str(int(initializations))
    from .board import stated_full_n

    try:
        count = stated_full_n(root, problem)
    except (OSError, ValueError):
        count = None
    if count:
        return str(count)
    return ""


def _english(items: list[str]) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def suite_paragraph(suite_name: str, group: list[dict], root: Path, by_slug: dict[str, list[dict]]) -> str:
    """A short count, then the problem names as a strip."""
    del root
    names = [human_title(problem, by_slug) for problem in group]
    count = len(group)
    noun = "problem" if count == 1 else "problems"
    items = "".join(f"<li>{_cell(name)}</li>" for name in names)
    return f"<p>{_cell(suite_name)} compares {count} {noun}.</p>" f'<ul class="suite-facts">{items}</ul>'


def _metric_value(row: dict) -> float | None:
    if row.get("rank_cost") is not None:
        return float(row["rank_cost"])
    if row.get("par2_mean") is not None:
        return float(row["par2_mean"])
    if row.get("published") is not None:
        return float(row["published"])
    return None


def _success(row: dict) -> float | None:
    if row.get("success_rate") is not None:
        return float(row["success_rate"])
    failed = row.get("nfailed")
    count = row.get("n") or 0
    if failed is None or not count:
        return None
    try:
        return (int(count) - int(failed)) / int(count)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _format_rate(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100.0 * float(value):.1f}%"


def code_summaries(problem: dict, ranked: list[dict], registry: list[dict]) -> list[dict]:
    """Best and median ranked cost, and the success rate of the best row, per code."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ranked:
        grouped[code_page_name(row, registry)].append(row)
    found: list[dict] = []
    for code, rows in grouped.items():
        scored = [(value, row) for row in rows if (value := _metric_value(row)) is not None]
        if not scored:
            continue
        scored.sort(key=lambda item: (item[0], str(item[1].get("label") or "")))
        values = [item[0] for item in scored]
        best_value, best_row = scored[0]
        par2 = best_row.get("par2_mean")
        found.append(
            {
                "code": code,
                "best": best_value,
                "median": float(statistics.median(values)),
                "success": _success(best_row),
                "par2": None if par2 is None else float(par2),
                "n": len(scored),
                "problem": str(problem.get("id") or ""),
            }
        )
    found.sort(key=lambda item: (item["best"], item["code"].lower()))
    return found


def _cell(value: object) -> str:
    return html.escape(str(value))


def suite_name_for(suite: str) -> str:
    from .site import suite_title

    return suite_title(suite)


def recorded_runs(root: Path, problem: dict, ranked: list[dict]) -> tuple[str, str]:
    """Run count from the problem file, else the count the ranked rows record.

    The second value is a footnote. It quotes the same number the cell shows.
    """
    stated = run_text(root, problem)
    if stated:
        return stated, ""
    by_count: dict[int, set[str]] = {}
    for row in ranked:
        try:
            count = int(row.get("n") or 0)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        code = str(row.get("code") or "").strip() or "a ranked row"
        by_count.setdefault(count, set()).add(code)
    if not by_count:
        return "", ""
    if len(by_count) == 1:
        count, codes = next(iter(by_count.items()))
        names = ", ".join(sorted(codes))
        return str(count), (
            f"The problem file does not state a run count. "
            f"The {names} rows record {count} runs."
        )
    detail = "; ".join(
        f"{count} on {', '.join(sorted(codes))}" for count, codes in sorted(by_count.items())
    )
    return "—", (
        "The problem file does not state a run count, and the ranked rows do not agree. "
        f"They record {detail}."
    )


def _th(label: str, tip: str = "") -> str:
    tip_attr = f' title="{_cell(tip)}"' if tip else ""
    return f"<th{tip_attr}>{_cell(label)}</th>"


def leaderboard_html(items: list[dict], *, show_problem: bool, bench: str) -> str:
    """Ranked codes. The bar uses the same log10 axis as the chart for that problem."""
    from .site_charts import cost_decades, log_cost_fraction

    notes: list[str] = []

    def marker(text: str) -> str:
        if text not in notes:
            notes.append(text)
        return f"<sup>{notes.index(text) + 1}</sup>"

    headers = [_th("Rank")]
    if show_problem:
        headers.append(_th("Problem"))
    headers.extend(
        [
            _th("Code"),
            _th("Best", "Lowest cost for this code. A shorter bar is a lower cost on the shared log scale."),
            _th("Median", "Median of this code's costs on that problem."),
            _th("Success", "Share of runs that finished."),
            _th("PAR-2", "Mean cost of the runs. A failure counts as twice the cap."),
        ]
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("group") or "")].append(float(item["best"]))
    spans = {group: cost_decades(values) for group, values in grouped.items()}
    lines = [
        '<div class="bench-scroll">',
        f'<table class="bench-sortable leader" data-bench="{_cell(bench)}">',
        "<caption>A shorter bar is a lower cost on the log scale for that problem. Select a column heading to sort.</caption>",
        "<thead><tr>" + "".join(headers) + "</tr></thead><tbody>",
    ]
    rank = 0
    previous = None
    par2_dash = "A dash in PAR-2 means that row has no PAR-2 score."
    rate_dash = "A dash in the success rate means that row did not record one."
    for item in items:
        group = str(item.get("group") or "")
        if group != previous:
            rank = 1
            previous = group
            started = True
        else:
            rank += 1
            started = False
        span = spans[group]
        fraction = None if span is None else log_cost_fraction(float(item["best"]), span[0], span[1])
        percent = 0.0 if fraction is None else 100.0 * fraction
        classes = []
        if rank == 1:
            classes.append("is-best")
        if started and show_problem:
            classes.append("group-start")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        rank_class = "rank r1" if rank == 1 else "rank r2" if rank == 2 else "rank r3" if rank == 3 else "rank"
        lines.append(f"<tr{class_attr}>")
        lines.append(
            f'<td class="rank-cell" data-label="Rank"><span class="{rank_class}">{rank}</span></td>'
        )
        if show_problem:
            lines.append(
                '<td class="problem" data-label="Problem">'
                f'<a href="{_cell(item.get("problem_href") or "")}">{_cell(item.get("problem_title") or "")}</a></td>'
            )
        lines.append(
            '<td class="code" data-label="Code">'
            f'<a href="{_cell(item.get("code_href") or "")}">{_cell(item.get("code") or "")}</a></td>'
        )
        if fraction is None:
            bar = ""
        else:
            bar = f'<span class="bar" aria-hidden="true" style="--w:{percent:.4f}%"><i></i></span>'
        lines.append(
            '<td class="num best" data-label="Best">'
            f"{bar}"
            f'<span class="bar-num">{_cell(_format_number(item["best"]) or "—")}</span>'
            "</td>"
        )
        lines.append(
            f'<td class="num" data-label="Median">{_cell(_format_number(item["median"]) or "—")}</td>'
        )
        rate = _format_rate(item.get("success"))
        rate_mark = "" if rate else marker(rate_dash)
        lines.append(f'<td class="num" data-label="Success">{_cell(rate or "—")}{rate_mark}</td>')
        par2 = _format_number(item.get("par2"))
        par2_mark = "" if par2 else marker(par2_dash)
        lines.append(f'<td class="num" data-label="PAR-2">{_cell(par2 or "—")}{par2_mark}</td>')
        lines.append("</tr>")
    if not items:
        span = 7 if show_problem else 6
        lines.append(f'<tr><td colspan="{span}">No ranked row.</td></tr>')
    lines.append("</tbody></table></div>")
    for index, note in enumerate(notes, start=1):
        lines.append(f'<p class="bench-note"><sup>{index}</sup> {_cell(note)}</p>')
    return "\n".join(lines)


def suite_tables(
    group: list[dict],
    document: dict,
    registry: list[dict],
    root: Path,
    by_slug: dict[str, list[dict]],
    suite: str,
) -> tuple[str, str, list[dict]]:
    """Leaderboard, problem table, and the plot series."""
    prepared: list[dict] = []
    for problem in group:
        view = (document.get("problems") or {}).get(problem["id"]) or {}
        runs, note = recorded_runs(root, problem, list(view.get("ranked") or []))
        prepared.append(
            {
                "problem": problem,
                "title": human_title(problem, by_slug),
                "href": f"{suite}/{problem['problem']}.html",
                "system": system_text(problem),
                "atoms": atom_text(problem),
                "runs": runs,
                "note": note,
                "view": view,
            }
        )
    show_atoms = any(item["atoms"] for item in prepared)
    show_runs = any(item["runs"] for item in prepared)
    headers = ["Problem", "System"]
    if show_atoms:
        headers.append("Atoms")
    if show_runs:
        headers.append("Runs")
    lines = [
        "<h2>Problems</h2>",
        '<div class="bench-scroll">',
        '<table class="bench-sortable" data-bench="problems">',
        "<caption>Problems in this suite.</caption>",
        "<thead><tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr></thead><tbody>",
    ]
    missing_atoms = False
    notes: list[str] = []
    for item in prepared:
        lines.append("<tr>")
        lines.append(
            f'<th scope="row" data-label="Problem"><a href="{_cell(item["href"])}">{_cell(item["title"])}</a></th>'
        )
        lines.append(f'<td data-label="System">{_cell(item["system"] or "—")}</td>')
        if show_atoms:
            if item["atoms"]:
                lines.append(f'<td class="num" data-label="Atoms">{_cell(item["atoms"])}</td>')
            else:
                missing_atoms = True
                lines.append('<td class="num" data-label="Atoms">—</td>')
        if show_runs:
            mark = ""
            if item["note"]:
                notes.append(item["note"])
                mark = f"<sup>{len(notes)}</sup>"
            lines.append(f'<td class="num" data-label="Runs">{_cell(item["runs"] or "—")}{mark}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    if missing_atoms:
        lines.append('<p class="bench-note">A dash means the problem file does not state an atom count.</p>')
    for index, note in enumerate(notes, start=1):
        lines.append(f'<p class="bench-note"><sup>{index}</sup> {_cell(note)}</p>')
    problems_html = "\n".join(lines)
    board_items: list[dict] = []
    series: list[dict] = []
    for problem in group:
        view = (document.get("problems") or {}).get(problem["id"]) or {}
        title = human_title(problem, by_slug)
        href = f"{suite}/{problem['problem']}.html"
        for item in code_summaries(problem, list(view.get("ranked") or []), registry):
            board_items.append(
                {
                    **item,
                    "group": str(problem["id"]),
                    "problem_title": title,
                    "problem_href": href,
                    "code_href": f"../codes/{_slug(item['code'])}.html",
                }
            )
            series.append({**item, "title": title})
    leaderboard = "<h2>Leaderboard</h2>\n" + leaderboard_html(board_items, show_problem=True, bench="codes")
    return leaderboard, problems_html, series


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def _style_axes(axis) -> None:
    """Light panel, dark ink. The panel stays readable on a dark page."""
    figure = axis.figure
    figure.patch.set_facecolor("#f7f4ee")
    axis.set_facecolor("#f7f4ee")
    ink = "#1c1915"
    axis.tick_params(colors=ink, labelsize=9)
    axis.xaxis.label.set_color(ink)
    axis.yaxis.label.set_color(ink)
    axis.title.set_color(ink)
    for spine in axis.spines.values():
        spine.set_color("#c8c0b4")
    axis.grid(True, axis="x", color="#d9d3c7", linewidth=0.6)
    axis.set_axisbelow(True)


def write_suite_plot(suite: str, series: list[dict], directory: Path) -> str | None:
    """One matplotlib figure on the same decade axis as the page chart."""
    if len(series) < 1:
        return None
    import math
    import os

    import matplotlib
    from matplotlib.ticker import FuncFormatter

    matplotlib.use("Agg", force=False)
    matplotlib.rcParams["svg.hashsalt"] = "optbench"
    matplotlib.rcParams["svg.fonttype"] = "none"
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib.pyplot as plt

    from .board import _save_figure
    from .site_charts import AXIS_TITLE, cost_decades, decade_label

    problems: list[str] = []
    for item in series:
        if item["title"] not in problems:
            problems.append(item["title"])
    counts = [sum(1 for row in series if row["title"] == title) for title in problems]
    height = max(2.6, 0.36 * max(counts) + 1.35)
    figure, axes = plt.subplots(
        nrows=len(problems),
        ncols=1,
        figsize=(7.2, height * len(problems)),
        squeeze=False,
        layout="constrained",
    )
    figure.patch.set_facecolor("#f7f4ee")

    def _tick(value: float, _position: object) -> str:
        if value <= 0:
            return ""
        return decade_label(int(round(math.log10(value))))

    for axis, title in zip(axes[:, 0], problems):
        rows = [row for row in series if row["title"] == title]
        labels = [row["code"] for row in rows]
        values = [float(row["best"]) for row in rows]
        _style_axes(axis)
        axis.set_title(title)
        axis.set_xlabel(AXIS_TITLE)
        axis.set_yticks(list(range(len(rows))), labels)
        axis.invert_yaxis()
        span = cost_decades(values)
        if span is None:
            axis.set_xticks([])
            continue
        k_lo, k_hi = span
        if k_hi == k_lo:
            axis.set_xlim(0, 1)
            axis.set_xticks([0.5])
            axis.set_xticklabels([decade_label(k_lo)])
            continue
        cheapest = min(value for value in values if value > 0)
        left = 10.0 ** k_lo
        axis.set_xscale("log")
        axis.set_xlim(left, 10.0 ** k_hi)
        for index, value in enumerate(values):
            if value <= 0:
                continue
            color = "#c2410c" if value == cheapest else "#1c1915"
            axis.barh(index, value - left, left=left, color=color, height=0.62)
        axis.set_xlim(left, 10.0 ** k_hi)
        axis.set_xticks([10.0 ** k for k in range(k_lo, k_hi + 1)])
        axis.xaxis.set_major_formatter(FuncFormatter(_tick))
        axis.margins(x=0)
        axis.set_xlim(left, 10.0 ** k_hi)
    directory.mkdir(parents=True, exist_ok=True)
    _save_figure(figure, directory, f"suite-{suite}")
    plt.close(figure)
    return f"suite-{suite}.svg"


def restyle_problem_plots(document: dict, directory: Path, titles: dict[str, str]) -> None:
    """Redraw the site copies. The files under results/ are left alone."""
    import os

    import matplotlib
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    matplotlib.use("Agg", force=False)
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib.pyplot as plt

    from .board import _plain_tick, _save_figure

    plain = FuncFormatter(_plain_tick)
    directory.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.hashsalt": "optbench", "svg.fonttype": "none"}):
        for problem_id, problem in (document.get("problems") or {}).items():
            title = titles.get(problem_id) or str(problem.get("title") or problem_id)
            slug = str(problem_id).replace("/", "-")
            ranked = list(problem.get("ranked") or [])
            profile = problem.get("profile") or {}
            curves = profile.get("rho") or {}
            if profile_worth_showing(problem):
                figure, axis = plt.subplots(figsize=(7.2, 4.2))
                ink_cycle = ("#0f6e6e", "#9a3412", "#1e3a5f", "#3f6212", "#6b21a8", "#9f1239")
                for index, (label, values) in enumerate(curves.items()):
                    axis.step(
                        profile.get("taus") or [],
                        values,
                        where="post",
                        label=label,
                        color=ink_cycle[index % len(ink_cycle)],
                    )
                _style_axes(axis)
                axis.set_xscale("log", base=2)
                axis.xaxis.set_major_formatter(plain)
                axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                axis.set_xlabel("Performance ratio")
                axis.set_ylabel("Fraction of starts")
                axis.set_ylim(0.0, 1.05)
                axis.set_title(title)
                if len(curves) <= 8:
                    axis.legend(fontsize=8, loc="lower right", frameon=False)
                figure.tight_layout()
                _save_figure(figure, directory, f"{slug}-profile")
                plt.close(figure)
            plotted = [
                row
                for row in ranked
                if row.get("par2_mean") is not None and row.get("success_rate") is not None
            ]
            if len(plotted) >= 1:
                figure, axis = plt.subplots(figsize=(7.2, 4.2))
                rates = [float(row["success_rate"]) for row in plotted]
                costs = [float(row["par2_mean"]) for row in plotted]
                axis.scatter(rates, costs, s=28, color="#0f6e6e", linewidths=0)
                _style_axes(axis)
                axis.xaxis.set_major_formatter(plain)
                axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                axis.set_xlabel("Success rate")
                axis.set_ylabel("PAR-2 cost")
                axis.set_title(title)
                figure.tight_layout()
                _save_figure(figure, directory, f"{slug}-cost")
                plt.close(figure)


def plot_figure(prefix: str, name: str, label: str) -> str:
    stem = name[:-4] if name.endswith(".svg") else name
    svg_name = name if name.endswith(".svg") else f"{stem}.svg"
    return (
        '<figure class="bench-plot">'
        f'<a href="{_cell(f"{prefix}/{svg_name}")}">'
        "<picture>"
        f'<source srcset="{_cell(f"{prefix}/{stem}.webp")}" type="image/webp">'
        f'<img src="{_cell(f"{prefix}/{stem}.png")}" alt="{_cell(label)}" loading="lazy" decoding="async">'
        "</picture>"
        "</a>"
        f"<figcaption>{_cell(label)}. Select the plot for the full SVG.</figcaption>"
        "</figure>"
    )


def metric_html(_problem: dict, primary: str | None, method_href: str) -> str:
    """The main metric, with the scoring rules linked once."""
    if primary:
        lead = f"The main metric is {_cell(metric_label(primary))}. "
    else:
        lead = ""
    return f'<p class="metric-line">{lead}<a href="{_cell(method_href)}">Scoring rules</a>.</p>'


def _sentences(text: str) -> list[str]:
    """Split on a sentence end that is followed by a capital letter."""
    parts: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] in ".!?" and (index + 1 == len(text) or text[index + 1] == " "):
            rest = text[index + 1 :].lstrip()
            if not rest or rest[0].isupper():
                sentence = text[start : index + 1].strip()
                if sentence:
                    parts.append(sentence)
                start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def success_note(problem: dict) -> str:
    """Problem-specific success rule, one sentence per paragraph."""
    version2 = (problem.get("scoring") or {}).get("v2") or {}
    lines = []
    success = str(version2.get("success") or "").strip()
    if success:
        for index, sentence in enumerate(_sentences(success)):
            if sentence[-1] not in ".!?":
                sentence += "."
            if index == 0:
                lines.append(f'<p class="bench-note">Success on this problem: {_cell(sentence)}</p>')
            else:
                lines.append(f'<p class="bench-note">{_cell(sentence)}</p>')
    else:
        cost = str(version2.get("cost") or "").strip()
        if cost:
            lines.append(f'<p class="bench-note">The cost is {_cell(cost)}.</p>')
    verification = version2.get("verification") or {}
    label = str(verification.get("label") or "").strip()
    url = str(verification.get("url") or "").strip()
    if label and url:
        lines.append(
            f'<p class="bench-note">The check links <a href="{html.escape(url, quote=True)}">{_cell(label)}</a>.</p>'
        )
    return "\n".join(lines)


def _first_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    for index, char in enumerate(cleaned):
        if char in ".!?" and (index + 1 == len(cleaned) or cleaned[index + 1] == " "):
            sentence = cleaned[: index + 1]
            if len(sentence) <= 500:
                return sentence
            break
    return ""


def code_blurb(name: str, pins: list[dict]) -> str:
    """What the code is, from the registry role and the first sentence of a note."""
    if name == "unknown" and not pins:
        return (
            "These archived files have no code field, and the entry name does not match a registry id. "
            "They stay on this page."
        )
    if not pins:
        return "The registry has no pin for this code string. The rows below use the spelling on the archived file."
    roles = []
    for pin in pins:
        role = str(pin.get("role") or "").strip()
        if role and role not in roles:
            roles.append(role)
    sentence = ""
    for pin in pins:
        sentence = _first_sentence(str(pin.get("note") or ""))
        if sentence:
            break
    lead = name
    if roles:
        lead += " is used for " + _english(roles)
    text = lead + "."
    if sentence and sentence not in text:
        text += " " + sentence
    aliases: list[str] = []
    for pin in pins:
        for alias in _alias_list(pin):
            if alias not in aliases and alias != name:
                aliases.append(alias)
    if aliases:
        text += " Archived rows also spell it " + _english(aliases) + "."
    return text


def versions_html(pins: list[dict]) -> str:
    if not pins:
        return "<p>No registry pin.</p>"
    lines = [
        "<h3>Versions</h3>",
        '<div class="bench-scroll"><table class="bench-sortable">',
        "<caption>Registry pins for this code.</caption>",
        "<thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>",
    ]
    for pin in pins:
        commit = str(pin.get("commit") or "")
        repo = str(pin.get("repository") or "")
        lines.append("<tr>")
        lines.append(f'<td data-label="Role">{_cell(pin.get("role") or "")}</td>')
        lines.append(f'<td data-label="Ref">{_cell(pin.get("ref") or "")}</td>')
        lines.append(f'<td data-label="Commit">{_cell(commit[:12])}</td>')
        lines.append(f'<td data-label="Source"><a href="{_cell(repo)}">{_cell(repo)}</a></td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)
