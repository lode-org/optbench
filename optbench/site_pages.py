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


def human_title(problem: dict, by_slug: dict[str, list[dict]] | None = None) -> str:
    """Title for a heading. A lowercase slug is replaced from the file or a sibling."""
    title = str(problem.get("title") or problem.get("problem") or "").strip()
    slug = str(problem.get("problem") or "")
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
    """One paragraph from the problem files in this suite."""
    names = [human_title(problem, by_slug) for problem in group]
    metrics: list[str] = []
    for problem in group:
        required = list((problem.get("metrics") or {}).get("required") or [])
        for key in required:
            if key == "nfailed":
                continue
            label = str(key).replace("_", " ")
            if label not in metrics:
                metrics.append(label)
    metric_text = _english(metrics[:6])
    runs = []
    for problem in group:
        count = run_text(root, problem)
        if count:
            runs.append(f"{human_title(problem, by_slug)} has {count} runs")
    text = f"{suite_name} covers {_english(names)}."
    if metric_text:
        text += f" The recorded metrics include {metric_text}."
    if runs:
        text += " " + _english(runs) + "."
    rules = {
        str(((problem.get("scoring") or {}).get("half_even") or {}).get("rule") or "")
        for problem in group
    }
    rules.discard("")
    if len(rules) == 1:
        rule = next(iter(rules))
        text += " Archived means follow this rule: " + rule[0].lower() + rule[1:]
    return text


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
        found.append(
            {
                "code": code,
                "best": best_value,
                "median": float(statistics.median(values)),
                "success": _success(best_row),
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


def suite_tables(
    group: list[dict],
    document: dict,
    registry: list[dict],
    root: Path,
    by_slug: dict[str, list[dict]],
    suite: str,
) -> tuple[str, list[dict]]:
    """Problem table and per-code summary. The plot series is the second value."""
    lines = [
        '<div class="bench-scroll">',
        '<table class="bench-sortable" data-bench="problems">',
        "<caption>Problems in this suite.</caption>",
        "<thead><tr><th>Problem</th><th>System</th><th>Atoms</th><th>Task</th><th>Runs</th></tr></thead><tbody>",
    ]
    task = suite_name_for(suite)
    for problem in group:
        name = str(problem["problem"])
        title = human_title(problem, by_slug)
        href = f"{suite}/{name}.html"
        atoms = atom_text(problem)
        lines.append("<tr>")
        lines.append(f'<th scope="row"><a href="{_cell(href)}">{_cell(title)}</a></th>')
        lines.append(f"<td>{_cell(system_text(problem))}</td>")
        lines.append(f'<td class="num">{_cell(atoms)}</td>')
        lines.append(f"<td>{_cell(task)}</td>")
        lines.append(f'<td class="num">{_cell(run_text(root, problem))}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    lines.append("<h2>Results by code</h2>")
    lines.append(
        "<p>Best is the lowest ranked cost for that code on that problem. "
        "Median is the median of that code's ranked costs. "
        "A harness row uses PAR-2. An archived row uses its published primary number. "
        "Success rate is the best row's rate, or the share of its runs that are not failures.</p>"
    )
    lines.append('<div class="bench-scroll">')
    lines.append('<table class="bench-sortable" data-bench="codes">')
    lines.append("<caption>One row per code on each problem.</caption>")
    lines.append(
        "<thead><tr><th>Problem</th><th>Code</th><th>Best</th><th>Median</th><th>Success rate</th><th>Rows</th></tr></thead><tbody>"
    )
    series: list[dict] = []
    for problem in group:
        view = (document.get("problems") or {}).get(problem["id"]) or {}
        title = human_title(problem, by_slug)
        for item in code_summaries(problem, list(view.get("ranked") or []), registry):
            href = f"../codes/{_slug(item['code'])}.html"
            lines.append("<tr>")
            lines.append(f"<td>{_cell(title)}</td>")
            lines.append(f'<td><a href="{_cell(href)}">{_cell(item["code"])}</a></td>')
            lines.append(f'<td class="num">{_cell(_format_number(item["best"]))}</td>')
            lines.append(f'<td class="num">{_cell(_format_number(item["median"]))}</td>')
            lines.append(f'<td class="num">{_cell(_format_rate(item["success"]))}</td>')
            lines.append(f'<td class="num">{_cell(item["n"])}</td>')
            lines.append("</tr>")
            series.append({**item, "title": title})
    if not series:
        lines.append('<tr><td colspan="6">No ranked row.</td></tr>')
    lines.append("</tbody></table></div>")
    return "\n".join(lines), series


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def write_suite_plot(suite: str, series: list[dict], directory: Path) -> str | None:
    """One matplotlib figure, same text and raster settings as the board plots."""
    if len(series) < 1:
        return None
    import os

    import matplotlib
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    matplotlib.use("Agg", force=False)
    matplotlib.rcParams["svg.hashsalt"] = "optbench"
    matplotlib.rcParams["svg.fonttype"] = "none"
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib.pyplot as plt

    from .board import _plain_tick, _save_figure

    problems: list[str] = []
    plotted: list[dict] = []
    for item in series:
        if item["title"] not in problems:
            problems.append(item["title"])
    for title in problems:
        rows = [row for row in series if row["title"] == title][:12]
        plotted.extend(rows)
    series = plotted
    height = max(2.6, 0.36 * max(len([row for row in series if row["title"] == title]) for title in problems) + 1.1)
    figure, axes = plt.subplots(
        nrows=len(problems),
        ncols=1,
        figsize=(7.2, height * len(problems)),
        squeeze=False,
    )
    plain = FuncFormatter(_plain_tick)
    for axis, title in zip(axes[:, 0], problems):
        rows = [row for row in series if row["title"] == title]
        labels = [row["code"] for row in rows]
        values = [row["best"] for row in rows]
        positions = list(range(len(rows)))
        axis.barh(positions, values, color="C0", height=0.62)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.xaxis.set_major_formatter(plain if min(values) > 0 else ScalarFormatter(useOffset=False))
        axis.set_xlabel("best ranked cost")
        axis.set_title(title)
        if min(values) > 0 and max(values) / min(values) >= 8:
            axis.set_xscale("log")
            axis.xaxis.set_major_formatter(plain)
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    _save_figure(figure, directory, f"suite-{suite}")
    plt.close(figure)
    return f"suite-{suite}.svg"


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


def metric_html(problem: dict, primary: str | None) -> str:
    """Metric names and the scoring sentences stored on the problem."""
    lines = ["<h3>Metrics</h3>"]
    if primary:
        lines.append(f"<p>The main metric is {_cell(primary.replace('_', ' '))}.</p>")
    scoring = problem.get("scoring") or {}
    half = scoring.get("half_even") or {}
    if half.get("rule"):
        lines.append(f"<p>{_cell(half['rule'])}</p>")
    version2 = scoring.get("v2") or {}
    if version2.get("cost"):
        lines.append(f"<p>{_cell(version2['cost'])}</p>")
    if version2.get("success"):
        lines.append(f"<p>{_cell(version2['success'])}</p>")
    required = list((problem.get("metrics") or {}).get("required") or [])
    if required:
        lines.append("<ul>")
        for key in required:
            lines.append(f"<li>{_cell(str(key).replace('_', ' '))}</li>")
        lines.append("</ul>")
    lines.append("<p>Select a column heading to sort the table.</p>")
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
        lines.append(f"<td>{_cell(pin.get('role') or '')}</td>")
        lines.append(f"<td>{_cell(pin.get('ref') or '')}</td>")
        lines.append(f"<td>{_cell(commit[:12])}</td>")
        lines.append(f'<td><a href="{_cell(repo)}">{_cell(repo)}</a></td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)
