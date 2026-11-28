"""Inline charts for the public site.

The figures are drawn from the ranked board. Colors come from CSS
variables, so the same SVG is readable in light and dark pages.
"""

from __future__ import annotations

import html
import math

from .site_pages import profile_worth_showing


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-") or "chart"


AXIS_TITLE = "cost (force calls, log scale)"
_READ = "A shorter bar is a lower cost on a log scale shared by the codes of this problem."
_READ_OFF = (
    "A shorter bar is a lower cost on a log scale shared by the codes of this problem, "
    "and a cost at or below zero is left off the scale."
)
_READ_NONE = "A cost at or below zero is left off the log scale, so this problem has no bar."
# optbench.css has one colour class per highlighted code, .s0 to .s5.
PALETTE_SIZE = 6
# The profile captions name that count, so the two sit together.
PALETTE_SENTENCE = "Colored lines are the six largest areas."
# A profile fraction runs from 0 to 1. Above 1 the chart keeps matplotlib's
# default data margin (axes.ymargin, 5% of the range), so a curve at 1 stays
# off the frame. The paper profiles use the same margin.
PROFILE_MARGIN = 0.05
PROFILE_TOP = 1.0 + PROFILE_MARGIN
_SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def decade_floor(value: float) -> int:
    """Largest integer k with 10**k <= value.

    ``floor(log10(1000))`` can come back one short, so the guess is
    corrected by comparing against the float power itself. The comparison
    is exact, so no tolerance is needed.
    """
    if not _positive(value):
        raise ValueError("decade_floor expects a positive finite cost")
    k = math.floor(math.log10(value))
    while 10.0 ** (k + 1) <= value:
        k += 1
    while 10.0 ** k > value:
        k -= 1
    return k


def decade_ceil(value: float) -> int:
    """Smallest integer k with 10**k >= value."""
    k = decade_floor(value)
    return k if 10.0 ** k == value else k + 1


def spans_a_decade(values) -> bool:
    """True when the positive values cover at least one factor of ten.

    That is the width of one decade on the cost bar axis, so a chart switches
    to a log scale at the same point where the bars gain a second decade.
    """
    values = [float(value) for value in values]
    if len(values) < 2 or not all(_positive(value) for value in values):
        return False
    return max(values) >= 10.0 * min(values)


def cost_decades(values) -> tuple[int, int] | None:
    """Decade at or below the minimum and decade at or above the maximum.

    Non-positive costs are not on a log scale. An empty positive set has no axis.
    """
    positive = [float(value) for value in values if _positive(float(value))]
    if not positive:
        return None
    return min(decade_floor(value) for value in positive), max(decade_ceil(value) for value in positive)


def log_cost_fraction(value: float, k_lo: int, k_hi: int) -> float | None:
    """Bar length from the shared decade axis. Zero is allowed. Nothing is clamped."""
    value = float(value)
    if not _positive(value):
        return None
    if k_hi == k_lo:
        # Every positive cost equals 10**k_lo, so every bar has the same zero length.
        return 0.0
    # The ends are exact powers of ten. Compare them directly, because
    # log10 of a power of ten can miss the integer by one ulp.
    if value == 10.0 ** k_lo:
        return 0.0
    if value == 10.0 ** k_hi:
        return 1.0
    return (math.log10(value) - float(k_lo)) / float(k_hi - k_lo)


def decade_label(k: int) -> str:
    return "10" + str(int(k)).translate(_SUPERS)


def _figure(slug: str, title: str, body: str, caption: str, legend: str = "") -> str:
    return (
        '<figure class="plate-figure">'
        f'<svg class="plate-chart" role="img" aria-labelledby="{_esc(slug)}-title" '
        f'viewBox="0 0 {body[0]} {body[1]}">'
        f'<title id="{_esc(slug)}-title">{_esc(title)}</title>'
        f"{body[2]}"
        "</svg>"
        f"<figcaption>{caption}</figcaption>"
        f"{legend}"
        "</figure>"
    )


def profile_svg(problem: dict) -> str:
    """Dolan-Moré profile. Empty when the problem has fewer than two codes."""
    if not profile_worth_showing(problem):
        return ""
    profile = problem.get("profile") or {}
    taus = [float(value) for value in profile.get("taus") or []]
    rho = {
        str(label): [float(value) for value in values]
        for label, values in (profile.get("rho") or {}).items()
        if values
    }
    taus = [value for value in taus if value > 0]
    if len(taus) < 2 or len(rho) < 2:
        return ""
    lo, hi = min(taus), max(taus)
    if hi <= lo:
        return ""
    area = profile.get("area_log10") or {}
    labels = sorted(rho, key=lambda label: (-float(area.get(label) or 0.0), label.lower()))
    highlight = labels[:PALETTE_SIZE]
    muted = [label for label in labels if label not in highlight]
    width, height = 640, 268
    left, right, top, bottom = 52, 16, 14, 36
    inner_w = width - left - right
    inner_h = height - top - bottom

    def x_of(tau: float) -> float:
        return left + (math.log(tau) - math.log(lo)) / (math.log(hi) - math.log(lo)) * inner_w

    def y_of(fraction: float) -> float:
        fraction = min(PROFILE_TOP, max(0.0, fraction))
        return top + (PROFILE_TOP - fraction) / PROFILE_TOP * inner_h

    parts: list[str] = []
    for fraction in (0.0, 0.5, 1.0):
        y = y_of(fraction)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        parts.append(
            f'<text class="num" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{_fmt(fraction)}</text>'
        )
    for tau in taus:
        x = x_of(tau)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + inner_h:.2f}"/>')
        parts.append(
            f'<text class="num" x="{x:.2f}" y="{top + inner_h + 16:.2f}" text-anchor="middle">{_fmt(tau)}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_h:.2f}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + inner_h:.2f}" x2="{width - right}" y2="{top + inner_h:.2f}"/>'
    )
    parts.append(
        f'<text class="axis-name" x="{left + inner_w / 2:.2f}" y="{height - 6}" text-anchor="middle">performance ratio (tau)</text>'
    )
    parts.append(
        f'<text class="axis-name" transform="rotate(-90 16 {top + inner_h / 2:.2f})" '
        f'x="16" y="{top + inner_h / 2:.2f}" text-anchor="middle">Fraction of starts</text>'
    )

    def curve(label: str, css: str) -> None:
        values = rho[label]
        count = min(len(taus), len(values))
        if count < 2:
            return
        commands = [f"M {x_of(taus[0]):.2f} {y_of(values[0]):.2f}"]
        for index in range(1, count):
            commands.append(f"H {x_of(taus[index]):.2f}")
            commands.append(f"V {y_of(values[index]):.2f}")
        parts.append(f'<path class="{css}" d="{" ".join(commands)}"><title>{_esc(label)}</title></path>')

    for label in muted:
        curve(label, "series muted")
    for index, label in enumerate(highlight):
        curve(label, f"series s{index}")

    legend_items = []
    for index, label in enumerate(highlight):
        legend_items.append(
            f'<li><i class="swatch s{index}"></i><span>{_esc(label)}</span></li>'
        )
    legend = '<ul class="chart-legend">' + "".join(legend_items) + "</ul>" if legend_items else ""
    display = str(problem.get("title") or problem.get("id") or "this problem")
    caption = "Performance profile. Higher means more starts solved within that performance ratio (tau)."
    if muted:
        caption += f" {PALETTE_SENTENCE} Thin lines are the other {len(muted)}."
    slug = _slug(str(problem.get("id") or display)) + "-profile"
    return _figure(slug, f"Performance profile for {display}", (width, height, "".join(parts)), _esc(caption), legend)


def cost_distribution_svg(rows: list[dict], title: str = "") -> str:
    """Histogram of rank cost. Empty when fewer than two costs can be drawn."""
    finite: list[float] = []
    missing = 0
    for row in rows:
        value = row.get("rank_cost")
        if value is None:
            missing += 1
            continue
        try:
            finite.append(float(value))
        except (TypeError, ValueError):
            missing += 1
    positive = [value for value in finite if value > 0]
    nonpositive = [value for value in finite if value <= 0]
    if nonpositive:
        used = finite
        scale = "linear"
        omitted = missing
    else:
        used = positive
        omitted = missing + (len(finite) - len(positive))
        scale = "log" if spans_a_decade(used) else "linear"
    if len(used) < 2:
        return ""
    lo, hi = min(used), max(used)
    # Sturges, J. Am. Stat. Assoc. 21, 65 (1926): ceil(log2 n) + 1 bins.
    bins = 1 if hi == lo else math.ceil(math.log2(len(used))) + 1
    if scale == "log" and hi > lo:
        log_lo, log_hi = math.log(lo), math.log(hi)
        edges = [math.exp(log_lo + (log_hi - log_lo) * index / bins) for index in range(bins + 1)]
    else:
        edges = [lo + (hi - lo) * index / bins for index in range(bins + 1)]
        scale = "linear"
    edges[0] = lo
    edges[-1] = hi
    counts = [0] * bins
    for value in used:
        if hi == lo:
            counts[0] += 1
            continue
        if scale == "log":
            span = math.log(edges[-1]) - math.log(edges[0])
            index = int((math.log(value) - math.log(edges[0])) / span * bins) if span else 0
        else:
            span = edges[-1] - edges[0]
            index = int((value - edges[0]) / span * bins) if span else 0
        if index >= bins:
            index = bins - 1
        if index < 0:
            index = 0
        counts[index] += 1
    width, height = 640, 220
    left, right, top, bottom = 44, 12, 12, 28
    inner_w = width - left - right
    inner_h = height - top - bottom
    peak = max(counts) or 1
    parts: list[str] = []
    # Counts are whole numbers, so the grid step starts at 1 and doubles until
    # four steps or fewer reach the peak. The count labels then sit at least a
    # quarter of the plot height apart.
    tick_step = 1
    while peak / tick_step > 4:
        tick_step *= 2
    tick = 0
    while tick <= peak:
        y = top + inner_h - (tick / peak) * inner_h
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="num" x="{left - 6}" y="{y + 4:.2f}" text-anchor="end">{tick}</text>')
        tick += tick_step
    slot = inner_w / bins
    # Each bar leaves this fraction of its bin empty on both sides, so two
    # neighbouring bins with the same count still read as two bars.
    gap = 0.12
    for index, count in enumerate(counts):
        bar_h = inner_h * count / peak
        x = left + index * slot + slot * gap
        bar_w = slot * (1 - 2 * gap)
        y = top + inner_h - bar_h
        start, end = edges[index], edges[index + 1]
        tip = f"{_fmt(start)} to {_fmt(end)}: {count}"
        parts.append(
            f'<rect class="hist" x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}">'
            f"<title>{_esc(tip)}</title></rect>"
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top + inner_h:.2f}" x2="{width - right}" y2="{top + inner_h:.2f}"/>'
    )
    parts.append(
        f'<text class="num" x="{left}" y="{height - 8}" text-anchor="start">{_fmt(lo)}</text>'
    )
    parts.append(
        f'<text class="num" x="{width - right}" y="{height - 8}" text-anchor="end">{_fmt(hi)}</text>'
    )
    display = title or "this problem"
    kind = "Logarithmic bins" if scale == "log" else "Linear bins"
    caption = f"Cost distribution of {len(used)} ranked rows. {kind}, from {_fmt(lo)} to {_fmt(hi)}."
    if omitted:
        caption += f" {omitted} ranked rows have no cost on this scale and are left off the chart."
    slug = _slug(display) + "-cost-dist"
    return _figure(
        slug,
        f"Cost distribution for {display}",
        (width, height, "".join(parts)),
        _esc(caption),
    )


def _cost_chart(title: str, rows: list[dict]) -> str:
    """One problem, one log10 axis, one sentence underneath."""
    if not rows:
        return ""
    values = [float(row["best"]) for row in rows]
    span = cost_decades(values)
    omitted = any(not _positive(value) for value in values)
    if span is None:
        caption = _READ_NONE
    elif omitted:
        caption = _READ_OFF
    else:
        caption = _READ
    width = 680
    label_w = 156
    number_w = 96
    bar_x = label_w + 12
    bar_max = width - bar_x - number_w
    parts: list[str] = []
    y = 18.0
    parts.append(f'<text class="group-label" x="0" y="{y:.0f}">{_esc(title)}</text>')
    y += 10
    positive = [value for value in values if _positive(value)]
    cheapest = min(positive) if positive else None
    k_lo, k_hi = span if span is not None else (0, 0)
    plot_top = y + 4
    row_mids: list[float] = []
    for _row in rows:
        row_mids.append(y + 14)
        y += 22
    if span is not None:
        y += 8
        axis_y = y
        parts.append(
            f'<line class="axis" x1="{bar_x:.2f}" y1="{axis_y:.2f}" x2="{bar_x + bar_max:.2f}" y2="{axis_y:.2f}"/>'
        )
        for k in range(k_lo, k_hi + 1):
            if k_hi == k_lo:
                tick_x = float(bar_x)
            else:
                tick_x = bar_x + (k - k_lo) / (k_hi - k_lo) * bar_max
            parts.append(
                f'<line class="grid" x1="{tick_x:.2f}" y1="{plot_top:.2f}" x2="{tick_x:.2f}" y2="{axis_y:.2f}"/>'
            )
            parts.append(
                f'<line class="axis" x1="{tick_x:.2f}" y1="{axis_y - 4:.2f}" x2="{tick_x:.2f}" y2="{axis_y:.2f}"/>'
            )
            parts.append(
                f'<text class="num" x="{tick_x:.2f}" y="{axis_y + 14:.2f}" text-anchor="middle">{_esc(decade_label(k))}</text>'
            )
        parts.append(
            f'<text class="axis-name" x="{bar_x + bar_max / 2:.2f}" y="{axis_y + 30:.2f}" text-anchor="middle">{_esc(AXIS_TITLE)}</text>'
        )
        y = axis_y + 36
    else:
        y += 8
    for row, value, mid in zip(rows, values, row_mids):
        code = str(row.get("code") or "")
        parts.append("<g>")
        parts.append(f"<title>{_esc(code)}: {_fmt(value)}</title>")
        parts.append(
            f'<text class="code-label" x="{label_w}" y="{mid + 4:.2f}" text-anchor="end">{_esc(code)}</text>'
        )
        fraction = None if span is None else log_cost_fraction(value, k_lo, k_hi)
        if fraction is not None:
            bar_w = bar_max * fraction
            css = "suite-bar is-best" if cheapest is not None and value == cheapest else "suite-bar"
            parts.append(
                f'<rect class="{css}" x="{bar_x:.2f}" y="{mid - 5:.2f}" width="{bar_w:.4f}" height="10"/>'
            )
        parts.append(
            f'<text class="num" x="{bar_x + bar_max + 8:.2f}" y="{mid + 4:.2f}">{_fmt(value)}</text>'
        )
        parts.append("</g>")
    slug = _slug(title) + "-cost-bars"
    return _figure(slug, f"Best cost for {title}", (width, y, "".join(parts)), _esc(caption))


def cost_bars_svg(groups: list[tuple[str, list[dict]]]) -> str:
    """One chart per problem. The axis is shared by the codes in that problem."""
    return "\n".join(_cost_chart(title, rows) for title, rows in groups if rows)


def suite_bars_svg(series: list[dict]) -> str:
    """Horizontal bars of best cost, one log scale per problem."""
    groups: list[tuple[str, list[dict]]] = []
    for item in series:
        title = str(item.get("title") or "")
        if not groups or groups[-1][0] != title:
            groups.append((title, []))
        groups[-1][1].append(item)
    return cost_bars_svg(groups)
