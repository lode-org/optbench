"""Dolan-Moré profile for each method family that has per-start records.

The curve is performance_profile from optbench.profile, the same
definition the board uses. Tau is the board grid, 1, 2, 4, 8, and 16.
Dolan and Moré, Math. Program. 91:201-213, 2002.
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path


def _common():
    path = Path(__file__).resolve().parent / "common.py"
    spec = importlib.util.spec_from_file_location("paper_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _common()


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from optbench.board import _aligned_costs, _geometry_key
    from optbench.families import FAMILIES, family_title
    from optbench.profile import performance_profile, profile_cost
    from optbench.site_charts import PALETTE_SENTENCE, PALETTE_SIZE, PROFILE_MARGIN, PROFILE_TOP

    common.apply_style()
    import matplotlib.pyplot as plt

    board = common.load_board()
    problems = common.load_problems()
    by_stem = common.start_records()
    by_problem: dict[str, list] = defaultdict(list)
    order = [family_id for family_id, _title in FAMILIES]
    for problem_id, problem in board["problems"].items():
        title = common.display_title(problem_id, problems, str(problem.get("title") or problem_id))
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in problem.get("ranked") or []:
            records = common.records_for_row(row, by_stem)
            if len(records) < 2:
                continue
            profile = [(_geometry_key(record.get("start")), profile_cost(record)) for record in records]
            if not profile:
                continue
            copied = dict(row)
            copied["_profile"] = profile
            grouped[str(row.get("family") or "")].append(copied)
        families = [family_id for family_id in order if family_id in grouped]
        families.extend(family_id for family_id in grouped if family_id not in families)
        for family_id in families:
            rows = grouped[family_id]
            costs, _reason = _aligned_costs(rows)
            if costs is None or len(costs) < 2:
                continue
            try:
                profile = performance_profile(costs)
            except ValueError:
                continue
            stem = f"profile-{common.problem_slug(problem_id)}-{family_id}"
            family_name = family_title(family_id)
            caption = (
                f"Dolan-Moré profile for {family_name} on {title}. "
                "The curve is the fraction of starts whose cost is at most tau times "
                "the best finite cost in this family. "
                "Tau is 1, 2, 4, 8, and 16. "
                "An unsolved start stays in the denominator and does not cross a finite tau. "
                "Dolan, E. D. and Moré, J. J., Math. Program. 91:201-213, 2002."
            )
            areas = profile["area_log10"]
            ranked_names = sorted(areas, key=lambda name: areas[name], reverse=True)
            colored = set(ranked_names[:PALETTE_SIZE])
            n_labeled = min(PALETTE_SIZE, len(ranked_names))
            ncol = 1 if n_labeled > 4 else 2
            legend_rows = (n_labeled + ncol - 1) // max(ncol, 1)
            fig, ax = plt.subplots(
                figsize=(common.COLUMN_IN, 2.9 + 0.24 * legend_rows),
                layout="constrained",
            )
            taus = profile["taus"]
            for name in ranked_names:
                values = profile["rho"][name]
                if name in colored:
                    color = common.OKABE[ranked_names.index(name) % len(common.OKABE)]
                    width = 1.4
                    label = name
                else:
                    color = common.MUTED
                    width = 0.7
                    label = None
                ax.step(taus, values, where="post", color=color, linewidth=width, label=label)
                ax.plot(taus, values, linestyle="none", marker="o", markersize=2.4, color=color)
            ax.set_xscale("log")
            ax.set_xlim(min(taus), max(taus))
            ax.set_ylim(-PROFILE_MARGIN, PROFILE_TOP)
            ax.set_xticks(list(taus))
            ax.set_xticklabels([str(int(tau)) for tau in taus])
            ax.set_xlabel("performance ratio (tau)")
            ax.set_ylabel("Fraction of starts")
            ax.set_title(f"{family_name}, {title}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, fontsize=5.5)
            if len(ranked_names) > PALETTE_SIZE:
                caption += f" {PALETTE_SENTENCE} The other curves are gray."
            saved = common.save_pair(fig, stem, caption)
            by_problem[problem_id].append(
                {
                    "kind": "family",
                    "problem": problem_id,
                    "family": family_id,
                    "svg": saved["svg"],
                    "pdf": saved["pdf"],
                    "caption": caption,
                }
            )
    common.write_part("family", by_problem)
    print(f"family profiles: {sum(len(items) for items in by_problem.values())}")


if __name__ == "__main__":
    main()
