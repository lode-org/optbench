"""Each board row names its unit, and a row is ranked only beside its own unit."""

import json
from pathlib import Path

from optbench import board
from optbench.board import _archive_rows, _registry, build_board, render_problem_section, stated_full_n
from optbench.problems import load_all
from optbench.site import _primary
from optbench.site_check import _check_unit, _parse_tables
from optbench.units import FORCE_CALLS, archive_unit, harness_unit, unit_errors

ROOT = Path(__file__).resolve().parents[1]
PIN = "a" * 40


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _board(root: Path, units: list[str], primary: str, entries: list[tuple[str, dict]]) -> dict:
    _write(
        root / "problems" / "neb" / "band.toml",
        [
            'id = "neb/band"',
            'title = "Band"',
            'suite = "neb"',
            'problem = "band"',
            'contract = "stated"',
            "",
            "[sample]",
            "starts = 2",
            "",
            "[convergence]",
            "force_norm = 0.01",
            "max_force_calls = 100",
            "",
            "[metrics]",
            f'required = ["{primary}"]',
            "",
            *units,
        ],
    )
    _write(
        root / "templates" / "toy" / "band.toml",
        [
            'id = "toy/band"',
            'engine = "toy"',
            'problem = "neb/band"',
            'code = "toy"',
            'code_version = "1.0"',
            f'commit = "{PIN}"',
            'tuning = "default"',
        ],
    )
    _write(root / "codes" / "registry.toml", ["[[code]]", 'id = "toy"', 'name = "toy"', f'commit = "{PIN}"'])
    _write(
        root / "results" / "campaign" / "toy-band.dat",
        [
            "force_calls 1",
            "nfailed 0",
            "code toy",
            "code_version 1.0",
            f"commit {PIN}",
            "template toy/band",
            "problem neb/band",
            "counter harness",
            "tuning default",
        ],
    )
    starts = [
        {"force_calls": 40, "success": True, "start": "0", "counter": "harness"},
        {"force_calls": 60, "success": True, "start": "1", "counter": "harness"},
    ]
    _write(root / "results" / "campaign" / "toy-band.starts.jsonl", [json.dumps(row) for row in starts])
    lines = []
    for entry, fields in entries:
        lines.append(
            json.dumps(
                {
                    "suite": "neb",
                    "problem": "band",
                    "entry": entry,
                    "path": f"website/neb/band/{entry}/benchmark.dat",
                    "fields": fields,
                    "text": "",
                }
            )
        )
    _write(root / "oracle" / "entries.jsonl", lines)
    return build_board(root)["problems"]["neb/band"]


PER_IMAGE = [
    "[units]",
    'harness = "force calls per band"',
    'archive = "force calls per image"',
    'source = "The archive divides each band total by the moving images."',
]


def test_an_archive_row_in_another_unit_is_ranked_in_its_own_table(tmp_path: Path):
    entries = [
        ("old-lbfgs", {"force_calls": "5", "nfailed": "0", "code": "Old", "algorithm": "lbfgs"}),
        ("old-fire", {"force_calls": "7", "nfailed": "0", "code": "Old", "algorithm": "fire"}),
    ]
    problem = _board(tmp_path, PER_IMAGE, "force_calls", entries)

    assert problem["unit"] == "force calls per band"
    assert [(row["origin"], row["unit"]) for row in problem["ranked"]] == [("harness", "force calls per band")]
    assert problem["ranked"][0]["family_rank"] == 1
    (group,) = problem["other_units"]
    assert group["unit"] == "force calls per image"
    assert [(row["entry"], row["unit"], row["rank_cost"]) for row in group["rows"]] == [
        ("old-lbfgs", "force calls per image", 5.0),
        ("old-fire", "force calls per image", 7.0),
    ]
    assert [row["family_rank"] for row in group["rows"]] == [1, 2]
    assert [row["origin"] for row in board.board_rows(problem)] == ["harness", "archive", "archive"]

    html = render_problem_section(problem, set())
    assert "<h3>Ranked in force calls per image</h3>" in html
    assert "The tables above count force calls per band" in html
    tables = [table for table in _parse_tables(html) if table["attrs"].get("data-board-table") == "unit"]
    assert len(tables) == 1
    document = {"problems": {"neb/band": problem}}
    assert _check_unit("page", tables[0], document) == []
    ranked = next(table for table in _parse_tables(html) if table["attrs"].get("data-board-table") == "ranked")
    assert "Unit" in [cell["text"] for cell in ranked["rows"][0]["cells"]]
    assert "force calls per image" not in json.dumps([cell["text"] for row in ranked["rows"] for cell in row["cells"]])

    tampered = html.replace('<td data-label="Published" data-field="published">5</td>', '<td data-label="Published" data-field="published">4</td>')
    table = next(table for table in _parse_tables(tampered) if table["attrs"].get("data-board-table") == "unit")
    assert _check_unit("page", table, document) == [
        "page neb/band force calls per image row 1 published is '4', board has '5'"
    ]


def test_without_a_units_table_archive_and_harness_rows_share_one_table(tmp_path: Path):
    entries = [("old-lbfgs", {"force_calls": "5", "nfailed": "0", "code": "Old", "algorithm": "lbfgs"})]
    problem = _board(tmp_path, [], "force_calls", entries)

    assert problem["unit"] == FORCE_CALLS
    assert problem["other_units"] == []
    assert [(row["origin"], row["unit"]) for row in problem["ranked"]] == [
        ("archive", FORCE_CALLS),
        ("harness", FORCE_CALLS),
    ]
    assert "Ranked in" not in render_problem_section(problem, set())


def test_a_row_that_is_not_a_force_call_count_leaves_mean_calls_blank(tmp_path: Path):
    units = ["[units]", 'archive = "quenches"', 'source = "The website heads it mean Q."']
    entries = [("old-bh", {"basin_hopping_steps_mean": "12", "nfailed": "0", "code": "Old", "algorithm": "bh"})]
    problem = _board(tmp_path, units, "basin_hopping_steps_mean", entries)

    (group,) = problem["other_units"]
    (row,) = group["rows"]
    assert (row["unit"], row["published"], row["rank_cost"], row["mean_calls"]) == ("quenches", 12.0, 12.0, None)
    cells = dict(board.ranked_cells(row, False))
    assert (cells["unit"], cells["published"], cells["mean_calls"]) == ("quenches", "12", "")


def test_the_shipped_problems_name_each_archive_unit():
    problems = load_all(ROOT / "problems")
    assert unit_errors(problems, _primary) == []
    differ = {ident: (harness_unit(problem), archive_unit(problem)) for ident, problem in problems.items()}
    differ = {ident: pair for ident, pair in differ.items() if pair[0] != pair[1]}
    assert differ == {
        "accelerated-md/aladatom": (FORCE_CALLS, "escape rate (1/s)"),
        "accelerated-md/standard_aladatom": (FORCE_CALLS, "escape rate (1/s)"),
        "globalopt/lj100-binary": (FORCE_CALLS, "lowest energy (epsilon)"),
        "globalopt/lj150": (FORCE_CALLS, "quenches"),
        "globalopt/lj98": (FORCE_CALLS, "quenches"),
        "mlopt/C2H4": ("loss-gradient evaluations", "epochs"),
        "mlopt/emt-pt38": ("loss-gradient evaluations", "epochs"),
        "neb/Pt-heptamer": ("force calls per band", "force calls per image"),
        "saddle-search/pt-island": ("force calls per search", "force calls per campaign"),
    }


def test_no_archive_row_shows_a_rate_energy_epoch_or_quench_count_as_mean_calls():
    problems = load_all(ROOT / "problems")
    full_n = {ident: stated_full_n(ROOT, problem) for ident, problem in problems.items()}
    rows = _archive_rows(ROOT, problems, _registry(ROOT), full_n)
    blank = sorted({row["problem"] for row in rows if row["mean_calls"] is None})
    assert blank == [
        "accelerated-md/aladatom",
        "accelerated-md/standard_aladatom",
        "globalopt/lj100-binary",
        "globalopt/lj150",
        "globalopt/lj98",
        "mlopt/C2H4",
        "mlopt/emt-pt38",
    ]
    assert sum(row["mean_calls"] is None for row in rows) == 20
    for row in rows:
        if row["mean_calls"] is None:
            assert _primary(problems[row["problem"]]) not in {"force_calls", "barrier_force_calls"}


def test_unit_errors_name_each_gap():
    def problem(primary: str, units=None) -> dict:
        record = {"metrics": {"required": [primary]}}
        if units is not None:
            record["units"] = units
        return record

    problems = {
        "a/rate": problem("true_rate"),
        "a/table": problem("force_calls", "per image"),
        "a/keys": problem("force_calls", {"archive": "x", "source": "s", "scale": 8}),
        "a/source": problem("force_calls", {"archive": "per image"}),
        "a/fine": problem("mean_epoch", {"archive": "epochs", "source": "s"}),
    }
    assert unit_errors(problems, _primary) == [
        "a/keys [units] has unknown keys scale",
        "a/rate ranks true_rate and its [units] table names no archive unit",
        "a/source [units] has no source",
        "a/table [units] is not a table",
    ]
