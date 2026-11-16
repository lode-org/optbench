"""Printed board rows for the band methods, a failed climbing step, and an asv file with no machine record."""

import json
from pathlib import Path

from optbench.board import build_board, render_problem_section, write_board
from optbench.rgsaddle_band import Report, _SURFACE_FAILED, accepted_step
from optbench.site_check import _cell_map, _parse_tables, _ranked_body

ROOT = Path(__file__).resolve().parents[1]
PIN = "a" * 40
KEPT = "a" * 40
ORPHAN = "b" * 40

# eOn rows that used to share one LBFGS label. Each line is the printed row.
SPLITS = {
    "double-ended-saddle-search/lj38": {
        "eOn (3.6.0) · OCI-NEB": ("4549.48", "crash 0, divergence 0, cap 44, wrong_stationary 0", "50"),
        "eOn (3.6.0) · energy-weighted CI-NEB": ("5000", "crash 0, divergence 0, cap 50, wrong_stationary 0", "50"),
        "eOn (3.6.0) · CI-NEB accept energy": (
            "4568.14",
            "crash 0, divergence 1, cap 43 (43 inferred), wrong_stationary 0",
            "50",
        ),
    },
    "double-ended-saddle-search/lj38_2ts_separation": {
        "eOn (develop 500350d9) · OCI-NEB": ("95094.3", "crash 0, divergence 0, cap 6, wrong_stationary 0", "51"),
        "eOn (develop 500350d9) · energy-weighted CI-NEB": (
            "15260.3",
            "crash 0, divergence 0, cap 0, wrong_stationary 0",
            "51",
        ),
        "eOn (3.6.0) · OCI-NEB": ("416487", "crash 0, divergence 42, cap 0, wrong_stationary 0", "51"),
        "eOn (3.6.0) · energy-weighted CI-NEB": (
            "343644",
            "crash 0, divergence 34, cap 0, wrong_stationary 0",
            "51",
        ),
        "eOn (3.6.0) · CI-NEB accept energy": (
            "415983",
            "crash 0, divergence 42, cap 0, wrong_stationary 0",
            "51",
        ),
    },
    "double-ended-saddle-search/pt-island": {
        "eOn (3.6.0) · OCI-NEB": ("3080.19", "crash 0, divergence 0, cap 21, wrong_stationary 0", "59"),
        "eOn (3.6.0) · energy-weighted CI-NEB": (
            "3206.76",
            "crash 0, divergence 0, cap 24, wrong_stationary 0",
            "59",
        ),
    },
}

CLIMBED = {
    "double-ended-saddle-search/lj38": (
        "rgsaddle (0.2.2) · CI-NEB energy-weighted",
        "2145.72",
        "crash 0, divergence 0, cap 11, wrong_stationary 0",
        "50",
    ),
    "double-ended-saddle-search/lj38_2ts_separation": (
        "rgsaddle (0.2.2) · CI-NEB energy-weighted",
        "266216",
        "crash 0, divergence 0, cap 27, wrong_stationary 0",
        "51",
    ),
}

_DOCUMENT = None


def _document():
    global _DOCUMENT
    if _DOCUMENT is None:
        _DOCUMENT = build_board(ROOT)
    return _DOCUMENT


def _printed_rows(problem_id: str) -> list[dict[str, str]]:
    html = render_problem_section(_document()["problems"][problem_id], set())
    tables = [table for table in _parse_tables(html) if table["attrs"].get("data-board-table") == "ranked"]
    assert len(tables) == 1
    return [_cell_map(row) for row in _ranked_body(tables[0])]


def test_each_band_method_is_its_own_printed_row():
    for problem_id, expected in SPLITS.items():
        rows = {row["label"]: row for row in _printed_rows(problem_id)}
        assert "eOn (3.6.0) · LBFGS" not in rows
        assert "eOn (develop 500350d9) · LBFGS" not in rows
        for label, (par2, failures, runs) in expected.items():
            row = rows[label]
            assert row["par2_mean"] == par2
            assert row["failures"] == failures
            assert row["n"] == runs
            assert row["family"] == "Band / NEB"


def test_a_failed_step_prints_the_climbed_image_as_a_cap(tmp_path: Path):
    """The zeroed report is image 0, under the force stop. The kept image is not."""
    for problem_id, (label, par2, failures, runs) in CLIMBED.items():
        row = next(item for item in _printed_rows(problem_id) if item["label"] == label)
        assert row["par2_mean"] == par2
        assert row["failures"] == failures
        assert row["n"] == runs

    kept = Report()
    kept.ci_index = 5
    zeroed = Report()
    zeroed.ci_index = 0
    forces = (0.0001, 0.0, 0.0, 0.0, 0.0, 1.94)
    failed = tmp_path / "failed"
    _band_tree(failed)
    _starts(failed, _record(accepted_step(kept, _SURFACE_FAILED, zeroed), forces, calls=10000, stop=0.001))
    row = _page_row(failed, "rgsaddle (0.2.2) · energy-weighted CI-NEB")
    assert row["failures"] == "crash 0, divergence 0, cap 1, wrong_stationary 0"
    assert row["par2_mean"] == "500000"
    assert row["n"] == "1"
    landed = tmp_path / "landed"
    _band_tree(landed)
    _starts(landed, _record(accepted_step(kept, 0, zeroed), forces, calls=10000, stop=0.001))
    quiet = _page_row(landed, "rgsaddle (0.2.2) · energy-weighted CI-NEB")
    assert quiet["failures"] == "crash 0, divergence 0, cap 0, wrong_stationary 0"
    assert quiet["par2_mean"] == "10000"


def test_an_asv_file_without_a_machine_record_stays_off_the_page(tmp_path: Path):
    _pair_problem(tmp_path)
    _method_template(tmp_path, "eon/develop-de-2ts-accept-energy", "CI-NEB accept energy")
    _method_template(tmp_path, "eon/develop-de-2ts-oci", "OCI-NEB")
    _registry(tmp_path)
    _asv_result(
        tmp_path,
        "bench_eon_develop.EonDevelopDe2ts",
        [4.0, 6.0],
        commit=KEPT,
        machine="workstation-a",
    )
    _asv_result(
        tmp_path,
        "bench_eon_develop.EonDevelopDe2tsOci",
        [80.0, 80.0],
        commit=ORPHAN,
        machine=None,
    )
    page = _page(tmp_path)
    row = _page_row(tmp_path, "eOn (3.6.0) · CI-NEB accept energy")
    assert row["failures"] == "crash 0, divergence 0, cap 0, wrong_stationary 0"
    assert row["par2_mean"] == "5"
    assert row["n"] == "2"
    assert "OCI-NEB" not in page
    assert "Not a problem template" not in page
    assert ORPHAN not in page


def _record(report: Report, forces: tuple[float, ...], *, calls: int, stop: float) -> dict:
    force = forces[int(report.ci_index)]
    if force > stop:
        return {
            "force_calls": calls,
            "success": False,
            "failure": "cap",
            "force_norm": force,
            "reasons": ["cap", "force_norm"],
            "start": "0",
            "counter": "harness",
        }
    return {
        "force_calls": calls,
        "success": True,
        "force_norm": force,
        "start": "0",
        "counter": "harness",
    }


def _band_tree(root: Path) -> None:
    _write(
        root / "problems" / "double-ended-saddle-search" / "pair.toml",
        [
            'id = "double-ended-saddle-search/pair"',
            'title = "Pair"',
            'suite = "double-ended-saddle-search"',
            'problem = "pair"',
            'contract = "stated"',
            "",
            "[sample]",
            "starts = 1",
            "",
            "[convergence]",
            "force_norm = 0.001",
            "max_force_calls = 250000",
            "",
            "[metrics]",
            'required = ["force_calls"]',
        ],
    )
    _write(
        root / "templates" / "rgsaddle" / "ew.toml",
        [
            'id = "rgsaddle/ew"',
            'engine = "rgsaddle-band"',
            'problem = "double-ended-saddle-search/pair"',
            'code = "rgsaddle"',
            'code_version = "0.2.2"',
            f'commit = "{PIN}"',
            'algorithm = "LBFGS"',
            'method = "energy-weighted CI-NEB"',
            'tuning = "default"',
        ],
    )
    _write(
        root / "codes" / "registry.toml",
        ["[[code]]", 'id = "rgsaddle"', 'name = "rgsaddle"', f'commit = "{PIN}"'],
    )
    _write(
        root / "results" / "campaign" / "ew.dat",
        [
            "force_calls 1",
            "nfailed 0",
            "code rgsaddle",
            "code_version 0.2.2",
            f"commit {PIN}",
            "template rgsaddle/ew",
            "problem double-ended-saddle-search/pair",
            "counter harness",
            "tuning default",
        ],
    )


def _starts(root: Path, row: dict) -> None:
    path = root / "results" / "campaign" / "ew.starts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _pair_problem(root: Path) -> None:
    _write(
        root / "problems" / "double-ended-saddle-search" / "pair.toml",
        [
            'id = "double-ended-saddle-search/pair"',
            'title = "Pair"',
            'suite = "double-ended-saddle-search"',
            'problem = "pair"',
            'contract = "stated"',
            "",
            "[sample]",
            "starts = 2",
            "",
            "[convergence]",
            "force_norm = 0.001",
            "max_force_calls = 250000",
            "",
            "[metrics]",
            'required = ["force_calls"]',
        ],
    )


def _method_template(root: Path, template_id: str, method: str) -> None:
    name = template_id.split("/")[-1]
    _write(
        root / "templates" / "eon" / f"{name}.toml",
        [
            f'id = "{template_id}"',
            'engine = "eon"',
            'problem = "double-ended-saddle-search/pair"',
            'code = "eOn"',
            'code_version = "3.6.0"',
            f'commit = "{PIN}"',
            'algorithm = "LBFGS"',
            f'method = "{method}"',
            'tuning = "default"',
        ],
    )


def _registry(root: Path) -> None:
    _write(
        root / "codes" / "registry.toml",
        ["[[code]]", 'id = "eon"', 'name = "eOn"', f'commit = "{PIN}"'],
    )


def _asv_result(root: Path, prefix: str, calls: list[float], *, commit: str, machine: str | None) -> None:
    folder = root / "results" / "asv" / (machine or "loose")
    folder.mkdir(parents=True, exist_ok=True)
    if machine is not None:
        (folder / "machine.json").write_text(
            json.dumps({"machine": machine, "version": 1}) + "\n",
            encoding="utf-8",
        )
    params = [["'0'"], [f"'{index}'" for index in range(len(calls))]]
    tracks = {
        "track_force_calls": list(calls),
        "track_success": [1.0 for _ in calls],
        "track_failure_cap": [0.0 for _ in calls],
        "track_failure_wrong_stationary": [0.0 for _ in calls],
        "track_failure_divergence": [0.0 for _ in calls],
        "track_failure_crash": [0.0 for _ in calls],
    }
    columns = ["result", "params", "version", "started_at", "duration"]
    results = {f"{prefix}.{name}": [value, params, "v", 0, 0.1] for name, value in tracks.items()}
    payload = {
        "commit_hash": commit,
        "env_name": "py",
        "date": 1,
        "params": {"machine": machine or "loose"},
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }
    (folder / f"{commit[:8]}-py.json").write_text(json.dumps(payload), encoding="utf-8")


def _page(root: Path) -> str:
    write_board(root, root / "board", site=False)
    return (root / "board" / "v2-board.org").read_text(encoding="utf-8")


def _page_row(root: Path, label: str) -> dict[str, str]:
    text = _page(root)
    found = []
    for table in _parse_tables(text):
        if table["attrs"].get("data-board-table") not in {"ranked", "headline"}:
            continue
        found.extend(_cell_map(row) for row in _ranked_body(table))
    return next(row for row in found if row.get("label") == label)


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_limit_runs_the_first_band_pairs(monkeypatch, tmp_path: Path):
    import optbench.rgsaddle_band as band
    from optbench.__main__ import run as cli_run

    monkeypatch.setattr(
        band,
        "cached_pairs",
        lambda *args, **kwargs: {"a": {"name": "a"}, "b": {"name": "b"}, "c": {"name": "c"}},
    )
    monkeypatch.setattr(band, "_job", lambda pair, settings: {"name": pair["name"]})
    seen = []

    def fake_execute(job):
        seen.append(job["name"])
        return {"start": job["name"]}

    monkeypatch.setattr(band, "_execute", fake_execute)
    monkeypatch.setattr(
        band,
        "write_outputs",
        lambda output, records, template, settings, problem: {
            "starts": len(records),
            "force_calls": 1,
            "nfailed": 0,
            "template": template["id"],
            "problem": problem["id"],
        },
    )
    cli_run(
        "neb/Pt-heptamer",
        "rgsaddle/ci-neb",
        root=ROOT,
        output=tmp_path / "benchmark.dat",
        limit=1,
    )
    assert seen == ["a"]


def test_a_crashed_band_search_does_not_write_nan():
    import json

    from optbench.rgsaddle_band import _crash_record

    record = _crash_record("pair", 3, 0, "RuntimeError")
    text = json.dumps(record, sort_keys=True)
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["failure"] == "crash"
    assert loaded["success"] is False
    assert loaded["energy"] is None
    assert loaded["force_norm"] is None
    assert loaded["force_calls"] == 3
