from pathlib import Path

from optbench import board
from optbench.check import load_oracle
from optbench.corrections import correction_errors, corrected, load_corrections

ROOT = Path(__file__).resolve().parents[1]
PAPER_NAMES = {"SD", "QM", "FIRE", "CG", "L-BFGS(line)", "L-BFGS(Hess)", "GL-BFGS(line)", "GL-BFGS(Hess)"}


def _ranked_archive_rows(problem_id: str) -> list[dict]:
    problems = board.load_all(ROOT / "problems")
    registry = board._registry(ROOT)
    full_n = {problem_id: board.stated_full_n(ROOT, problems[problem_id])}
    rows = [row for row in board._archive_rows(ROOT, problems, registry, full_n) if row["problem"] == problem_id]
    for row in rows:
        board._mark_short(row, full_n[problem_id])
        board._annotate_row(row, problems[problem_id], registry)
    return board._collapse_methods(rows)


def test_every_correction_names_its_archived_value_and_a_source():
    entries = load_oracle(ROOT / "oracle" / "entries.jsonl")
    assert correction_errors(ROOT, entries) == []
    assert {item["value"] for item in load_corrections(ROOT)} >= PAPER_NAMES


def test_each_gman_neb_optimizer_is_its_own_row():
    gman = [row for row in _ranked_archive_rows("neb/Pt-heptamer") if row["code"] == "Gman"]
    assert sorted(row["method"] for row in gman) == sorted(PAPER_NAMES)
    by_entry = {row["entry"]: row["method"] for row in gman}
    assert by_entry["gman-fire"] == "FIRE"
    assert by_entry["gman-lbfgs-hess"] == "L-BFGS(Hess)"
    assert by_entry["gman-glbfgs-hess"] == "GL-BFGS(Hess)"


def test_a_correction_applies_only_while_the_archive_has_its_value(tmp_path: Path):
    (tmp_path / "oracle").mkdir()
    (tmp_path / "oracle" / "corrections.toml").write_text(
        '[[correction]]\npath = "a/benchmark.dat"\nfield = "algorithm"\nwas = "lbfgs"\nvalue = "FIRE"\nsource = "opt.py"\n'
        '[[correction]]\npath = "b/benchmark.dat"\nfield = "algorithm"\nwas = "lbfgs"\nvalue = "CG"\nsource = ""\n',
        encoding="utf-8",
    )
    corrections = load_corrections(tmp_path)
    assert corrected("a/benchmark.dat", {"algorithm": "lbfgs"}, corrections) == {"algorithm": "FIRE"}
    assert corrected("a/benchmark.dat", {"algorithm": "fire"}, corrections) == {"algorithm": "fire"}
    entries = [{"path": "a/benchmark.dat", "fields": {"algorithm": "fire"}}]
    assert correction_errors(tmp_path, entries) == [
        "oracle/corrections.toml a/benchmark.dat algorithm expects 'lbfgs', the archive has 'fire'",
        "oracle/corrections.toml b/benchmark.dat algorithm has no source",
        "oracle/corrections.toml b/benchmark.dat algorithm names no oracle entry",
    ]
