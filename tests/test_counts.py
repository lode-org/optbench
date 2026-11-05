import ast
import re
from pathlib import Path

from optbench import board
from optbench.check import load_oracle
from optbench.counts import count_errors, entry_count, load_counts, text_count

ROOT = Path(__file__).resolve().parents[1]


def _entry_dir(path: str) -> Path:
    return ROOT / "archive" / "entries" / Path(path).parent.relative_to("website")


def _archive_rows(problem_id: str) -> dict[str, dict]:
    problems = board.load_all(ROOT / "problems")
    full_n = {problem_id: board.stated_full_n(ROOT, problems[problem_id])}
    rows = board._archive_rows(ROOT, problems, board._registry(ROOT), full_n)
    return {row["entry"]: row for row in rows if row["problem"] == problem_id}


def _whole_number_means(printed: str, runs: int) -> bool:
    low = int(float(printed) * runs) - 2
    return any("%.6g" % (total / runs) == printed for total in range(low, low + 5))


def test_every_count_names_an_entry_and_a_source():
    entries = load_oracle(ROOT / "oracle" / "entries.jsonl")
    assert count_errors(ROOT, entries) == []
    paths = [item["path"] for item in load_counts(ROOT)]
    assert len(paths) == 24
    assert {Path(path).parts[1] for path in paths} == {"saddle-search", "mlopt"}


def test_each_count_is_what_the_archived_scripts_state():
    oracle = {entry["path"]: entry["fields"] for entry in load_oracle(ROOT / "oracle" / "entries.jsonl")}
    for item in load_counts(ROOT):
        folder = _entry_dir(item["path"])
        n = item["n"]
        if (folder / "calc_benchmarks.py").is_file():
            tree = ast.parse((folder / "calc_benchmarks.py").read_text(encoding="utf-8"))
            lists = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.List)]
            assert [len(values) for values in lists] == [n] * 4, item["path"]
        elif (folder / "report.sh").is_file():
            assert "for i in {1..10}" in (folder / "run.sh").read_text(encoding="utf-8")
            assert "for d in run-?" in (folder / "report.sh").read_text(encoding="utf-8")
            fields = oracle[item["path"]]
            fits = [runs for runs in range(1, 11) if all(_whole_number_means(fields[key], runs) for key in ("unique_saddles", "total_searches"))]
            assert fits == [n], item["path"]
        elif item["counted"] == "repetitions":
            assert "get_data_for_web_benchmark.m" in (folder / "run.sh").read_text(encoding="utf-8")
            script = (ROOT / "softsaddle" / "upstream" / "test_run" / "get_data_for_web_benchmark.m").read_text(encoding="utf-8")
            assert re.search(rf"^n_repetitions\s*=\s*{n};", script, re.M)
            config = ROOT / "softsaddle" / "upstream" / "test_run" / "recorded" / folder.name / "iniConfig.m"
            assert re.search(rf"^n_repetition\s*=\s*{n};", config.read_text(encoding="utf-8"), re.M)
        else:
            assert f"python mlrf0_optb.py {n}\n" in (folder / "run.sh").read_text(encoding="utf-8")


def test_htst_takes_n_from_the_average_line_after_its_mean():
    assert _archive_rows("hTST/Al")["eon-dimer-cg2"]["n"] == 432
    assert _archive_rows("hTST/Al")["eon-dimer-cg"]["n"] == 639
    pt = {entry: row["n"] for entry, row in _archive_rows("hTST/Pt").items()}
    assert pt == {
        "alpha_dimer-qm": 1,
        "eon-dimer-cg": 13,
        "eon-dimer-cg2": 9,
        "eon-dimer-lbfgs": 13,
        "eon-dimer-lbfgs2": 9,
        "eon-lanczos-cg": 10,
        "eon-lanczos-cg2": 11,
        "eon-lanczos-lbfgs": 8,
        "eon-lanczos-lbfgs2": 6,
        "quench_dimer-qm": 10,
        "tsase-dimer-qm": 4,
    }
    text = "barrier_force_calls 2.107e+04 \naverage over 432 \nrate_force_calls  7.663e+05 \naverage over 11 \n"
    assert text_count(text, "barrier_force_calls") == 432
    assert text_count(text, "rate_force_calls") == 11
    assert text_count("force_calls 12\n", "force_calls") is None


def test_saddle_search_and_mlopt_rows_carry_the_stated_count():
    saddle = {entry: row["n"] for entry, row in _archive_rows("saddle-search/pt-island").items()}
    assert {n for entry, n in saddle.items() if entry.startswith("adaptive_HSphere")} == {10}
    assert {n for entry, n in saddle.items() if entry.startswith("eon")} == {9}
    for problem_id in ("mlopt/emt-pt38", "mlopt/C2H4"):
        assert {row["n"] for row in _archive_rows(problem_id).values()} == {50}
    problems = board.load_all(ROOT / "problems")
    assert board.stated_full_n(ROOT, problems["mlopt/emt-pt38"]) == 50
    assert board.stated_full_n(ROOT, problems["mlopt/C2H4"]) is None


def test_a_count_needs_an_entry_a_source_and_no_count_of_its_own(tmp_path: Path):
    (tmp_path / "oracle").mkdir()
    (tmp_path / "oracle" / "counts.toml").write_text(
        '[[count]]\npath = "a/benchmark.dat"\nn = 9\ncounted = "campaigns"\nsource = "calc_benchmarks.py"\n'
        '[[count]]\npath = "a/benchmark.dat"\nn = 0\ncounted = ""\nsource = "run.sh"\n'
        '[[count]]\npath = "b/benchmark.dat"\nn = 3\ncounted = "windows"\nsource = ""\n'
        '[[count]]\npath = "c/benchmark.dat"\nn = 4\ncounted = "windows"\nsource = "report.sh"\n',
        encoding="utf-8",
    )
    entries = [
        {"path": "a/benchmark.dat", "text": "force_calls 12\n"},
        {"path": "b/benchmark.dat", "text": "barrier_force_calls 7\naverage over 5\n"},
    ]
    assert count_errors(tmp_path, entries) == [
        "oracle/counts.toml a/benchmark.dat is counted twice",
        "oracle/counts.toml a/benchmark.dat n is not a positive whole number",
        "oracle/counts.toml a/benchmark.dat has no counted",
        "oracle/counts.toml b/benchmark.dat has no source",
        "oracle/counts.toml b/benchmark.dat also states its own count",
        "oracle/counts.toml c/benchmark.dat names no oracle entry",
    ]
    counts = load_counts(tmp_path)
    assert entry_count(entries[0], "force_calls", counts) == 9
    assert entry_count(entries[1], "barrier_force_calls", counts) == 5
