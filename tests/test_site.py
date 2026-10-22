import json
from pathlib import Path

from optbench.board import ranked_cells
from optbench.check import load_oracle
from optbench.problems import load_problem
from optbench.site import campaign_problem, problem_html, render
from optbench.site_check import check_site

ROOT = Path(__file__).resolve().parents[1]


def test_nav_pages_have_titles_and_the_suites_fold_under_archive():
    org = ROOT / "docs" / "orgmode"
    for name in ("index.org", "run.org", "archive.org", "codes.org", "board.org"):
        text = (org / name).read_text(encoding="utf-8")
        assert "title:nil" not in text
        assert text.startswith("#+TITLE:")
    index = (org / "index.org").read_text(encoding="utf-8")
    assert ":hidden:" in index
    assert "\n   board\n" in index
    assert "generated/suites/minimization" in index
    assert "generated/suites/neb" in index
    assert "generated/suites/accelerated-md" in index
    archive = (org / "archive.org").read_text(encoding="utf-8")
    assert "generated/suites/minimization" in archive
    assert "generated/suites/accelerated-md" in archive
    conf = (ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8")
    assert '"globaltoc_expand_depth": 0' in conf


def test_campaign_names_map_to_problems():
    assert campaign_problem("eon-client-morse-lbfgs") == "minimization/morse-bulk"
    assert campaign_problem("eon-3.5.0-neb-pt-lbfgs") == "neb/Pt-heptamer"
    assert campaign_problem("sella-2.6.0-ts-lj38") == "tsopt/LJ38"
    assert campaign_problem("eon-3.5.0-de-lj38-ci-neb") == "double-ended-saddle-search/lj38"
    assert campaign_problem("eon-3.5.0-saddle-pt-dimer") == "saddle-search/pt-island"
    assert campaign_problem("eon-3.5.0-saddle-pt-classic") == "saddle-search/pt-island"
    assert campaign_problem("rgsaddle-0.2.2-dimer-lbfgs-ts-lj38") == "tsopt/LJ38"
    assert campaign_problem("rgsaddle-0.2.2-dimer-lbfgs-ts-pt") == "tsopt/pt-island"
    assert campaign_problem("rgsaddle-0.2.2-mode-dimer-loweig") == "lowest-eigenvector/lj38"
    assert campaign_problem("rgsaddle-0.2.2-dimer-lbfgs-saddle-pt") == "saddle-search/pt-island"


def test_lj38_table_ranks_the_archived_scores():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    html = problem_html(problem, oracle)
    assert "eon-cg" in html
    assert ">453<" in html
    assert "held out of the original table" in html
    assert "0.01" in html
    assert "bench-plot-data" in html
    assert '"scale": "log"' in html


def test_rendered_pages_use_the_board_rank():
    render(ROOT)
    document = json.loads((ROOT / "results" / "board" / "index.json").read_text(encoding="utf-8"))
    lj38 = document["problems"]["minimization/lj38"]["ranked"][0]
    morse = document["problems"]["minimization/morse-bulk"]["ranked"][0]
    neb = document["problems"]["neb/Pt-heptamer"]["ranked"][0]
    cells = dict(ranked_cells(lj38, False))
    sota = (ROOT / "docs" / "include" / "sota.org").read_text(encoding="utf-8")
    assert f'data-board-table="headline" data-problem="minimization/lj38"' in sota
    assert 'data-field="current"' not in sota
    assert cells["template"] in sota
    assert f'data-field="par2_mean">{cells["par2_mean"]}<' in sota
    assert "Present sources. The cell is the record." not in sota
    assert "rgmin-lbfgs-0.3.0" not in sota
    assert "eon-3.5.0-lbfgs-m100" not in sota
    collection = (ROOT / "docs" / "include" / "collection.org").read_text(encoding="utf-8")
    assert "Morse bulk" in collection
    assert morse["template"] in collection
    assert "No present run is recorded." not in collection
    assert "No committed harness run." in collection
    minimization = (ROOT / "docs" / "orgmode" / "generated" / "suites" / "minimization.org").read_text(encoding="utf-8")
    assert f'data-board-table="ranked" data-problem="minimization/lj38"' in minimization
    assert f'data-field="par2_mean">{cells["par2_mean"]}<' in minimization
    assert 'data-field="current"' not in minimization
    assert "<th>Current</th>" not in minimization
    assert 'data-field="wilson_low"' not in minimization
    assert 'data-field="wilson_high"' not in minimization
    assert f'data-field="wilson">{cells["wilson"]}<' in minimization
    assert "<th>Wilson</th>" in minimization
    assert lj38["current"] is True
    assert "minimization-lj38-profile.svg" in minimization
    assert "minimization-lj38-profile.webp" in minimization
    assert 'loading="lazy"' in minimization
    assert 'data-board-table="summary"' in minimization
    assert f'data-par2="{cells["par2_mean"]}"' in minimization
    assert "<details" in minimization
    assert "Legacy 2014 archive" in minimization
    assert "eon-cg" in minimization
    assert "par2-wilson-bootstrap" in minimization
    assert "Present runs. Fewer force calls first." not in minimization
    neb_page = (ROOT / "docs" / "orgmode" / "generated" / "suites" / "neb.org").read_text(encoding="utf-8")
    assert neb["template"] in neb_page
    assert "not yet on the harness counter" in neb_page
    assert "Legacy 2014 scores. Fewer calls is to the left." in neb_page
    tsopt = (ROOT / "docs" / "orgmode" / "generated" / "suites" / "tsopt.org").read_text(encoding="utf-8")
    assert "Sella 0.1.1 legacy" in tsopt
    assert "2.6.0 legacy" not in tsopt
    assert "tsopt-LJ38-profile.svg" in tsopt
    saddle = (ROOT / "docs" / "orgmode" / "generated" / "suites" / "saddle-search.org").read_text(encoding="utf-8")
    assert 'data-board-table="ranked" data-problem="saddle-search/pt-island"' in saddle
    assert "rgsaddle/saddle-dimer-rotskip-tuned" in saddle
    assert 'data-field="par2_mean">10127.1<' in saddle
    assert "No full harness row." not in saddle
    assert "eon-3.5.0-saddle-pt-dimer" in saddle
    assert "not yet on the harness counter" in saddle
    htst_ranked = {row["template"] for row in document["problems"]["hTST/Pt"]["ranked"]}
    assert "eon-client/htst-pt-dimer-lbfgs" in htst_ranked
    assert "eon-client/htst-pt-lanczos-lbfgs" in htst_ranked
    htst_page = (ROOT / "docs" / "orgmode" / "generated" / "suites" / "hTST.org").read_text(encoding="utf-8")
    assert "eon-client/htst-pt-dimer-lbfgs" in htst_page
    assert "eon-client/htst-pt-lanczos-lbfgs" in htst_page
    assert "Per-search cost" not in saddle
    assert "noise/rgmin-dogleg-s0" in minimization
    assert 'data-field="par2_mean">81.881<' in minimization
    board = (ROOT / "docs" / "include" / "v2-board.org").read_text(encoding="utf-8")
    assert 'data-board-table="summary"' in board
    assert f'data-par2="{cells["par2_mean"]}"' in board
    assert "min-width: 40rem" not in (ROOT / "docs" / "source" / "_static" / "optbench.css").read_text(encoding="utf-8")
    codes = (ROOT / "docs" / "include" / "codes.org").read_text(encoding="utf-8")
    assert "Rohit Goswami" in codes
    assert "4d8b34baf1ba" in codes
    assert "1d71c9445018" in codes
    assert "f15726f1c798" in codes
    assert "c0db2b8d333c" in codes
    assert 'data-board-table="code-best"' in codes
    assert not (ROOT / "docs" / "include" / "featured.org").exists()
    assert not (ROOT / "docs" / "include" / "terra-pass.org").exists()
    for problem in document["problems"].values():
        if problem.get("ranked") and not problem.get("profile"):
            suite = problem["id"].split("/")[0]
            page = (ROOT / "docs" / "orgmode" / "generated" / "suites" / f"{suite}.org").read_text(encoding="utf-8")
            assert problem["profile_gap"]
            assert problem["profile_gap"] in page
    assert check_site(ROOT) == []


def test_a_stale_summary_number_fails_the_check():
    render(ROOT)
    document = json.loads((ROOT / "results" / "board" / "index.json").read_text(encoding="utf-8"))
    top = document["problems"]["minimization/lj38"]["ranked"][0]
    par2 = dict(ranked_cells(top, False))["par2_mean"]
    path = ROOT / "docs" / "include" / "v2-board.org"
    original = path.read_text(encoding="utf-8")
    needle = f'data-par2="{par2}"'
    assert needle in original
    path.write_text(original.replace(needle, 'data-par2="1"', 1), encoding="utf-8")
    try:
        errors = check_site(ROOT)
        assert errors
        assert any("par2" in line and "minimization/lj38" in line for line in errors)
    finally:
        path.write_text(original, encoding="utf-8")


def test_a_stale_ranked_number_fails_the_check():
    render(ROOT)
    document = json.loads((ROOT / "results" / "board" / "index.json").read_text(encoding="utf-8"))
    top = document["problems"]["minimization/lj38"]["ranked"][0]
    par2 = dict(ranked_cells(top, False))["par2_mean"]
    path = ROOT / "docs" / "orgmode" / "generated" / "suites" / "minimization.org"
    original = path.read_text(encoding="utf-8")
    needle = f'data-field="par2_mean">{par2}<'
    assert needle in original
    path.write_text(original.replace(needle, 'data-field="par2_mean">1<', 1), encoding="utf-8")
    try:
        errors = check_site(ROOT)
        assert errors
        assert any("par2_mean" in line and "minimization/lj38" in line for line in errors)
    finally:
        path.write_text(original, encoding="utf-8")
