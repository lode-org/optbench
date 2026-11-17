"""The command line: what each command prints, writes, and refuses.

A command that succeeds ends the process with status 0. A refusal ends it
with a message, and the message is what the person at the terminal reads, so
the messages are compared as written.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from optbench.__main__ import _eon_client, _head, main
from optbench.campaign import list_templates
from optbench.dat import parse

pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.invalid",
    "GIT_COMMITTER_NAME": "Test Author",
    "GIT_COMMITTER_EMAIL": "author@example.invalid",
}


def _run(argv: list[str]):
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    return stopped.value.code


def _repo(path: Path) -> str:
    path.mkdir()
    for args in (["init", "-q", "-b", "main"], ["commit", "-q", "--allow-empty", "-m", "start"]):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, env=GIT_ENV)
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, env=GIT_ENV).strip()


def test_templates_prints_one_id_per_line_in_the_order_they_are_listed(capsys):
    assert _run(["templates", "--root", str(ROOT)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == [template["id"] for template in list_templates(ROOT)]
    assert "ase/lj38-bfgs-default" in printed


@pytest.mark.parametrize(
    ("argv", "output"),
    [
        (["run", "no/such-problem", "no/such-template"], ["--output", "bench.dat"]),
        (["campaign-ase"], ["--output-dir", "campaign"]),
    ],
)
def test_a_negative_limit_is_refused_before_anything_is_looked_up_or_written(argv, output, tmp_path):
    destination = [output[0], str(tmp_path / output[1])]
    code = _run([*argv, "--root", str(tmp_path), "--limit=-1", *destination])
    assert code == "limit must be zero or positive"
    assert list(tmp_path.iterdir()) == []


def test_run_prints_one_summary_line_and_writes_the_dat_and_its_starts(tmp_path, capsys):
    output = tmp_path / "bench.dat"
    code = _run(
        [
            "run",
            "minimization/lj38",
            "ase/lj38-bfgs-default",
            "--root",
            str(ROOT),
            "--limit",
            "1",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    shown = re.fullmatch(
        r"minimization/lj38 ase/lj38-bfgs-default force_calls=(?P<calls>[0-9.]+) nfailed=(?P<failed>\d+) starts=1",
        line,
    )
    assert shown, line
    fields = parse(output.read_text(encoding="utf-8"))
    assert float(fields["force_calls"]) == float(shown["calls"])
    assert fields["template"] == "ase/lj38-bfgs-default"
    assert fields["problem"] == "minimization/lj38"
    assert len(output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_template_that_runs_a_client_needs_one_and_says_how_to_give_it(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("EONCLIENT", raising=False)
    code = _run(
        ["run", "minimization/lj38", "eon-client/lbfgs-m20", "--root", str(ROOT), "--output", str(tmp_path / "bench.dat")]
    )
    assert code == "pass --client, set EONCLIENT, or put eonclient on PATH"


def test_the_client_is_the_one_asked_for_then_the_one_on_the_path_then_the_environment_one(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    on_path = bin_dir / "eonclient"
    on_path.write_text("#!/bin/sh\n", encoding="utf-8")
    on_path.chmod(0o755)
    named = tmp_path / "elsewhere" / "eonclient"

    monkeypatch.delenv("EONCLIENT", raising=False)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert _eon_client(None) == on_path
    assert _eon_client(named) == named

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("EONCLIENT", str(named))
    assert _eon_client(None) == named
    assert _eon_client(on_path) == on_path

    monkeypatch.delenv("EONCLIENT")
    with pytest.raises(SystemExit, match="pass --client, set EONCLIENT, or put eonclient on PATH"):
        _eon_client(None)


def _starts(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "\n", encoding="utf-8")
    return path


def test_score_reduces_a_starts_file_to_the_v2_summary_and_prints_the_headline(tmp_path, capsys):
    starts = _starts(
        tmp_path / "starts.jsonl",
        [
            {"start": "a", "force_calls": 10, "success": True},
            {"start": "b", "force_calls": 20, "success": True},
            {"start": "c", "force_calls": 30, "success": True},
            {"start": "d", "force_calls": 100, "success": False, "failure": "cap"},
        ],
    )
    output = tmp_path / "score.json"
    assert _run(["score", str(starts), "--cap", "100", "--output", str(output)]) == 0
    assert capsys.readouterr().out == "successes=3/4 par2_mean=65.0 median=25.0\n"
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert (summary["n"], summary["successes"], summary["nfailed"], summary["n_cap"]) == (4, 3, 1, 1)
    assert (summary["par2_mean"], summary["median"], summary["success_only_mean"]) == (65.0, 25.0, 20.0)
    assert summary["success_rate"] == 0.75
    assert summary["cap"] == 100


def test_score_will_not_charge_a_failure_without_a_cap(tmp_path):
    starts = _starts(tmp_path / "starts.jsonl", [{"start": "d", "force_calls": 100, "success": False, "failure": "cap"}])
    with pytest.raises(ValueError, match="a failure needs a cap"):
        main(["score", str(starts), "--output", str(tmp_path / "score.json")])
    assert not (tmp_path / "score.json").exists()


def test_quadratic_writes_the_v2_sample_under_the_commit_it_is_given(tmp_path, capsys):
    output = tmp_path / "quadratic.json"
    assert _run(["quadratic", "--output", str(output), "--root", str(tmp_path), "--commit", "abc"]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert capsys.readouterr().out == f"wrote {output} runs={len(document['runs'])}\n"
    assert document["schema"] == "optbench.result/v2"
    assert document["problem"] == "quadratic/bowl"
    assert len(document["runs"]) == 5
    assert {run["commit"] for run in document["runs"]} == {"abc"}


def test_quadratic_takes_the_commit_from_the_checkout_when_none_is_given(tmp_path):
    head = _repo(tmp_path / "checkout")
    output = tmp_path / "quadratic.json"
    assert _run(["quadratic", "--output", str(output), "--root", str(tmp_path / "checkout")]) == 0
    assert {run["commit"] for run in json.loads(output.read_text(encoding="utf-8"))["runs"]} == {head}


def test_a_directory_that_is_not_a_checkout_has_no_head(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert _head(tmp_path) == ""
    assert _head(tmp_path / "missing") == ""
    head = _repo(tmp_path / "checkout")
    assert _head(tmp_path / "checkout") == head
    assert len(head) == 40


def test_statistics_scores_the_committed_asv_tracks_and_checks_the_commit_it_is_asked_about(tmp_path, capsys):
    output = tmp_path / "statistics.json"
    base = ["statistics", "--root", str(ROOT), "--output", str(output)]
    assert _run(base) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    recorded = document["provenance"]["commit"]
    displacement = document["displacement"]["metrics"]
    dimer = document["dimer"]["metrics"]
    assert capsys.readouterr().out == (
        f"wrote {output} commit={recorded} displacement_median={displacement['median']} "
        f"dimer_median={dimer['median']} "
        f"displacement_failed={displacement['nfailed']} dimer_failed={dimer['nfailed']}\n"
    )
    assert _run([*base, "--commit", recorded[:8]]) == 0
    assert _run([*base, "--commit", "0" * 8]) == f"asv results are commit {recorded}, not {'0' * 8}"


def test_sitesize_says_when_there_is_no_build_and_when_the_build_has_no_pages(tmp_path, capsys):
    assert _run(["sitesize", "--root", str(tmp_path)]) == 2
    assert capsys.readouterr().out == f"site build is missing: {tmp_path / 'docs' / 'build'}\n"
    (tmp_path / "docs" / "build").mkdir(parents=True)
    assert _run(["sitesize", "--root", str(tmp_path)]) == 2
    assert capsys.readouterr().out == f"no html pages in {tmp_path / 'docs' / 'build'}\n"


def test_sitesize_reads_a_relative_build_under_the_root_and_an_absolute_one_where_it_is(tmp_path, capsys):
    page = "<html><body><p>one page</p></body></html>"
    under_root = tmp_path / "root" / "out"
    under_root.mkdir(parents=True)
    (under_root / "index.html").write_text(page, encoding="utf-8")
    assert _run(["sitesize", "--root", str(tmp_path / "root"), "--build", "out"]) == 0
    assert "site weight within budget" in capsys.readouterr().out

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "index.html").write_text("<html><body>" + "<p>x</p>" * 6000 + "</body></html>", encoding="utf-8")
    assert _run(["sitesize", "--root", str(tmp_path / "root"), "--build", str(elsewhere)]) == 1
    assert "site weight exceeds the budget" in capsys.readouterr().out


def test_rendering_thumbnails_with_no_saved_board_stops_with_the_file_it_needs(tmp_path):
    code = _run(["board", "render", "--root", str(tmp_path)])
    assert isinstance(code, str)
    assert str(tmp_path / "results" / "board" / "index.json") in code
