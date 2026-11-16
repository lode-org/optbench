"""Behaviour of the uncovered writers on the committed problems."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_board_command_prints_the_pt_heptamer_leader(tmp_path: Path, capsys) -> None:
    from optbench.__main__ import board

    board(root=ROOT, output=tmp_path / "board")
    text = capsys.readouterr().out
    assert "neb/Pt-heptamer" in text
    assert "par2_mean=318.9230769230769" in text
    assert "success_rate=1.0" in text
    page = (tmp_path / "board" / "index.json").read_text(encoding="utf-8")
    assert '"ranked_count"' in page
    assert "neb/Pt-heptamer" in page


def test_render_writes_thumbnails_and_stops_without_a_board(tmp_path: Path, capsys) -> None:
    from optbench.__main__ import render

    render(root=ROOT, output=tmp_path / "plots")
    text = capsys.readouterr().out
    written = [path for path in (tmp_path / "plots").iterdir() if path.is_file()]
    png = [path for path in written if path.suffix == ".png"]
    webp = [path for path in written if path.suffix == ".webp"]
    assert png and webp
    assert "neb-Pt-heptamer-cost.png" in {path.name for path in png}
    assert f"wrote {len(png) + len(webp)} thumbnails" in text
    with pytest.raises(SystemExit, match="index.json"):
        render(root=tmp_path, output=tmp_path / "missing")


def test_metric_labels_are_the_names_the_site_prints() -> None:
    from optbench.site_pages import metric_label

    assert metric_label("force_calls_median") == "Median force calls"
    assert metric_label(" mean_energy_rmse ") == "Mean energy RMSE"
    assert metric_label("custom_rmse_score") == "Custom RMSE Score"
    assert metric_label("rmse") == "RMSE"
    assert metric_label("") == ""
    assert metric_label("___") == "___"


def test_tuning_label_reads_the_eon_memory_and_an_explicit_token() -> None:
    from optbench.v2 import tuning_from_template, tuning_label

    assert tuning_from_template(ROOT, "eon-svn/de-2ts-neb") == "default"
    assert tuning_from_template(ROOT, "gpridge/pt-heptamer-ci-neb") == "adjusted"
    assert tuning_from_template(ROOT, "eon/develop-de-2ts") == ""
    assert tuning_from_template(ROOT, "") == ""
    assert tuning_from_template(ROOT, "no/such-template") == ""
    assert tuning_label({"tuning": "tuned"}) == "tuned"
    assert tuning_label({"code": "eOn", "algorithm": "LBFGS", "method": {"lbfgs_memory": 20}}) == "default"
    assert tuning_label({"code": "Eon", "algorithm": "LBFGS", "method": {"lbfgs_memory": 100}}) == "adjusted"
    assert tuning_label({"code": "eOn", "algorithm": "LBFGS", "method": {"lbfgs_memory": 50}}) == "unlabelled"
    assert tuning_label({"code": "ASE", "algorithm": "LBFGS", "method": {"lbfgs_memory": 20}}) == "unlabelled"
    assert tuning_label({"code": "eOn", "algorithm": "FIRE", "method": {"lbfgs_memory": 100}}) == "unlabelled"
    assert tuning_label({}) == "unlabelled"


def test_track_files_keep_saved_commits_and_skip_a_bad_file(tmp_path: Path) -> None:
    from optbench.asv_score import iter_track_files

    saved = list(iter_track_files(ROOT / "results" / "asv"))
    assert saved
    assert any(item.commit_hash.startswith("de7ce86b") for item in saved)
    assert any(
        "track_force_calls" in key for item in saved for key in item.get_all_result_keys()
    )
    machine = tmp_path / "bench"
    machine.mkdir()
    (machine / "machine.json").write_text(
        '{"version": 1, "machine": "bench", "arch": "x86_64", "cpu": "test", '
        '"num_cpu": "1", "os": "test", "ram": "1"}\n',
        encoding="utf-8",
    )
    (machine / "good-env.json").write_text(
        '{"version": 2, "commit_hash": "abc123", "params": {"machine": "bench"}, '
        '"result_columns": ["result", "params", "version"], '
        '"results": {"Bench.track_force_calls": [1.0, ["s"], 1]}}\n',
        encoding="utf-8",
    )
    (machine / "bad-env.json").write_text('{"version": 1}\n', encoding="utf-8")
    found = list(iter_track_files(tmp_path))
    assert [item.commit_hash for item in found] == ["abc123"]
