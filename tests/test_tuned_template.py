"""A tuned template is rejected wherever it sits, not only on the ranked table."""

from pathlib import Path

from optbench.board import omit_retired_tuned_track
from optbench.check import check, tuned_row_errors, tuned_template_errors


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_tuned_name_or_tuned_true_is_rejected_anywhere(tmp_path: Path, capsys):
    _write(
        tmp_path,
        "templates/ase/lj38-lbfgs.toml",
        'id = "ase/lj38-lbfgs"\nengine = "ase"\ntuning = "default"\n',
    )
    assert tuned_template_errors(tmp_path) == []

    _write(
        tmp_path,
        "templates/ase/lj38-lbfgs.toml",
        'id = "ase/lj38-lbfgs"\nengine = "ase"\ntuning = "tuned"\n',
    )
    _write(
        tmp_path,
        "templates/ase/lj38-lbfgs-tuned.toml",
        'id = "ase/lj38-lbfgs"\nengine = "ase"\n',
    )
    _write(
        tmp_path,
        "templates/kept/stock.toml",
        'id = "kept/stock-tuned"\nengine = "ase"\n',
    )
    _write(
        tmp_path,
        "templates/kept/flag.toml",
        'id = "kept/flag"\nengine = "ase"\n\n[optimizer]\ntuned = true\n',
    )
    assert tuned_template_errors(tmp_path) == [
        "tuned template templates/ase/lj38-lbfgs-tuned.toml",
        "tuned template templates/ase/lj38-lbfgs.toml",
        "tuned template templates/kept/flag.toml",
        "tuned template templates/kept/stock.toml",
    ]

    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "entries.jsonl").write_text("", encoding="utf-8")
    (oracle / "archive_gaps.txt").write_text("", encoding="utf-8")
    (tmp_path / "problems").mkdir()
    assert check(tmp_path / "problems", oracle / "entries.jsonl") == 1
    captured = capsys.readouterr()
    assert "tuned template templates/ase/lj38-lbfgs-tuned.toml" in captured.out
    assert "site tables match the board" not in captured.out


def test_tuned_rows_fail_on_every_list_and_a_retired_track_is_omitted():
    document = {
        "problems": {
            "minimization/lj38": {
                "ranked": [{"template": "rgmin/lj38-dogleg"}],
                "partial": [{"template": "ase/lj38-lbfgs-tuned"}],
                "history": [{"template": "rgmin/lj38-bfgs-tuned"}],
            }
        },
        "partial": [],
        "history": [],
        "not_comparable": [
            {"template": "sella/lj38-min", "tuned": True},
            {"template": "eon-client/lbfgs-m100", "tuning": "tuned"},
        ],
        "unparsed": [],
        "unmapped": [{"entry": "bench_rgmin.RgminLj38LbfgsTuned", "template": ""}],
    }
    assert tuned_row_errors(document) == [
        "tuned row minimization/lj38 partial ase/lj38-lbfgs-tuned",
        "tuned row minimization/lj38 history rgmin/lj38-bfgs-tuned",
        "tuned row not_comparable sella/lj38-min",
        "tuned row not_comparable eon-client/lbfgs-m100",
    ]
    assert omit_retired_tuned_track("RgminLj38LbfgsTuned", {}) is True
    assert omit_retired_tuned_track(
        "RgminLj38LbfgsTuned",
        {"template": "rgmin/lj38-lbfgs-tuned", "problem": "minimization/lj38"},
    ) is False
    assert omit_retired_tuned_track("RgminLj38Lbfgs", {}) is False
    assert omit_retired_tuned_track("Lj38EonLbfgsM100", {}) is True
    assert omit_retired_tuned_track("HeptamerRgsaddleCiNebLbfgs", {}) is True
    assert omit_retired_tuned_track(
        "RgsaddleLj38DimerLbfgs",
        {"template": "rgsaddle/dimer-lbfgs"},
    ) is False
    assert omit_retired_tuned_track("Lj38EonLbfgsM20", {}) is False
