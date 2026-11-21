"""The comparison report is the ledger renderer, not a hand-written table."""

import importlib.util
import json
from pathlib import Path

from optbench.catalog_report import markdown, rows

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_report_is_the_ledger_renderer():
    path = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-vs-179e90f2-report.md"
    text = path.read_text(encoding="utf-8")
    assert text == markdown(ROOT)
    assert "1 core and 1 thread" in text
    assert "27/18171" in text
    assert "eon-classic" not in text
    assert "7ec1840" not in text
    assert "withheld" not in text.lower()
    payload = json.loads(
        (ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-vs-179e90f2-report.json").read_text(encoding="utf-8")
    )
    assert payload["rows"] == rows(ROOT)
    assert "including the 14 initial-minimisation calls" in text
    assert "179e90f2 lanczos stopped at 24 connected saddles, below the goal of 27." in text
    assert "aa6dd937 lanczos reached the goal of 27 connected saddles." in text


def test_the_board_index_reads_the_library_sidecar():
    sidecar_path = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-pt-island-lib-sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    board = json.loads((ROOT / "results" / "board" / "index.json").read_text(encoding="utf-8"))
    assert board["rgsaddle_lib_sha256"] == sidecar["rgsaddle_lib_sha256"]
    from optbench.board import _library_sha256

    assert _library_sha256(ROOT) == sidecar["rgsaddle_lib_sha256"]
    assert _library_sha256(ROOT / "missing") is None


def test_the_catalog_csv_matches_csv_text():
    path = ROOT / "results" / "campaign" / "rgsaddle-develop-aa6dd937-pt-island.csv"
    committed = path.read_bytes()
    script = ROOT / "results" / "campaign" / "rgsaddle_develop_catalog.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_develop_catalog_csv", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.csv_text().encode("utf-8") == committed
    assert path.read_bytes() == committed
    text = committed.decode("utf-8")
    assert "7ec1840" not in text
    assert "eon-classic" not in text
    assert "dimer-fire" in text
    assert "lanczos" in text
    assert "de80b814c908c4fa3b3ea9f76b7e4115d323824eb4081221b258c22ed148c1a0" in text


def test_each_aa6dd937_ledger_names_the_library_sha():
    from optbench.board import _library_hash_bound

    directory = ROOT / "results" / "campaign"
    sidecar = json.loads(
        (directory / "rgsaddle-develop-aa6dd937-pt-island-lib-sha256.json").read_text(encoding="utf-8")
    )
    commands = json.loads(
        (directory / "rgsaddle-develop-aa6dd937-pt-island-commands.json").read_text(encoding="utf-8")
    )
    sha = sidecar["rgsaddle_lib_sha256"]
    for method in ("dimer-fire", "dimer-lbfgs", "dimer-rotskip", "lanczos"):
        stem = f"rgsaddle-develop-aa6dd937-saddle-{method}"
        fields = {}
        for line in (directory / f"{stem}.dat").read_text(encoding="utf-8").splitlines():
            if line.startswith("rgsaddle_lib_sha256 "):
                fields["rgsaddle_lib_sha256"] = line.split(" ", 1)[1]
        assert commands[method]["rgsaddle_lib_sha256"] == sha
        assert fields["rgsaddle_lib_sha256"] == sha
        assert _library_hash_bound(directory, stem, fields) is True
    assert _library_hash_bound(directory, "rgsaddle-develop-aa6dd937-saddle-lanczos", {}) is False
