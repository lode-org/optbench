"""Track checks read the parameter strings that asv 0.6.6 saved."""

import itertools
import json
from pathlib import Path

import pytest

import benchmarks.bench_lammpsneb as bench_lammpsneb
import benchmarks.bench_rgsaddle as bench_rgsaddle
from optbench.asv_score import decoded_params
from optbench.failures import FAILURES

ROOT = Path(__file__).resolve().parents[1]

SAVED = (
    (bench_rgsaddle, "19e5e5f2"),
    (bench_lammpsneb, "de7400eb"),
)


def _saved(commit: str) -> dict:
    paths = sorted((ROOT / "results" / "asv").glob(f"*/{commit}-*.json"))
    assert len(paths) == 1, paths
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _band_rows(document: dict, module) -> dict[str, dict[str, list]]:
    rows: dict[str, dict[str, list]] = {}
    for name, row in document["results"].items():
        parts = name.split(".")
        if parts[-1] in module.TRACKS and parts[-2] in module.CLASS_TEMPLATE:
            rows.setdefault(parts[-2], {})[parts[-1]] = row
    return rows


def _write_sidecars(document: dict, module) -> int:
    """One sidecar per saved start, at the archive name, holding the saved tracks."""
    columns = document["result_columns"]
    result, params = columns.index("result"), columns.index("params")
    written = 0
    for class_name, tracks in _band_rows(document, module).items():
        names = module.all_starts(module.CLASS_PROBLEM[class_name])
        stored = tracks["track_force_calls"][params]
        assert stored[0] == ["'0'"]
        assert stored[1] == [repr(name) for name in names[: len(stored[1])]]
        for index, start in enumerate(names[: len(stored[1])]):
            values = {method: row[result][index] for method, row in tracks.items()}
            failure = [name for name in FAILURES if values[f"track_failure_{name}"] == 1.0]
            record = {
                "force_calls": values["track_force_calls"],
                "success": values["track_success"] == 1.0,
                "failure": failure[0] if failure else None,
            }
            path = module.sidecar_path(module.CLASS_PROBLEM[class_name], module.CLASS_TEMPLATE[class_name], "0", start)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            written += 1
    return written


def test_decoded_params_undo_the_repr_quotes():
    assert decoded_params([["'0'"], ["'min01'", "'pair_00'"]]) == [["0"], ["min01", "pair_00"]]
    assert decoded_params([["1"], ["'0001'"]]) == [[1], ["0001"]]


@pytest.mark.parametrize(("module", "commit"), SAVED, ids=[module.__name__ for module, _commit in SAVED])
def test_verify_asv_tracks_accepts_a_saved_asv_file(tmp_path: Path, monkeypatch, module, commit):
    monkeypatch.setenv("OPTBENCH_RESULT_ROOT", str(tmp_path))
    document = _saved(commit)
    assert _write_sidecars(document, module) > 0
    module.verify_asv_tracks(document)
    class_name = sorted(_band_rows(document, module))[0]
    key = next(
        name for name in document["results"] if name.endswith(f".{class_name}.track_force_calls")
    )
    document["results"][key][0][0] += 1.0
    with pytest.raises(ValueError, match="the sidecar has"):
        module.verify_asv_tracks(document)


def test_attach_wall_seconds_finds_the_sidecar_of_a_saved_start(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPTBENCH_RESULT_ROOT", str(tmp_path))
    document = _saved("19e5e5f2")
    written = _write_sidecars(document, bench_rgsaddle)
    timed = {
        name: row
        for name, row in document["results"].items()
        if name.endswith(".time_search") and name.split(".")[-2] in bench_rgsaddle.CLASS_TEMPLATE
    }
    columns = document["result_columns"]
    expected = sum(
        len(list(itertools.product(*row[columns.index("params")]))) for row in timed.values()
    )
    assert expected == written
    updated = bench_rgsaddle.attach_wall_seconds(
        {"result_columns": columns, "results": timed}, tmp_path / "missing"
    )
    assert updated == written
