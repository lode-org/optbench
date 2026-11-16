"""A later asv writer must keep tracks already stored for the same commit."""

import json
from pathlib import Path

from optbench.sella_loweig import write_asv
from optbench.v2 import HOST_LABEL_ENV


def _stored(root: Path) -> dict:
    path = next(item for item in (root / "results" / "asv").rglob("*.json") if item.name != "machine.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_sella_keeps_tracks_already_in_the_result_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    monkeypatch.setattr("optbench.sella_loweig.ROOT", tmp_path)
    from optbench.asv_machine import machine_dir

    folder = machine_dir(tmp_path / "results" / "asv")
    commit = "abc1234567890abc1234567890abc1234567890a"
    dest = folder / f"{commit[:8]}-uv-pip-py3.12.json"
    dest.write_text(
        json.dumps(
            {
                "commit_hash": commit,
                "env_name": "uv-pip-py3.12",
                "date": 1,
                "params": {"machine": "workstation-a"},
                "python": "3.12",
                "requirements": {},
                "env_vars": {},
                "result_columns": ["result", "params", "version", "started_at", "duration"],
                "results": {"bench_geometric.Kept.track_force_calls": [[1.0], [["'0'"]], "v", 1, 0.2]},
                "durations": {},
                "version": 2,
            }
        ),
        encoding="utf-8",
    )
    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    row = {"start": "0000", "force_calls": 4, "success": True, "failure": None, "seconds": 1.5}
    (campaign / "sella-0.1.1-lj38-loweig.starts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    write_asv(commit)
    stored = _stored(tmp_path)
    assert "bench_geometric.Kept.track_force_calls" in stored["results"]
    assert "bench_sella_loweig.Lj38Loweig.track_force_calls" in stored["results"]
