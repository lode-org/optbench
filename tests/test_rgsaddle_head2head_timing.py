from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_head2head_module():
    path = Path(__file__).resolve().parents[1] / "results" / "campaign" / "rgsaddle_head2head.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_head2head", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timing_resume_keeps_host_and_false_wall_flag(tmp_path, monkeypatch):
    head2head = _load_head2head_module()
    monkeypatch.setattr(head2head, "ROOT", tmp_path)
    monkeypatch.setattr(head2head.socket, "gethostname", lambda: "host-a")
    method = "lanczos"
    arm = "old"
    timing_path = tmp_path / "build" / "rgsaddle-h2h" / method / arm / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text("[]\n", encoding="utf-8")

    started = head2head._timing_start(method, arm, {"wall_comparable": False})
    head2head._timing_finish(method, arm, started, {"wall_comparable": True})
    first = json.loads(timing_path.read_text(encoding="utf-8"))
    assert first["host"] == "host-a"
    assert first["wall_comparable"] is False

    monkeypatch.setattr(head2head.socket, "gethostname", lambda: "host-b")
    started = head2head._timing_start(method, arm, {"wall_comparable": True})
    head2head._timing_finish(method, arm, started, {"wall_comparable": True})
    second = json.loads(timing_path.read_text(encoding="utf-8"))
    assert second["host"] == "host-a"
    assert second["wall_comparable"] is False


def test_run_eon_does_not_add_outer_timing_segments(tmp_path, monkeypatch):
    head2head = _load_head2head_module()
    monkeypatch.setattr(head2head, "ROOT", tmp_path)

    class FakeCatalog:
        def __init__(self):
            self._output_path = lambda _method: tmp_path / "unused.dat"
            self._checkpoint_dir = lambda _method: tmp_path / "unused"
            self.PROVENANCE_PATH = tmp_path / "unused-provenance.json"

        def run_eon(self, _cpu, _client, _provenance_facts=None):
            checkpoint = self._checkpoint_dir("eon-classic")
            checkpoint.mkdir(parents=True, exist_ok=True)
            timing = {
                "method": "eon-classic",
                "wall_comparable": True,
                "wall_seconds_total": 12.5,
            }
            (checkpoint / "timing.json").write_text(json.dumps(timing, sort_keys=True) + "\n", encoding="utf-8")
            segments = checkpoint / "timing.segments.jsonl"
            segments.write_text(
                json.dumps({"segment_started_unix": 1.0}, sort_keys=True)
                + "\n"
                + json.dumps({"segment_finished_unix": 2.0, "segment_wall_seconds": 1.0}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

    fake = FakeCatalog()
    monkeypatch.setattr(head2head, "catalog", fake)
    client = tmp_path / "eonclient"
    client.write_text("", encoding="utf-8")
    monkeypatch.setattr(head2head, "_sha256", lambda _path: "abc")

    head2head.run_eon("old", 0, client)

    checkpoint = tmp_path / "build" / "rgsaddle-h2h" / "eon-classic" / "old"
    segments = [json.loads(line) for line in (checkpoint / "timing.segments.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(segments) == 2
    timing = json.loads((checkpoint / "timing.json").read_text(encoding="utf-8"))
    assert timing["wall_comparable"] is False
