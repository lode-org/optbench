from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path


def _load_catalog_module():
    path = Path(__file__).resolve().parents[1] / "results" / "campaign" / "rgsaddle_develop_catalog.py"
    spec = importlib.util.spec_from_file_location("rgsaddle_develop_catalog", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timing_segments_accumulate_across_resumes(tmp_path, monkeypatch):
    catalog = _load_catalog_module()
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    method = "timing-test"

    started_1 = catalog._timing_start(method, extra={"method": method})
    time.sleep(0.01)
    catalog._timing_finish(method, started_1, extra={"method": method})

    timing_path = tmp_path / "build" / "rgsaddle" / method / "timing.json"
    first = json.loads(timing_path.read_text(encoding="utf-8"))
    first_started = float(first["started_unix"])
    first_total = float(first["wall_seconds_total"])
    assert first_total > 0.0

    started_2 = catalog._timing_start(method, extra={"method": method})
    time.sleep(0.01)
    catalog._timing_finish(method, started_2, extra={"method": method})

    second = json.loads(timing_path.read_text(encoding="utf-8"))
    second_total = float(second["wall_seconds_total"])
    assert float(second["started_unix"]) == first_started
    assert second_total > first_total

    segments_path = tmp_path / "build" / "rgsaddle" / method / "timing.segments.jsonl"
    segments = [json.loads(line) for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(segments) == 4
    assert "segment_started_unix" in segments[0]
    assert "segment_finished_unix" in segments[1]
