"""The geomeTRIC asv writer stores its machine record with the result."""

import json
from pathlib import Path

import pytest

from optbench.asv_machine import machine_record
from optbench.geometric_asv import write_results
from optbench.noise_asv import write_results as write_noise
from optbench.v2 import HOST_LABEL_ENV


def test_a_legacy_crashed_geometric_zero_is_unmeasured():
    from optbench.geometric_asv import _seconds

    assert _seconds({"start": "0", "success": True, "seconds": 0.0}) == 0.0
    assert (
        _seconds({"start": "1", "success": False, "failure": "crash", "force_calls": 0, "seconds": 0.0})
        is None
    )


def _campaign(root: Path, name: str, template: str, count: int) -> None:
    campaign = root / "results" / "campaign"
    campaign.mkdir(parents=True, exist_ok=True)
    record = {"force_calls": 4, "success": True, "seconds": 0.01}
    lines = "\n".join(json.dumps(record) for _ in range(count)) + "\n"
    (campaign / f"{name}.starts.jsonl").write_text(lines, encoding="utf-8")
    (campaign / f"{name}.dat").write_text(f"template {template}\n", encoding="utf-8")


def test_geometric_results_write_a_machine_record(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    root = tmp_path
    _campaign(root, "geometric-morse-cart", "geometric/morse-cart", 100)
    _campaign(root, "geometric-morse-tric", "geometric/morse-tric", 100)
    _campaign(root, "geometric-lj38-cart", "geometric/lj38-cart", 1000)
    _campaign(root, "geometric-lj38-tric", "geometric/lj38-tric", 1000)
    for coords in ("cart", "tric"):
        for sigma in ("s0", "s1em2", "s1em3", "s1em4"):
            _campaign(
                root,
                f"noise-geometric-{coords}-{sigma}",
                f"noise/geometric-{coords}-{sigma}",
                1000,
            )
    written = write_results(root, "abc1234567890abc1234567890abc1234567890a", started_at=1)
    machine = json.loads((written.parent / "machine.json").read_text(encoding="utf-8"))
    record = {**machine_record("workstation-a"), "version": 1}
    assert machine == record
    stored = json.loads(written.read_text(encoding="utf-8"))
    assert stored["params"]["machine"] == "workstation-a"
    assert written.parent.name == "workstation-a"


def test_a_missing_duration_is_not_written_as_zero(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    root = tmp_path
    _campaign(root, "geometric-morse-cart", "geometric/morse-cart", 100)
    _campaign(root, "geometric-morse-tric", "geometric/morse-tric", 100)
    _campaign(root, "geometric-lj38-cart", "geometric/lj38-cart", 1000)
    _campaign(root, "geometric-lj38-tric", "geometric/lj38-tric", 1000)
    for coords in ("cart", "tric"):
        for sigma in ("s0", "s1em2", "s1em3", "s1em4"):
            _campaign(
                root,
                f"noise-geometric-{coords}-{sigma}",
                f"noise/geometric-{coords}-{sigma}",
                1000,
            )
    path = root / "results" / "campaign" / "geometric-lj38-cart.starts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["seconds"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="no measured seconds"):
        write_results(root, "abc1234567890abc1234567890abc1234567890a", started_at=1)


def _noise_rows(campaign: Path, stem: str, template: str) -> None:
    record = {"force_calls": 4, "success": True, "failure": None, "seconds": 0.2}
    lines = "\n".join(json.dumps(record) for _ in range(1000)) + "\n"
    (campaign / f"{stem}.starts.jsonl").write_text(lines, encoding="utf-8")
    (campaign / f"{stem}.dat").write_text(f"template {template}\n", encoding="utf-8")


def test_a_geometric_noise_campaign_is_not_a_noise_track(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    monkeypatch.setattr(
        "optbench.noise_asv.specs",
        lambda: [{"id": "noise/rgmin-fire-s0"}],
    )
    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    _noise_rows(campaign, "noise-rgmin-fire-s0", "noise/rgmin-fire-s0")
    _noise_rows(campaign, "noise-geometric-cart-s0", "noise/geometric-cart-s0")
    written = write_noise(tmp_path, "abc1234567890abc1234567890abc1234567890a", started_at=1)
    stored = json.loads(written.read_text(encoding="utf-8"))
    keys = list(stored["results"])
    assert any(key.startswith("bench_noise.NoiseRgminFireS0.") for key in keys)
    assert not any("Geometric" in key for key in keys)


def test_a_noise_result_keeps_geometric_tracks(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(HOST_LABEL_ENV, "workstation-a")
    commit = "abc1234567890abc1234567890abc1234567890a"
    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    _campaign(tmp_path, "geometric-morse-cart", "geometric/morse-cart", 100)
    _campaign(tmp_path, "geometric-morse-tric", "geometric/morse-tric", 100)
    _campaign(tmp_path, "geometric-lj38-cart", "geometric/lj38-cart", 1000)
    _campaign(tmp_path, "geometric-lj38-tric", "geometric/lj38-tric", 1000)
    for coords in ("cart", "tric"):
        for sigma in ("s0", "s1em2", "s1em3", "s1em4"):
            _campaign(
                tmp_path,
                f"noise-geometric-{coords}-{sigma}",
                f"noise/geometric-{coords}-{sigma}",
                1000,
            )
    written = write_results(tmp_path, commit, started_at=1)
    # The geometric noise files belong to the geometric writer. Hide them
    # so this check is the shared result file, not the noise glob.
    for path in campaign.glob("noise-geometric-*"):
        path.unlink()
    monkeypatch.setattr("optbench.noise_asv.specs", lambda: [{"id": "noise/rgmin-fire-s0"}])
    _noise_rows(campaign, "noise-rgmin-fire-s0", "noise/rgmin-fire-s0")
    write_noise(tmp_path, commit, started_at=2)
    stored = json.loads(written.read_text(encoding="utf-8"))
    keys = list(stored["results"])
    assert any(key.startswith("bench_geometric.") for key in keys)
    assert any(key.startswith("bench_noise.NoiseRgminFireS0.") for key in keys)
