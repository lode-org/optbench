"""The geomeTRIC asv writer stores its machine record with the result."""

import json
from pathlib import Path

from optbench.asv_machine import machine_record
from optbench.geometric_asv import write_results
from optbench.v2 import HOST_LABEL_ENV


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
