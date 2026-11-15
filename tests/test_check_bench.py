"""The check timing samples read the checkout they were given."""

import os
from pathlib import Path

from benchmarks.bench_check import (
    campaign_json_paths,
    checkout,
    ensure_checkout,
    load_campaign_json,
    parse_site,
)

# The benchmark file lives in this repo, so the default checkout is this repo.
from benchmarks import bench_check


def test_check_timing_reads_the_checkout(monkeypatch) -> None:
    root = bench_check.checkout()
    assert root == Path(bench_check.__file__).resolve().parents[1]
    assert (root / "oracle" / "entries.jsonl").is_file()
    assert (root / "results" / "campaign").is_dir()
    assert (root / "results" / "asv").is_dir()
    monkeypatch.setenv("OPTBENCH_CHECKOUT", str(root))
    assert checkout() == root
    bound = ensure_checkout(root)
    assert bound == root
    import optbench

    assert Path(optbench.__file__).resolve().parents[1] == root
    paths = campaign_json_paths(root)
    assert len(paths) > 50
    assert all(path.is_file() and root in path.parents for path in paths)
    assert load_campaign_json(root) > 100000
    assert parse_site(root) > 10
    assert "OPTBENCH_CHECKOUT" in os.environ
