"""The check timing samples read the checkout they were given."""

import os
import sys
from pathlib import Path

from benchmarks.bench_check import (
    _place_checkout,
    campaign_json_paths,
    checkout,
    drop_memos,
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


def test_drop_memos_clears_the_geometry_cache() -> None:
    from optbench.board import _GEOMETRY_CACHE, _geometry_key

    _geometry_key("cluster_0000")
    assert _GEOMETRY_CACHE["cluster_0000"] == "0"
    drop_memos()
    assert _GEOMETRY_CACHE == {}


def test_rebind_does_not_duplicate_the_checkout(monkeypatch) -> None:
    text = str(checkout())
    before = sys.path.count(text)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _place_checkout(text)
    _place_checkout(text)
    assert sys.path.count(text) == max(before, 1)
