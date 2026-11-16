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


def test_peakmem_entry_points_run_on_the_checkout() -> None:
    import resource

    from benchmarks.bench_check import Check, CheckJson, CheckReplay, CheckSite, CheckTracks

    root = checkout()
    samples = (
        (Check, "peakmem_check", "root"),
        (CheckJson, "peakmem_json", "root"),
        (CheckTracks, "peakmem_tracks", "root"),
        (CheckReplay, "peakmem_replay", "prepared"),
        (CheckSite, "peakmem_site", "root"),
    )
    for cls, name, attr in samples:
        bench = cls()
        bench.setup()
        getattr(bench, name)()
        if attr == "root":
            assert bench.root == root
        else:
            assert bench.prepared
    # Linux reports the peak resident set in kilobytes. A real sample is
    # well above an empty interpreter.
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8 * 1024


def test_a_plot_peak_matches_across_two_fresh_processes(tmp_path: Path) -> None:
    import subprocess

    from benchmarks.bench_check import plot_process_args, plot_process_env

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; matplotlib.use('Agg'); "
            "from matplotlib.font_manager import findfont; findfont('DejaVu Sans'); "
            "import matplotlib.pyplot; import PIL.Image; PIL.Image.init()",
        ],
        check=True,
        cwd=tmp_path,
    )
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from benchmarks.bench_check import traced_plot_peak\n"
        "rows=[('minimization/toy', [{'code':'a','label':'alpha','success_rate':1.0,"
        "'par2_mean':3.0,'_profile':[['0',1.0],['1',2.0]]}])]\n"
        "print(traced_plot_peak(rows, Path(sys.argv[1])))\n"
    )
    env = plot_process_env()
    env["PYTHONPATH"] = str(checkout())
    peaks = []
    for name in ("one", "two"):
        proc = subprocess.run(
            [*plot_process_args(sys.executable), "-c", code, str(tmp_path / name)],
            check=True,
            cwd=checkout(),
            env=env,
            capture_output=True,
            text=True,
        )
        peaks.append(int(proc.stdout.strip()))
    assert peaks[0] == peaks[1]


def test_rebind_does_not_duplicate_the_checkout(monkeypatch) -> None:
    text = str(checkout())
    before = sys.path.count(text)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _place_checkout(text)
    _place_checkout(text)
    assert sys.path.count(text) == max(before, 1)
