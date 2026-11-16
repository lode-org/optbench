"""Wall time and peak memory of ``optbench check`` and of its stages.

The checkout is ``OPTBENCH_CHECKOUT`` when that is set, otherwise the
tree that contains this file. ``optbench`` is loaded from that tree, and
every sample reads that tree's campaign files, asv results, and site
pages. Setup reads those files into the page cache, runs the stage once,
and then drops the in-process memo tables. The timed call is the next
sample. ``repeat`` is 3 so one asv run keeps three samples. A
``peakmem_`` sample is the process high-water mark after that same call.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
from pathlib import Path


def checkout(root: Path | None = None) -> Path:
    """The tree being measured, not whichever tree first imported this file."""
    if root is not None:
        return Path(root).resolve()
    env = os.environ.get("OPTBENCH_CHECKOUT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def _loaded_optbench_root() -> Path | None:
    module = sys.modules.get("optbench")
    if module is None or not getattr(module, "__file__", None):
        return None
    return Path(module.__file__).resolve().parents[1]


def _place_checkout(text: str) -> None:
    """Put ``text`` first on ``sys.path`` without adding a second copy."""
    if text not in sys.path:
        sys.path.insert(0, text)
    elif sys.path[0] != text:
        sys.path.remove(text)
        sys.path.insert(0, text)


def ensure_checkout(root: Path | None = None) -> Path:
    """Import ``optbench`` from ``root``. Reload it when another tree is loaded."""
    root = checkout(root)
    if _loaded_optbench_root() == root:
        return root
    text = str(root)
    _place_checkout(text)
    for name in list(sys.modules):
        if name == "optbench" or name.startswith("optbench."):
            del sys.modules[name]
    importlib.invalidate_caches()
    loaded = importlib.import_module("optbench")
    found = Path(loaded.__file__).resolve().parents[1]
    if found != root:
        raise RuntimeError(f"optbench loaded from {found}, not {root}")
    return root


@contextlib.contextmanager
def _at_checkout(root: Path):
    """Run git-using commands from the checkout. asv's cwd is not that tree."""
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def drop_memos() -> None:
    """Forget in-process parse and template memos so the next sample refills them."""
    board = sys.modules.get("optbench.board")
    git_text = getattr(board, "_git_text", None) if board is not None else None
    procs = getattr(git_text, "_procs", None) if git_text is not None else None
    if isinstance(procs, dict):
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
        procs.clear()
    campaign = sys.modules.get("optbench.campaign")
    cache = getattr(campaign, "_TEMPLATE_CACHE", None) if campaign is not None else None
    if isinstance(cache, dict):
        cache.clear()
    score = sys.modules.get("optbench.asv_score")
    unwrap = getattr(score, "_UNWRAP_CACHE", None) if score is not None else None
    if isinstance(unwrap, dict):
        unwrap.clear()
    geometry = getattr(board, "_GEOMETRY_CACHE", None) if board is not None else None
    if isinstance(geometry, dict):
        geometry.clear()


def campaign_json_paths(root: Path) -> list[Path]:
    """Start files the board parses one JSON line at a time."""
    root = ensure_checkout(root)
    from optbench.board import _campaign_groups

    paths: list[Path] = []
    for group in _campaign_groups(root / "results" / "campaign").values():
        for kind in ("starts_jsonl", "starts"):
            path = group.get(kind)
            if path is not None:
                paths.append(path)
    return paths


def _asv_json_paths(root: Path) -> list[Path]:
    directory = root / "results" / "asv"
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*.json") if path.is_file())


def _site_paths(root: Path) -> list[Path]:
    pages: list[Path] = []
    include = root / "docs" / "include"
    for name in ("sota.org", "v2-board.org", "collection.org", "codes.org"):
        pages.append(include / name)
    generated = root / "docs" / "orgmode" / "generated"
    for folder in ("suites", "codes"):
        directory = generated / folder
        if directory.is_dir():
            pages.extend(sorted(directory.rglob("*.org")))
    org = root / "docs" / "orgmode"
    for name in ("index.org", "codes.org", "archive.org", "board.org", "run.org", "changelog.org"):
        pages.append(org / name)
    return [path for path in pages if path.is_file()]


def warm(root: Path) -> None:
    """Read the stage inputs so the timed call is not the first touch."""
    paths = campaign_json_paths(root) + _asv_json_paths(root) + _site_paths(root)
    for path in paths:
        path.read_bytes()


def load_campaign_json(root: Path) -> int:
    """Parse every campaign start file with this checkout's line reader."""
    root = ensure_checkout(root)
    from optbench.board import _json_lines

    count = 0
    for path in campaign_json_paths(root):
        count += len(_json_lines(path))
    return count


def load_tracks(root: Path) -> int:
    """Turn every saved asv track into start rows. Returns the row count."""
    root = ensure_checkout(root)
    from asv.results import iter_results

    from optbench.asv_score import track_rows

    count = 0
    for result in iter_results(str(root / "results" / "asv")):
        seen: set[str] = set()
        for key in result.get_all_result_keys():
            text = str(key)
            if not text.endswith("track_force_calls"):
                continue
            class_name = text[: -len(".track_force_calls")].split(".")[-1]
            if class_name in seen:
                continue
            seen.add(class_name)
            try:
                count += len(track_rows(result, class_name))
            except SystemExit:
                continue
    return count


def prepare_replay(root: Path):
    """Load this checkout's campaign and asv rows and collect their scores.

    Building the list is not part of the replay sample.
    """
    root = ensure_checkout(root)
    from optbench.board import _ScoreSession, _asv_rows, _campaign_rows
    from optbench.campaign import list_templates
    from optbench.problems import load_all

    problems = load_all(root / "problems") if (root / "problems").is_dir() else {}
    templates = (
        {item["id"]: item for item in list_templates(root)} if (root / "templates").is_dir() else {}
    )
    session = _ScoreSession()
    try:
        with _at_checkout(root):
            _campaign_rows(root, problems, templates, session)
            _asv_rows(root, problems, templates, session)
        prepared = []
        for batch in session.batches:
            if not batch.held or batch.events is None:
                continue
            prepared.append((batch.events, list(batch.collect())))
        return prepared
    finally:
        session.close()


def run_replay(prepared) -> None:
    """Replay one prepared board in event order."""
    from optbench.board import _replay

    for events, results in prepared:
        _replay(events, [], [], [], list(results))


def plot_rows(root: Path) -> list[tuple[str, list]]:
    """Ranked rows in problem order.

    Building the board is not part of the plot sample. The score pool finishes
    in arrival order, so two builds of the same rows do not leave the same heap.
    """
    root = ensure_checkout(root)
    from optbench.board import build_board

    document = build_board(root)
    prepared: list[tuple[str, list]] = []
    for problem_id, problem in sorted(document["problems"].items()):
        ranked = problem.get("_ranked_rows") or []
        if ranked:
            prepared.append((problem_id, ranked))
    return prepared


def plot_process_env(base: dict | None = None) -> dict[str, str]:
    """Environment for one plot sample.

    ``PYTHONHASHSEED`` is fixed. Matplotlib also keys font and artist caches by
    object identity, so the process has to be started with address layout fixed
    (``setarch -R``) or those caches allocate in a different order. The font
    list on disk is loaded at import; it has to exist already, or the process
    that builds it does not match a process that reads it. The working directory
    is the checkout. Figures are written elsewhere, so the sample does not
    create a directory on the import path.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    return env


def plot_process_args(python: str) -> list[str]:
    """argv prefix for one plot sample, before ``-c`` or the script path."""
    import platform

    return ["setarch", platform.machine(), "-R", python]


def traced_plot_peak(rows: list[tuple[str, list]], directory: Path) -> int:
    """Traced peak bytes of writing ``rows``.

    The imports below are the same ones the plot writer loads on its first
    call. They finish before the trace, which is the post-import window used
    for the other Python peaks. ``gc.collect()`` runs after that and before
    the trace. No figure is drawn first: a real board builds the font and text
    caches while it writes.
    """
    import gc
    import tracemalloc

    import matplotlib

    matplotlib.use("Agg", force=False)
    pyplot = importlib.import_module("matplotlib.pyplot")
    for name in (
        "matplotlib.ticker",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
        "matplotlib.backends._backend_agg",
    ):
        importlib.import_module(name)
    importlib.import_module("PIL.Image").init()
    pyplot.switch_backend("Agg")

    from optbench.board import _profile_for, _write_plots

    directory = Path(directory)
    gc.collect()
    tracemalloc.start()
    try:
        for problem_id, ranked in rows:
            # write_board builds the profile in this same pass, then draws it.
            _write_plots(problem_id, ranked, _profile_for(ranked), directory)
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


def parse_site(root: Path) -> int:
    """Parse the rendered site pages. Returns the page count."""
    root = ensure_checkout(root)
    from optbench.site_check import read_site_pages

    return len(read_site_pages(root))


def run_check(root: Path) -> None:
    """Run ``optbench check`` on this checkout. A failing check is an error."""
    root = ensure_checkout(root)
    from optbench.check import check

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), _at_checkout(root):
        code = check(root / "problems", root / "oracle" / "entries.jsonl")
    if code != 0:
        raise RuntimeError(buffer.getvalue())


class _Samples:
    """Three timed samples. Setup warms the cache and drops memos first."""

    repeat = 3
    rounds = 1
    number = 1
    min_run_count = 1
    sample_time = 0
    warmup_time = 0
    timeout = 300

    def _bind(self) -> Path:
        root = ensure_checkout(checkout())
        warm(root)
        drop_memos()
        return root


class Check(_Samples):
    """End-to-end ``optbench check``."""

    def setup(self) -> None:
        root = self._bind()
        run_check(root)
        drop_memos()
        self.root = root

    def time_check(self) -> None:
        run_check(self.root)

    def peakmem_check(self) -> None:
        run_check(self.root)


class CheckJson(_Samples):
    """Campaign start files through the checkout's JSON-line reader."""

    def setup(self) -> None:
        root = self._bind()
        load_campaign_json(root)
        drop_memos()
        self.root = root

    def time_json(self) -> None:
        load_campaign_json(self.root)

    def peakmem_json(self) -> None:
        load_campaign_json(self.root)


class CheckTracks(_Samples):
    """Saved asv tracks turned into start rows."""

    def setup(self) -> None:
        root = self._bind()
        load_tracks(root)
        drop_memos()
        self.root = root

    def time_tracks(self) -> None:
        load_tracks(self.root)

    def peakmem_tracks(self) -> None:
        load_tracks(self.root)


class CheckReplay(_Samples):
    """Ordered replay of a board built from this checkout."""

    def setup(self) -> None:
        root = self._bind()
        run_replay(prepare_replay(root))
        drop_memos()
        self.prepared = prepare_replay(root)

    def time_replay(self) -> None:
        run_replay(self.prepared)

    def peakmem_replay(self) -> None:
        run_replay(self.prepared)


class CheckSite(_Samples):
    """Rendered site pages and their tables."""

    def setup(self) -> None:
        root = self._bind()
        parse_site(root)
        drop_memos()
        self.root = root

    def time_site(self) -> None:
        parse_site(self.root)

    def peakmem_site(self) -> None:
        parse_site(self.root)
