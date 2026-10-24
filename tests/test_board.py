"""Behaviour of the v2 board built from committed result files."""

import json
from pathlib import Path

import pytest

from optbench.board import MissingAsvCommit, build_board, headline_table, render_problem_section, write_board
from optbench.score import rate_interval

PIN = "a" * 40
OTHER = "b" * 40


def _problem(root: Path, starts: int = 2, cap: int = 10) -> None:
    path = root / "problems" / "minimization" / "toy.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'id = "minimization/toy"',
                'title = "Toy"',
                'suite = "minimization"',
                'problem = "toy"',
                'contract = "stated"',
                "",
                "[sample]",
                f"starts = {starts}",
                "",
                "[convergence]",
                "force_norm = 0.01",
                f"max_force_calls = {cap}",
                "",
                "[metrics]",
                'required = ["force_calls"]',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _template(root: Path, template_id: str = "toy/fast", *, images: int | None = None) -> None:
    path = root / "templates" / f"{template_id}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'id = "{template_id}"',
        'engine = "toy"',
        'problem = "minimization/toy"',
        'code = "toy"',
        'code_version = "1.0"',
        f'commit = "{PIN}"',
        'tuning = "default"',
    ]
    if images is not None:
        lines.append('reducer = "per_image"')
        lines.append("")
        lines.append("[neb]")
        lines.append(f"moving_images = {images}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _registry(root: Path, commit: str = PIN) -> None:
    path = root / "codes" / "registry.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "[[code]]",
                'id = "toy"',
                'name = "toy"',
                f'commit = "{commit}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _dat(
    root: Path,
    stem: str,
    commit: str,
    *,
    counter: str = "harness",
    template: str = "toy/fast",
    version: str = "1.0",
    code: str = "toy",
    problem: str = "minimization/toy",
) -> Path:
    path = root / "results" / "campaign" / f"{stem}.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "force_calls 1",
                "nfailed 0",
                f"code {code}",
                f"code_version {version}",
                f"commit {commit}",
                f"template {template}",
                f"problem {problem}",
                f"counter {counter}",
                "tuning default",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _starts(root: Path, stem: str, rows: list[dict]) -> None:
    path = root / "results" / "campaign" / f"{stem}.starts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(calls: float, success: bool, start: str, failure: str | None = None) -> dict:
    record = {"force_calls": calls, "success": success, "start": start, "counter": "harness"}
    if not success:
        record["failure"] = failure or "cap"
    return record


def test_full_harness_row_is_ranked_by_par2(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN)
    _starts(
        tmp_path,
        "toy-fast",
        [_row(3, True, "0"), _row(10, False, "1", "cap")],
    )

    document = build_board(tmp_path)
    problem = document["problems"]["minimization/toy"]
    assert problem["short"] == []
    assert problem["not_comparable"] == []
    assert len(problem["ranked"]) == 1
    row = problem["ranked"][0]
    assert row["template"] == "toy/fast"
    assert row["par2_mean"] == (3 + 20) / 2
    assert row["mean_calls"] == (3 + 10) / 2
    assert row["median_calls"] == (3 + 10) / 2
    assert row["min_calls"] == 3
    assert row["max_calls"] == 10
    assert row["n"] == 2
    assert row["nfailed"] == 1
    assert row["failures"]["cap"] == 1
    assert row["success_rate"] == 0.5
    low, high = rate_interval(1, 2)
    assert low < row["success_rate"] < high or low == high
    assert row["registry_match"] is True
    assert row["short"] is False
    assert "rate_low" not in row or row.get("rate_low") is None


def test_a_full_row_shows_a_repeated_energy_without_leaving_the_table(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    path = _dat(tmp_path, "toy-fast", PIN)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "finding 1 of 2 starts at -1.00000000; lowest energy -2.00000000\n",
        encoding="utf-8",
    )
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(4, True, "1")])

    document = build_board(tmp_path)
    problem = document["problems"]["minimization/toy"]
    assert problem["short"] == []
    assert problem["ranked"][0]["finding"] == "1 of 2 starts at -1.00000000; lowest energy -2.00000000"
    html = render_problem_section(problem, set())
    assert "<h3>Findings</h3>" in html
    assert "1 of 2 starts at -1.00000000; lowest energy -2.00000000" in html
    assert "1 of 2 starts at -1.00000000; lowest energy -2.00000000" in headline_table(problem)


def test_short_grid_stays_off_the_ranked_table(tmp_path: Path):
    _problem(tmp_path, starts=4, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN)
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(4, True, "1")])

    document = build_board(tmp_path)
    problem = document["problems"]["minimization/toy"]
    assert problem["ranked"] == []
    assert len(problem["short"]) == 1
    short = problem["short"][0]
    assert short["n"] == 2
    assert short["short_reason"] == "scored 2 of 4 starts"
    assert document["short"][0]["template"] == "toy/fast"


def test_duplicate_files_are_kept_and_the_registry_commit_is_current(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path, PIN)
    kept = _dat(tmp_path, "toy-fast-old", OTHER)
    current = _dat(tmp_path, "toy-fast", PIN)
    _starts(tmp_path, "toy-fast-old", [_row(4, True, "0"), _row(4, True, "1")])
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(3, True, "1")])

    document = build_board(tmp_path)
    ranked = document["problems"]["minimization/toy"]["ranked"]
    assert len(ranked) == 1
    assert ranked[0]["commit"] == PIN
    assert ranked[0]["par2_mean"] == 3
    assert kept.is_file()
    assert current.is_file()

    tied = tmp_path / "results" / "campaign" / "toy-fast-tied.dat"
    tied.write_text(current.read_text(encoding="utf-8").replace(PIN, OTHER), encoding="utf-8")
    _starts(tmp_path, "toy-fast-tied", [_row(3, True, "0"), _row(3, True, "1")])
    tied_board = build_board(tmp_path)["problems"]["minimization/toy"]["ranked"]
    assert [row["commit"] for row in tied_board] == [PIN]


def test_client_counter_is_not_on_the_ranked_board(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN, counter="client")
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(4, True, "1")])

    document = build_board(tmp_path)
    problem = document["problems"]["minimization/toy"]
    assert problem["ranked"] == []
    assert len(problem["not_comparable"]) == 1
    assert problem["not_comparable"][0]["counter"] == "client"
    assert document["short"] == []


def test_a_dat_with_no_value_is_listed_and_left_in_place(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    path = tmp_path / "results" / "campaign" / "broken.dat"
    path.parent.mkdir(parents=True)
    path.write_text("force_calls\n", encoding="utf-8")

    document = build_board(tmp_path)
    assert document["ranked_count"] == 0
    assert len(document["unparsed"]) == 1
    assert document["unparsed"][0]["path"] == "results/campaign/broken.dat"
    assert "no value" in document["unparsed"][0]["note"]
    assert path.read_text(encoding="utf-8") == "force_calls\n"


def test_a_problem_without_a_sample_count_does_not_invent_one(tmp_path: Path):
    path = tmp_path / "problems" / "saddle-search" / "island.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                'id = "saddle-search/island"',
                'title = "Island"',
                'suite = "saddle-search"',
                'problem = "island"',
                'contract = "stated"',
                "",
                "[metrics]",
                'required = ["force_calls"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    template = tmp_path / "templates" / "toy" / "fast.toml"
    template.parent.mkdir(parents=True)
    template.write_text(
        "\n".join(
            [
                'id = "toy/fast"',
                'engine = "toy"',
                'problem = "saddle-search/island"',
                'code = "toy"',
                'code_version = "1.0"',
                f'commit = "{PIN}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _registry(tmp_path)
    short = _dat(tmp_path, "short", PIN)
    short.write_text(
        short.read_text(encoding="utf-8").replace("problem minimization/toy", "problem saddle-search/island"),
        encoding="utf-8",
    )
    wide = _dat(tmp_path, "wide", PIN)
    wide.write_text(
        wide.read_text(encoding="utf-8").replace("problem minimization/toy", "problem saddle-search/island"),
        encoding="utf-8",
    )
    _starts(tmp_path, "short", [_row(3, True, "0"), _row(4, True, "1")])
    _starts(tmp_path, "wide", [_row(3, True, str(index)) for index in range(4)])

    problem = build_board(tmp_path)["problems"]["saddle-search/island"]
    assert problem["full_n"] is None
    assert problem["short"] == []
    assert [row["n"] for row in problem["ranked"]] == [4]


def test_ranked_false_stays_off_the_table(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    path = _dat(tmp_path, "toy-fast", PIN)
    path.write_text(path.read_text(encoding="utf-8") + "ranked false\n", encoding="utf-8")
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(4, True, "1")])

    problem = build_board(tmp_path)["problems"]["minimization/toy"]
    assert problem["ranked"] == []
    assert problem["short"][0]["short_reason"] == "result file sets ranked false"
    path.write_text(path.read_text(encoding="utf-8") + "potential ljcluster (shifted, cutoff 15)\n", encoding="utf-8")
    problem = build_board(tmp_path)["problems"]["minimization/toy"]
    assert problem["ranked"] == []
    assert problem["short"][0]["short_reason"] == (
        "result file sets ranked false; potential ljcluster (shifted, cutoff 15)"
    )


def test_per_image_band_uses_the_image_cost(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=100)
    _template(tmp_path, images=2)
    _registry(tmp_path)
    dat = _dat(tmp_path, "toy-fast", PIN)
    dat.write_text(dat.read_text(encoding="utf-8") + "reducer per_image\n", encoding="utf-8")
    _starts(
        tmp_path,
        "toy-fast",
        [
            {"force_calls": 10, "success": True, "start": "0", "counter": "harness"},
            {"force_calls": 20, "success": True, "start": "1", "counter": "harness"},
        ],
    )

    row = build_board(tmp_path)["problems"]["minimization/toy"]["ranked"][0]
    assert row["call_unit"] == "total"
    assert row["par2_mean"] == (10 + 20) / 2
    assert row["mean_calls"] == (10 + 20) / 2
    assert row["mean_per_image"] == (5 + 10) / 2


def _asv_track(root: Path, prefix: str, calls: list[float], *, commit: str = "c" * 40) -> None:
    machine = root / "results" / "asv" / "testbox"
    machine.mkdir(parents=True, exist_ok=True)
    (machine / "machine.json").write_text(
        json.dumps({"machine": "testbox", "version": 1}) + "\n",
        encoding="utf-8",
    )
    params = [["'0'"], [f"'{index}'" for index in range(len(calls))]]
    columns = ["result", "params", "version", "started_at", "duration"]
    zeros = [0.0] * len(calls)
    tracks = {
        "track_force_calls": list(calls),
        "track_success": [1.0] * len(calls),
        "track_failure_cap": zeros,
        "track_failure_wrong_stationary": zeros,
        "track_failure_divergence": zeros,
        "track_failure_crash": zeros,
    }
    results = {f"{prefix}.{name}": [value, params, "v", 0, 0.1] for name, value in tracks.items()}
    payload = {
        "commit_hash": commit,
        "env_name": "py",
        "date": 1,
        "params": {"machine": "testbox"},
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }
    (machine / f"{commit[:8]}-py.json").write_text(json.dumps(payload), encoding="utf-8")


def _asv_one_start(root: Path) -> None:
    _asv_track(root, "bench_lj38.Lj38EonLbfgsM100", [4.0])


def test_one_start_asv_grid_is_partial(tmp_path: Path):
    _problem(tmp_path, starts=1000, cap=10000)
    problem = tmp_path / "problems" / "minimization" / "toy.toml"
    text = problem.read_text(encoding="utf-8").replace('id = "minimization/toy"', 'id = "minimization/lj38"')
    text = text.replace('problem = "toy"', 'problem = "lj38"')
    text = text.replace('title = "Toy"', 'title = "Lennard-Jones 38 clusters"')
    problem.write_text(text, encoding="utf-8")
    template = tmp_path / "templates" / "eon-client" / "lbfgs-m100.toml"
    template.parent.mkdir(parents=True)
    template.write_text(
        "\n".join(
            [
                'id = "eon-client/lbfgs-m100"',
                'engine = "eon"',
                'problem = "minimization/lj38"',
                'code = "eOn"',
                'code_version = "3.5.0"',
                f'commit = "{PIN}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _asv_one_start(tmp_path)

    document = build_board(tmp_path)
    short_rows = [row for row in document["short"] if row["entry"].endswith("Lj38EonLbfgsM100")]
    assert len(short_rows) == 1
    assert short_rows[0]["n"] == 1
    assert short_rows[0]["short"] is True
    assert short_rows[0]["short_reason"] == "scored 1 of 1000 starts"
    ranked = [
        row
        for problem_rows in document["problems"].values()
        for row in problem_rows["ranked"]
        if str(row["entry"]).endswith("Lj38EonLbfgsM100")
    ]
    assert ranked == []


def test_profile_lines_up_the_same_starts(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=100)
    _template(tmp_path, "toy/cheap")
    _template(tmp_path, "toy/dear")
    _registry(tmp_path)
    _dat(tmp_path, "cheap", PIN, template="toy/cheap")
    _dat(tmp_path, "dear", PIN, template="toy/dear")
    _starts(tmp_path, "cheap", [_row(2, True, "cluster_0000.con"), _row(2, True, "cluster_0001.con")])
    _starts(tmp_path, "dear", [_row(4, True, "0"), _row(4, True, "1")])

    written = write_board(tmp_path, tmp_path / "board")
    problem = written["problems"]["minimization/toy"]
    rho = problem["profile"]["rho"]
    cheap = next(label for label in rho if "toy/cheap" in label)
    dear = next(label for label in rho if "toy/dear" in label)
    assert rho[cheap][0] == 1.0
    assert rho[dear][0] == 0.0
    assert rho[cheap][1] == 1.0
    assert rho[dear][1] == 1.0
    profile_svg = (tmp_path / "board" / "plots" / "minimization-toy-profile.svg").read_text(encoding="utf-8")
    assert "DejaVuSans-" not in profile_svg
    assert profile_svg.count("<use") < 40
    assert (tmp_path / "board" / "plots" / "minimization-toy-profile.png").stat().st_size > 0
    assert (tmp_path / "board" / "plots" / "minimization-toy-profile.webp").read_bytes().startswith(b"RIFF")
    assert (tmp_path / "board" / "plots" / "minimization-toy-cost.svg").stat().st_size > 0
    assert (tmp_path / "board" / "plots" / "minimization-toy-cost.png").stat().st_size > 0
    assert (tmp_path / "board" / "plots" / "minimization-toy-cost.webp").read_bytes().startswith(b"RIFF")
    include = (tmp_path / "docs" / "include" / "v2-board.org").read_text(encoding="utf-8")
    assert "toy (1.0)" in include
    assert "PAR-2" in include
    page = json.loads((tmp_path / "board" / "index.json").read_text(encoding="utf-8"))
    assert page["ranked_count"] == 2


def test_plot_bytes_match_across_two_writes(tmp_path: Path):
    from optbench.board import _write_plots

    profile = {"taus": [1.0, 2.0], "rho": {"a": [0.0, 1.0]}, "solved": 1, "area_log10": 0.0}
    ranked = [{"success_rate": 1.0, "par2_mean": 3.0, "median": 3.0}]
    first = tmp_path / "a"
    second = tmp_path / "b"
    _write_plots("minimization/toy", ranked, profile, first)
    _write_plots("minimization/toy", ranked, profile, second)
    for name in (
        "minimization-toy-profile.svg",
        "minimization-toy-cost.svg",
        "minimization-toy-profile.png",
        "minimization-toy-cost.png",
        "minimization-toy-profile.webp",
        "minimization-toy-cost.webp",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_a_partial_problem_drops_a_plot_left_from_an_earlier_run(tmp_path: Path):
    _problem(tmp_path, starts=4, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN)
    _starts(tmp_path, "toy-fast", [_row(1, True, "0"), _row(1, True, "1")])
    plots = tmp_path / "board" / "plots"
    plots.mkdir(parents=True)
    stale = plots / "minimization-toy-profile.svg"
    stale.write_text("stale", encoding="utf-8")
    site = tmp_path / "docs" / "source" / "_static" / "board"
    site.mkdir(parents=True)
    site_stale = site / "minimization-toy-cost.png"
    site_stale.write_text("stale", encoding="utf-8")
    site_webp = site / "minimization-toy-profile.webp"
    site_webp.write_bytes(b"stale-webp")

    written = write_board(tmp_path, tmp_path / "board")

    assert written["problems"]["minimization/toy"]["ranked"] == []
    assert not stale.exists()
    assert not site_stale.exists()
    assert not site_webp.exists()
    assert list(plots.iterdir()) == []
    assert list(site.iterdir()) == []


def test_only_the_newest_full_version_is_ranked(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    old = _dat(tmp_path, "old", PIN, version="0.4.0")
    current = _dat(tmp_path, "current", PIN, version="0.4.1")
    _dat(tmp_path, "short", PIN, version="0.4.2")
    _starts(tmp_path, "old", [_row(4, True, "0"), _row(4, True, "1")])
    _starts(tmp_path, "current", [_row(3, True, "0"), _row(3, True, "1")])
    _starts(tmp_path, "short", [_row(1, True, "0")])

    problem = build_board(tmp_path)["problems"]["minimization/toy"]
    assert [row["code_version"] for row in problem["ranked"]] == ["0.4.1", "0.4.0"]
    assert [row["code_version"] for row in problem["short"]] == ["0.4.2"]
    assert old.is_file()
    assert current.is_file()


def test_two_codes_on_one_band_rank_the_same_total(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10000)
    problem_path = tmp_path / "problems" / "minimization" / "toy.toml"
    problem_path.write_text(problem_path.read_text(encoding="utf-8") + "\n[band]\ninterior_images = 8\n", encoding="utf-8")
    _template(tmp_path, "toy/ase")
    _template(tmp_path, "toy/eon")
    _registry(tmp_path)
    _dat(tmp_path, "ase-band", PIN, template="toy/ase", code="ASE")
    _dat(tmp_path, "eon-band", PIN, template="toy/eon", code="eOn")
    _starts(
        tmp_path,
        "ase-band",
        [
            {"force_calls": 80, "success": True, "start": "0", "counter": "harness"},
            {"force_calls": 160, "success": True, "start": "1", "counter": "harness"},
        ],
    )
    _starts(
        tmp_path,
        "eon-band",
        [
            {
                "force_calls": 10,
                "raw": 80,
                "count_key": "force_calls_neb",
                "success": True,
                "start": "0",
                "counter": "harness",
            },
            {
                "force_calls": 20,
                "raw": 160,
                "count_key": "force_calls_neb",
                "success": True,
                "start": "1",
                "counter": "harness",
            },
        ],
    )
    _dat(tmp_path, "bare", OTHER, template="toy/ase")
    _starts(
        tmp_path,
        "bare",
        [{"force_calls": 10, "success": True, "start": "0", "counter": "harness", "call_unit": "per_image"}],
    )
    # Drop the image count for that third file by giving it a problem with no band.
    bare = tmp_path / "results" / "campaign" / "bare.dat"
    bare.write_text(
        bare.read_text(encoding="utf-8").replace("problem minimization/toy", "problem minimization/plain"),
        encoding="utf-8",
    )
    plain = tmp_path / "problems" / "minimization" / "plain.toml"
    plain.write_text(
        problem_path.read_text(encoding="utf-8")
        .replace('id = "minimization/toy"', 'id = "minimization/plain"')
        .replace('problem = "toy"', 'problem = "plain"')
        .replace("[band]\ninterior_images = 8\n", ""),
        encoding="utf-8",
    )
    plain_template = tmp_path / "templates" / "toy" / "plain.toml"
    plain_template.write_text(
        (tmp_path / "templates" / "toy" / "ase.toml").read_text(encoding="utf-8").replace("toy/ase", "toy/plain").replace(
            "minimization/toy", "minimization/plain"
        ),
        encoding="utf-8",
    )
    bare.write_text(bare.read_text(encoding="utf-8").replace("template toy/ase", "template toy/plain"), encoding="utf-8")

    document = build_board(tmp_path)
    ranked = document["problems"]["minimization/toy"]["ranked"]
    assert len(ranked) == 2
    units = {row["call_unit"] for row in ranked}
    means = {row["mean_calls"] for row in ranked}
    per = {row["mean_per_image"] for row in ranked}
    assert units == {"total"}
    assert means == {120}
    assert per == {15}
    assert "minimization/plain" not in document["problems"]
    assert any(
        row["problem"] == "minimization/plain" and "image count" in row["note"] for row in document["unparsed"]
    )


def test_a_campaign_and_its_asv_track_are_one_row(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10000)
    problem = tmp_path / "problems" / "minimization" / "toy.toml"
    text = problem.read_text(encoding="utf-8").replace('id = "minimization/toy"', 'id = "minimization/lj38"')
    text = text.replace('problem = "toy"', 'problem = "lj38"')
    problem.write_text(text, encoding="utf-8")
    template = tmp_path / "templates" / "rgmin" / "lj38-lbfgs.toml"
    template.parent.mkdir(parents=True)
    template.write_text(
        "\n".join(
            [
                'id = "rgmin/lj38-lbfgs"',
                'engine = "rgmin"',
                'problem = "minimization/lj38"',
                'code = "rgmin"',
                'code_version = "0.4.1"',
                f'commit = "{PIN}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _dat(
        tmp_path,
        "rgmin-lj38",
        PIN,
        template="rgmin/lj38-lbfgs",
        version="0.4.1",
        code="rgmin",
        problem="minimization/lj38",
    )
    _starts(tmp_path, "rgmin-lj38", [_row(10, True, "0"), _row(20, True, "1")])
    _asv_track(tmp_path, "bench_rgmin.RgminLj38Lbfgs", [10.0, 20.0])

    ranked = build_board(tmp_path)["problems"]["minimization/lj38"]["ranked"]
    assert len(ranked) == 1
    assert ranked[0]["source"] == "campaign, asv"
    assert ranked[0]["n"] == 2
    assert ranked[0]["mean_calls"] == 15
    assert ranked[0]["par2_mean"] == 15


def test_a_missing_asv_commit_is_not_pinned_from_the_working_tree(tmp_path: Path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _asv_track(tmp_path, "bench_rgmin.RgminLj38Lbfgs", [10.0, 20.0], commit="d" * 40)
    with pytest.raises(MissingAsvCommit, match="not in this checkout"):
        build_board(tmp_path)


def test_a_one_start_asv_track_stays_out_of_the_ranked_mean(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10000)
    problem = tmp_path / "problems" / "minimization" / "toy.toml"
    text = problem.read_text(encoding="utf-8").replace('id = "minimization/toy"', 'id = "minimization/lj38"')
    text = text.replace('problem = "toy"', 'problem = "lj38"')
    problem.write_text(text, encoding="utf-8")
    template = tmp_path / "templates" / "rgmin" / "lj38-lbfgs.toml"
    template.parent.mkdir(parents=True)
    template.write_text(
        "\n".join(
            [
                'id = "rgmin/lj38-lbfgs"',
                'engine = "rgmin"',
                'problem = "minimization/lj38"',
                'code = "rgmin"',
                'code_version = "0.4.1"',
                f'commit = "{PIN}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _dat(
        tmp_path,
        "rgmin-lj38",
        PIN,
        template="rgmin/lj38-lbfgs",
        version="0.4.0",
        code="rgmin",
        problem="minimization/lj38",
    )
    _starts(tmp_path, "rgmin-lj38", [_row(10, True, "0"), _row(20, True, "1")])
    _asv_track(tmp_path, "bench_rgmin.RgminLj38Lbfgs", [100.0])

    problem_rows = build_board(tmp_path)["problems"]["minimization/lj38"]
    assert len(problem_rows["ranked"]) == 1
    assert problem_rows["ranked"][0]["n"] == 2
    assert problem_rows["ranked"][0]["mean_calls"] == 15
    assert problem_rows["ranked"][0]["code_version"] == "0.4.0"
    assert problem_rows["ranked"][0]["source"] == "campaign"
    smoke = problem_rows["short"]
    assert len(smoke) == 1
    assert smoke[0]["n"] == 1
    assert smoke[0]["mean_calls"] == 100
    assert smoke[0]["code_version"] == "0.4.1"
    assert smoke[0]["short_reason"] == "scored 1 of 2 starts"


def test_shipped_band_starts_share_one_call_unit():
    import statistics

    from optbench.problems import load_all
    from optbench.score import normalize_start

    root = Path(__file__).resolve().parents[1]
    problems = load_all(root / "problems")
    campaign = root / "results" / "campaign"

    ase_path = campaign / "ase-3.29.0-neb-pt-heptamer-default.starts.jsonl"
    ase_rows = [json.loads(line) for line in ase_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ase_min = next(row for row in ase_rows if row["start"] == "min01")
    ase_norm = normalize_start(ase_min, 8)
    assert ase_min["force_calls"] == 378
    assert ase_norm["force_calls"] == ase_min["force_calls"]
    assert ase_norm["per_image"] == ase_min["force_calls"] / 8
    assert ase_norm["call_unit"] == "total"
    dat_fields = {}
    for line in (campaign / "ase-3.29.0-neb-pt-heptamer-default.dat").read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] not in dat_fields:
            dat_fields[parts[0]] = parts[1]
    start_mean = statistics.fmean(normalize_start(row, 8)["force_calls"] for row in ase_rows)
    assert start_mean > float(dat_fields["force_calls"]) * 2
    assert abs(start_mean - float(dat_fields["force_calls_raw"])) < 1

    eon_path = campaign / "eon-main-neb-pt-lbfgs.starts"
    eon_rows = [json.loads(line) for line in eon_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eon_min = next(row for row in eon_rows if row["geometry"] == "min01")
    eon_norm = normalize_start(eon_min, 8)
    assert eon_norm["force_calls"] == eon_min["raw"]
    assert eon_norm["per_image"] == eon_min["force_calls"]
    assert eon_norm["call_unit"] == "total"
    assert eon_min["raw"] == 288
    assert eon_min["force_calls"] == 36

    de_path = campaign / "ase-3.29.0-de-lj38-default.starts.jsonl"
    de_row = json.loads(de_path.read_text(encoding="utf-8").splitlines()[0])
    de_norm = normalize_start(de_row, 5)
    assert de_row["force_calls"] == 2500
    assert de_norm["force_calls"] == de_row["force_calls"]
    assert de_norm["per_image"] == de_row["force_calls"] / 5
    assert de_norm["call_unit"] == "total"

    seen_images: dict[str, int] = {}
    for path in sorted(campaign.iterdir()):
        name = path.name
        if name.endswith(".starts.jsonl"):
            stem = name[: -len(".starts.jsonl")]
        elif name.endswith(".starts"):
            stem = name[: -len(".starts")]
        else:
            continue
        dat = campaign / f"{stem}.dat"
        if not dat.is_file():
            continue
        fields = {}
        for line in dat.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0] not in fields:
                fields[parts[0]] = parts[1]
        problem_id = fields.get("problem", "")
        if not problem_id.startswith(("neb/", "double-ended-saddle-search/")):
            continue
        images = int((problems[problem_id].get("band") or {})["interior_images"])
        previous = seen_images.setdefault(problem_id, images)
        assert previous == images
        for raw in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
            norm = normalize_start(raw, images)
            assert norm["call_unit"] == "total"
            if raw.get("count_key") == "force_calls_neb":
                assert norm["force_calls"] == float(raw["raw"])
            elif raw.get("force_calls") is not None:
                assert norm["force_calls"] == float(raw["force_calls"])
            if norm["force_calls"]:
                assert abs(norm["force_calls"] / norm["per_image"] - images) < 1e-9


def test_the_current_ase_benchmark_records_the_band_total():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    current = (root / "benchmarks" / "bench_ase_sella.py").read_text(encoding="utf-8")
    assert 'float(record["force_calls"]) / images' not in current
    old = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            "806ecaca75a456ba67e7f95621c51ab84e94adaf:benchmarks/bench_ase_sella.py",
        ],
        text=True,
    )
    assert 'float(record["force_calls"]) / images' in old
