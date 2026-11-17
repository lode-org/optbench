"""How the rgsaddle driver finds its starts, builds a spec for each, and reports a run.

The starts are the archived displaced clusters that the repository holds
under ``inputs``: 200 LJ38 starts, the 49 Pt-island starts that share one
reactant, and the 200 LJ38 starts that come with a reference lowest mode.
The library that climbs them is not built in CI. A run that needs a climb
is given scripted ones, and everything around the climb is the driver's own.
"""

from __future__ import annotations

import copy
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest

import optbench.rgsaddle_minmode as driver
from optbench.campaign import load_template, resolve_problem
from optbench.rgsaddle_minmode import (
    HarnessSurface,
    _assert_source_commit,
    _bind,
    _extract,
    _finite_point,
    _free_view,
    _geometries,
    _library_path,
    _potential_for,
    _seed,
    _spec_from_geometry,
    _stored,
    _summarize,
    _unit,
    run_template,
)
from optbench.surfaces import lennard_jones, load_con, potential_from_problem

ROOT = Path(__file__).resolve().parents[1]
LJ38_ARCHIVE = "website/tsopt/LJ38/coords-con.tar.gz"


@pytest.fixture(scope="module")
def starts(tmp_path_factory):
    """Each archive is unpacked once and its rows kept."""
    rows = {}
    for problem_id in ("tsopt/LJ38", "tsopt/pt-island", "lowest-eigenvector/lj38"):
        work = tmp_path_factory.mktemp(problem_id.replace("/", "-"))
        rows[problem_id] = _geometries(ROOT, resolve_problem(ROOT, problem_id), work)
    return rows


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as packed:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            packed.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _root_holding(tmp_path: Path, archive_path: str, members: dict[str, bytes]) -> Path:
    """A repository root whose manifest lists one archive."""
    root = tmp_path / "repo"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "held.tar.gz").write_bytes(_tar_bytes(members))
    manifest = {"path": archive_path, "stored": "inputs/held.tar.gz"}
    (root / "inputs" / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def _git(directory: Path, *arguments: str) -> str:
    identity = ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false"]
    done = subprocess.run(["git", *identity, *arguments], cwd=directory, capture_output=True, text=True, check=True)
    return done.stdout.strip()


# The manifest and the archives


def test_the_manifest_gives_the_stored_file_of_an_archived_path():
    stored = _stored(ROOT, LJ38_ARCHIVE)
    assert stored.is_file()
    assert stored.name == "coords-con.tar.gz"
    assert (ROOT / "inputs") in stored.parents


def test_a_path_the_manifest_does_not_list_is_not_found(tmp_path):
    root = _root_holding(tmp_path, "listed/a.tar.gz", {"x": b"x"})
    manifest = root / "inputs" / "manifest.jsonl"
    manifest.write_text("\n   \n" + manifest.read_text(encoding="utf-8"), encoding="utf-8")
    assert _stored(root, "listed/a.tar.gz") == root / "inputs" / "held.tar.gz"
    with pytest.raises(FileNotFoundError, match="listed/b.tar.gz"):
        _stored(root, "listed/b.tar.gz")


def test_an_archive_is_unpacked_once_and_its_members_are_kept(tmp_path):
    archive = tmp_path / "held.tar.gz"
    archive.write_bytes(_tar_bytes({"coords-con/0000.con": b"first\n"}))
    destination = tmp_path / "geometries"
    assert _extract(archive, destination) == destination
    member = destination / "coords-con" / "0000.con"
    assert member.read_bytes() == b"first\n"
    member.write_bytes(b"edited\n")
    assert _extract(archive, destination) == destination
    assert member.read_bytes() == b"edited\n"


def test_a_folder_and_a_file_that_is_not_an_archive_are_returned_as_they_are(tmp_path):
    folder = tmp_path / "already-here"
    folder.mkdir()
    assert _extract(folder, tmp_path / "unused") == folder
    plain = tmp_path / "start.con"
    plain.write_text("not a tar\n", encoding="utf-8")
    assert _extract(plain, tmp_path / "unused-too") == plain


def test_a_member_that_would_leave_the_destination_is_refused(tmp_path):
    archive = tmp_path / "held.tar.gz"
    archive.write_bytes(_tar_bytes({"../escaped.con": b"outside\n"}))
    with pytest.raises(tarfile.FilterError):
        _extract(archive, tmp_path / "inside")
    assert not (tmp_path / "escaped.con").exists()


# Finding the starts of each problem


@pytest.mark.parametrize(
    ("problem_id", "count", "first", "last"),
    [
        ("tsopt/LJ38", 200, "0000", "0199"),
        ("tsopt/pt-island", 49, "initial_0", "initial_48"),
        ("lowest-eigenvector/lj38", 200, "0000", "0199"),
    ],
)
def test_each_problem_lists_all_its_starts_in_order_and_each_file_is_there(starts, problem_id, count, first, last):
    rows = starts[problem_id]
    assert len(rows) == count
    assert rows[0]["name"] == first
    assert rows[-1]["name"] == last
    assert len({row["name"] for row in rows}) == count
    assert all(Path(row["path"]).is_file() for row in rows)


def test_the_lj38_starts_are_38_free_atoms_each(starts):
    for row in (starts["tsopt/LJ38"][0], starts["tsopt/LJ38"][199]):
        positions, fixed, _cell = load_con(Path(row["path"]))
        assert positions.shape == (38, 3)
        assert not fixed.any()
        assert row["mode"] is None


def test_the_pt_island_starts_share_one_reactant(starts):
    rows = starts["tsopt/pt-island"]
    assert len({str(row["reactant"]) for row in rows}) == 1
    reactant = Path(rows[0]["reactant"])
    assert reactant.name == "reactant.con"
    assert reactant.is_file()
    assert all(row["mode"] is None for row in rows)
    start, start_fixed, _ = load_con(Path(rows[0]["path"]))
    base, base_fixed, _ = load_con(reactant)
    assert start.shape == base.shape
    assert np.array_equal(start_fixed, base_fixed)
    assert start_fixed.any() and not start_fixed.all()
    assert np.allclose(start[start_fixed], base[base_fixed], atol=1e-5)
    assert not np.allclose(start[~start_fixed], base[~base_fixed], atol=1e-2)


def test_the_mode_starts_bring_a_reference_mode_and_the_shared_initial_mode(starts):
    rows = starts["lowest-eigenvector/lj38"]
    assert len({str(row["initial_mode"]) for row in rows}) == 1
    for row in (rows[0], rows[199]):
        assert Path(row["mode"]).name == f"{row['name']}_mode"
        assert Path(row["mode"]).is_file()
        assert Path(row["initial_mode"]).name == "initial_mode"
        assert Path(row["initial_mode"]).is_file()


def test_a_problem_that_lists_one_geometry_is_one_start_named_after_its_file():
    problem = {"id": "x/one", "input": {"geometries": ["website/saddle-search/pt-island/pt-island.con"]}}
    (row,) = _geometries(ROOT, problem, Path("unused"))
    assert row["name"] == "pt-island.con"
    assert row["mode"] is None
    assert Path(row["path"]) == _stored(ROOT, "website/saddle-search/pt-island/pt-island.con")


def test_a_problem_without_an_archive_or_a_geometry_has_no_starts(tmp_path):
    with pytest.raises(ValueError, match="x/none has no geometries"):
        _geometries(ROOT, {"id": "x/none", "input": {}}, tmp_path)
    with pytest.raises(ValueError, match="x/none has no geometries"):
        _geometries(ROOT, {"id": "x/none"}, tmp_path)


def test_an_archive_with_no_known_layout_is_refused(tmp_path):
    problem = copy.deepcopy(resolve_problem(ROOT, "tsopt/LJ38"))
    problem["id"] = "tsopt/unlisted"
    with pytest.raises(ValueError, match="no geometry layout for tsopt/unlisted"):
        _geometries(ROOT, problem, tmp_path)


def test_the_first_start_an_archive_lacks_is_named(tmp_path):
    present = {f"coords-con/{index:04d}.con": b"stand-in\n" for index in (0, 1, 2, 4, 5)}
    root = _root_holding(tmp_path, LJ38_ARCHIVE, present)
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    with pytest.raises(FileNotFoundError, match=r"0003\.con$"):
        _geometries(root, problem, tmp_path / "work")


# What one start becomes


def test_an_lj38_start_becomes_a_free_cluster_spec_that_follows_the_template(starts):
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    template = load_template(ROOT, "rgsaddle/dimer-lbfgs")
    row = starts["tsopt/LJ38"][3]
    spec = _spec_from_geometry(row, problem, template, 10000, 0.001)
    positions, fixed, cell = load_con(Path(row["path"]))
    assert spec["name"] == "0003"
    assert np.array_equal(spec["positions"], positions)
    assert np.array_equal(spec["fixed"], fixed)
    assert np.array_equal(spec["cell"], cell)
    assert spec["potential"] == problem["potential"]
    assert (spec["session"], spec["kind"], spec["method"]) == ("minmode", "dimer", "lbfgs")
    assert (spec["task"], spec["mode_seed"]) == ("saddle", "gradient")
    assert spec["saddle"] == template["ini"]["SaddleSearch"]
    assert spec["optimizer"] == template["ini"]["Optimizer"]
    assert spec["sella"] == {}
    assert (spec["cap"], spec["force_tol"], spec["overlap"]) == (10000, 0.001, 0.99)
    assert spec["reactant"] is None
    assert spec["file_mode"] is None
    assert spec["reference_mode"] is None


def test_a_prfo_template_names_the_prfo_session_and_keeps_its_sella_section(starts):
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    template = load_template(ROOT, "rgsaddle/prfo")
    spec = _spec_from_geometry(starts["tsopt/LJ38"][0], problem, template, 10000, 0.001)
    assert (spec["session"], spec["method"], spec["kind"]) == ("prfo", "prfo", "dimer")
    assert spec["saddle"] == {}
    assert spec["sella"] == template["ini"]["Sella"]


def test_a_problem_that_states_no_potential_gets_lennard_jones_in_reduced_units(starts):
    problem = {key: value for key, value in resolve_problem(ROOT, "tsopt/LJ38").items() if key != "potential"}
    spec = _spec_from_geometry(starts["tsopt/LJ38"][0], problem, load_template(ROOT, "rgsaddle/dimer-lbfgs"), 10000, 0.001)
    assert spec["potential"] == {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0}


def test_a_pt_island_start_carries_the_free_atoms_of_its_reactant_and_seeds_along_the_move(starts):
    problem = resolve_problem(ROOT, "tsopt/pt-island")
    template = load_template(ROOT, "rgsaddle/pt-dimer-lbfgs")
    row = starts["tsopt/pt-island"][3]
    spec = _spec_from_geometry(row, problem, template, 10000, 0.001)
    base, base_fixed, _ = load_con(Path(row["reactant"]))
    assert spec["fixed"].any()
    assert spec["reactant"].shape == (3 * int((~base_fixed).sum()),)
    assert np.array_equal(spec["reactant"], base[~base_fixed].reshape(-1))
    assert spec["mode_seed"] == "reactant"
    assert spec["potential"]["kind"] == "morse"


def test_a_pt_island_start_with_no_stated_seed_is_seeded_along_the_move_too(starts):
    problem = resolve_problem(ROOT, "tsopt/pt-island")
    spec = _spec_from_geometry(starts["tsopt/pt-island"][0], problem, load_template(ROOT, "rgsaddle/pt-prfo"), 10000, 0.001)
    assert spec["mode_seed"] == "reactant"
    assert spec["sella"]["mode_seed"] == "reactant"


def test_a_seed_taken_from_a_file_is_not_replaced_by_the_move(starts):
    problem = resolve_problem(ROOT, "tsopt/pt-island")
    template = copy.deepcopy(load_template(ROOT, "rgsaddle/pt-dimer-lbfgs"))
    template["ini"]["SaddleSearch"]["mode_seed"] = "file"
    spec = _spec_from_geometry(starts["tsopt/pt-island"][0], problem, template, 10000, 0.001)
    assert spec["mode_seed"] == "file"
    assert spec["file_mode"] is None


def _mode_values(path: Path) -> np.ndarray:
    values = [float(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    mode = np.asarray(values)
    return mode / np.linalg.norm(mode)


def test_a_lowest_mode_start_is_a_mode_task_with_its_reference_mode_as_a_unit_vector(starts):
    problem = resolve_problem(ROOT, "lowest-eigenvector/lj38")
    template = load_template(ROOT, "rgsaddle/mode-dimer")
    row = starts["lowest-eigenvector/lj38"][7]
    spec = _spec_from_geometry(row, problem, template, 10000, 0.001)
    assert spec["task"] == "mode"
    assert spec["reference_mode"].shape == (114,)
    assert np.linalg.norm(spec["reference_mode"]) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(spec["reference_mode"], _mode_values(Path(row["mode"])), atol=1e-12)
    assert spec["file_mode"] is None
    assert spec["overlap"] == 0.99


def test_the_initial_mode_is_read_only_when_the_template_seeds_from_a_file(starts):
    problem = resolve_problem(ROOT, "lowest-eigenvector/lj38")
    template = copy.deepcopy(load_template(ROOT, "rgsaddle/mode-dimer"))
    template["ini"]["SaddleSearch"]["mode_seed"] = "file"
    row = starts["lowest-eigenvector/lj38"][0]
    spec = _spec_from_geometry(row, problem, template, 10000, 0.001)
    assert spec["mode_seed"] == "file"
    assert np.allclose(spec["file_mode"], _mode_values(Path(row["initial_mode"])), atol=1e-12)


def _pinned_row(root: Path) -> dict:
    positions = [[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [2.24, 0.0, 0.0]]
    head = ["pinned", "0 0 0", "30 30 30", "90 90 90", "0 0", "0 0 0", "1", "3", "39.9", "Ar", "Coordinates of Component 1"]
    atoms = [f"{x} {y} {z} {int(i == 0)} {i}" for i, (x, y, z) in enumerate(positions)]
    (root / "start.con").write_text("\n".join([*head, *atoms, ""]), encoding="utf-8")
    (root / "start_mode").write_text("#eigenvalue -1.0\n" + "\n".join(["9", "9", "9", "3", "0", "4", "0", "0", "0"]) + "\n", encoding="utf-8")
    (root / "initial_mode").write_text("\n".join(["7", "7", "7", "0", "5", "0", "12", "0", "0"]) + "\n", encoding="utf-8")
    return {"name": "pinned", "path": root / "start.con", "mode": root / "start_mode", "initial_mode": root / "initial_mode"}


MODE_PROBLEM = {"id": "lowest-eigenvector/lj38", "potential": {"kind": "lennard-jones"}}


def test_a_mode_on_pinned_atoms_keeps_only_the_free_atoms(tmp_path):
    spec = _spec_from_geometry(_pinned_row(tmp_path), MODE_PROBLEM, {"id": "t", "ini": {}}, 100, 1e-3)
    reference = spec["reference_mode"]
    assert reference.shape == (6,)
    assert np.allclose(reference / np.linalg.norm(reference), [0.6, 0.0, 0.8, 0.0, 0.0, 0.0])
    assert spec["fixed"].tolist() == [True, False, False]
    assert spec["file_mode"] is None


def test_the_initial_mode_on_pinned_atoms_keeps_only_the_free_atoms_too(tmp_path):
    template = {"id": "t", "ini": {"SaddleSearch": {"mode_seed": "file"}}}
    spec = _spec_from_geometry(_pinned_row(tmp_path), MODE_PROBLEM, template, 100, 1e-3)
    file_mode = spec["file_mode"]
    assert file_mode.shape == (6,)
    assert np.allclose(file_mode / np.linalg.norm(file_mode), [0.0, 5.0 / 13.0, 0.0, 12.0 / 13.0, 0.0, 0.0])


def test_a_template_that_states_nothing_gets_the_minmode_dimer_on_lbfgs_with_the_gradient_seed(tmp_path):
    spec = _spec_from_geometry(_pinned_row(tmp_path), MODE_PROBLEM, {"id": "t"}, 100, 1e-3)
    assert (spec["session"], spec["kind"], spec["method"], spec["mode_seed"]) == ("minmode", "dimer", "lbfgs", "gradient")
    assert (spec["saddle"], spec["optimizer"], spec["sella"]) == ({}, {}, {})


# What a start is seeded with, and the surface it is climbed on


def test_a_unit_vector_is_the_input_scaled_and_a_vector_with_no_direction_is_the_first_axis():
    assert np.allclose(_unit(np.array([3.0, 0.0, 4.0])), [0.6, 0.0, 0.8])
    for useless in (np.zeros(3), np.array([0.0, 1e-15, 0.0]), np.array([np.nan, 1.0, 0.0]), np.array([np.inf, 0.0, 0.0])):
        assert _unit(useless).tolist() == [1.0, 0.0, 0.0]


def _well(coords):
    energy = float(np.sum((np.asarray(coords) - 1.0) ** 2))
    return energy, 2.0 * (np.asarray(coords) - 1.0)


def test_a_start_is_seeded_along_the_move_the_file_or_the_force_in_that_order_of_asking():
    x = np.array([3.0, 1.0, 1.0, 1.0])
    reactant = np.array([3.0, 1.0, 1.0, 0.0])
    from_file = np.array([0.0, 0.0, 2.0, 0.0])
    surface = HarnessSurface(_well)
    assert np.allclose(_seed(x, surface, "reactant", reactant, from_file), [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(_seed(x, surface, "file", reactant, from_file), [0.0, 0.0, 1.0, 0.0])
    assert surface.calls == 0
    assert np.allclose(_seed(x, surface, "gradient", reactant, from_file), [1.0, 0.0, 0.0, 0.0])
    assert surface.calls == 1


def test_a_seed_falls_back_to_the_force_when_the_move_or_the_file_is_missing():
    x = np.array([1.0, 1.0, 3.0])
    assert np.allclose(_seed(x, HarnessSurface(_well), "reactant", None, None), [0.0, 0.0, 1.0])
    assert np.allclose(_seed(x, HarnessSurface(_well), "file", np.zeros(3), None), [0.0, 0.0, 1.0])


def test_the_free_view_is_the_free_coordinates_and_a_copy_of_everything():
    positions = np.arange(12, dtype=float).reshape(4, 3)
    fixed = np.array([True, False, False, True])
    free, flat, full = _free_view(positions, fixed)
    assert free.tolist() == [1, 2]
    assert flat.tolist() == [3, 4, 5, 6, 7, 8]
    assert np.array_equal(full, positions)
    flat[0] = -1.0
    full[0, 0] = -1.0
    assert positions[1, 0] == 3.0 and positions[0, 0] == 0.0


def test_a_cluster_with_every_atom_fixed_cannot_be_climbed():
    with pytest.raises(ValueError, match="every atom is fixed"):
        _free_view(np.zeros((2, 3)), np.array([True, True]))


def test_the_free_coordinates_of_an_island_start_are_energy_and_force_of_the_whole_island(starts):
    problem = resolve_problem(ROOT, "tsopt/pt-island")
    positions, fixed, cell = load_con(Path(starts["tsopt/pt-island"][5]["path"]))
    free = np.flatnonzero(~fixed)
    spec = {"potential": problem["potential"]}
    surface = _potential_for(spec, positions, free, cell)
    energy, gradient = surface.energy_forces(positions[free].reshape(-1))
    full_energy, full_gradient = potential_from_problem(problem)(positions, cell)
    assert energy == pytest.approx(full_energy, rel=1e-12)
    assert np.allclose(gradient, full_gradient[free].reshape(-1), atol=1e-12)
    assert len(gradient) == 3 * len(free)
    assert surface.calls == 1


def test_moving_a_free_coordinate_moves_only_that_atom_and_the_pinned_atoms_stay():
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [2.8, 0.0, 0.0]])
    free = np.array([1])
    cell = np.array([30.0, 30.0, 30.0])

    def raw(coords, _cell):
        return lennard_jones(coords, epsilon=1.0, sigma=1.0, cutoff=40.0, cell=cell, pbc=(False, False, False))

    evaluate = _bind(raw, positions, free, cell)
    moved = evaluate(np.array([1.4, 0.0, 0.0]))
    expected_energy, expected_gradient = raw(np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]]), cell)
    assert moved[0] == pytest.approx(expected_energy, rel=1e-12)
    assert np.allclose(moved[1], expected_gradient[1], atol=1e-12)
    assert evaluate(np.array([1.3, 0.0, 0.0]))[0] != moved[0]
    assert positions[1, 0] == 1.3
    positions[0, 0] = -5.0
    assert evaluate(np.array([1.4, 0.0, 0.0]))[0] == moved[0]


def test_a_surface_handed_in_replaces_the_problem_potential():
    spec = {"evaluate": _well, "potential": {"kind": "morse", "D": 1.0, "alpha": 1.0, "r0": 1.0}}
    surface = _potential_for(spec, np.zeros((1, 3)), np.array([0]), np.array([1.0, 1.0, 1.0]))
    assert surface.energy_forces(np.array([2.0, 1.0, 1.0]))[0] == 1.0


def test_a_point_with_a_non_finite_coordinate_is_not_a_point():
    assert _finite_point(np.array([0.0, -1.0, 2.0]))
    assert not _finite_point(np.array([0.0, np.nan, 2.0]))
    assert not _finite_point(np.array([np.inf, 0.0, 2.0]))


# The library and the source it was built from


def test_the_library_must_be_named_and_must_be_a_file(monkeypatch, tmp_path):
    monkeypatch.delenv("RGSADDLE_LIB", raising=False)
    with pytest.raises(RuntimeError, match="set RGSADDLE_LIB"):
        _library_path()
    monkeypatch.setenv("RGSADDLE_LIB", str(tmp_path))
    with pytest.raises(RuntimeError, match="RGSADDLE_LIB is not a file"):
        _library_path()
    library = tmp_path / "librgsaddle.so"
    library.write_bytes(b"")
    monkeypatch.setenv("RGSADDLE_LIB", str(library))
    assert _library_path() == library


def test_the_source_commit_is_checked_only_when_a_source_and_a_commit_are_given(monkeypatch, tmp_path):
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "start")
    head = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setenv("RGSADDLE_SRC", str(tmp_path))
    _assert_source_commit(head)
    _assert_source_commit("")
    with pytest.raises(RuntimeError, match=f"rgsaddle HEAD {head} does not match the template commit {'0' * 40}"):
        _assert_source_commit("0" * 40)
    monkeypatch.delenv("RGSADDLE_SRC")
    _assert_source_commit("0" * 40)


# The summary of a run


TEMPLATE = {
    "id": "rgsaddle/dimer-lbfgs",
    "code": "rgsaddle",
    "code_version": "0.2.2",
    "commit": "c" * 40,
    "algorithm": "Dimer L-BFGS",
    "tuning": "default",
}


def _record(name, calls, *, success=True, failure=None, rotations=0, evaluations=0):
    return {
        "start": name,
        "force_calls": calls,
        "success": success,
        "failure": failure,
        "reasons": [] if success else [failure or "cap"],
        "rotations": rotations,
        "library_evaluations": evaluations,
    }


def _lines(text: str) -> dict[str, str]:
    return {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in text.splitlines()}


def test_the_summary_reports_successes_by_their_half_even_mean_and_failures_by_their_class():
    records = [
        _record("0000", 100, rotations=4, evaluations=300),
        _record("0001", 201, rotations=5, evaluations=400),
        _record("0002", 303, rotations=6, evaluations=500),
        _record("0003", 10000, success=False, failure="cap", rotations=7, evaluations=600),
        _record("0004", 57, success=False, failure="crash"),
    ]
    text, metrics = _summarize(records, TEMPLATE, 10000, "tsopt/LJ38")
    fields = _lines(text)
    assert fields["force_calls"] == "201"
    assert (fields["force_calls_min"], fields["force_calls_median"], fields["force_calls_max"]) == ("100", "201", "303")
    assert fields["nfailed"] == "2"
    assert float(fields["par2_mean"]) == pytest.approx((100 + 201 + 303 + 20000 + 20000) / 5, abs=1e-9)
    assert fields["successes"] == "3"
    assert (fields["n_cap"], fields["n_crash"], fields["n_divergence"], fields["n_wrong_stationary"]) == ("1", "1", "0", "0")
    assert float(fields["rate_low"]) == pytest.approx(metrics["rate_low"], abs=1e-10)
    assert float(fields["rate_high"]) == pytest.approx(metrics["rate_high"], abs=1e-10)
    assert float(fields["par2_median"]) == pytest.approx(metrics["median"], abs=1e-10)
    assert fields["rotation_steps"] == "22"
    assert fields["library_evaluations"] == "1800"
    assert fields["harness_calls"] == str(100 + 201 + 303 + 10000 + 57)
    assert text.endswith("\n")


def test_the_summary_names_the_code_the_template_and_the_problem():
    text, _metrics = _summarize([_record("0000", 10)], TEMPLATE, 10000, "tsopt/LJ38")
    fields = _lines(text)
    assert fields["counter"] == "harness"
    assert fields["code"] == "rgsaddle"
    assert fields["code_version"] == "0.2.2"
    assert fields["commit"] == "c" * 40
    assert fields["algorithm"] == "Dimer L-BFGS"
    assert fields["tuning"] == "default"
    assert fields["template"] == "rgsaddle/dimer-lbfgs"
    assert fields["problem"] == "tsopt/LJ38"


def test_a_summary_of_ties_rounds_to_the_even_count():
    assert _lines(_summarize([_record("a", 2), _record("b", 3)], TEMPLATE, 100, "p")[0])["force_calls"] == "2"
    assert _lines(_summarize([_record("a", 1), _record("b", 2)], TEMPLATE, 100, "p")[0])["force_calls"] == "2"


def test_a_run_with_no_success_has_no_call_counts_and_a_template_without_names_gets_defaults():
    records = [_record("0000", 10000, success=False, failure="cap"), _record("0001", 10000, success=False, failure="cap")]
    text, metrics = _summarize(records, {"id": "rgsaddle/bare"}, 10000, "tsopt/LJ38")
    fields = _lines(text)
    assert not any(name.startswith("force_calls") for name in fields)
    assert fields["nfailed"] == "2"
    assert fields["successes"] == "0"
    assert float(fields["par2_mean"]) == 20000.0
    assert fields["code"] == "rgsaddle"
    assert fields["code_version"] == ""
    assert fields["commit"] == ""
    assert metrics["nfailed"] == 2


def test_a_run_that_reports_no_rotations_or_library_evaluations_adds_up_to_zero():
    records = [{"start": "0000", "force_calls": 10, "success": True, "failure": None, "rotations": None, "library_evaluations": None}]
    fields = _lines(_summarize(records, TEMPLATE, 100, "p")[0])
    assert fields["rotation_steps"] == "0"
    assert fields["library_evaluations"] == "0"


# A run over the archived starts, with scripted climbs


@pytest.fixture
def climbs(monkeypatch):
    """Climbs that stand in for the library: the second start ends at the cap and the others succeed."""
    monkeypatch.delenv("RGSADDLE_SRC", raising=False)
    specs = []

    def scripted(spec):
        specs.append(spec)
        index = len(specs) - 1
        if index == 1:
            return _record(spec["name"], 10000, success=False, failure="cap") | {"x": [0.0, 1.0]}
        return _record(spec["name"], 120 + 30 * index, rotations=2, evaluations=40) | {"x": [0.0, 1.0]}

    monkeypatch.setattr(driver, "_dispatch", scripted)
    return specs


def test_a_run_climbs_each_start_in_order_and_writes_the_summary_and_the_starts(climbs, tmp_path):
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    template = load_template(ROOT, "rgsaddle/dimer-lbfgs")
    output = tmp_path / "out" / "lj38.dat"
    result = run_template(ROOT, problem, template, output=output, work=tmp_path / "scratch", limit=3)
    assert [spec["name"] for spec in climbs] == ["0000", "0001", "0002"]
    assert (result["starts"], result["nfailed"], result["force_calls"]) == (3, 1, 150)
    assert result["template"] == "rgsaddle/dimer-lbfgs"
    assert result["problem"] == "tsopt/LJ38"
    assert [row["start"] for row in result["records"]] == ["0000", "0001", "0002"]
    assert result["par2_mean"] == pytest.approx((120 + 20000 + 180) / 3)
    fields = _lines(output.read_text(encoding="utf-8"))
    assert (fields["force_calls"], fields["nfailed"], fields["template"]) == ("150", "1", "rgsaddle/dimer-lbfgs")
    assert (tmp_path / "scratch" / "geometries" / "coords-con" / "0000.con").is_file()


def test_a_run_in_which_nothing_succeeds_has_no_mean_call_count(monkeypatch, tmp_path):
    monkeypatch.delenv("RGSADDLE_SRC", raising=False)
    monkeypatch.setattr(driver, "_dispatch", lambda spec: _record(spec["name"], 10000, success=False, failure="cap"))
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    result = run_template(ROOT, problem, load_template(ROOT, "rgsaddle/dimer-lbfgs"), output=tmp_path / "none.dat", work=tmp_path / "w", limit=2)
    assert result["force_calls"] is None
    assert result["nfailed"] == 2
    assert result["par2_mean"] == 20000.0


def test_each_start_row_names_its_provenance_and_leaves_the_final_point_out(climbs, tmp_path):
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    template = load_template(ROOT, "rgsaddle/dimer-lbfgs")
    output = tmp_path / "lj38.dat"
    result = run_template(ROOT, problem, template, output=output, work=tmp_path / "scratch", limit=2)
    rows = [json.loads(line) for line in (tmp_path / "lj38.starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["start"] for row in rows] == ["0000", "0001"]
    for row in rows:
        assert "x" not in row
        assert row["code"] == "rgsaddle"
        assert row["code_version"] == template["code_version"]
        assert row["commit"] == template["commit"]
        assert row["tuning"] == "default"
        assert row["template"] == "rgsaddle/dimer-lbfgs"
    assert all("x" in record for record in result["records"])


def test_a_run_without_a_limit_climbs_every_start_and_hands_each_the_cap_and_the_stop(climbs, tmp_path):
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    result = run_template(ROOT, problem, load_template(ROOT, "rgsaddle/dimer-lbfgs"), output=tmp_path / "all.dat", work=tmp_path / "scratch")
    assert result["starts"] == len(climbs) == 200
    assert {(spec["cap"], spec["force_tol"]) for spec in climbs} == {(10000, 0.001)}


def test_a_mode_run_hands_the_default_stop_to_a_problem_that_states_none(climbs, tmp_path):
    problem = resolve_problem(ROOT, "lowest-eigenvector/lj38")
    result = run_template(ROOT, problem, load_template(ROOT, "rgsaddle/mode-dimer"), output=tmp_path / "mode.dat", work=tmp_path / "scratch", limit=2)
    assert result["starts"] == 2
    assert {(spec["task"], spec["force_tol"], spec["cap"]) for spec in climbs} == {("mode", 0.001, 10000)}


def test_the_scratch_folder_defaults_to_one_named_for_the_template_beside_the_output(climbs, tmp_path):
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    run_template(ROOT, problem, load_template(ROOT, "rgsaddle/dimer-lbfgs"), output=tmp_path / "out" / "lj38.dat", limit=1)
    assert (tmp_path / "out" / "work" / "rgsaddle-dimer-lbfgs" / "geometries" / "coords-con" / "0000.con").is_file()


def test_a_problem_with_no_catalogue_has_no_repetition_to_select(climbs, tmp_path):
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    with pytest.raises(RuntimeError, match="tsopt/LJ38 has no repetitions to select"):
        run_template(ROOT, problem, load_template(ROOT, "rgsaddle/dimer-lbfgs"), output=tmp_path / "x.dat", work=tmp_path / "w", repetition=1)
    assert climbs == []


def test_a_problem_that_counts_a_different_number_of_starts_than_it_has_is_refused(climbs, tmp_path):
    problem = copy.deepcopy(resolve_problem(ROOT, "tsopt/LJ38"))
    problem["sample"]["starts"] = 199
    with pytest.raises(RuntimeError, match="tsopt/LJ38 has 200 geometries, the problem states 199"):
        run_template(ROOT, problem, load_template(ROOT, "rgsaddle/dimer-lbfgs"), output=tmp_path / "x.dat", work=tmp_path / "w")
    assert climbs == []


def test_a_run_from_a_source_at_another_commit_stops_before_it_climbs(climbs, monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "commit", "--quiet", "--allow-empty", "-m", "start")
    monkeypatch.setenv("RGSADDLE_SRC", str(source))
    problem = resolve_problem(ROOT, "tsopt/LJ38")
    with pytest.raises(RuntimeError, match="does not match the template commit"):
        run_template(ROOT, problem, load_template(ROOT, "rgsaddle/dimer-lbfgs"), output=tmp_path / "x.dat", work=tmp_path / "w", limit=1)
    assert climbs == []
    assert not (tmp_path / "x.dat").exists()
