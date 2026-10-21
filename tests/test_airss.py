"""AIRSS cell generation and the harness two-point step."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_airss import AirssLj38, selected_starts
from optbench.potentials import lennard_jones
from optbench.runners.airss_go import (
    TARBALL_SHA256,
    cell_text,
    find_buildcell,
    load_wales_coordinates,
    positions_from_cell,
    reached_minimum,
    reference_energy,
    relax_positions,
    seed_words,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wales_lj38_energy_is_the_published_minimum():
    coords = load_wales_coordinates()
    assert coords.shape == (38, 3)
    energy = reference_energy(coords)
    assert energy == pytest.approx(-173.92842631234794, abs=1e-9)
    assert energy == pytest.approx(-173.9284263, abs=5e-7)
    direct, _gradient = lennard_jones(coords)
    assert direct == pytest.approx(energy)


def test_energy_window_accepts_the_minimum_and_rejects_the_icosahedron():
    reference = reference_energy()
    assert reached_minimum(reference, reference, 0.001)
    assert reached_minimum(reference - 1e-4, reference, 0.001)
    assert reached_minimum(reference + 0.001, reference, 0.001)
    assert reached_minimum(reference + 0.001 + 1e-9, reference, 0.001) is False
    assert reached_minimum(-173.252378, reference, 0.001) is False
    assert reached_minimum(float("nan"), reference, 0.001) is False


def test_seed_words_are_eight_positive_integers_and_depend_on_the_start():
    first = seed_words(0, 0)
    assert len(first) == 8
    assert all(1 <= word < 2147483647 for word in first)
    assert seed_words(0, 0) == first
    assert seed_words(1, 0) != first
    assert seed_words(0, 1) != first


def test_cell_text_carries_the_example_directives_and_eight_seeds():
    text = cell_text(
        natoms=38,
        box=100,
        posamp=1.875,
        minsep=0.75,
        symmops="2-5",
        nform=1,
        seed=seed_words(3, 4),
    )
    assert "#CLUSTER" in text
    assert "#POSAMP=1.875" in text
    assert "#MINSEP=0.75" in text
    assert "#SYMMOPS=2-5" in text
    assert "#NFORM=1" in text
    assert "NUM=38" in text
    seed_line = next(line for line in text.splitlines() if line.startswith("#SEED="))
    assert len(seed_line.split("=", 1)[1].split()) == 8


def test_positions_from_cell_scales_fractional_coordinates():
    text = """%BLOCK LATTICE_ABC
    100.00000    100.00000    100.00000
     90.00000     90.00000     90.00000
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_FRAC
  Al        0.5000000        0.5000000        0.5000000
  Al        0.5100000        0.5000000        0.5000000
%ENDBLOCK POSITIONS_FRAC
"""
    pos = positions_from_cell(text, 2)
    assert pos == pytest.approx(np.array([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0]]))


def test_two_point_step_relaxes_a_dimer_to_the_pair_minimum():
    pos = np.array([[50.0, 50.0, 50.0], [51.5, 50.0, 50.0]])
    point, energy, calls, converged = relax_positions(pos, 100.0)
    distance = float(np.linalg.norm(point[0] - point[1]))
    assert converged
    assert calls > 0
    assert energy == pytest.approx(-1.0, abs=1e-8)
    assert distance == pytest.approx(2.0 ** (1.0 / 6.0), abs=1e-5)


def test_two_point_step_keeps_the_wales_minimum_inside_the_window():
    coords = load_wales_coordinates()
    point = coords - coords.mean(axis=0) + 50.0
    _point, energy, calls, converged = relax_positions(point, 100.0)
    reference = reference_energy(coords)
    assert converged
    assert calls < 500
    assert reached_minimum(energy, reference, 0.001)
    assert energy == pytest.approx(reference, abs=1e-3)


def test_template_pins_the_tarball_and_the_asv_grid_is_one_start(monkeypatch, tmp_path):
    import tomllib

    template = tomllib.loads((ROOT / "templates" / "airss" / "lj38.toml").read_text(encoding="utf-8"))
    assert template["commit"] == TARBALL_SHA256
    assert template["engine"] == "airss"
    assert template["problem"] == "globalopt/lj38"
    assert template["airss"]["posamp"] == 1.875
    assert template["airss"]["symmops"] == "2-5"
    assert AirssLj38.template_id == "airss/lj38"
    assert AirssLj38.problem_id == "globalopt/lj38"
    assert AirssLj38.param_names == ["seed", "start"]
    assert AirssLj38.params[0] == ["0"]
    assert AirssLj38.params[1] == ["0"]
    assert AirssLj38.cap == 2000000
    assert AirssLj38.repeat == 1
    assert AirssLj38.track_force_calls.unit == "force calls"
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert selected_starts() == [str(index) for index in range(100)]
    monkeypatch.delenv("AIRSS_BUILDCELL", raising=False)
    import optbench.runners.airss_go as airss

    monkeypatch.setattr(airss.shutil, "which", lambda _name: None)
    monkeypatch.setattr(airss.Path, "home", lambda: tmp_path)
    with pytest.raises(NotImplementedError):
        AirssLj38().setup("0", "0")


def test_lj75_budget_writes_partial_rows(tmp_path, monkeypatch):
    import json

    import optbench.runners.airss_go as airss
    from optbench.campaign import run

    def fake(job):
        return airss._capped(job, calls=100 + job["index"], energy=-300.0 - job["index"], structures=20)

    monkeypatch.setattr(airss, "search_start", fake)
    monkeypatch.setenv("AIRSS_BUILDCELL", "/bin/true")
    output = tmp_path / "lj75.dat"
    summary = run(
        ROOT,
        "globalopt/lj75",
        "airss/lj75",
        output=output,
        workers=1,
        work=tmp_path / "work",
    )
    text = output.read_text(encoding="utf-8")
    assert "partial true" in text
    assert "cap 2000000" in text
    assert "n 200" in text
    assert summary["starts"] == 200
    assert summary["nfailed"] == 200
    assert summary["par2_mean"] == 4000000
    rows = [
        json.loads(line)
        for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in rows[:2]] == ["coords.1", "coords.2"]
    assert rows[-1]["start"] == "coords.200"
    assert rows[0]["energy"] == -300.0
    assert rows[-1]["energy"] == -499.0
    assert all(row["success"] is False and row["failure"] == "cap" and row["counter"] == "harness" for row in rows)


def test_buildcell_returns_38_atoms_for_a_fixed_seed():
    try:
        binary = find_buildcell()
    except FileNotFoundError:
        pytest.skip("buildcell is not installed")
    from optbench.runners.airss_go import build_positions

    text = cell_text(
        natoms=38,
        box=100,
        posamp=1.875,
        minsep=0.75,
        symmops="2-5",
        nform=1,
        seed=seed_words(0, 0),
    )
    pos = build_positions(binary, text, 38)
    assert pos.shape == (38, 3)
    assert np.all(pos >= 0.0)
    assert np.all(pos < 100.0)
    again = build_positions(binary, text, 38)
    assert again == pytest.approx(pos)
