"""A LAMMPS climbing-image band, judged from the files a run leaves behind.

LAMMPS is replaced at the process boundary: the stand-in launcher writes the
logs and the final dump that ``mpirun`` would, and the harness counts pair
computes from the stats blocks, reads the climbing replica, and judges its
geometry with the pair potential. Geometries are a pinned pair around one
free atom, whose Hessian index follows from the pair curvature, and the real
LJ38 minima.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from optbench import lammps_neb
from optbench.campaign import geometry_archive, load_template, resolve_problem
from optbench.lammps_neb import (
    COMMIT,
    _box_bounds,
    _coords_from_dump,
    _dat,
    _error_line,
    _last_pe,
    _number,
    _read_dump,
    _registry,
    _settings,
    _success_mean,
    calls_from_blocks,
    iteration_budget,
    lammps_executable,
    minimization_blocks,
    publish_sidecars,
    render_neb_input,
    run,
    search_geometry,
    write_outputs,
)
from optbench.pes import load_pairs

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = "double-ended-saddle-search/lj38"
TEMPLATE = "lammpsneb/ci-fire"
STAGE_ONE = 11
STAGE_TWO = 7
PER_REPLICA = (1 + STAGE_ONE + 1) + (1 + STAGE_TWO + 1)


def _settings_for(problem_name=PROBLEM, template_name=TEMPLATE) -> dict:
    return _settings(
        load_template(ROOT, template_name),
        resolve_problem(ROOT, problem_name),
        _registry(ROOT)["commit"],
    )


def _stats(stop: str, neval: int) -> str:
    return (
        "Minimization stats:\n"
        f"  Stopping criterion = {stop}\n"
        f"  Energy initial, next-to-last, final = \n         -1.0 -1.0 -1.0\n"
        f"  Force two-norm initial, final = 1.0 0.001\n"
        f"  Iterations, force evaluations = {neval + 1} {neval}\n"
    )


def _replica_log(blocks) -> str:
    return "LAMMPS (30 Sep 2026)\n" + "\n".join(_stats(stop, neval) for stop, neval in blocks)


def _dump(positions, free=None) -> str:
    rows = [
        "ITEM: TIMESTEP",
        "0",
        "ITEM: NUMBER OF ATOMS",
        str(len(positions)),
        "ITEM: BOX BOUNDS ff ff ff",
        "-100 100",
        "-100 100",
        "-100 100",
        "ITEM: ATOMS id type xu yu zu",
    ]
    for index, (x, y, z) in enumerate(positions, start=1):
        kind = 1 if free is None or free[index - 1] else 2
        rows.append(f"{index} {kind} {x:.17g} {y:.17g} {z:.17g}")
    return "\n".join(rows) + "\n"


def _trio(half_gap: float, drift: float = 0.0) -> np.ndarray:
    return np.array([[-half_gap, 0.0, 0.0], [0.0, drift, 0.0], [half_gap, 0.0, 0.0]])


FREE = np.array([False, True, False])
REACTANT = np.array([[-1.5, 0.0, 0.0], [0.0, -0.3, 0.0], [1.5, 0.0, 0.0]])
PRODUCT = np.array([[-1.5, 0.0, 0.0], [0.0, 0.3, 0.0], [1.5, 0.0, 0.0]])


class Launcher:
    """Stands in for ``mpirun`` + LAMMPS. Every field is what one run leaves behind."""

    def __init__(
        self,
        *,
        climber=None,
        replica=4,
        blocks=None,
        returncode=0,
        skip_logs=(),
        universe=None,
        dump=None,
    ):
        self.climber = climber
        self.replica = replica
        self.blocks = blocks if blocks is not None else [("force tolerance", STAGE_ONE), ("force tolerance", STAGE_TWO)]
        self.returncode = returncode
        self.skip_logs = set(skip_logs)
        self.universe = universe
        self.dump = dump
        self.launched: list[dict] = []

    def __call__(self, binary, work, n_images, timeout):
        self.launched.append(
            {
                "n_images": n_images,
                "files": {
                    name: (work / name).read_text(encoding="utf-8")
                    for name in ("start.data", "final.neb", "in.neb")
                },
            }
        )
        for index in range(n_images):
            if index not in self.skip_logs:
                (work / f"log.lammps.{index}").write_text(_replica_log(self.blocks), encoding="utf-8")
        text = self.universe
        if text is None:
            text = f"Climbing replica = {self.replica}\n" if self.replica is not None else ""
        (work / "log.lammps").write_text(text, encoding="utf-8")
        if self.climber is not None and self.replica is not None:
            body = self.dump if self.dump is not None else _dump(self.climber, FREE)
            (work / f"final.{self.replica - 1}.dump").write_text(body, encoding="utf-8")
        return self.returncode


@pytest.fixture
def launch(monkeypatch):
    def install(launcher: Launcher) -> Launcher:
        monkeypatch.setattr(lammps_neb, "_launch_neb", launcher)
        return launcher

    return install


def _band(tmp_path, launcher: Launcher, *, settings=None, reactant=REACTANT, product=PRODUCT, free=FREE) -> dict:
    return search_geometry(
        name="trio",
        reactant=reactant,
        product=product,
        free=free,
        box=None,
        settings=settings or _settings_for(),
        n1=140,
        n2=141,
        binary=Path("lmp"),
        work=tmp_path / "band",
    )


def test_a_band_that_climbs_onto_an_index_one_saddle_is_a_success(tmp_path, launch):
    launcher = launch(Launcher(climber=_trio(1.5)))
    record = _band(tmp_path, launcher)
    assert record["success"] is True
    assert record["failure"] is None
    assert record["reasons"] == []
    assert record["hessian_index"] == 1
    assert record["force_norm"] == pytest.approx(0.0, abs=1e-9)
    assert record["converged"] is True
    assert record["climbing_replica"] == 4
    assert record["force_calls"] == 7 * PER_REPLICA
    assert record["verify_calls"] == 6
    assert (record["n1"], record["n2"], record["n_images"]) == (140, 141, 7)
    assert record["returncode"] == 0
    assert record["schema"] == "optbench.start/v2"


def test_the_band_is_handed_over_as_a_data_file_a_final_image_and_an_input_script(tmp_path, launch):
    launcher = launch(Launcher(climber=_trio(1.5)))
    _band(tmp_path, launcher)
    (seen,) = launcher.launched
    assert seen["n_images"] == 7
    data = seen["files"]["start.data"].splitlines()
    assert "3 atoms" in data
    assert "2 atom types" in data
    atoms = data[data.index("Atoms") + 2 : data.index("Atoms") + 5]
    assert atoms[1].split() == ["2", "1", "0", "-0.3", "0"]
    assert [row.split()[1] for row in atoms] == ["2", "1", "2"]
    assert seen["files"]["final.neb"].splitlines()[:2] == ["1", "2 0 0.3 0"]
    script = seen["files"]["in.neb"].splitlines()
    assert "neb 0.0 0.001 140 141 1 final final.neb" in script
    assert "fix freeze frozen setforce 0.0 0.0 0.0" in script
    assert "min_style fire" in script
    assert "boundary f f f" in script


def test_pair_computes_are_counted_from_the_stats_blocks_of_every_replica(tmp_path, launch):
    ended = launch(Launcher(climber=_trio(1.5), blocks=[("max iterations", 20), ("force tolerance", 5)]))
    record = _band(tmp_path, ended)
    assert record["force_calls"] == 7 * ((1 + 20) + (1 + 5 + 1))
    assert calls_from_blocks([("max iterations", 20), ("force tolerance", 5)]) == 28


def test_a_stationary_point_of_the_wrong_index_is_not_the_saddle(tmp_path, launch):
    record = _band(tmp_path, launch(Launcher(climber=_trio(1.2))))
    assert record["success"] is False
    assert record["failure"] == "wrong_stationary"
    assert record["reasons"] == ["hessian_index"]
    assert record["hessian_index"] == 0
    second = _band(tmp_path / "second", launch(Launcher(climber=_trio(1.0))))
    assert second["failure"] == "wrong_stationary"
    assert second["hessian_index"] == 2


def test_a_band_that_stops_with_force_left_is_a_divergence_and_no_hessian_is_taken(tmp_path, launch):
    record = _band(tmp_path, launch(Launcher(climber=_trio(1.5, 0.1))))
    assert record["failure"] == "divergence"
    assert record["reasons"] == ["force_norm"]
    assert record["force_norm"] > 0.001
    assert record["hessian_index"] is None
    assert record["verify_calls"] == 0


def test_a_band_that_spent_the_budget_is_a_cap_whatever_it_reached(tmp_path, launch):
    settings = dict(_settings_for(), cap=100)
    record = _band(tmp_path, launch(Launcher(climber=_trio(1.5))), settings=settings)
    assert record["failure"] == "cap"
    assert record["reasons"] == ["cap"]
    assert record["success"] is False
    assert record["force_calls"] == 7 * PER_REPLICA
    assert record["hessian_index"] is None
    far = _band(tmp_path / "far", launch(Launcher(climber=_trio(1.5, 0.1))), settings=settings)
    assert far["failure"] == "cap"
    assert far["reasons"] == ["cap", "force_norm"]


@pytest.mark.parametrize(
    ("launcher", "expected"),
    [
        (Launcher(climber=_trio(1.5), skip_logs=(3,)), ""),
        (Launcher(climber=_trio(1.5), blocks=[("force tolerance", STAGE_ONE)]), ""),
        (Launcher(climber=_trio(1.5), replica=None), ""),
        (Launcher(climber=_trio(1.5), replica=8), ""),
        (Launcher(climber=_trio(1.5), replica=0), ""),
        (Launcher(climber=_trio(1.5), returncode=1, universe="Climbing replica = 4\nERROR: Lost atoms: original 3\n"), "ERROR: Lost atoms: original 3"),
        (Launcher(climber=None, replica=4), ""),
    ],
    ids=["a-replica-log-is-missing", "a-stage-is-missing", "no-climbing-replica", "replica-past-the-last", "replica-zero", "lammps-errored", "no-final-dump"],
)
def test_a_band_that_is_not_complete_is_a_crash(tmp_path, launch, launcher, expected):
    record = _band(tmp_path, launch(launcher))
    assert record["success"] is False
    assert record["failure"] == "crash"
    assert record["reasons"][0] == "crash"
    assert record["force_norm"] is None
    assert record["hessian_index"] is None
    if expected:
        assert record["error"] == expected
        assert record["returncode"] == 1


def test_a_final_dump_with_the_wrong_atoms_names_the_file_and_the_counts(tmp_path, launch):
    short = _dump(_trio(1.5)[:2])
    record = _band(tmp_path, launch(Launcher(climber=_trio(1.5), dump=short)))
    assert record["failure"] == "crash"
    assert record["error"] == "final.3.dump has 2 atoms, the data file has 3"


def test_a_run_that_outlives_the_timeout_is_reported_as_one(tmp_path, launch):
    record = _band(tmp_path, launch(Launcher(climber=_trio(1.5), returncode=None)))
    assert record["failure"] == "crash"
    assert record["error"] == "timeout"
    assert record["returncode"] is None


def test_endpoints_that_do_not_agree_are_refused_before_anything_runs(tmp_path, launch):
    launcher = launch(Launcher(climber=_trio(1.5)))
    with pytest.raises(ValueError, match="trio endpoints do not match"):
        _band(tmp_path, launcher, product=PRODUCT[:2])
    with pytest.raises(ValueError, match="trio has no free atoms"):
        _band(tmp_path, launcher, free=np.zeros(3, dtype=bool))
    moved = PRODUCT.copy()
    moved[0, 0] += 0.01
    with pytest.raises(ValueError, match=r"trio moves a fixed atom by 0\.0"):
        _band(tmp_path, launcher, product=moved)
    assert launcher.launched == []


def test_a_search_or_a_probe_without_a_binary_says_so(monkeypatch):
    monkeypatch.setattr(lammps_neb, "lammps_executable", lambda: None)
    with pytest.raises(FileNotFoundError, match="LAMMPS binary is not available"):
        search_geometry(
            name="trio",
            reactant=REACTANT,
            product=PRODUCT,
            free=FREE,
            box=None,
            settings=_settings_for(),
            n1=2,
            n2=2,
        )
    with pytest.raises(FileNotFoundError, match="LAMMPS binary is not available"):
        lammps_neb.probe_pair("lj", REACTANT, None)


def test_the_binary_comes_from_the_environment_then_the_default_then_the_path(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen-lmp"
    chosen.write_text("#!/bin/sh\n", encoding="utf-8")
    default = tmp_path / "default-lmp"
    monkeypatch.setattr(lammps_neb, "_DEFAULT_BIN", default)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("LAMMPS_BIN", raising=False)
    assert lammps_executable() is None

    on_path = tmp_path / "bin"
    on_path.mkdir()
    found = on_path / "lmp"
    found.write_text("#!/bin/sh\n", encoding="utf-8")
    found.chmod(found.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(on_path))
    assert lammps_executable() == found

    default.write_text("#!/bin/sh\n", encoding="utf-8")
    assert lammps_executable() == default

    monkeypatch.setenv("LAMMPS_BIN", str(chosen))
    assert lammps_executable() == chosen
    monkeypatch.setenv("LAMMPS_BIN", str(tmp_path / "missing-lmp"))
    assert lammps_executable() is None


def test_an_iteration_budget_that_cannot_fit_the_cap_is_refused():
    with pytest.raises(ValueError, match="NEB needs at least two replicas"):
        iteration_budget(1, 2500)
    with pytest.raises(ValueError, match=r"call reserve 1\.0 is outside \[0, 1\)"):
        iteration_budget(7, 2500, reserve=1.0)
    with pytest.raises(ValueError, match=r"call reserve -0\.1 is outside"):
        iteration_budget(7, 2500, reserve=-0.1)
    with pytest.raises(ValueError, match="cap 20 cannot fund two stages on 7 replicas"):
        iteration_budget(7, 20)


def test_a_stats_block_without_a_stopping_criterion_is_refused():
    with pytest.raises(ValueError, match="force evaluations appear before a stopping criterion"):
        minimization_blocks("Iterations, force evaluations = 12 11\n")
    assert minimization_blocks("nothing to count\n") == []


def test_a_template_is_checked_against_the_registry_and_the_problem():
    template = load_template(ROOT, TEMPLATE)
    problem = resolve_problem(ROOT, PROBLEM)
    commit = _registry(ROOT)["commit"]
    assert commit == COMMIT
    with pytest.raises(ValueError, match="commit does not match the lammpsneb registry"):
        _settings(template, problem, "0" * 40)
    with pytest.raises(ValueError, match="engine is geodesic"):
        _settings(dict(template, engine="geodesic"), problem, commit)
    band = dict(template["band"])
    with pytest.raises(ValueError, match="is not a climbing-image run"):
        _settings(dict(template, band={**band, "climbing": False}), problem, commit)
    with pytest.raises(ValueError, match="min_style steepest"):
        _settings(dict(template, band={**band, "min_style": "steepest"}), problem, commit)
    with pytest.raises(ValueError, match="parallel spring sideways"):
        _settings(dict(template, band={**band, "parallel": "sideways"}), problem, commit)
    with pytest.raises(ValueError, match="potential eam"):
        _settings(template, dict(problem, band={**problem["band"], "potential": "eam"}), commit)


def test_the_settings_follow_the_template_and_the_problem():
    fire = _settings_for(PROBLEM, "lammpsneb/ci-fire")
    assert fire["call_reserve"] == 0.2
    assert fire["n_images"] == fire["n_interior"] + 2 == 7
    assert fire["units"] == "lj"
    assert fire["timestep"] == 0.005
    quick = _settings_for("neb/Pt-heptamer", "lammpsneb/ci-quickmin")
    assert quick["call_reserve"] == 0.0
    assert quick["units"] == "metal"
    assert quick["timestep"] == 0.001
    assert quick["potential"] == "morse_pt"


def test_a_lennard_jones_box_pads_both_endpoints_and_a_slab_box_must_hold_the_cutoff():
    low, high = _box_bounds("lj", REACTANT, PRODUCT, None)
    assert low == pytest.approx([-101.5, -100.3, -100.0])
    assert high == pytest.approx([101.5, 100.3, 100.0])
    lo, hi = _box_bounds("morse_pt", REACTANT, PRODUCT, np.array([21.0, 22.0, 30.0]))
    assert lo.tolist() == [0.0, 0.0, 0.0]
    assert hi.tolist() == [21.0, 22.0, 30.0]
    with pytest.raises(ValueError, match="morse_pt needs the orthorhombic cell"):
        _box_bounds("morse_pt", REACTANT, PRODUCT, None)
    with pytest.raises(ValueError, match=r"Morse cutoff 9\.5 does not fit the cell \[18\.0, 22\.0, 30\.0\]"):
        _box_bounds("morse_pt", REACTANT, PRODUCT, np.array([18.0, 22.0, 30.0]))


def test_a_slab_band_is_periodic_and_a_cluster_band_is_open():
    cluster = render_neb_input(_settings_for(), n1=140, n2=141, frozen=False)
    assert "boundary f f f" in cluster
    assert "group frozen type 2" not in cluster
    slab = render_neb_input(_settings_for("neb/Pt-heptamer", "lammpsneb/ci-quickmin"), n1=2, n2=3, frozen=True)
    assert "boundary p p p" in slab
    assert "group frozen type 2" in slab
    assert "pair_modify shift yes" in slab
    assert "neb 0.0 0.01 2 3 1 final final.neb" in slab.splitlines()


def test_a_dump_gives_each_atom_by_id_whatever_order_it_lists_them(tmp_path):
    path = tmp_path / "forces.dump"
    path.write_text(
        "ITEM: TIMESTEP\n0\nITEM: ATOMS id type fx fy fz\n2 1 0.5 0.25 0.125\n1 1 -1 -2 -3\nITEM: TIMESTEP\n1\n",
        encoding="utf-8",
    )
    atoms = _read_dump(path, ("fx", "fy", "fz"))
    assert atoms[1].tolist() == [-1.0, -2.0, -3.0]
    assert atoms[2].tolist() == [0.5, 0.25, 0.125]
    dump = tmp_path / "final.dump"
    dump.write_text(_dump([(3.0, 2.0, 1.0), (6.0, 5.0, 4.0)]), encoding="utf-8")
    assert _coords_from_dump(dump, 2).tolist() == [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]
    with pytest.raises(ValueError, match="final.dump has 2 atoms, the data file has 3"):
        _coords_from_dump(dump, 3)
    dump.write_text(_dump([(3.0, 2.0, 1.0), (6.0, 5.0, 4.0)]).replace("\n2 1 ", "\n7 1 "), encoding="utf-8")
    with pytest.raises(ValueError, match="has 2 atoms"):
        _coords_from_dump(dump, 2)


def test_the_last_energy_in_a_log_is_the_one_a_probe_reports():
    log = "\n".join(
        [
            "Step PotEng",
            "       0   -173.2523784",
            "Loop time of 1e-06 on 1 procs for 0 steps",
            "       1   -173.2523785",
            "tail",
        ]
    )
    assert _last_pe(log) == -173.2523785
    with pytest.raises(ValueError, match="LAMMPS log has no potential energy"):
        _last_pe("Step PotEng\nno numbers here\n")


def test_the_first_error_or_lost_atoms_line_is_what_a_failed_run_reports():
    log = "setting up\nWARNING: a thing\nERROR: Out of range atoms - cannot compute PPPM (src/x.cpp:1)\nERROR: later\n"
    assert _error_line(log) == "ERROR: Out of range atoms - cannot compute PPPM (src/x.cpp:1)"
    assert _error_line("step 1\nLost atoms: original 38 current 37\n") == "Lost atoms: original 38 current 37"
    assert _error_line("all fine\n") == ""


def test_numbers_print_whole_when_they_are_whole():
    assert _number(3.0) == "3"
    assert _number(2.5) == "2.5"
    assert _number(1.0 / 3.0) == "0.3333333333"


def _record(start: str, calls: int, *, success: bool, failure=None, index=None) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": start,
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": [] if success else [failure],
        "energy": None,
        "force_norm": None,
        "hessian_index": index,
        "converged": success,
        "climbing_replica": 4,
        "n1": 140,
        "n2": 141,
        "n_images": 7,
        "returncode": 0,
        "error": "",
        "seed": "0",
        "problem": PROBLEM,
        "template": TEMPLATE,
        "code": "LAMMPS",
        "code_version": "30 Sep 2026",
        "commit": COMMIT,
        "algorithm": "NEB",
        "tuning": "default",
    }


def test_the_dat_reports_the_round_half_even_mean_of_the_successes_and_counts_the_rest():
    template = load_template(ROOT, TEMPLATE)
    problem = resolve_problem(ROOT, PROBLEM)
    settings = _settings_for()
    records = [
        _record("pair_00", 100, success=True, index=1),
        _record("pair_01", 105, success=True, index=1),
        _record("pair_02", 2500, success=False, failure="cap"),
        _record("pair_03", 90, success=False, failure="wrong_stationary", index=0),
    ]
    lines = _dat(records, template, settings, problem).splitlines()
    assert "force_calls 102" in lines
    assert "force_calls_min 100" in lines
    assert "force_calls_max 105" in lines
    assert "force_calls_median 102.5" in lines
    assert "force_calls_per_image 20" in lines
    assert "call_unit total" in lines
    assert "successes 2" in lines
    assert "nfailed 2" in lines
    assert "n 4" in lines
    assert "transition_states 2" in lines
    assert "counter harness" in lines
    assert f"problem {PROBLEM}" in lines
    assert f"template {TEMPLATE}" in lines
    assert "images 5" in lines


def test_a_dat_with_no_success_states_no_call_counts():
    template = load_template(ROOT, TEMPLATE)
    problem = resolve_problem(ROOT, PROBLEM)
    records = [_record("pair_00", 2500, success=False, failure="cap")]
    lines = _dat(records, template, _settings_for(), problem).splitlines()
    assert not [line for line in lines if line.startswith("force_calls")]
    assert "successes 0" in lines
    assert "nfailed 1" in lines


def test_the_success_mean_keeps_whole_numbers_and_the_median_as_it_is():
    records = [
        _record("pair_00", 100, success=True, index=1),
        _record("pair_01", 101, success=True, index=1),
        _record("pair_02", 7, success=False, failure="cap"),
    ]
    summary = _success_mean(records)
    assert summary == {
        "successes": 2,
        "nfailed": 1,
        "n": 3,
        "transition_states": 2,
        "force_calls": 100,
        "force_calls_min": 100,
        "force_calls_max": 101,
        "force_calls_median": 100.5,
    }
    assert _success_mean([_record("pair_02", 7, success=False, failure="cap")]) == {
        "successes": 0,
        "nfailed": 1,
        "n": 1,
        "transition_states": 0,
    }


def _pair_names() -> list[str]:
    problem = resolve_problem(ROOT, PROBLEM)
    return [pair.name for pair in load_pairs(geometry_archive(ROOT, problem), "xyz-pairs", 50)]


def test_the_outputs_are_a_dat_a_starts_file_and_a_v2_score(tmp_path):
    template = load_template(ROOT, TEMPLATE)
    problem = resolve_problem(ROOT, PROBLEM)
    records = [
        _record("pair_00", 100, success=True, index=1),
        _record("pair_01", 2500, success=False, failure="cap"),
    ]
    summary = write_outputs(tmp_path / "out" / "band.dat", records, template, _settings_for(), problem)
    assert summary == {
        "problem": PROBLEM,
        "template": TEMPLATE,
        "force_calls": 100,
        "nfailed": 1,
        "starts": 2,
        "successes": 1,
        "output": str(tmp_path / "out" / "band.dat"),
    }
    document = json.loads((tmp_path / "out" / "band.v2.json").read_text(encoding="utf-8"))
    assert document["schema"] == "optbench.score/v2"
    assert document["problem"] == PROBLEM
    assert document["counter"] == "harness"
    assert document["success_mean"]["successes"] == 1
    assert document["score"]["n_cap"] == 1
    assert document["provenance"]["parameters"]["n1"] == 140
    assert document["provenance"]["parameters"]["n2"] == 141
    starts = (tmp_path / "out" / "band.starts.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["start"] for line in starts] == ["pair_00", "pair_01"]


def test_sidecars_are_reduced_in_the_order_of_the_problem_and_must_be_complete(tmp_path):
    names = _pair_names()
    assert len(names) == 50
    sidecars = tmp_path / "sidecars"
    for index, name in reversed(list(enumerate(names))):
        record = _record(name, 100 + index, success=index % 2 == 0, failure=None if index % 2 == 0 else "cap", index=1 if index % 2 == 0 else None)
        (sidecars / "a" / "b").mkdir(parents=True, exist_ok=True)
        (sidecars / "a" / "b" / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    out = tmp_path / "published"
    (summary,) = publish_sidecars(ROOT, sidecars, out)
    assert summary["starts"] == 50
    assert summary["successes"] == 25
    assert Path(summary["output"]).name == "lammps-30Sep2026-double-ended-saddle-search-lj38-ci-fire.dat"
    starts = [
        json.loads(line)["start"]
        for line in Path(summary["output"]).with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert starts == names

    (sidecars / "a" / "b" / f"{names[7]}.json").unlink()
    with pytest.raises(ValueError, match=f"{TEMPLATE} on {PROBLEM} has 49 records, the problem has 50 pairs"):
        publish_sidecars(ROOT, sidecars, tmp_path / "again")


def test_a_directory_with_no_sidecars_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no sidecar records in"):
        publish_sidecars(ROOT, tmp_path, tmp_path / "out")


def _atoms_of(data: str) -> list[tuple[float, float, float]]:
    lines = data.splitlines()
    start = lines.index("Atoms") + 2
    rows = []
    for line in lines[start:]:
        if not line.strip():
            break
        _index, _kind, x, y, z = line.split()
        rows.append((float(x), float(y), float(z)))
    return rows


class MinimumLauncher(Launcher):
    """A band that ends on the reactant it started from."""

    def __call__(self, binary, work, n_images, timeout):
        data = (work / "start.data").read_text(encoding="utf-8")
        self.climber = _atoms_of(data)
        self.dump = _dump(self.climber)
        return super().__call__(binary, work, n_images, timeout)


def test_a_run_writes_one_sidecar_per_start_and_a_resume_runs_only_what_is_left(tmp_path, monkeypatch, capsys):
    launcher = MinimumLauncher()
    monkeypatch.setattr(lammps_neb, "_launch_neb", launcher)
    monkeypatch.setattr(lammps_neb, "lammps_executable", lambda: Path("lmp"))
    output = tmp_path / "band.dat"
    first = run(ROOT, PROBLEM, TEMPLATE, output=output, limit=2)
    assert len(launcher.launched) == 2
    assert first["starts"] == 2
    assert first["successes"] == 0
    assert first["force_calls"] is None
    parts = sorted(path.name for path in output.with_suffix(".parts").glob("*.json"))
    assert parts == ["pair_00.json", "pair_01.json"]
    sidecar = json.loads((output.with_suffix(".parts") / "pair_00.json").read_text(encoding="utf-8"))
    assert sidecar["failure"] == "wrong_stationary"
    assert sidecar["hessian_index"] == 0
    assert sidecar["force_calls"] == 7 * PER_REPLICA
    assert sidecar["problem"] == PROBLEM
    assert sidecar["template"] == TEMPLATE
    assert sidecar["seed"] == "0"
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith(f"1/2 {TEMPLATE} pair_00 calls={7 * PER_REPLICA} success=False failure=wrong_stationary")

    second = run(ROOT, PROBLEM, TEMPLATE, output=output, limit=3, resume=True)
    assert len(launcher.launched) == 3
    assert second["starts"] == 3
    records = [
        json.loads(line)
        for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in records] == ["pair_00", "pair_01", "pair_02"]

    run(ROOT, PROBLEM, TEMPLATE, output=output, limit=3)
    assert len(launcher.launched) == 6


def test_a_named_start_that_the_problem_lacks_is_a_key_error():
    with pytest.raises(KeyError, match=f"{PROBLEM} has no start pair_99"):
        lammps_neb.search_named(ROOT, PROBLEM, TEMPLATE, "pair_99")


def test_a_start_that_cannot_run_is_recorded_as_a_crash_with_the_reason(tmp_path, monkeypatch):
    def broken(binary, work, n_images, timeout):
        raise RuntimeError("mpirun is not here")

    monkeypatch.setattr(lammps_neb, "_launch_neb", broken)
    monkeypatch.setattr(lammps_neb, "lammps_executable", lambda: Path("lmp"))
    record = lammps_neb.search_named(ROOT, PROBLEM, TEMPLATE, "pair_00", seed="3", n1=2, n2=2, work=tmp_path)
    assert record["success"] is False
    assert record["failure"] == "crash"
    assert record["error"] == "RuntimeError: mpirun is not here"
    assert record["seed"] == "3"
    assert (record["n1"], record["n2"]) == (2, 2)
    assert record["template"] == TEMPLATE


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def mpirun_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "mpi-bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]))

    def install(body: str) -> Path:
        return _script(bin_dir / "mpirun", body)

    return install


def _launch_band(tmp_path, **kwargs) -> dict:
    return search_geometry(
        name="trio",
        reactant=REACTANT,
        product=PRODUCT,
        free=FREE,
        box=None,
        settings=_settings_for(),
        n1=140,
        n2=141,
        binary=Path("lmp"),
        work=tmp_path / "band",
        **kwargs,
    )


def test_mpirun_gets_one_rank_per_replica_one_thread_each_and_the_partition_flags(tmp_path, mpirun_on_path):
    mpirun_on_path('printf "%s\\n" "$@" > argv.txt\nprintf "%s\\n" "$OMP_NUM_THREADS" > threads.txt\n')
    record = _launch_band(tmp_path, timeout=30.0)
    work = tmp_path / "band"
    assert (work / "argv.txt").read_text(encoding="utf-8").splitlines() == [
        "--oversubscribe",
        "--bind-to",
        "none",
        "-np",
        "7",
        "lmp",
        "-partition",
        "7x1",
        "-in",
        "in.neb",
        "-log",
        "log.lammps",
        "-screen",
        "none",
    ]
    assert (work / "threads.txt").read_text(encoding="utf-8").strip() == "1"
    assert record["failure"] == "crash"
    assert record["returncode"] == 0


def test_what_the_launcher_prints_before_it_fails_is_the_error_of_the_start(tmp_path, mpirun_on_path):
    mpirun_on_path('echo "ERROR: Lost atoms: original 3 current 2"\nexit 3\n')
    record = _launch_band(tmp_path, timeout=30.0)
    assert record["failure"] == "crash"
    assert record["returncode"] == 3
    assert record["error"] == "ERROR: Lost atoms: original 3 current 2"


def test_a_launcher_that_outlives_its_timeout_is_stopped_and_called_a_timeout(tmp_path, mpirun_on_path):
    mpirun_on_path("sleep 60\n")
    record = _launch_band(tmp_path, timeout=0.5)
    assert record["failure"] == "crash"
    assert record["returncode"] is None
    assert record["error"] == "timeout"


def test_a_machine_without_mpirun_cannot_start_a_band(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="mpirun is not on PATH"):
        _launch_band(tmp_path)
    monkeypatch.setattr(lammps_neb, "lammps_executable", lambda: Path("lmp"))
    pair = load_pairs(geometry_archive(ROOT, resolve_problem(ROOT, PROBLEM)), "xyz-pairs", 50)[0]
    record = lammps_neb._execute(pair, _settings_for(), work=tmp_path / "again")
    assert record["failure"] == "crash"
    assert record["error"] == "FileNotFoundError: mpirun is not on PATH"


@pytest.fixture
def fake_lammps(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    capture.mkdir()
    monkeypatch.setenv("FAKE_CAPTURE", str(capture))
    binary = _script(
        tmp_path / "lmp",
        'cp in.probe "$FAKE_CAPTURE/in.probe"\n'
        'cp start.data "$FAKE_CAPTURE/start.data"\n'
        'cp "$FAKE_LOG" log.lammps\n'
        '[ -n "$FAKE_DUMP" ] && cp "$FAKE_DUMP" forces.dump\n'
        'exit "${FAKE_STATUS:-0}"\n',
    )

    def prepare(log: str, dump: str | None = None, status: int = 0) -> Path:
        log_path = tmp_path / "fake.log"
        log_path.write_text(log, encoding="utf-8")
        monkeypatch.setenv("FAKE_LOG", str(log_path))
        if dump is None:
            monkeypatch.setenv("FAKE_DUMP", "")
        else:
            dump_path = tmp_path / "fake.dump"
            dump_path.write_text(dump, encoding="utf-8")
            monkeypatch.setenv("FAKE_DUMP", str(dump_path))
        monkeypatch.setenv("FAKE_STATUS", str(status))
        return binary

    prepare.capture = capture
    return prepare


FORCES = "ITEM: ATOMS id type fx fy fz\n2 1 4.0 5.0 6.0\n1 1 1.0 2.0 3.0\n"
THERMO = "Step PotEng\n       0   -1.0158730158730158\n"


def test_a_probe_returns_the_energy_of_the_log_and_the_gradient_as_minus_the_force(fake_lammps):
    binary = fake_lammps(THERMO, FORCES)
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    energy, gradient = lammps_neb.probe_pair("lj", coords, None, binary=binary)
    assert energy == -1.0158730158730158
    assert gradient.tolist() == [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]
    script = (fake_lammps.capture / "in.probe").read_text(encoding="utf-8").splitlines()
    assert "units lj" in script
    assert "boundary f f f" in script
    assert "pair_style lj/cut 40.0" in script
    assert "run 0" in script
    data = (fake_lammps.capture / "start.data").read_text(encoding="utf-8").splitlines()
    assert "2 atoms" in data
    assert "1 atom types" in data


def test_a_slab_probe_is_periodic_in_the_cell_it_was_given(fake_lammps):
    binary = fake_lammps(THERMO, FORCES)
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    lammps_neb.probe_pair("morse_pt", coords, np.array([21.0, 22.0, 30.0]), binary=binary)
    script = (fake_lammps.capture / "in.probe").read_text(encoding="utf-8").splitlines()
    assert "units metal" in script
    assert "boundary p p p" in script
    data = (fake_lammps.capture / "start.data").read_text(encoding="utf-8").splitlines()
    assert "0 21 xlo xhi" in data
    assert "0 22 ylo yhi" in data
    assert "0 30 zlo zhi" in data


def test_a_probe_that_fails_reports_the_lammps_error_or_the_exit_code(fake_lammps):
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    binary = fake_lammps("setup\nERROR: Lost atoms: original 2 current 1\n", status=1)
    with pytest.raises(RuntimeError, match="ERROR: Lost atoms: original 2 current 1"):
        lammps_neb.probe_pair("lj", coords, None, binary=binary)
    binary = fake_lammps("setup\n", status=7)
    with pytest.raises(RuntimeError, match="LAMMPS probe exited 7"):
        lammps_neb.probe_pair("lj", coords, None, binary=binary)


def test_a_probe_whose_dump_misses_an_atom_is_refused(fake_lammps):
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    binary = fake_lammps(THERMO, "ITEM: ATOMS id type fx fy fz\n1 1 1.0 2.0 3.0\n")
    with pytest.raises(RuntimeError, match="LAMMPS force dump does not list every atom"):
        lammps_neb.probe_pair("lj", coords, None, binary=binary)


def test_a_registry_without_the_code_is_refused(tmp_path):
    (tmp_path / "codes").mkdir()
    (tmp_path / "codes" / "registry.toml").write_text(
        '[[code]]\nid = "other"\ncommit = "abc"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="codes/registry.toml has no lammpsneb entry"):
        _registry(tmp_path)


def test_an_unknown_potential_has_no_input_script():
    with pytest.raises(ValueError, match="unknown potential eam"):
        render_neb_input(dict(_settings_for(), potential="eam"), n1=2, n2=2, frozen=False)


def test_a_problem_without_interior_images_reports_no_per_image_cost():
    template = load_template(ROOT, TEMPLATE)
    problem = resolve_problem(ROOT, PROBLEM)
    records = [_record("pair_00", 100, success=True, index=1)]
    lines = _dat(records, template, dict(_settings_for(), n_interior=0), problem).splitlines()
    assert "force_calls 100" in lines
    assert not [line for line in lines if line.startswith("force_calls_per_image")]
    assert "call_unit total" not in lines
    assert "images 0" in lines
