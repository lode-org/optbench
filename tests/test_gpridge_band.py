from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from optbench import gpridge_band as native
from optbench.pes import Pair


@pytest.mark.parametrize("name", [
    "pt-heptamer-gp-neb", "pt-heptamer-ci-neb",
    "pt-island-gp-neb", "pt-island-ci-neb",
])
def test_committed_templates_match_registry(name):
    from optbench.campaign import load_template
    from optbench.engines.common import registry_pin, require_pin

    root = Path(__file__).resolve().parents[1]
    template = load_template(root, f"gpridge/{name}")
    require_pin(template, registry_pin(root, "gpridge"))


def pair():
    return Pair("start", np.array([[0., 0., 0.], [3., 0., 0.]]),
                np.array([[1., 0., 0.], [3., 0., 0.]]),
                np.array([True, False]), np.array([20., 20., 30.]))


def settings():
    return {"n_interior": 8, "force_norm": .01, "max_move": .2, "cap": 10000}


def test_command_keeps_the_problem_stop_and_geometry(tmp_path):
    argv = native.command("gprn", tmp_path, settings(), {})
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert flags["--nimages"] == "8"
    assert flags["--convergence-force-norm"] == "l2"
    assert flags["--converged-force"] == "0.01"
    assert flags["--climbing-image-converged-only"] == "false"
    assert flags["--spring-constant"] == flags["--spring-constant-min"] == "5"
    assert flags["--neb-lbfgs-max-move"] == "0.2"
    assert flags["--gp-nonperiodic"] == "false"
    assert flags["--endpoint-assign"] == flags["--sidpp"] == "false"


@pytest.mark.parametrize("name", ["pt-heptamer-gp-neb", "pt-island-gp-neb"])
def test_gp_templates_select_the_publication_fit_and_arithmetic(tmp_path, name):
    from optbench.campaign import load_template

    root = Path(__file__).resolve().parents[1]
    template = load_template(root, f"gpridge/{name}")
    argv = native.command("gprn", tmp_path, settings(), template["gpridge"])
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert flags["--gp-model-profile"] == "gprd_zbl"
    assert flags["--hyperparameter-mode"] == "rgmin:lbfgs"
    assert flags["--linalg-mode"] == "library"
    assert flags["--gp-hssvd"] == "false"
    assert flags["--gp-active-posterior"] == "true"
    assert flags["--gp-prior-mean-statistic"] == "max"
    assert flags["--seed-full-chain"] == "true"
    assert flags["--gp-zbl-prior-mean"] == "false"
    assert flags["--climbing-image-converged-only"] == "false"


def test_input_roundtrip_retains_fixed_atoms_cell_and_cutoff(tmp_path):
    import tomllib
    from ase.io import read

    directory = tmp_path / "case"
    native.write_inputs(directory, pair())
    for name, positions in (("reactant", pair().reactant), ("product", pair().product)):
        atoms = read(directory / f"{name}.con", format="eon")
        np.testing.assert_allclose(atoms.positions, positions, atol=1e-12)
        np.testing.assert_allclose(atoms.cell.lengths(), pair().box)
        assert atoms.constraints[0].index.tolist() == [1]
    params = tomllib.loads((directory / "potential.toml").read_text())["oracle"]["parameters"]["morse"]
    assert params == {"de": .7102, "a": 1.6047, "re": 2.897, "cutoff": 9.5}
    assert (directory / "environment-types.txt").read_text() == "0 78 0\n1 78 0\n"


def test_missing_and_separate_search_counts():
    assert native.parse_calls("search unfinished") == (None, None)
    assert native.parse_calls("Oracle calls (to convergence): 17  (endpoints=2)\n"
                              "Oracle calls (with validation): 29 (validation=12)\n") == (17, 29)


def write_band(path, positions, gradient=None, energy=0.):
    with h5py.File(path, "w") as out:
        out["summary/max_energy_image"] = [1.]
        out["converged/1/positions"] = np.asarray(positions).reshape(-1)
        out["converged/1/gradient"] = np.zeros(3) if gradient is None else gradient
        out["converged/1/energy"] = [energy]


def test_independent_check_accepts_only_an_index_one_stationary_point(tmp_path, monkeypatch):
    def saddle(positions, box):
        coeff = np.array([-1., 2., 3.])
        gradient = np.zeros_like(positions)
        gradient[0] = coeff * positions[0]
        return .5 * float(np.dot(gradient[0], positions[0])), gradient

    monkeypatch.setattr(native, "morse_pt", saddle)
    path = tmp_path / "band.h5"
    write_band(path, [0., 0., 0.])
    record = native.verify_band(path, pair(), settings())
    assert record["reasons"] == []
    assert record["hessian_index"] == 1
    assert record["verify_calls"] == 7
    assert record["gradient_error"] == 0
    write_band(path, [0.1, 0., 0.], [-.1, 0., 0.], -.005)
    record = native.verify_band(path, pair(), settings())
    assert record["reasons"] == ["force_norm"]
    assert record["hessian_index"] is None


def test_surface_mismatch_and_moved_substrate_are_rejected(tmp_path):
    path = tmp_path / "band.h5"
    write_band(path, [0., 0., 0.])
    assert "potential_mismatch" in native.verify_band(path, pair(), settings())["reasons"]
    write_band(path, [[0., 0., 0.], [3.1, 0., 0.]])
    with pytest.raises(ValueError, match="fixed substrate"):
        native.verify_band(path, pair(), settings())


@pytest.mark.parametrize("energy,gradient", [
    (float("nan"), [0., 0., 0.]),
    (float("inf"), [0., 0., 0.]),
    (0., [float("nan"), 0., 0.]),
    (0., [float("inf"), 0., 0.]),
])
def test_nonfinite_native_values_are_rejected(tmp_path, energy, gradient):
    path = tmp_path / "band.h5"
    write_band(path, [0., 0., 0.], gradient, energy)
    with pytest.raises(ValueError, match="non-finite native energy or gradient"):
        native.verify_band(path, pair(), settings())


def test_nonfinite_independent_surface_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "band.h5"
    write_band(path, [0., 0., 0.])
    monkeypatch.setattr(native, "morse_pt", lambda positions, box:
                        (float("nan"), np.zeros_like(positions)))
    with pytest.raises(ValueError, match="non-finite independent"):
        native.verify_band(path, pair(), settings())


def test_runtime_requires_successful_validation_and_exact_executable(tmp_path, monkeypatch):
    import json

    binary = tmp_path / "gprn"
    binary.write_bytes(b"verified executable")
    manifest = tmp_path / "validation"
    manifest.mkdir()
    (manifest / "provenance.json").write_text(json.dumps({"source_revision": "revision"}))
    (manifest / "programs.sha256").write_text(f"{native.digest(binary)}  {binary}\n")
    monkeypatch.setenv("GPRIDGE_GPRN", str(binary))
    monkeypatch.setenv("GPRIDGE_VALIDATION", str(manifest))
    (manifest / "result.json").write_text('{"exit_code": 1}')
    with pytest.raises(ValueError, match="validation did not succeed"):
        native.runtime({"commit": "revision"})
    (manifest / "result.json").write_text('{"exit_code": 0}')
    assert native.runtime({"commit": "revision"})["sha256"] == native.digest(binary)
    binary.write_bytes(b"different executable")
    with pytest.raises(ValueError, match="does not match"):
        native.runtime({"commit": "revision"})


def test_timeout_terminates_child_processes(tmp_path):
    import os
    import select
    import signal
    import sys
    import time

    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "\"import os, time; print('ready', os.getpgrp(), flush=True); time.sleep(60)\"])\n"
        "time.sleep(60)\n"
    )
    reader, writer = os.pipe()
    group = None
    closed = False
    try:
        with os.fdopen(writer, "w") as log:
            status = native.run_search([sys.executable, str(script)], tmp_path,
                                       os.environ.copy(), log, 2.)
        assert status == 124
        deadline = time.monotonic() + 2.
        received = b""
        while time.monotonic() < deadline:
            readable, _, _ = select.select([reader], [], [], deadline - time.monotonic())
            if not readable:
                break
            chunk = os.read(reader, 4096)
            if not chunk:
                closed = True
                break
            received += chunk
            if received.startswith(b"ready ") and b"\n" in received:
                group = int(received.splitlines()[0].split()[1])
        assert received.startswith(b"ready ")
        assert closed, "a descendant retains the output pipe after timeout"
    finally:
        os.close(reader)
        if group is not None and not closed:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_packaged_runtime_keeps_the_validated_source_hash(tmp_path, monkeypatch):
    import json

    binary = tmp_path / "gprn"
    binary.write_bytes(b"relocated executable")
    library = tmp_path / "libgpr.so"
    library.write_bytes(b"packaged library")
    manifest = tmp_path / "validation"
    manifest.mkdir()
    (manifest / "provenance.json").write_text('{"source_revision": "revision"}')
    (manifest / "result.json").write_text('{"exit_code": 0}')
    (manifest / "programs.sha256").write_text("original-hash  /build/gprn\n")
    package = {"source_revision": "revision", "files": {
        "gprn": {"sha256": native.digest(binary), "source_sha256": "original-hash"},
        "libgpr.so": {"sha256": native.digest(library)},
    }}
    (tmp_path / "runtime.json").write_text(json.dumps(package))
    monkeypatch.setenv("GPRIDGE_GPRN", str(binary))
    monkeypatch.setenv("GPRIDGE_VALIDATION", str(manifest))
    assert native.runtime({"commit": "revision"})["validated_sha256"] == "original-hash"
    package["files"]["gprn"]["source_sha256"] = "different-build"
    (tmp_path / "runtime.json").write_text(json.dumps(package))
    with pytest.raises(ValueError, match="native-validation"):
        native.runtime({"commit": "revision"})
    package["files"]["gprn"]["source_sha256"] = "original-hash"
    (tmp_path / "runtime.json").write_text(json.dumps(package))
    library.write_bytes(b"wrong library")
    with pytest.raises(ValueError, match="package file mismatch"):
        native.runtime({"commit": "revision"})


@pytest.mark.parametrize("selection,flag", [
    ("auto", None), ("radius", "--radius-only"), ("window", "--evict-frozen"),
])
def test_frozen_selection_maps_to_the_native_cli(tmp_path, selection, flag):
    argv = native.command("gprn", tmp_path, settings(), {
        "frozen_selection": selection, "frozen_neighbors": 0,
        "frozen_activation_distance": 5.0,
    })
    assert argv[argv.index("--nn-frozen") + 1] == "0"
    assert argv[argv.index("--gp-frozen-activation-distance") + 1] == "5.0"
    assert ("--radius-only" in argv) == (flag == "--radius-only")
    assert ("--evict-frozen" in argv) == (flag == "--evict-frozen")


@pytest.mark.parametrize("neighbors", [2, 16])
def test_explicit_frozen_neighbor_count_reaches_the_native_cli(tmp_path, neighbors):
    argv = native.command("gprn", tmp_path, settings(), {"frozen_neighbors": neighbors})
    assert argv[argv.index("--nn-frozen") + 1] == str(neighbors)
    assert "--radius-only" not in argv
    assert "--evict-frozen" not in argv


@pytest.mark.parametrize("options", [
    {"frozen_selection": "nearest"}, {"frozen_selection": None},
    {"frozen_neighbors": -1}, {"frozen_neighbors": 2.5},
    {"frozen_neighbors": "2"}, {"frozen_neighbors": True},
    {"frozen_neighbors": 2, "frozen_selection": "radius"},
    {"frozen_neighbors": 2, "frozen_selection": "window"},
    {"frozen_activation_distance": -1},
    {"frozen_activation_distance": float("nan")},
    {"frozen_activation_distance": float("inf")},
    {"frozen_activation_distance": "5"}, {"frozen_activation_distance": True},
])
def test_invalid_frozen_policy_is_rejected_before_launch(tmp_path, options):
    with pytest.raises(ValueError, match="frozen_"):
        native.command("gprn", tmp_path, settings(), options)


@pytest.mark.parametrize("system", ["heptamer", "island"])
@pytest.mark.parametrize("variant,neighbors,selection", [
    ("auto", 0, "auto"), ("radius", 0, "radius"),
    ("k2", 2, "auto"), ("k16", 16, "auto"),
])
def test_frozen_comparison_templates_change_only_environment_selection(
        tmp_path, system, variant, neighbors, selection):
    from optbench.campaign import load_template
    from optbench.engines.common import registry_pin, require_pin

    root = Path(__file__).resolve().parents[1]
    base = load_template(root, f"gpridge/pt-{system}-gp-neb")
    comparison = load_template(root, f"gpridge/pt-{system}-gp-neb-{variant}")
    require_pin(comparison, registry_pin(root, "gpridge"))
    assert comparison["problem"] == base["problem"]
    assert comparison["gpridge"] == {
        **base["gpridge"], "frozen_neighbors": neighbors,
        "frozen_selection": selection, "frozen_activation_distance": 5.0,
    }
    argv = native.command("gprn", tmp_path, settings(), comparison["gpridge"])
    base_argv = native.command("gprn", tmp_path, settings(), base["gpridge"])
    assert argv[:len(base_argv)] == base_argv
    assert argv[argv.index("--nn-frozen") + 1] == str(neighbors)
    assert ("--radius-only" in argv) == (selection == "radius")


@pytest.mark.parametrize("program", ["gprd", "potserv"])
def test_runtime_validates_the_selected_native_program(tmp_path, monkeypatch, program):
    binary = tmp_path / program
    binary.write_bytes(program.encode())
    manifest = tmp_path / "validation"
    manifest.mkdir()
    (manifest / "provenance.json").write_text('{"source_revision": "revision"}')
    (manifest / "result.json").write_text('{"exit_code": 0}')
    (manifest / "programs.sha256").write_text(f"{native.digest(binary)}  {binary}\n")
    monkeypatch.setenv(f"GPRIDGE_{program.upper()}", str(binary))
    monkeypatch.setenv("GPRIDGE_VALIDATION", str(manifest))
    assert native.runtime({"commit": "revision"}, program)["binary"] == str(binary)
    binary.write_bytes(b"different")
    with pytest.raises(ValueError, match="native-validation"):
        native.runtime({"commit": "revision"}, program)


def test_runtime_validates_a_nested_packaged_server(tmp_path, monkeypatch):
    import json

    relative = "subprojects/rgpot/CppCore/potserv"
    binary = tmp_path / relative
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"packaged server")
    manifest = tmp_path / "validation"
    manifest.mkdir()
    (manifest / "provenance.json").write_text('{"source_revision": "revision"}')
    (manifest / "result.json").write_text('{"exit_code": 0}')
    (manifest / "programs.sha256").write_text("original-server /build/potserv\n")
    (tmp_path / "runtime.json").write_text(json.dumps({
        "source_revision": "revision", "files": {
            relative: {"sha256": native.digest(binary), "source_sha256": "original-server"},
        },
    }))
    monkeypatch.setenv("GPRIDGE_POTSERV", str(binary))
    monkeypatch.setenv("GPRIDGE_VALIDATION", str(manifest))
    identity = native.runtime({"commit": "revision"}, "potserv")
    assert identity["validated_sha256"] == "original-server"
    assert identity["runtime_manifest_sha256"] == native.digest(tmp_path / "runtime.json")


@pytest.mark.parametrize("system", ["heptamer", "island"])
def test_extended_fit_baseline_keeps_publication_physical_settings(system):
    from optbench.campaign import load_template
    from optbench.engines.common import registry_pin, require_pin

    root = Path(__file__).resolve().parents[1]
    publication = load_template(root, f"gpridge/pt-{system}-gp-neb-auto")
    extended = load_template(root, f"gpridge/pt-{system}-gp-neb-extended")
    require_pin(extended, registry_pin(root, "gpridge"))
    assert extended["problem"] == publication["problem"]
    assert extended["gpridge"] == {
        **publication["gpridge"], "arithmetic": "reproducible", "model_profile": "extended",
        "hyperparameter_mode": "inla", "hssvd": True, "active_posterior": False,
        "prior_mean_statistic": "mean", "seed_full_chain": False,
    }


@pytest.mark.parametrize("system", ["heptamer", "island"])
def test_atomic_templates_preserve_the_physical_comparison(tmp_path, system):
    from optbench.campaign import load_template
    from optbench.engines.common import registry_pin, require_pin

    root = Path(__file__).resolve().parents[1]
    publication = load_template(root, f"gpridge/pt-{system}-gp-neb-auto")
    atomic = load_template(root, f"gpridge/pt-{system}-gp-neb-atomic")
    require_pin(atomic, registry_pin(root, "gpridge"))
    assert atomic["problem"] == publication["problem"]
    assert atomic["gpridge"] == {
        **publication["gpridge"], "active_posterior": False,
        "atomic_environments": True,
    }
    assert atomic["gpridge"]["timeout"] == 1800
    base_argv = native.command("gprn", tmp_path, settings(), publication["gpridge"])
    argv = native.command("gprn", tmp_path, settings(), atomic["gpridge"])
    base_flags = dict(zip(base_argv[1::2], base_argv[2::2]))
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert flags == {
        **base_flags, "--gp-active-posterior": "false",
        "--gp-atomic-environments": "true", "--uncertainty-mode": "map",
        "--hyperparameter-objective": "nlml", "--gp-learn-noise": "false",
        "--rff-features": "0", "--gp-virtual-obs": "false",
        "--gp-robust-nu": "0", "--gp-local-tier": "false",
    }


def test_atomic_model_is_opt_in(tmp_path):
    assert "--gp-atomic-environments" not in native.command(
        "gprn", tmp_path, settings(), {})
    assert native.atomic_arguments({"atomic_environments": False}) == [
        "--gp-atomic-environments", "false"]


@pytest.mark.parametrize("options", [
    {"atomic_environments": "true"}, {"atomic_environments": 1},
    {"atomic_environments": None}, {"model_profile": "extended"},
    {"hyperparameter_mode": "inla"}, {"active_posterior": True},
])
def test_atomic_model_rejects_unsupported_fit_settings(tmp_path, options):
    valid = {"atomic_environments": True, "model_profile": "gprd_zbl",
             "hyperparameter_mode": "rgmin:lbfgs", "active_posterior": False}
    with pytest.raises(ValueError, match="atomic_environments"):
        native.command("gprn", tmp_path, settings(), {**valid, **options})
