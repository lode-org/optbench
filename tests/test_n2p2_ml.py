"""Fingerprints, the squared-energy loss, and one n2p2 Kalman step."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from optbench.n2p2_ml import (
    EPOCH_CAP,
    FingerprintSpec,
    G1,
    G2,
    MlDataset,
    element_width,
    fingerprints_frame,
    g2_value,
    init_weights,
    image_jacobian,
    loss_and_gradient,
    n_weights,
    parameter_slices,
    parse_fpparas,
    reference_kalman_update,
    run_start,
    start_rng,
)

ROOT = Path(__file__).resolve().parents[1]


def _n2p2_root() -> Path:
    configured = os.environ.get("N2P2_ROOT")
    if configured:
        return Path(configured)
    return ROOT / "third_party" / "n2p2"


N2P2_ROOT = _n2p2_root()
N2P2 = N2P2_ROOT / "src" / "libnnptrain" / "KalmanFilter.cpp"


def _equilateral() -> tuple[np.ndarray, np.ndarray]:
    height = 2.5 * math.sqrt(3.0) / 2.0
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, height, 0.0]], dtype=np.float64)
    return positions, np.array([30.0, 30.0, 30.0])


def test_parser_counts_the_archived_fingerprints():
    pt = parse_fpparas((ROOT / "archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0/fpParas.dat").read_text())
    assert pt.elements == ("Pt",)
    assert pt.n_fp == (8,)
    assert [feature.index for feature in pt.g1] == [0, 1, 2, 3]
    assert [feature.index for feature in pt.g2] == [4, 5, 6, 7]
    ethene = parse_fpparas((ROOT / "archive/entries/mlopt/C2H4/pyamff-adam10-7-fcoeff0/fpParas.dat").read_text())
    assert ethene.elements == ("C", "H")
    assert ethene.n_fp == (7, 7)
    carbon = [feature for feature in ethene.g2 if feature.center == 0]
    assert [(feature.neighbor_a, feature.neighbor_b, feature.index) for feature in carbon] == [
        (0, 0, 3),
        (0, 1, 4),
        (0, 1, 5),
        (1, 1, 6),
    ]
    assert carbon[1].eta == carbon[2].eta
    assert carbon[1].zeta == carbon[2].zeta
    assert carbon[1].lam == carbon[2].lam
    assert n_weights(pt.n_fp) == 175
    assert n_weights(ethene.n_fp) == 330
    assert element_width(8) == 175
    assert element_width(7) == 165


def test_fcoeff0_fpparas_copies_match():
    for problem, stem in (("emt-pt38", "pt"), ("C2H4", "c2h4")):
        names = [
            f"pyamff-adam10-7-fcoeff0/fpParas.dat",
            f"pyamff-lbfgs10-7-fcoeff0/fpParas.dat",
            f"pyamff-rprop10-7-fcoeff0/fpParas.dat",
        ]
        base = (ROOT / "archive/entries/mlopt" / problem / names[0]).read_bytes()
        for name in names[1:]:
            assert (ROOT / "archive/entries/mlopt" / problem / name).read_bytes() == base
        del stem


def test_equilateral_g1_and_g2_match_the_fortran_products():
    positions, lengths = _equilateral()
    text = "\n".join(
        [
            "#FP type",
            "BP",
            "#Elements",
            "Pt",
            "0.0",
            "#nG1s nG2s",
            "1 1",
            "#G1",
            "G1 Pt Pt 0.05 0.0 5.0",
            "#G2",
            "G2 Pt Pt Pt 0.005 1.0 1.0 0.0 5.0",
        ]
    )
    spec = parse_fpparas(text)
    frame = fingerprints_frame(positions, ["Pt", "Pt", "Pt"], lengths, spec)
    distance = 2.5
    rc = 5.0
    g1 = math.exp(-0.05 * distance * distance / (rc * rc)) * 0.5 * (math.cos(math.pi * distance / rc) + 1.0)
    assert frame[:, 0] == pytest.approx(2.0 * g1)
    feature = spec.g2[0]
    vij = positions[1] - positions[0]
    vik = positions[2] - positions[0]
    expected = g2_value(distance, distance, distance, vij, vik, feature)
    cosine = 0.5
    theta = math.acos(cosine)
    base = 1.0 + 1.0 * math.cos(theta)
    radial = (math.exp(-distance * distance / (rc * rc)) ** 3) ** 0.005
    cutoff = (0.5 * (math.cos(math.pi * distance / rc) + 1.0)) ** 3
    formula = base ** (1.0 - 1.0) * base * radial * cutoff * 2.0 ** (1.0 - 1.0)
    assert expected == pytest.approx(formula)
    assert frame[:, 1] == pytest.approx(expected)


def test_duplicate_angular_slots_receive_the_same_value():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.2, 1.1, 0.0], [-0.4, -0.9, 0.2]],
        dtype=np.float64,
    )
    spec = parse_fpparas((ROOT / "archive/entries/mlopt/C2H4/pyamff-adam10-7-fcoeff0/fpParas.dat").read_text())
    frame = fingerprints_frame(positions, ["C", "C", "H", "H"], np.array([20.0, 20.0, 20.0]), spec)
    carbon = frame[0]
    assert carbon[4] != 0.0
    assert carbon[4] == pytest.approx(carbon[5])
    assert carbon.shape == (7,)


def test_loss_gradient_matches_a_finite_difference():
    rng = np.random.Generator(np.random.PCG64(11))
    n_fp = (2,)
    weights = init_weights(n_fp, rng)
    element_index = np.array([0, 0], dtype=np.int32)
    fingerprints = rng.normal(size=(2, 2, 2))
    energy = np.array([0.3, -0.2])
    loss, _rmse, grad = loss_and_gradient(weights, element_index, n_fp, fingerprints, energy, 2)
    step = 1e-6
    numeric = np.zeros_like(weights)
    for index in range(weights.size):
        bumped = weights.copy()
        bumped[index] += step
        bumped_loss, _rmse, _grad = loss_and_gradient(bumped, element_index, n_fp, fingerprints, energy, 2)
        numeric[index] = (bumped_loss - loss) / step
    assert grad == pytest.approx(numeric, rel=1e-5, abs=1e-5)
    peaks = [float(np.max(np.abs(grad[start:stop]))) for start, stop in parameter_slices(n_fp)]
    assert len(peaks) == 6


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_library_matches_the_reference_loss_and_one_filter_pass():
    from optbench.n2p2_ml import KALMAN_ETAMAX, KALMAN_ETATAU, KALMAN_ETA0, KALMAN_Q0, KALMAN_QMIN, KALMAN_QTAU, KalmanSession

    rng = np.random.Generator(np.random.PCG64(19))
    n_fp = (3, 2)
    element_index = np.array([0, 1, 1], dtype=np.int32)
    fingerprints = rng.normal(size=(2, 3, 3))
    energy = np.array([1.5, -0.4])
    dataset = MlDataset(("A", "B"), n_fp, element_index, fingerprints, energy)
    weights = init_weights(n_fp, rng)
    loss, rmse, grad = loss_and_gradient(weights, element_index, n_fp, fingerprints, energy, 3)
    session = KalmanSession(dataset, 0.02)
    try:
        session.set_weights(weights)
        evaluation = session.evaluate()
        assert evaluation["loss"] == pytest.approx(loss, rel=1e-10, abs=1e-10)
        assert evaluation["energy_rmse"] == pytest.approx(rmse, rel=1e-10, abs=1e-10)
        assert evaluation["grad"] == pytest.approx(grad, rel=1e-9, abs=1e-9)
        slices = parameter_slices(n_fp)
        peaks = [float(np.max(np.abs(grad[start:stop]))) for start, stop in slices]
        assert evaluation["tensor_maxabs"] == pytest.approx(peaks)
        updates = dataset.n_images
        state_w = weights.copy()
        state_p = np.eye(state_w.size) / 1.0e-2
        eta = KALMAN_ETA0
        q = KALMAN_Q0
        order = np.arange(updates)
        for image in order:
            pred, jacobian = image_jacobian(state_w, element_index, n_fp, fingerprints[int(image)])
            state_w, state_p, eta, q = reference_kalman_update(
                state_w,
                state_p,
                jacobian,
                float(energy[image]) - pred,
                eta,
                q,
                etatau=KALMAN_ETATAU / updates,
                etamax=KALMAN_ETAMAX,
                qtau=KALMAN_QTAU / updates,
                qmin=KALMAN_QMIN,
            )
        assert session.epoch(order) == 0
        assert session.get_weights() == pytest.approx(state_w, rel=1e-9, abs=1e-9)
        assert session.updates() == updates
    finally:
        session.close()


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_harness_counts_each_loss_evaluation_until_the_cap():
    rng = start_rng(0)
    n_fp = (2,)
    weights_unused = init_weights(n_fp, rng)
    del weights_unused
    dataset = MlDataset(
        ("Pt",),
        n_fp,
        np.zeros(2, dtype=np.int32),
        np.zeros((4, 2, 2)),
        np.array([0.2, 0.2, 0.2, 0.2]),
    )
    record = run_start(dataset, 3, cap=4, tol=1e-30, shuffle=False)
    assert record["force_calls"] == 4
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert record["kalman_epochs"] == 3
    assert record["seed"] == 20261015 + 3
    assert EPOCH_CAP == 10000


def test_weight_draw_is_uniform_on_the_half_open_interval():
    weights = init_weights((8,), start_rng(4))
    assert weights.shape == (175,)
    assert weights.min() >= -1.0
    assert weights.max() < 1.0
    again = init_weights((8,), start_rng(4))
    assert again == pytest.approx(weights)


def test_spec_objects_are_the_feature_records():
    spec = FingerprintSpec(("Pt",), (1,), (G1(0, 0, 0.05, 0.0, 5.0, 0),), (G2(0, 0, 0, 0.005, 1.0, 1.0, 0.0, 5.0, 1),), 5.0)
    assert spec.stride == 1


def test_force_fpparas_bytes_match_the_energy_files():
    for problem in ("emt-pt38", "C2H4"):
        energy = (ROOT / "archive/entries/mlopt" / problem / "pyamff-adam10-7-fcoeff0/fpParas.dat").read_bytes()
        force = (ROOT / "archive/entries/mlopt" / problem / "pyamff-adam10-7-fcoeff0_1/fpParas.dat").read_bytes()
        assert force == energy


def test_projected_force_updates_follow_the_documented_ratio():
    from optbench.n2p2_ml import projected_force_updates

    assert projected_force_updates(500, 38) == 1500
    assert projected_force_updates(570, 6) == 1710
    assert projected_force_updates(2, 2) == 6


def test_fingerprint_derivatives_match_a_central_difference():
    positions, lengths = _equilateral()
    text = "\n".join(
        [
            "#FP type",
            "BP",
            "#Elements",
            "Pt",
            "0.0",
            "#nG1s nG2s",
            "1 1",
            "#G1",
            "G1 Pt Pt 0.05 0.0 5.0",
            "#G2",
            "G2 Pt Pt Pt 0.005 1.0 1.0 0.0 5.0",
        ]
    )
    spec = parse_fpparas(text)
    symbols = ["Pt", "Pt", "Pt"]
    values, dg = fingerprints_frame(positions, symbols, lengths, spec, derivatives=True)
    step = 1e-6
    for atom in range(3):
        for component in range(3):
            bumped = positions.copy()
            bumped[atom, component] += step
            plus = fingerprints_frame(bumped, symbols, lengths, spec)
            bumped[atom, component] -= 2.0 * step
            minus = fingerprints_frame(bumped, symbols, lengths, spec)
            numeric = (plus - minus) / (2.0 * step)
            assert dg[:, atom, component, : spec.stride] == pytest.approx(numeric, rel=1e-5, abs=1e-6)
    assert values.shape == (3, spec.stride)


def test_directional_weight_gradient_matches_a_central_difference():
    from optbench.n2p2_ml import _atom_forward, atom_directional_grad

    rng = np.random.Generator(np.random.PCG64(23))
    n_fp = 3
    block = init_weights((n_fp,), rng)
    x = rng.normal(size=n_fp)
    direction = rng.normal(size=n_fp)
    analytic = atom_directional_grad(x, direction, block, n_fp)
    step = 1e-6
    numeric = np.zeros_like(block)
    for index in range(block.size):
        bumped = block.copy()
        bumped[index] += step
        _energy, _grad, plus = _atom_forward(x, bumped, n_fp)
        bumped[index] -= 2.0 * step
        _energy, _grad, minus = _atom_forward(x, bumped, n_fp)
        numeric[index] = float((plus - minus) @ direction) / (2.0 * step)
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7)


def test_combined_force_loss_matches_a_finite_difference():
    from optbench.n2p2_ml import combined_loss_and_gradient

    rng = np.random.Generator(np.random.PCG64(29))
    n_fp = (2,)
    weights = init_weights(n_fp, rng)
    element_index = np.array([0, 0], dtype=np.int32)
    fingerprints = rng.normal(size=(2, 2, 2))
    energy = np.array([0.4, -0.1])
    forces = rng.normal(size=(2, 2, 3))
    dg = rng.normal(size=(2, 2, 2, 3, 2)) * 0.1
    loss, _energy_rmse, _force_rmse, grad = combined_loss_and_gradient(
        weights, element_index, n_fp, fingerprints, energy, forces, dg, 2
    )
    step = 1e-6
    numeric = np.zeros_like(weights)
    for index in range(weights.size):
        bumped = weights.copy()
        bumped[index] += step
        bumped_loss, _e, _f, _g = combined_loss_and_gradient(
            bumped, element_index, n_fp, fingerprints, energy, forces, dg, 2
        )
        numeric[index] = (bumped_loss - loss) / step
    assert grad == pytest.approx(numeric, rel=1e-4, abs=1e-5)


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_recommended_kalman_block_matches_the_constants():
    from optbench.n2p2_ml import (
        FORCE_ENERGY_RATIO,
        FORCE_WEIGHT,
        KALMAN_EPSILON,
        KALMAN_ETA0,
        KALMAN_ETAMAX,
        KALMAN_ETATAU,
        KALMAN_Q0,
        KALMAN_QMIN,
        KALMAN_QTAU,
        N2P2_RANDOM_SEED,
    )

    text = (N2P2_ROOT / "examples" / "input.nn.recommended").read_text(encoding="utf-8")
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        values[parts[0]] = parts[1:]
    assert float(values["kalman_epsilon"][0]) == KALMAN_EPSILON
    assert float(values["kalman_q0"][0]) == KALMAN_Q0
    assert float(values["kalman_qtau"][0]) == KALMAN_QTAU
    assert float(values["kalman_qmin"][0]) == KALMAN_QMIN
    assert float(values["kalman_eta"][0]) == KALMAN_ETA0
    assert float(values["kalman_etatau"][0]) == KALMAN_ETATAU
    assert float(values["kalman_etamax"][0]) == KALMAN_ETAMAX
    assert values["kalman_type"][0] == "0"
    assert values["updater_type"][0] == "1"
    assert float(values["force_weight"][0]) == FORCE_WEIGHT
    assert float(values["force_energy_ratio"][0]) == FORCE_ENERGY_RATIO
    assert int(values["random_seed"][0]) == N2P2_RANDOM_SEED
    assert values["normalize_data_set"][0] == "force"
    assert "NOT RECOMMENDED" in text


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_n2p2_shuffle_is_deterministic_and_advances():
    from optbench.n2p2_ml import N2P2_RANDOM_SEED, N2p2Order

    first = N2p2Order(N2P2_RANDOM_SEED)
    second = N2p2Order(N2P2_RANDOM_SEED)
    try:
        left = first.permutation(32)
        again = second.permutation(32)
        assert left.tolist() == again.tolist()
        nxt = first.permutation(32)
        assert nxt.tolist() != left.tolist()
        pcg = np.random.Generator(np.random.PCG64(N2P2_RANDOM_SEED)).permutation(32)
        assert left.tolist() != pcg.tolist()
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_schedule_shuffle_does_not_advance_the_candidate_engine():
    from optbench.n2p2_ml import N2P2_RANDOM_SEED, N2p2Order

    mixed = N2p2Order(N2P2_RANDOM_SEED)
    plain = N2p2Order(N2P2_RANDOM_SEED)
    try:
        assert mixed.permutation(16).tolist() == plain.permutation(16).tolist()
        schedule = mixed.shuffle_schedule(np.array([1, 1, 1, 0, 0], dtype=np.int32))
        assert sorted(schedule.tolist()) == [0, 0, 1, 1, 1]
        assert schedule.tolist() != [1, 1, 1, 0, 0]
        assert mixed.permutation(16).tolist() == plain.permutation(16).tolist()
    finally:
        mixed.close()
        plain.close()


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_library_force_step_matches_the_reference_filter():
    from optbench.n2p2_ml import (
        FORCE_WEIGHT,
        KALMAN_EPSILON,
        KALMAN_ETAMAX,
        KALMAN_ETATAU,
        KALMAN_ETA0,
        KALMAN_Q0,
        KALMAN_QMIN,
        KALMAN_QTAU,
        KalmanSession,
        force_component_jacobian,
        projected_force_updates,
    )

    rng = np.random.Generator(np.random.PCG64(31))
    n_fp = (3,)
    element_index = np.array([0, 0], dtype=np.int32)
    fingerprints = rng.normal(size=(2, 2, 3))
    energy = np.array([0.7, -0.2])
    forces = rng.normal(size=(2, 2, 3))
    dg = rng.normal(size=(2, 2, 2, 3, 3)) * 0.05
    dataset = MlDataset(("A",), n_fp, element_index, fingerprints, energy, forces, dg)
    weights = init_weights(n_fp, rng)
    n_force = projected_force_updates(dataset.n_images, dataset.n_atoms)
    assert n_force == 6
    session = KalmanSession(dataset, 0.1, loss="force")
    try:
        session.set_weights(weights)
        evaluation = session.evaluate()
        from optbench.n2p2_ml import combined_loss_and_gradient

        loss, energy_rmse, force_rmse, grad = combined_loss_and_gradient(
            weights, element_index, n_fp, fingerprints, energy, forces, dg, 2
        )
        assert evaluation["loss"] == pytest.approx(loss, rel=1e-10, abs=1e-10)
        assert evaluation["energy_rmse"] == pytest.approx(energy_rmse, rel=1e-10, abs=1e-10)
        assert evaluation["force_rmse"] == pytest.approx(force_rmse, rel=1e-10, abs=1e-10)
        assert evaluation["grad"] == pytest.approx(grad, rel=1e-8, abs=1e-8)
        energy_order = np.array([1, 0], dtype=np.int32)
        force_ids = np.array([0, 4, 7, 3, 10, 1], dtype=np.int32)
        schedule = np.array([1, 0, 1, 1, 0, 1, 1, 1], dtype=np.int32)
        state_w = weights.copy()
        state_p = np.eye(state_w.size) / KALMAN_EPSILON
        eta = KALMAN_ETA0
        q = KALMAN_Q0
        total = dataset.n_images + n_force
        energy_cursor = 0
        force_cursor = 0
        components = dataset.n_atoms * 3
        for kind in schedule:
            if int(kind) == 0:
                image = int(energy_order[energy_cursor])
                energy_cursor += 1
                pred, jacobian = image_jacobian(state_w, element_index, n_fp, fingerprints[image])
                residual = float(energy[image]) - pred
                column = jacobian
            else:
                packed = int(force_ids[force_cursor])
                force_cursor += 1
                image = packed // components
                slot = packed % components
                atom = slot // 3
                component = slot % 3
                value, column = force_component_jacobian(
                    state_w, element_index, n_fp, fingerprints[image], dg[image], atom, component
                )
                residual = FORCE_WEIGHT * (float(forces[image, atom, component]) - value)
                column = FORCE_WEIGHT * column
            state_w, state_p, eta, q = reference_kalman_update(
                state_w,
                state_p,
                column,
                residual,
                eta,
                q,
                etatau=KALMAN_ETATAU / total,
                etamax=KALMAN_ETAMAX,
                qtau=KALMAN_QTAU / total,
                qmin=KALMAN_QMIN,
            )
        assert session.epoch_mixed(energy_order, force_ids, schedule) == 0
        assert session.get_weights() == pytest.approx(state_w, rel=1e-8, abs=1e-8)
        assert session.updates() == int(schedule.size)
    finally:
        session.close()
