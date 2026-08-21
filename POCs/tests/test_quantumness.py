"""Robust quantumness / ZZ-structure tests (mostly no Qiskit)."""

import numpy as np

from qml_v2.quantumness import (
    QuantumnessConfig,
    center_kernel,
    decide_verdict,
    frobenius_distance,
    kernel_alignment,
    labels_pm1,
    logistic_gate,
    quantumness_score,
    quantumness_score_robust,
    spectral_entropy,
)


def test_labels_pm1():
    y = np.array([0, 1, 0, 1])
    assert set(labels_pm1(y).tolist()) == {-1.0, 1.0}


def test_centered_alignment_perfect_block():
    y = np.array([1, 1, -1, -1], dtype=float)
    K = np.outer(y, y).astype(float)
    a = kernel_alignment(K, y, centered=True)
    assert a > 0.99


def test_center_kernel_row_sums_near_zero():
    rng = np.random.RandomState(0)
    K = rng.rand(6, 6)
    K = 0.5 * (K + K.T)
    C = center_kernel(K)
    assert np.allclose(C.sum(axis=0), 0.0, atol=1e-9)


def test_spectral_entropy_identity_high():
    n = 8
    assert abs(spectral_entropy(np.eye(n)) - np.log(n)) < 1e-6


def test_frobenius_normalized_zero_when_equal():
    K = np.eye(4) * 3.0
    assert frobenius_distance(K, K, normalize=True) == 0.0


def test_robust_score_gates_geometry():
    # Negative ΔA: geometry must not flip to try-ZZ
    neg = quantumness_score_robust(-0.2, 0.5, lambda_geom=0.35, gate_kappa=20.0)
    assert neg["Q_rob"] < 0
    assert neg["geom_gate"] < 0.05
    # Positive ΔA: geometry can boost
    pos = quantumness_score_robust(0.2, 0.5, lambda_geom=0.35, gate_kappa=20.0)
    assert pos["Q_rob"] > 0.2
    assert pos["geom_gate"] > 0.95


def test_decide_verdict_ci():
    assert decide_verdict(0.1, 0.05, Q_lo=0.02, advantage_pp=10.0) == "try-ZZ"
    assert decide_verdict(0.1, 0.05, Q_lo=-0.01, advantage_pp=10.0) == "ZZ-unstable"
    assert decide_verdict(0.1, 0.05, Q_lo=0.02, advantage_pp=-10.0) == "ZZ-unstable"
    assert decide_verdict(0.1, -0.05, Q_lo=0.05) == "ZZ-geometry-only"
    assert decide_verdict(-0.1, -0.05, Q_lo=-0.2) == "classical"


def test_logistic_gate_limits():
    assert logistic_gate(-1.0, 20.0) < 0.01
    assert logistic_gate(1.0, 20.0) > 0.99


def test_legacy_score_still_works():
    s = quantumness_score(0.5, 0.1, 0.4, 2.0, 1.5, d_F_ref=1.0, n=40)
    assert "Q_score" in s and "Q_AH" in s


def test_config_defaults():
    cfg = QuantumnessConfig()
    assert cfg.n_repeats == 5
    assert cfg.lambda_geom > 0
