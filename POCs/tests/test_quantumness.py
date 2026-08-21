"""Quantumness / ZZ-structure indicator tests (no Qiskit required for most)."""

import numpy as np

from qml_v2.quantumness import (
    QuantumnessConfig,
    frobenius_distance,
    kernel_alignment,
    labels_pm1,
    quantumness_score,
    spectral_entropy,
)


def test_labels_pm1():
    y = np.array([0, 1, 0, 1])
    yp = labels_pm1(y)
    assert set(yp.tolist()) == {-1.0, 1.0}


def test_kernel_alignment_perfect():
    y = np.array([1, 1, -1, -1], dtype=float)
    # Block-diagonal perfect alignment with yy^T structure
    K = np.outer(y, y).astype(float)
    a = kernel_alignment(K, y)
    assert a > 0.99


def test_spectral_entropy_identity_high():
    n = 8
    K = np.eye(n)
    h = spectral_entropy(K)
    # uniform spectrum → H = log(n)
    assert abs(h - np.log(n)) < 1e-6


def test_frobenius_zero_when_equal():
    K = np.eye(4)
    assert frobenius_distance(K, K) == 0.0


def test_quantumness_score_sign():
    # Positive ΔA and d_F → positive Q (with mild negative ΔH)
    scored_pos = quantumness_score(0.5, 0.1, 0.4, 2.0, 1.5, d_F_ref=1.0, n=40)
    scored_neg = quantumness_score(0.1, 0.5, 0.05, 1.0, 2.5, d_F_ref=1.0, n=40)
    assert scored_pos["Q_score"] > 0
    assert scored_neg["Q_score"] < 0
    assert scored_pos["Q_AH"] > scored_neg["Q_AH"]


def test_config_defaults():
    cfg = QuantumnessConfig()
    assert cfg.n_sample == 80
    assert abs(cfg.w_delta_a + cfg.w_d_f + cfg.w_delta_h - 1.0) < 1e-9
