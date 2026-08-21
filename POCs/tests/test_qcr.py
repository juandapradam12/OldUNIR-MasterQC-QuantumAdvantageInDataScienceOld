"""QCR pre-map unit tests (no Qiskit)."""

import numpy as np

from qml_v2.qcr import (
    QCRConfig,
    apply_rotation,
    apply_zz_interactions,
    diagnose_bottleneck,
)


def test_zz_interactions_shape():
    rng = np.random.RandomState(0)
    X = rng.uniform(-np.pi, np.pi, size=(40, 2))
    out = apply_zz_interactions(X, n_qubits=2)
    assert out.shape == (40, 2)
    assert out.min() >= -np.pi - 1e-6
    assert out.max() <= np.pi + 1e-6


def test_rotation_preserves_shape():
    rng = np.random.RandomState(1)
    X = rng.uniform(-np.pi, np.pi, size=(30, 2))
    out = apply_rotation(X, theta=0.3)
    assert out.shape == (30, 2)


def test_diagnose_prefers_interactions_when_delta_a_nonpositive():
    assert diagnose_bottleneck({"delta_A": -0.1, "d_F": 0.4}) == "zz_interactions"
    assert diagnose_bottleneck({"delta_A": 0.1, "d_F": 0.1}) == "rotation"
    assert diagnose_bottleneck({"delta_A": 0.1, "d_F": 0.4}) == "reuploading"


def test_qcr_config_defaults():
    cfg = QCRConfig()
    assert cfg.n_qubits == 2
    assert 2 in cfg.reps_grid
