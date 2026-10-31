"""Regression protocol tests (no Qiskit)."""

import numpy as np

from qml_v2.classification import encode_for_zz_kernel
from qml_v2.regression import RegConfig, load_friedman, subsample


def test_reg_config_defaults():
    cfg = RegConfig()
    assert cfg.n_repeats == 10
    assert cfg.n_total == 200


def test_friedman_matched_width_no_pca():
    X, y = load_friedman(60, 5, random_state=0)
    assert X.shape[1] == 5
    Xtr, Xte = X[:40], X[40:]
    out_tr, out_te, var = encode_for_zz_kernel(Xtr, Xte, n_qubits=5, random_state=0, already_zz_space=False)
    assert out_tr.shape[1] == 5
    assert var == 1.0


def test_subsample_size():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(500, 3))
    y = rng.normal(size=500)
    Xs, ys = subsample(X, y, n_total=100, random_state=1)
    assert len(ys) == 100
