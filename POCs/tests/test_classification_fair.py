"""Protocol tests that do not require Qiskit."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit

from src.classification_fair import (
    FairConfig,
    encode_for_zz_kernel,
    subsample_stratified,
)


def test_encode_shapes_and_range():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 7))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    y = np.array([0] * 40 + [1] * 40)
    tr, te = next(sss.split(X, y))
    Xtr, Xte, var = encode_for_zz_kernel(
        X[tr], X[te], n_qubits=2, random_state=0, already_zz_space=False
    )
    assert Xtr.shape == (len(tr), 2)
    assert Xte.shape == (len(te), 2)
    assert 0 < var <= 1
    assert Xtr.min() >= -np.pi - 1e-9
    assert Xtr.max() <= np.pi + 1e-9


def test_native_space_not_rescaled():
    Xtr = np.array([[-np.pi, 0.0], [np.pi, 1.0]])
    Xte = np.array([[0.0, -1.0]])
    out_tr, out_te, var = encode_for_zz_kernel(
        Xtr, Xte, n_qubits=2, random_state=0, already_zz_space=True
    )
    assert var == 1.0
    np.testing.assert_array_equal(out_tr, Xtr)
    np.testing.assert_array_equal(out_te, Xte)


def test_subsample_preserves_both_classes():
    X, y = make_classification(
        n_samples=400, n_features=6, n_informative=4, n_redundant=0, random_state=1
    )
    Xs, ys = subsample_stratified(X, y, n_total=100, random_state=1)
    assert len(ys) == 100
    assert set(ys) == {0, 1}


def test_fair_config_defaults():
    cfg = FairConfig()
    assert cfg.n_qubits == 2
    assert cfg.n_repeats == 5
    assert cfg.svm_C == 1.0
