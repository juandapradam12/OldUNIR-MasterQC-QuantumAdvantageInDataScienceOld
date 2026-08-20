"""Protocol tests that do not require Qiskit."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit

from qml_v2.classification import (
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
    assert cfg.n_repeats == 10
    assert cfg.n_total == 200
    assert cfg.svm_C == 1.0
    assert cfg.class_weight == "balanced"


def test_scores_majority_collapse_has_low_f1():
    from qml_v2.classification import _scores

    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0])  # always majority
    acc, f1, auc, pos = _scores(y_true, y_pred)
    assert acc > 0.5
    assert f1 < 0.5
    assert pos == 0.0


def test_encode_no_pca_when_width_matches_qubits():
    rng = np.random.RandomState(0)
    Xtr = rng.normal(size=(40, 3))
    Xte = rng.normal(size=(10, 3))
    out_tr, out_te, var = encode_for_zz_kernel(
        Xtr, Xte, n_qubits=3, random_state=0, already_zz_space=False
    )
    assert out_tr.shape == (40, 3)
    assert out_te.shape == (10, 3)
    assert var == 1.0

