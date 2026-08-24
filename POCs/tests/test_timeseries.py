"""Time-series lag protocol tests (no Qiskit)."""

import numpy as np

from qml_v2.timeseries import (
    TSConfig,
    enrich_classical_features,
    make_lag_matrix,
    make_sine_series,
    time_split_indices,
)


def test_lag_matrix_shapes():
    y = np.arange(20, dtype=float)
    X, t = make_lag_matrix(y, n_lags=3)
    assert X.shape == (17, 3)
    assert t.shape == (17,)
    np.testing.assert_array_equal(X[0], [0, 1, 2])
    assert t[0] == 3


def test_sine_series_length():
    s = make_sine_series(n_samples=100, random_state=0)
    assert s.shape == (100,)


def test_time_split_is_ordered():
    tr, te = time_split_indices(100, 0.3)
    assert tr.max() < te.min()
    assert len(tr) + len(te) == 100


def test_enrich_adds_columns():
    X = np.random.RandomState(0).normal(size=(50, 2))
    hour = np.arange(50) % 24
    extra = np.random.RandomState(1).normal(size=(50, 3))
    out = enrich_classical_features(X, hour=hour, extra_lags=extra)
    assert out.shape[0] == 50
    assert out.shape[1] > X.shape[1]


def test_ts_config_defaults():
    cfg = TSConfig()
    assert cfg.n_lags == 2
    assert cfg.n_qubits == 2
