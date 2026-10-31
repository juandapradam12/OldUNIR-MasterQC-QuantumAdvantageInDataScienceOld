"""Time-series lag protocol tests (no Qiskit)."""

import numpy as np
import pandas as pd
import pytest

from qml_v2.timeseries import (
    TSConfig,
    enrich_classical_features,
    lag_block,
    load_skg_corridors,
    make_lag_matrix,
    make_sine_series,
    naive_references,
    repeat_windows,
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


def test_lag_block_aligns_table_a_with_target():
    cfg = TSConfig(n_lags=2, n_extra_lags=4)
    series = np.arange(30, dtype=float)
    Xa, y, Xb = lag_block(series, cfg, hour=np.arange(30) % 24)
    assert Xa.shape == (24, 2) and Xb.shape[0] == 24
    np.testing.assert_array_equal(Xa[:, -1], y - 1)
    assert Xb.shape[1] == 2 + 4 + 3 + 2


def test_repeat_windows_are_contiguous_and_seeded():
    cfg = TSConfig(n_total=20, n_repeats=3)
    series = np.arange(100, dtype=float)
    blocks = [lag_block(series, cfg)]
    first = [(r, ya.copy()) for r, _, ya, _ in repeat_windows(blocks, cfg)]
    again = [(r, ya.copy()) for r, _, ya, _ in repeat_windows(blocks, cfg)]
    assert len(first) == 3
    for (_, a), (_, b) in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
        assert len(a) == 20 and np.all(np.diff(a) == 1)


def test_naive_references_persistence_on_a_ramp():
    cfg = TSConfig(n_total=40, n_repeats=2)
    blocks = [lag_block(np.arange(60, dtype=float), cfg)]
    refs = naive_references(blocks, cfg)
    assert np.allclose(refs["persistence"], 1.0)
    assert (refs["train_mean"] > refs["persistence"]).all()


def test_load_skg_corridors_orders_by_readings(tmp_path):
    rows = [
        (tid, f"2023-11-01 0{minute // 60}:{minute % 60:02d}:00", float(minute))
        for tid, n in ((1, 5), (2, 8))
        for minute in range(n)
    ]
    path = tmp_path / "speeds.csv"
    pd.DataFrame(rows, columns=["tid", "read_time", "speed"]).to_csv(path, index=False)
    corridors = load_skg_corridors(2, path)
    assert [len(speed) for speed, _ in corridors] == [8, 5]
    with pytest.raises(FileNotFoundError):
        load_skg_corridors(1, tmp_path / "missing.csv")
