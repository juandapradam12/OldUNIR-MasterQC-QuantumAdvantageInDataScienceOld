"""Fair clustering unit tests (no Qiskit kernel eval)."""

import numpy as np

from qml_v2.clustering import (
    ClusterConfig,
    ClusterRepeatResult,
    _fit_predict_models,
    make_blobs_dataset,
    make_moons_dataset,
    summarize,
)


def test_moons_blobs_shapes():
    X, y = make_moons_dataset(n_samples=80, random_state=0)
    assert X.shape == (80, 2)
    assert set(np.unique(y)) == {0, 1}
    Xb, yb = make_blobs_dataset(n_samples=80, random_state=0)
    assert Xb.shape[0] == 80 and Xb.shape[1] == 4


def test_classical_fit_predict_two_clusters():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 0.3, (40, 2)), rng.normal(3, 0.3, (40, 2))])
    preds = _fit_predict_models(X, n_clusters=2, random_state=0, gamma=1.0, K_zz=None)
    assert set(preds) == {"KMeans", "Spectral-RBF", "Agglomerative"}
    for p in preds.values():
        assert p.shape == (80,)
        assert len(np.unique(p)) <= 2


def test_spectral_zz_accepts_precomputed_kernel():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(30, 2))
    K = np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    preds = _fit_predict_models(X, n_clusters=2, random_state=1, gamma=1.0, K_zz=K)
    assert "Spectral-ZZ" in preds
    assert preds["Spectral-ZZ"].shape == (30,)


def test_summarize_delta():
    rows = [
        ClusterRepeatResult(0, "KMeans", "A_matched_features", 0.5, 0.5, 40, 2),
        ClusterRepeatResult(1, "KMeans", "A_matched_features", 0.6, 0.55, 40, 2),
        ClusterRepeatResult(0, "Spectral-ZZ", "A_matched_features", 0.8, 0.7, 40, 2),
        ClusterRepeatResult(1, "Spectral-ZZ", "A_matched_features", 0.9, 0.75, 40, 2),
    ]
    s = summarize(rows)
    zz = s[s["model"] == "Spectral-ZZ"].iloc[0]
    assert zz["delta_ari_vs_best_classical"] > 0


def test_cluster_config_defaults():
    cfg = ClusterConfig()
    assert cfg.n_clusters == 2
    assert cfg.n_qubits == 2
