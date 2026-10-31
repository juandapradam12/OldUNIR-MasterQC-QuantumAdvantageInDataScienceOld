"""Fair anomaly-detection unit tests (no Qiskit kernel eval)."""

import numpy as np

from qml_v2.anomaly import (
    AnomalyConfig,
    AnomalyRepeatResult,
    _roc_ap_f1,
    _sklearn_outlier_to_binary,
    fit_score_classical,
    inject_feature_outliers,
    make_blob_anomaly,
    make_moons_anomaly,
    set_contamination,
    summarize,
)


def test_set_contamination_rate():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    y = np.array([0] * 150 + [1] * 50)
    Xs, ys = set_contamination(X, y, contamination=0.10, n_total=100, random_state=0)
    assert len(ys) == 100
    assert abs(ys.mean() - 0.10) < 1e-9
    assert set(ys) == {0, 1}


def test_blob_and_moons_shapes():
    Xb, yb = make_blob_anomaly(n_total=80, contamination=0.1, random_state=0)
    assert Xb.shape == (80, 4)
    assert abs(yb.mean() - 0.1) < 1e-9
    Xm, ym = make_moons_anomaly(n_total=80, contamination=0.1, random_state=0)
    assert Xm.shape == (80, 2)
    assert abs(ym.mean() - 0.1) < 1e-9


def test_inject_does_not_use_business_labels():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(100, 5))
    X2, y = inject_feature_outliers(X, contamination=0.1, random_state=1)
    assert X2.shape == X.shape
    assert y.sum() == 10
    # injected rows should be far from the original mean
    out = X2[y == 1]
    mu = X.mean(axis=0)
    dist = np.linalg.norm(out - mu, axis=1).mean()
    assert dist > 3.0


def test_classical_fit_on_inliers_only():
    rng = np.random.RandomState(0)
    X_in = rng.normal(0, 0.4, size=(60, 2))
    X_te = np.vstack([rng.normal(0, 0.4, size=(20, 2)), rng.uniform(-6, 6, size=(8, 2))])
    y_te = np.array([0] * 20 + [1] * 8)
    scores = fit_score_classical(
        X_in,
        X_te,
        contamination=0.1,
        nu=0.05,
        n_estimators=50,
        n_neighbors=10,
        random_state=0,
    )
    assert set(scores) == {"IsolationForest", "OCSVM-RBF", "LOF"}
    auc, ap, f1 = _roc_ap_f1(y_te, scores["IsolationForest"][0], scores["IsolationForest"][1])
    assert auc > 0.7


def test_sklearn_outlier_mapping():
    pred = np.array([1, 1, -1, 1, -1])
    np.testing.assert_array_equal(_sklearn_outlier_to_binary(pred), [0, 0, 1, 0, 1])


def test_summarize_delta():
    rows = [
        AnomalyRepeatResult(0, "IsolationForest", "A_matched_features", 0.6, 0.3, 0.2, 40, 20, 2, 2),
        AnomalyRepeatResult(1, "IsolationForest", "A_matched_features", 0.7, 0.4, 0.3, 40, 20, 2, 2),
        AnomalyRepeatResult(0, "OCSVM-ZZ", "A_matched_features", 0.85, 0.6, 0.4, 40, 20, 2, 2),
        AnomalyRepeatResult(1, "OCSVM-ZZ", "A_matched_features", 0.90, 0.7, 0.5, 40, 20, 2, 2),
    ]
    s = summarize(rows)
    zz = s[s["model"] == "OCSVM-ZZ"].iloc[0]
    assert zz["delta_auc_vs_best_classical"] > 0


def test_anomaly_config_defaults():
    cfg = AnomalyConfig()
    assert cfg.contamination == 0.10
    assert cfg.nu == 0.05
    assert cfg.n_qubits == 2
