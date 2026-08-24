# Anomaly detection — fair one-class OCSVM-ZZ vs classical

Protocol: [`POCs/qml_v2/anomaly.py`](../POCs/qml_v2/anomaly.py)

## Not fraud

Fraud detection is a **labelled rare-class classification** problem
(``classification.py``). This module is the general case: **novelty /
one-class detection**.

| | Anomaly (this module) | Fraud (not this module) |
|---|---|---|
| Train labels | inliers only | both classes |
| Detector | IsolationForest, OCSVM, LOF | SVC / GB / QSVM |
| Evaluation labels | held-out only | train + test |
| SKG / Telco | **injected** feature outliers | would use churn/gravity — avoided |

## Idea

Fit on **train inliers only**. Scaler / PCA also fit on inliers only.
Evaluate ROC-AUC and average precision on a test split that contains outliers.

| Model | Role |
|---|---|
| IsolationForest | Classical density / isolation baseline |
| OCSVM-RBF | Classical kernel one-class |
| LOF | Classical local density |
| **OCSVM-ZZ** | One-class SVM with ZZ fidelity kernel |

## Run

```bash
cd POCs
python3 -m qml_v2.anomaly --quick
python3 -m qml_v2.anomaly --n-total 160 --n-repeats 5
```

## Datasets

| Dataset | Construction | Expectation |
|---|---|---|
| `zz_phase` | Majority ZZ phase = inlier; other phase = outlier (same support) | OCSVM-ZZ may win Table A |
| `moons` / `blobs` | Manifold / Gaussian + synthetic outliers | classical |
| `telco_injected` / `skg_injected` | Practice tables; **discard business labels**; inject far spikes | Table B IsolationForest |

SKG / Telco remain **practice data only**.

Results: [`anomaly_v2_results.md`](anomaly_v2_results.md).
