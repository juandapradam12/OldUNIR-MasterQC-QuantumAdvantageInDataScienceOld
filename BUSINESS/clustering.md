# Clustering — fair Spectral-ZZ vs classical

Protocol: [`POCs/qml_v2/clustering.py`](../POCs/qml_v2/clustering.py)

## Idea

Unsupervised clustering with a matched-feature Table A (ZZ-ready angles) and
a business Table B (classical full features, matched n). Ground-truth labels
are used **only** for ARI / NMI — never for fitting.

| Model | Role |
|---|---|
| KMeans | Classical centroid baseline |
| Spectral-RBF | Classical spectral (RBF affinity) |
| Agglomerative | Classical hierarchical |
| **Spectral-ZZ** | Spectral clustering with ZZ fidelity kernel affinity |

## Run

```bash
cd POCs
python3 -m qml_v2.clustering --quick
python3 -m qml_v2.clustering --n-total 120 --n-repeats 5
```

## Datasets

| Dataset | Expectation |
|---|---|
| `quantum_native` | Spectral-ZZ should win Table A if labels come from ZZ |
| `moons` / `blobs` | Classical spectral / KMeans should win |
| `telco` / `skg` | Practice data only — churn labels as eval proxy |

Results: [`clustering_v2_results.md`](clustering_v2_results.md).

## Headline (n=120, 5 repeats)

| Dataset | Table A winner | Note |
|---------|----------------|------|
| quantum_native | Spectral-ZZ (ARI≈0.05) | Débil pero > classical (~0) |
| moons / blobs | classical | Spectral-RBF / Agglomerative |
| telco / skg | classical / tie | ARI≈0 — churn no es estructura de clusters |

SKG/Telco: labels de churn solo como proxy de evaluación (datos de práctica).
