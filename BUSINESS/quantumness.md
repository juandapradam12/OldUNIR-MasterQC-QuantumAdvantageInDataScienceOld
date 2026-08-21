# Quantumness — detect ZZ structure with Q(D)

TFM record: [`POCs/notebooks/08_test_cuanticidad_datasets.ipynb`](../POCs/notebooks/08_test_cuanticidad_datasets.ipynb)  
Protocol: [`POCs/qml_v2/quantumness.py`](../POCs/qml_v2/quantumness.py)

## Idea

Before training QSVM, ask whether the **ZZ fidelity kernel** is structurally preferred over a classical RBF on the same ZZ-ready features.

\[
\mathcal{Q}(D)=w_1\Delta A+w_2\hat d_F+w_3\Delta H
\]

| Term | Meaning |
|---|---|
| \(\Delta A\) | Kernel alignment: ZZ vs RBF vs labels |
| \(\hat d_F\) | Frobenius gap ZZ vs best RBF approximant |
| \(\Delta H\) | Spectral entropy (anti-concentration) |

\(\mathcal{Q}>0\) → ZZ-preferred; \(\mathcal{Q}<0\) → classical-ok.

Also reports:
- \(g\) (geometric difference)
- \(\mathcal{Q}_{AH}\) = score **without** \(\Delta H\) (entropy is the noisiest TFM term)
- matched QSVM vs SVM-RBF advantage on a holdout of the subsample

\(\Delta H\) enters \(\mathcal{Q}\) scaled by \(\log n\).

## Run

```bash
cd POCs
python3 -m qml_v2.quantumness --quick
python3 -m qml_v2.quantumness --n-sample 80
python3 -m qml_v2.quantumness --datasets alpha_sweep --n-sample 80
```

Results: [`quantumness_v2_results.md`](quantumness_v2_results.md).

## Validation

- **quantum_native** should score \(\mathcal{Q}>0\) and positive QSVM advantage.
- **alpha sweep** \(\alpha\in[0,1]\): Spearman\((\alpha,\mathcal{Q})\) should be positive (structure dial).
- Telco / SKG: expect classical-ok or weak \(\mathcal{Q}\) under 2-qubit PCA.
