# Quantumness — robust ZZ structure score Q_rob

TFM record: [`POCs/notebooks/08_test_cuanticidad_datasets.ipynb`](../POCs/notebooks/08_test_cuanticidad_datasets.ipynb)  
Protocol: [`POCs/qml_v2/quantumness.py`](../POCs/qml_v2/quantumness.py)

## Hardening

| Issue | Fix |
|---|---|
| \(d_F\) alone → false “quantum” | Geometry **gated** by \(\Delta A\) |
| \(\Delta H\) volatile | Diagnostic only (out of \(\mathcal{Q}_{\mathrm{rob}}\)) |
| Uncentered alignment | **Centered** Gram matrices |
| Classical σ peeking at labels | Baseline = **Frobenius-best RBF** (no label fit) |
| Single subsample noise | `n_repeats` + percentile CI |
| Alignment ≠ accuracy | Verdict guardrail on empirical advantage |

## Score

\[
\mathcal{Q}_{\mathrm{rob}}
=
\Delta A_c
+
\lambda\cdot\sigma(\kappa\cdot\Delta A_c)\cdot\hat d_F
\]

## Verdicts

| Verdict | Rule |
|---|---|
| `try-ZZ` | CI lower \(>0\), \(\Delta A>0\), advantage ≳ 0 |
| `ZZ-unstable` | mean looks good but CI or advantage disagree |
| `ZZ-geometry-only` | Q>0 but \(\Delta A\le 0\) |
| `classical` | otherwise |

## Run

```bash
cd POCs
python3 -m qml_v2.quantumness --quick
python3 -m qml_v2.quantumness --n-sample 80 --n-repeats 5
```

Results: [`quantumness_v2_results.md`](quantumness_v2_results.md).
