# Regression — QML v2

TFM record: [`POCs/notebooks/03_regresion_VQC_vs_clasico.ipynb`](../POCs/notebooks/03_regresion_VQC_vs_clasico.ipynb)  
Protocol: [`POCs/qml_v2/regression.py`](../POCs/qml_v2/regression.py)

## What notebook 03 did wrong (same pattern as classification)

- **Friedman:** QK-SVR on a subset; classical also often on a subset in places, but VQC on an even smaller slice; hyperparameter search only for quantum.
- **Quantum-native:** continuous \(y=\langle ZZ\rangle\) — fairer cell; still the positive-control story.
- **SKG speeds:** classical on the **full** corridor table; QK-SVR on ~150 rows after PCA to 2. Absolute Mac path.

## QML v2 tables

- **Table A:** same rows, same `n_qubits` features in ZZ space, QK-SVR vs SVR-RBF / Ridge / GB / RF. Metric = \(R^2\), RMSE, MAE.
- **Table B:** same rows; classical keeps all columns. QK-SVR stays at `n_qubits`.
- **friedman_full:** 5 features → 5 qubits, **no PCA** (matched-width arm).
- **SKG:** features = `tid`, `readings`, `hour` only (dropped `los_enc` / `elapsed_time` near-target leakage).
- Defaults: `n_total=200`, `n_repeats=10`.

```bash
cd POCs
python3 -m qml_v2.regression --quick
python3 -m qml_v2.regression --n-total 200 --n-repeats 10
python3 -m qml_v2.regression --datasets friedman_full --n-total 100 --n-repeats 3
```

Results land in [`regression_v2_results.md`](regression_v2_results.md).

## Previous results (n=100, 3 repeats, 2 qubits) — pre-hardening baseline

| Dataset | Table A | Read |
|---|---|---|
| Quantum-native (\(y=\langle ZZ\rangle\)) | QK-SVR **R²=0.977** vs RF **0.200** (Δ **+0.78**) | Positive control works for regression too (under a fair protocol) |
| Friedman #1 | QK-SVR **−0.39** vs Ridge **0.17** on PCA-2; Table B SVR-RBF **0.79** on 5 features | PCA-2 destroys Friedman structure; full features stay classical |
| SKG corridor speeds | QK-SVR **−0.17** vs SVR-RBF **0.47**; Table B RF **0.57** | Classical wins on practice data (old feature set with leakage) |

Note: the TFM claimed quantum-native regression failed; with matched features and no destructive re-scaling of ZZ angles, QK-SVR recovers the \(\langle ZZ\rangle\) target. Re-run after hardening for updated SKG / friedman_full numbers.
