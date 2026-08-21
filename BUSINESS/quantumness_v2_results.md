# Quantumness Q(D) results (QML v2 — robust)

Primary score: `Q_rob = ΔA_c + λ · gate(ΔA_c) · d̂_F`
(centered alignment vs best classical; geometry gated by ΔA).
Verdict uses CI lower bound over stratified repeats.

Config: `{"n_qubits": 2, "zz_reps": 2, "entanglement": "full", "n_sample": 80, "test_size": 0.3, "random_state": 42, "n_repeats": 5, "lambda_geom": 0.35, "gate_kappa": 20.0, "d_f_ref": 1.0, "svm_C": 1.0, "w_delta_a": 0.4, "w_d_f": 0.3, "w_delta_h": 0.3, "ci_low": 0.1, "ci_high": 0.9, "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0], "alpha_n": 80}`

## quantum_native

```
Q_rob=+0.2084±0.0245  CI[+0.1815,+0.2340]  → try-ZZ
ΔA=+0.2058±0.0244  d_F=0.0077  gate=0.982  geom_term=+0.0026  g=3764.3752
classical_best=rbfσ=10.249  acc_QSVM=0.842  acc_RBF=0.542  advantage_pp=+30.00±4.08
(legacy Q_TFM=+0.1883, Q_ungated=+0.1265)
```

## telco

```
Q_rob=-0.1081±0.0304  CI[-0.1394,-0.0714]  → classical
ΔA=-0.1084±0.0302  d_F=0.0080  gate=0.116  geom_term=+0.0003  g=3800.3278
classical_best=rbfσ=14.403  acc_QSVM=0.750  acc_RBF=0.750  advantage_pp=+0.00±0.00
(legacy Q_TFM=+0.1269, Q_ungated=-0.0618)
```

## skg

```
Q_rob=-0.0249±0.0136  CI[-0.0387,-0.0087]  → classical
ΔA=-0.0259±0.0134  d_F=0.0081  gate=0.375  geom_term=+0.0011  g=3449.8552
classical_best=rbfσ=18.981  acc_QSVM=0.517  acc_RBF=0.558  advantage_pp=-4.17±10.87
(legacy Q_TFM=+0.1285, Q_ungated=-0.0123)
```

## alpha sweep

Spearman(α, Q_rob) = `1.000`

```
 alpha     Q_rob  Q_rob_lo  Q_rob_hi   delta_A  geom_gate  advantage_pp   verdict
  0.00 -0.544302 -0.544302 -0.544302 -0.544302   0.000019    -30.000000 classical
  0.25 -0.355058 -0.355058 -0.355058 -0.355060   0.000823    -33.333333 classical
  0.50 -0.141958 -0.141958 -0.141958 -0.142106   0.055090    -12.500000 classical
  0.75  0.119529  0.119529  0.119529  0.117079   0.912263     14.166667    try-ZZ
  1.00  0.216854  0.216854  0.216854  0.214205   0.986402     19.166667    try-ZZ
```

