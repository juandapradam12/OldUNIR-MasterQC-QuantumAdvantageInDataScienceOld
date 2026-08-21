# Quantumness Q(D) results (QML v2)

ZZ fidelity kernel vs best RBF approximant on the same ZZ-ready features.
Q = w1·ΔA + w2·d̂_F + w3·ΔH.  Q>0 → ZZ-preferred.

Config: `{"n_qubits": 2, "zz_reps": 2, "entanglement": "full", "n_sample": 80, "test_size": 0.3, "random_state": 42, "w_delta_a": 0.4, "w_d_f": 0.3, "w_delta_h": 0.3, "d_f_ref": 1.0, "svm_C": 1.0, "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0], "alpha_n": 80}`

## quantum_native

```
Q=+0.1490 (try-ZZ)  Q_AH=+0.2357 (try-ZZ)
ΔA=+0.1088  d_F=0.3626  ΔH_norm=-0.0109  g=1515.1261
acc_QSVM=0.917  acc_RBF=0.583  advantage_pp=+33.33
```

## telco

```
Q=+0.0627 (try-ZZ)  Q_AH=+0.2049 (try-ZZ)
ΔA=+0.0459  d_F=0.3639  ΔH_norm=-0.2161  g=143.3792
acc_QSVM=0.750  acc_RBF=0.750  advantage_pp=+0.00
```

## skg

```
Q=+0.0184 (ZZ-geometry-only)  Q_AH=+0.1426 (ZZ-geometry-only)
ΔA=-0.0423  d_F=0.3276  ΔH_norm=-0.2097  g=71.6820
acc_QSVM=0.667  acc_RBF=0.625  advantage_pp=+4.17
```

## alpha sweep

Spearman(α, Q) = `1.000`

```
 alpha   Q_score      Q_AH   delta_A      d_F  delta_H_norm  advantage_pp
  0.00 -0.046267 -0.006795 -0.370587 0.356996     -0.017104    -25.000000
  0.25  0.000056  0.051109 -0.254779 0.356996     -0.017104    -37.500000
  0.50  0.056953  0.122230 -0.112536 0.356996     -0.017104     -8.333333
  0.75  0.120885  0.202144  0.047293 0.356996     -0.017104     -4.166667
  1.00  0.146902  0.234666  0.112335 0.356996     -0.017104     12.500000
```

