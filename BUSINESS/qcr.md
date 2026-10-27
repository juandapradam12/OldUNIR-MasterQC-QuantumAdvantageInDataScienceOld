# QCR — Quantum-Compatible Representation (pre-maps)

TFM record: [`POCs/notebooks/09_mapeo_cuantico_compatible.ipynb`](../POCs/notebooks/09_mapeo_cuantico_compatible.ipynb)  
Protocol: [`POCs/qml_v2/qcr.py`](../POCs/qml_v2/qcr.py)  
Score: robust [`quantumness.py`](../POCs/qml_v2/quantumness.py) \(\mathcal{Q}_{\mathrm{rob}}\)

## Idea

If baseline verdict is `classical`, try a pre-map \(\psi\) so the ZZ kernel sees more structure, then re-score.

| Strategy | What it does | When recommended |
|---|---|---|
| `zz_interactions` | \(z_{jk}=x_j(\pi-x_k)\), PCA back to \(n_q\) | \(\Delta A\le 0\) |
| `rotation` | 2D orthogonal θ maximizing Q_rob on a **selection** split | \(\Delta A>0\) but low \(d_F\) |
| `reuploading` | choose ZZFeatureMap `reps` maximizing Q_rob | otherwise |
| `qcr_auto_best` | best empirical \(\Delta Q_{\mathrm{rob}}\) | always reported |
| `qcr_heuristic` | bottleneck rule (can disagree with auto) | diagnostic |

Hyperparameters (θ, reps) are chosen on a selection subsample; final numbers use multi-repeat \(\mathcal{Q}_{\mathrm{rob}}\) + advantage.

## Run

```bash
cd POCs
python3 -m qml_v2.qcr --quick
python3 -m qml_v2.qcr --n-sample 80 --n-repeats 3
```

## Results (n=80, 3 repeats)

Ver [`qcr_v2_results.md`](qcr_v2_results.md). Headline:

| Dataset | Efecto QCR |
|---------|------------|
| native_xor | Ya try-ZZ; rotación da boost menor |
| telco_churn | Sin efecto útil (sigue classical) |
| skg_churn | Rotación → try-ZZ geométrico (+5.6 pp vs RBF); **sin** lift de accuracy vs GB/RF. SKG **no** es cuántico-nativo. |

La heurística falló en SKG (sugirió `zz_interactions`); **`qcr_auto_best`** encuentra la rotación, pero eso es señal de geometría — no rescate operativo.

## Classification probe (cierra Q_rob → ML)

```bash
cd POCs
python3 -m qml_v2.qcr --class-probe skg --class-n-total 200 --class-n-repeats 5
```

Compara Table A baseline vs `A_matched_features+qcr_rot` (θ seleccionado por Q_rob,
scaler solo en train).

**SKG result (n=200, 5 repeats, θ≈2.22 rad):**

| Table | QSVM acc | Best classical | Verdict |
|-------|----------|----------------|---------|
| A baseline | 0.500±0.086 | LR 0.513 | tie |
| A + QCR rot | 0.483±0.031 | LR 0.517 | classical |

Subir \(\mathcal{Q}_{\mathrm{rob}}\) **no** se tradujo en accuracy QSVM aquí.
QCR sigue siendo útil como filtro de geometría; no sustituye Table A/B.

## Relación con el resto del TFM

```
quantumness (¿hay señal ZZ?)
    ↓ no / borderline
qcr (¿un premapa barato la crea?)
    ↓ sí
classification / regression con ZZ (Table A/B)
```

SKG y Telco siguen siendo **datos de práctica**, no clientes. QCR no
cambia eso: solo dice si merece la pena invertir en el feature map ZZ.
