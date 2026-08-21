# QCR v2 — resultados (POCs/qml_v2)

**Fecha:** 2026-03-21  
**Protocolo:** `qml_v2.qcr` · `n=80` · `repeats=3` · `seed=42` · robust \(\mathcal{Q}\)

```bash
cd POCs && python3 -m qml_v2.qcr --n-sample 80 --n-repeats 3
```

## Resumen

| Dataset | Baseline | Mejor premapa | ΔQ_rob | Verdict post | Advantage post | Heurística |
|---------|----------|----------------|--------|--------------|----------------|------------|
| native_xor | try-ZZ | rotation (θ≈74°) | +0.0018 | try-ZZ | +30.0 pp | rotation ✓ |
| telco_churn | classical | rotation (θ≈74°) | ≈0 | classical | −15.6 pp | rotation ✓ |
| skg_churn | classical | **rotation (θ≈74°)** | **+0.018** | **try-ZZ** | **+5.6 pp** | zz_interactions ✗ |

## Lectura

1. **SKG:** el único caso donde QCR cambia el veredicto. Baseline
   \(\mathcal{Q}_{\mathrm{rob}}<0\) sin ventaja; tras rotación global →
   `try-ZZ` y +5.6 pp de ventaja kernel. La heurística falló (sugirió
   `zz_interactions` por \(d_F\) alto) — usar **`qcr_auto_best`**.
2. **Telco:** QCR no mueve la aguja. Cuello de botella de datos/features,
   no de geometría ZZ. Coherente con Table A/B y quantumness.
3. **Native:** ya era `try-ZZ`; rotación da un boost menor. Reuploading
   empeora (sobre-expresión del feature map).

## Implicación TFM

QCR es útil como **rescate condicional** (SKG), no como default. En el
pipeline operativo:

```
quantumness → si classical y ΔA≈0 → qcr_auto_best → re-evaluar Q_rob
           → si sigue classical → no forzar ZZ / QSVM
```
