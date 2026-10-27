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
| skg_churn | classical | rotation (θ≈74°) | +0.018 | try-ZZ (geom.) | +5.6 pp vs RBF | zz_interactions ✗ |

## Lectura

1. **SKG:** QCR cambia el veredicto *geométrico* (classical → try-ZZ) y
   +5.6 pp de ventaja kernel vs RBF, pero SKG **no** es cuántico-nativo:
   el probe Table A no traduce eso en lift de accuracy vs GB/RF.
2. **Telco:** QCR no mueve la aguja; permanece classical.
3. **Native:** ya era `try-ZZ`; rotación da un boost menor.

## Classification probe (SKG)

θ≈2.22 rad · Table A · n=200 · 5 repeats:

- Baseline: QSVM ≈ classical (tie).
- Tras rotación: QSVM **no mejora** (0.483 vs 0.500).

Conclusión operativa: `try-ZZ` vía QCR es señal de geometría, no garantía
de lift supervisado. Siempre cerrar con Table A/B.

## Implicación TFM

QCR es un **filtro de geometría** (a veces mueve SKG a try-ZZ), no un
rescate operativo ni un default de despliegue. En el pipeline:

```
quantumness → si classical y ΔA≈0 → qcr_auto_best → re-evaluar Q_rob
           → si sigue classical → no forzar ZZ / QSVM
```
