# Procedencia de métricas (criterio de integridad)

**Alterar resultados o datos es inadmisible en este TFM.**

Los CSV de este directorio son la fuente de verdad para figuras regeneradas.
Deben coincidir con la salida impresa de los notebooks (no con valores
“embellecidos”).

| Archivo | Origen |
|---------|--------|
| `06_qkernel_resumen_3datasets.csv` | Notebook 06 / clasificación (Δ vs mejor clásico, pp accuracy) |
| `08_test_cuanticidad_resultados.csv` | Notebook 08, tabla resumen (protocolo NB08) |
| `08_barrido_sintetico_alpha.csv` | Notebook 08, barrido α |
| `09_mapeos_resultados.csv` | Notebook 09, tabla de estrategias |
| `estructura_por_caso_uso.csv` | Script `medir_estructura_por_caso.py`: ΔA, d_F, ΔH en clasif./reg./anomalías (protocolo unificado qml_v2) |

Reglas:
1. Si un artefacto falta, **no inventarlo**. Re-ejecutar el notebook/script o omitir la figura.
2. `advantage_pp_rbf` (NB08) ≠ `advantage_pp_op` (clasificación / NB06): no mezclar en una ρ ni en figuras de estructura.
3. Figuras de estructura cuántica usan **solo** ΔA, d_F, ΔH (mismas métricas entre casos de uso). No mezclar accuracy / R² / AUC ahí.
4. Matrices $K$ no serializadas: no regenerar heatmaps a mano.
5. Series de rotación: solo PNG exportada por NB09; no curvas proxy.
