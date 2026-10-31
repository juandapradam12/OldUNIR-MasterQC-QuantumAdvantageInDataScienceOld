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
| `07_sintesis_resultados.csv` | `scripts/sintesis_resultados.py`: consolida los nueve casos leyendo solo CSV/JSON ejecutados; es la única fuente del panel y del mapa de calor de síntesis |
| `01_clasificacion_*.csv`, `03_regresion_*.csv`, `04_portafolio_resultados.csv`, `06_*.csv` | Notebooks 01, 03, 04 y 06 reejecutados de principio a fin (Qiskit 2.6.1, kernel exacto) |
| `qml_v2_*_summary.csv`, `fair_*.csv` | Módulos `python -m qml_v2.<módulo>` (semillas y repeticiones en los `*_config.json`) |
| `fair_scalability_native*.csv` | `python -m qml_v2.scalability` (6 tamaños × 10 repeticiones; kernel de fidelidad exacto verificado contra `FidelityQuantumKernel`) |
| `10_mds_qaoa_*` | `python -m qml_v2.mds_qaoa` (simulador exacto verificado contra Qiskit) |
| `estructura_por_caso_uso.csv` | Script `medir_estructura_por_caso.py`: ΔA, ΔH, d_F, g (+control nulo de ΔA) en clasif./reg./anomalías/optimización (paisaje QUBO); lógica en `qml_v2/estructura_casos.py`; notebook 11 |
| `qml_v2_portfolio_*` | `python -m qml_v2.portfolio` (Caso 4: 10 semillas, dos penalizaciones; precios de `data/output/input/portfolio_prices.csv`); los `tiempo_s` son orientativos (3 procesos en paralelo) |
| `04_portafolio_resultados.csv` | Notebook 04, una ejecución (semilla 0) |
| `05_tsp_resultados.csv` | Notebook 05 (Colombia, 12 capitales; QAOA y SA-QUBO sobre las 4 primeras) |
| `qml_v2_tsp_*` | `python -m qml_v2.tsp_optimization` |
| `07_sintesis_resultados.csv` | `scripts/sintesis_resultados.py` (lee los demás CSV; ningún valor a mano) |

Reglas:
1. Si un artefacto falta, **no inventarlo**. Re-ejecutar el notebook/script o omitir la figura.
2. `advantage_pp_rbf` (NB08) ≠ `advantage_pp_op` (clasificación / NB06): no mezclar en una ρ ni en figuras de estructura.
3. Figuras de estructura cuántica usan **solo** ΔA, ΔH, d_F, g (+Q informativo) (mismas métricas entre casos de uso). En optimización se mide el paisaje QUBO (bitstrings→energía), no QAOA. No mezclar accuracy / R² / AUC ahí.
4. Matrices $K$ no serializadas: no regenerar heatmaps a mano.
5. Series de rotación: solo PNG exportada por NB09; no curvas proxy.
6. Los scripts de figuras (`generar_figuras_*.py`) leen estos CSV y fallan si falta alguno; no contienen valores de respaldo. Las únicas figuras no empíricas son las teóricas del Capítulo 5 (complejidad, conteo de Grover, superposición), rotuladas como tales.
7. Los muestreadores con *shots* deben llevar semilla; los kernels de los Notebooks 01, 03 y 06 usan la fidelidad exacta.
8. `scripts/verificar_cifras_tesis.py` comprueba que las cifras de los capítulos coinciden con estos CSV; debe terminar sin faltas antes de compilar.
9. El notebook 07 no contiene valores propios: ejecuta la síntesis y el panel a partir de los CSV de los demás casos.
