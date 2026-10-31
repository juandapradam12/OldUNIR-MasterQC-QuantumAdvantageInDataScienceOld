# POCs — Experimentos del TFM

Código y resultados de la comparación cuántico vs clásico sobre nueve casos de uso. Todos los números de la tesis salen de las salidas ejecutadas en `data/output/metrics/`.

## Puesta en marcha

```bash
cd POCs
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt           # versiones mínimas
pip install -r requirements-lock.txt      # alternativa: versiones exactas del entorno de reejecución (Python 3.12.3)
pytest tests/ -q
ruff check . && ruff format --check .
```

## Tecnologías

- **Cuántico:** Qiskit 2.6.1 (`qiskit-algorithms` 0.4.0, `qiskit-machine-learning` 0.9.1, `qiskit-optimization` 0.7.0, `qiskit-finance` 0.4.1)
- **Clásico:** scikit-learn 1.9.1, XGBoost 3.4.1, SciPy 1.18.1
- **Análisis:** pandas 3.1.0, NumPy 2.5.4, matplotlib 3.11.2, seaborn 0.13.2, NetworkX 3.7
- **Optimización:** PuLP 2.9.0 (CBC) y `scipy.optimize.milp` (HiGHS)
- **Entorno de reejecución:** Python 3.12.3 en Linux; versiones exactas en `requirements-lock.txt` (el entorno del autor: Python 3.14, Qiskit 2.3.0; véase el Anexo A de la tesis)

## Estructura

```
POCs/
├── qml_v2/                 # Protocolos justos (un módulo por familia de problema)
│   ├── outputs.py          # Rutas, CSV/informe comunes y CLI compartida
│   ├── kernels.py          # Kernel de fidelidad ZZ y modelos sklearn sobre Gram precalculada
│   ├── classification.py   # Casos 1 y 6: QSVM frente a clásicos (Tablas A y B)
│   ├── regression.py       # Caso 2: QK-SVR
│   ├── optimization.py     # Caso 3: asignación (PuLP, Greedy, QAOA)
│   ├── portfolio.py        # Caso 4: cartera (VQE/QAOA, 10 semillas)
│   ├── tsp_optimization.py # Caso 5: TSP con depósito fijo
│   ├── mds_qaoa.py         # Caso 7: conjunto dominante mínimo
│   ├── anomaly.py          # Caso 8: OCSVM-ZZ one-class
│   ├── timeseries.py       # Caso 9: forecasting de un paso por lags
│   ├── quantumness.py      # Puntuación de cuanticidad Q(D)
│   ├── qcr.py              # Remapeos cuántico-compatibles
│   ├── estructura_casos.py # Indicadores de estructura por caso de uso
│   ├── clustering.py, maxcutm_optimization.py
├── notebooks/              # 01-12: registro ejecutado de cada experimento
├── scripts/                # Figuras, síntesis y verificación de cifras
├── tests/
└── data/
    ├── input/              # Datasets públicos (Telco se descarga si falta)
    └── output/{figures,metrics}/
```

Los datos de SKG están en `../Datos/` y los precios de Yahoo Finance en `data/input/`.

## Casos de uso y cuadernos

| Caso | Problema | Módulo | Cuaderno |
|:---:|---|---|---|
| 1 | Clasificación | `classification` | 01, 01b |
| 2 | Regresión | `regression` | 03 |
| 3 | Optimización (asignación) | `optimization` | 02 |
| 4 | Cartera | `portfolio` | 04 |
| 5 | TSP | `tsp_optimization` | 05 |
| 6 | Clasificación (umbral) | `classification` | 06 |
| 7 | Conjunto dominante mínimo | `mds_qaoa` | 10 |
| 8 | Anomalías | `anomaly` | — (`python -m qml_v2.anomaly`) |
| 9 | Series temporales | `timeseries` | 12 |

Los cuadernos 07 (síntesis), 08 (cuanticidad Q(D)), 09 (remapeos QCR) y 11 (estructura por caso de uso) agregan o diagnostican los resultados de los demás.

## Ejecución

```bash
python -m qml_v2.classification --quick     # humo; sin --quick reproduce las tablas de la tesis
python -m qml_v2.timeseries                 # caso 9
python POCs/scripts/sintesis_resultados.py  # tabla de síntesis (07_sintesis_resultados.csv)
python POCs/scripts/verificar_cifras_tesis.py --auditoria   # cifras del texto frente a los CSV
```

Cada módulo escribe sus métricas en `data/output/metrics/` y un informe en `../BUSINESS/`.
