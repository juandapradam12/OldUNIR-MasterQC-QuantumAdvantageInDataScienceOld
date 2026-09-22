# Computación cuántica vs clásica en ciencia de datos

**TFM · UNIR · Máster en Inteligencia Artificial**  
Juan David Prada Malagón · Dir. Ricardo Rafael Domínguez Reyes · Contexto aplicado: SKG Tecnología

---

## La pregunta del trabajo

> *¿En qué condiciones, y para qué tipos de problemas de ciencia de datos, los algoritmos cuánticos de la era NISQ ofrecen una ventaja real y medible frente a los algoritmos clásicos?*

No es un producto de “Quantum ML para operaciones”. Es un **estudio comparativo empírico y sistemático**: nueve casos de uso, protocolos justos, baselines clásicos fuertes y evidencia reproducible —incluyendo cuándo la cuántica **no** gana.

---

## Abre-bocas (para interesados)

| Qué encontrarás | Detalle |
|---|---|
| **Hallazgo central** | La ventaja es *real pero condicionada*: aparece cuando el **feature map está alineado con el generador de los datos**, o en problemas **combinatorios / QUBO**. Sin esa alineación, el clásico suele ser superior. |
| **Alcance** | Clasificación (QSVM), regresión (QK-SVR), optimización (QAOA/VQE/TSP), anomalías one-class (OCSVM-ZZ) y forecasting por lags — **9 casos**. |
| **Datos** | Tres capas: sintéticos con estructura cuántica nativa, datasets públicos de referencia y datos de práctica de SKG Tecnología. |
| **Aportes originales** | Score de cuanticidad \(\mathcal{Q}(D)\) / \(\mathcal{Q}_{\mathrm{rob}}\) y Framework QCR (**planteado formalmente**; conjetura abierta). |
| **Entregables** | PDF del TFM + código Qiskit (`POCs/qml_v2/`) + figuras versionadas + tests. |

**Leer ya:** [`Documento/main.pdf`](Documento/main.pdf)  
**Código:** [`POCs/qml_v2/`](POCs/qml_v2/) · **Figuras:** [`POCs/data/output/figures/`](POCs/data/output/figures/)

---

## Síntesis de resultados (lectura rápida)

| CU | Algoritmo | Lectura |
|:---:|---|---|
| 1 | QSVM | Ventaja fuerte con datos alineados al mapa; clásico en datos reales sin alineación |
| 2 | QK-SVR | Clásico preferible en regresión operativa típica |
| 3 | QAOA asignación | −30,3 % coste vs greedy (protocolo justificado) |
| 4 | VQE portafolio | Empate / ligera mejora en escala pequeña |
| 5 | QAOA TSP | Empate óptimo vs NN+2-opt — valor de *modelado*, no de speedup |
| 6 | QSVM umbral | Condicional a estructura nativa / umbral |
| 7 | QAOA MDS | Paradigma híbrido cuántico–clásico |
| 8 | OCSVM-ZZ | Gana solo si la novedad está alineada al mapa (`zz_phase`) |
| 9 | QK-SVR lags | Gana solo si el target está alineado al mapa (`zz_lag`) |

**Conclusión práctica del TFM:** algoritmos cuánticos como *herramienta adicional y selectiva*, no como sustitutos del stack clásico.

---

## Repositorio

```
Documento/main.pdf          # TFM compilado
Documento/Capitulos/        # Fuentes LaTeX
POCs/qml_v2/                # Protocolos justos (Python / Qiskit)
POCs/scripts/               # Regeneración de figuras
POCs/data/output/figures/   # PNGs en git
POCs/tests/                 # Suite de regresión
BUSINESS/                   # Notas técnicas por familia experimental
```

```bash
cd POCs
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -q
```

---

## Palabras clave

computación cuántica · quantum machine learning · ventaja cuántica · QSVM · quantum kernel · QAOA · VQE · optimización combinatoria · OCSVM-ZZ · score de cuanticidad · Framework QCR · NISQ · algoritmo híbrido cuántico-clásico

---

*Evidencia condicionada, no hype. El PDF desarrolla el argumento completo.*
