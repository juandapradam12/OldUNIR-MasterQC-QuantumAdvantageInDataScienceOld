# Quantum ML aplicado a operaciones reales

**¿Cuándo aporta la computación cuántica… y cuándo no?**

Este repositorio responde esa pregunta con evidencia, no con hype: **nueve casos de uso** que confrontan algoritmos cuánticos (kernels ZZ, QAOA, VQE) frente a baselines clásicos fuertes, bajo protocolos justos y reproducibles.

> **Hallazgo central:** la ventaja no es “usar cuántica”. Aparece cuando el **feature map está alineado con el generador de los datos**, o cuando el problema es **combinatorio / QUBO**. En datos operativos sin esa alineación, el clásico gana —y eso también es un resultado de negocio.

---

## Para clientes e interesados (30 segundos)

| Si te preocupa… | Lo que entregamos |
|---|---|
| ¿Invertir en cuántica sin saber el ROI? | Un **mapa de decisión**: cuándo activar el módulo cuántico y cuándo quedarte con ML clásico |
| ¿Es solo teoría? | **Código Qiskit ejecutable** + PDF del TFM + figuras versionadas |
| ¿Sirve para mi operación? | Casos en **clasificación, regresión, asignación/optimización, anomalías y forecasting** |
| ¿Hype vs evidencia? | Ventajas medidas (p. ej. **+38 pp** en kernels alineados; **−30 %** coste QAOA vs greedy) y también **derrotas honestas** en datos reales sin alineación |

**Documento listo:** [`Documento/main.pdf`](Documento/main.pdf)  
**Código reproducible:** [`POCs/qml_v2/`](POCs/qml_v2/)  
**Empresa de práctica:** SKG Tecnología · **UNIR** — Máster en IA · Autor: Juan David Prada Malagón

---

## Resultados que importan (síntesis)

| Caso | Qué se compara | Lectura de negocio |
|:---:|---|---|
| 1 | QSVM vs SVM / boosting | Ventaja fuerte solo con datos alineados al mapa |
| 2 | QK-SVR vs Ridge / GB | Clásico preferible en datos reales típicos |
| 3 | QAOA vs greedy (asignación) | **−30,3 %** coste en protocolo justificado |
| 4 | VQE portafolio vs Markowitz | Empate / ligera mejora en escala pequeña |
| 5 | QAOA TSP vs NN+2-opt | Empate óptimo — valor de **modelado QUBO**, no de speedup |
| 6 | Kernel advantage (Liu-style) | Condicional al umbral / estructura nativa |
| 7 | QAOA conjunto dominante | Patrón **híbrido** cuántico–clásico |
| 8 | OCSVM-ZZ vs IsolationForest | Gana solo si la novedad es de tipo *fase* (alineada al mapa) |
| 9 | QK-SVR por lags | Gana solo en series cuyo target está alineado al mapa |

**Regla operativa:** clásico por defecto; cuántica **selectiva** tras diagnóstico de alineación (score \(\mathcal{Q}\)) o formulación QUBO viable.

---

## Qué hay en el repo

```
TFM_ComputacionCuantica/
├── Documento/main.pdf          # TFM compilado (listo para leer)
├── Documento/Capitulos/        # Fuentes LaTeX
├── POCs/qml_v2/                # Protocolos justos (Python, Qiskit)
├── POCs/scripts/               # Regeneración de figuras
├── POCs/data/output/figures/   # PNGs versionados en git
├── POCs/tests/                 # Suite de regresión (58 tests)
└── BUSINESS/                   # Notas de walkthrough por familia
```

### Correr rápido

```bash
cd POCs
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -q
PYTHONPATH=. python3 -m qml_v2.feature_map_alignment --n-total 120 --n-repeats 5
```

Compilar el PDF (opcional; ya está en git):

```bash
cd Documento && tectonic -X compile main.tex
```

---

## Stack

Qiskit · scikit-learn · QAOA / VQE · Quantum Kernels (ZZFeatureMap) · protocolos Table A/B · pytest

---

## Créditos

**Autor:** Juan David Prada Malagón  
**Director:** Ricardo Rafael Domínguez Reyes  
**Universidad:** UNIR — Máster en Inteligencia Artificial  
**Contexto aplicado:** SKG Tecnología (datos de práctica operativa)

---

*No prometemos “ventaja cuántica universal”. Prometemos criterios claros para saber cuándo usarla.*
