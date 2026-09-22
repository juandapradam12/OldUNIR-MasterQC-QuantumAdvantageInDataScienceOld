# Quantum vs classical algorithms in data science

**Master's thesis (TFM) · UNIR · MSc in Artificial Intelligence**  
Juan David Prada Malagón · Advisor: Ricardo Rafael Domínguez Reyes · Applied context: SKG Tecnología

---

## Research question

> *Under what conditions, and for which classes of data-science problems, do NISQ-era quantum algorithms offer a real, measurable advantage over classical algorithms?*

This is **not** a “Quantum ML for operations” product. It is an **empirical comparative study**: nine use cases, fair protocols, strong classical baselines, and reproducible evidence — including when quantum methods **do not** win.

---

## Elevator pitch

| What you get | Detail |
|---|---|
| **Core finding** | Advantage is *real but conditional*: it appears when the **feature map is aligned with the data generator**, or on **combinatorial / QUBO** problems. Without that alignment, classical methods usually win. |
| **Scope** | Classification (QSVM), regression (QK-SVR), optimization (QAOA / VQE / TSP), one-class anomaly detection (OCSVM-ZZ), and lag forecasting — **9 cases**. |
| **Data layers** | Synthetic data with native quantum structure, public reference datasets, and SKG Tecnología practice data. |
| **Original contributions** | Quantumness score \(\mathcal{Q}(D)\) / \(\mathcal{Q}_{\mathrm{rob}}\) and the QCR Framework (**formally posed**; open conjecture). |
| **Deliverables** | Thesis PDF + Qiskit code (`POCs/qml_v2/`) + versioned figures + tests. |

**Read now:** [`Documento/main.pdf`](Documento/main.pdf)  
**Code:** [`POCs/qml_v2/`](POCs/qml_v2/) · **Figures:** [`POCs/data/output/figures/`](POCs/data/output/figures/)

---

## Results at a glance

| Case | Algorithm | Takeaway |
|:---:|---|---|
| 1 | QSVM | Strong advantage on map-aligned data; classical wins on real data without alignment |
| 2 | QK-SVR | Classical preferred for typical operational regression |
| 3 | QAOA assignment | −30.3% cost vs greedy (fair protocol) |
| 4 | VQE portfolio | Tie / slight improvement at small scale |
| 5 | QAOA TSP | Optimal tie vs NN+2-opt — *modeling* value, not speedup |
| 6 | QSVM (threshold) | Conditional on native structure / threshold |
| 7 | QAOA MDS | Hybrid quantum–classical pattern |
| 8 | OCSVM-ZZ | Wins only when novelty is map-aligned (`zz_phase`) |
| 9 | QK-SVR lags | Wins only when the target is map-aligned (`zz_lag`) |

**Practical conclusion:** quantum algorithms as an *additional, selective* tool — not a replacement for the classical stack.

---

## Repository layout

```
Documento/main.pdf          # Compiled thesis
Documento/Capitulos/        # LaTeX sources
POCs/qml_v2/                # Fair protocols (Python / Qiskit)
POCs/scripts/               # Figure regeneration
POCs/data/output/figures/   # PNGs tracked in git
POCs/tests/                 # Regression suite
BUSINESS/                   # Technical notes by experiment family
```

```bash
cd POCs
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -q
```

---

## Keywords

quantum computing · quantum machine learning · quantum advantage · QSVM · quantum kernel · QAOA · VQE · combinatorial optimization · OCSVM-ZZ · quantumness score · QCR Framework · NISQ · hybrid quantum–classical algorithms

---

*Conditional evidence, not hype. The PDF develops the full argument.*
