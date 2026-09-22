# Quantum vs classical algorithms in data science

**Master's thesis (TFM) · UNIR · MSc in Artificial Intelligence**  
Juan David Prada Malagón · Advisor: Ricardo Rafael Domínguez Reyes · Applied context: SKG Tecnología

---

## Research question

> *Under what conditions, and for which classes of data-science problems, do NISQ-era quantum algorithms offer a real, measurable advantage over classical algorithms?*

This repository is an **empirical comparative study** — not a “Quantum ML for operations” product. It covers three pillars:

1. **Fair quantum vs classical benchmarks** across nine use cases  
2. **Hybrid quantum–classical algorithms** as the viable integration pattern in the NISQ era  
3. **Original research proposals**: the quantumness score \(\mathcal{Q}(D)\) and the QCR Framework

---

## Elevator pitch

| Pillar | What it delivers |
|---|---|
| **Comparative evidence** | Advantage is *real but conditional*: it appears when the **feature map is aligned with the data generator**, or on **combinatorial / QUBO** problems. Without that alignment, classical methods usually win. |
| **Hybrid paradigm** | Variational loops (QAOA / VQE + classical optimizers such as COBYLA) with a **minimum interface**: the quantum module explores the solution space; the classical module orchestrates parameters, preprocessing, and final decisions. Demonstrated on assignment, portfolio, TSP, and minimum dominating set (Case 7). |
| **Research proposals** | **Quantumness score** \(\mathcal{Q}(D)\) / \(\mathcal{Q}_{\mathrm{rob}}\) — predicts when a quantum kernel is likely to help. **QCR Framework** — formally posed protocol + open conjecture for quantum-compatible remappings (not claimed as proven). |
| **Scope** | Classification (QSVM), regression (QK-SVR), combinatorial optimization (QAOA / VQE / TSP), one-class anomalies (OCSVM-ZZ), lag forecasting — **9 cases**. |
| **Deliverables** | Thesis PDF + Qiskit code (`POCs/qml_v2/`) + versioned figures + tests. |

**Read now:** [`Documento/main.pdf`](Documento/main.pdf)  
**Code:** [`POCs/qml_v2/`](POCs/qml_v2/) · **Figures:** [`POCs/data/output/figures/`](POCs/data/output/figures/)

---

## Hybrid quantum–classical algorithms

In the NISQ era, the practical integration path is **hybrid**, not “replace classical with quantum”:

- **QAOA / VQE + classical optimizer** — shared loop for QUBO-formulable problems (assignment, portfolio, TSP, dominating set)
- **Selective activation** — quantum module on when the instance is QUBO-compatible and scale-feasible; classical stack otherwise
- **Prototype architecture** — explicit activation criteria in the thesis (Ch. 4–5), generalizable beyond the SKG practice context

Cases **3, 4, 5, and 7** are the hybrid spine of the work; Case **7** (minimum dominating set, \(P(\mathrm{opt})\approx 73\%\)) is the clearest illustration of the QAOA–COBYLA pattern.

---

## Original research proposals

| Proposal | Status in this TFM | Role |
|---|---|---|
| **Quantumness score \(\mathcal{Q}(D)\)** (and robust \(\mathcal{Q}_{\mathrm{rob}}\) in QML v2) | Empirically evaluated; Spearman \(\rho_S = 0.833\) (\(p=0.005\), \(n=9\)) on the synthetic sweep | Diagnostic: estimate whether data structure favors a quantum kernel before paying the cost |
| **QCR Framework** (quantum-compatible remappings) | **Formally posed** — protocol + open conjecture; preliminary evidence only | Research line: can classical data be remapped to raise \(\mathcal{Q}\)? Strong validation is post-TFM |

Together with the alignment principle (feature map ↔ data generator), these proposals turn “try quantum and hope” into a **selective, diagnosable** workflow.

---

## Results at a glance

| Case | Algorithm | Takeaway |
|:---:|---|---|
| 1 | QSVM | Strong advantage on map-aligned data; classical wins on real data without alignment |
| 2 | QK-SVR | Classical preferred for typical operational regression |
| 3 | QAOA assignment (**hybrid**) | −30.3% cost vs greedy (fair protocol) |
| 4 | VQE portfolio (**hybrid**) | Tie / slight improvement at small scale |
| 5 | QAOA TSP (**hybrid**) | Optimal tie vs NN+2-opt — *modeling* value, not speedup |
| 6 | QSVM (threshold) | Conditional on native structure / threshold |
| 7 | QAOA MDS (**hybrid**) | Hybrid quantum–classical pattern on an NP-hard coverage problem |
| 8 | OCSVM-ZZ | Wins only when novelty is map-aligned (`zz_phase`) |
| 9 | QK-SVR lags | Wins only when the target is map-aligned (`zz_lag`) |

**Practical conclusion:** quantum as an *additional, selective* tool inside a **hybrid** stack — activated after alignment / \(\mathcal{Q}\) diagnosis or when the problem is QUBO-viable.

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

quantum computing · quantum machine learning · hybrid quantum–classical algorithms · quantum advantage · QSVM · quantum kernel · QAOA · VQE · combinatorial optimization · quantumness score · QCR Framework · NISQ · OCSVM-ZZ

---

*Conditional evidence, hybrid integration, and open research proposals — not hype. The PDF develops the full argument.*
