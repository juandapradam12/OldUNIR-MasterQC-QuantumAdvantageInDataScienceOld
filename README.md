# Quantum vs classical algorithms in data science

**Master's thesis (TFM) · UNIR · MSc in Artificial Intelligence**  
Juan David Prada Malagón · Advisor: Ricardo Rafael Domínguez Reyes · Applied context: SKG Tecnología

---

## Research question

> *Under what conditions, and for which classes of data-science problems, do NISQ-era quantum algorithms offer a real, measurable advantage over classical algorithms?*

This repository documents an **empirical comparative study** — what was studied, how it was analyzed, and what follows for practice and research. It is not a “Quantum ML for operations” product.

---

## What was studied and analyzed

Three connected lines of work:

### 1. Comparative evaluation (nine use cases)

Quantum methods were compared head-to-head with strong classical baselines on the same data and metrics:

| Area | Quantum side | Classical baselines |
|---|---|---|
| Supervised classification | Quantum kernel SVM (QSVM) | SVM-RBF, Random Forest, boosting |
| Regression | Quantum-kernel SVR (QK-SVR) | Ridge, SVR-RBF, tree ensembles |
| Combinatorial optimization | QAOA, VQE (assignment, portfolio, TSP, dominating set) | Greedy, local search, ILP / Markowitz / NN+2-opt |
| One-class anomaly detection | OCSVM with ZZ kernel | Isolation Forest, LOF, OCSVM-RBF |
| Lag forecasting | QK-SVR on lag windows | Ridge, SVR-RBF |

Each case used up to three data layers when applicable: synthetic data with **native quantum structure**, **public reference** datasets, and **SKG Tecnología** practice data.

### 2. Hybrid quantum–classical methods

The thesis analyzes how quantum and classical components work **together**, not as substitutes:

- The quantum module proposes candidate solutions (typically for QUBO-formulable problems).
- A classical optimizer adjusts parameters and keeps preprocessing / final decisions.
- Activation is **selective**: use the hybrid path when the problem fits QUBO and the instance scale is feasible; otherwise stay with the classical pipeline.

Cases **3, 4, 5, and 7** develop this hybrid pattern; Case **7** (minimum dominating set) is the clearest end-to-end example.

### 3. Original research proposals

| Proposal | What it is | Status in this TFM |
|---|---|---|
| **Quantumness score \(\mathcal{Q}(D)\)** (and robust \(\mathcal{Q}_{\mathrm{rob}}\)) | A diagnostic score to estimate whether a dataset’s geometry favors a quantum kernel before running expensive experiments | Empirically evaluated (e.g. Spearman \(\rho_S = 0.833\), \(p=0.005\), \(n=9\) on the synthetic sweep) |
| **QCR Framework** | A formal protocol and open conjecture on remapping classical data to raise \(\mathcal{Q}\) (quantum-compatible remappings) | **Formally posed** with preliminary evidence — strong validation is left as a research line |

---

## Core finding

Advantage is **real but conditional**. It appears when the **feature map is aligned with how the data were generated**, or when the problem is combinatorial / QUBO and the hybrid design is sound. Without that alignment, classical methods usually win — and those negative results are part of the analysis.

**Practical conclusion:** treat quantum methods as an *additional, selective* tool inside a **hybrid** stack, activated after alignment / \(\mathcal{Q}\) diagnosis or when a QUBO formulation is viable.

---

## Results at a glance

| Case | Method | Takeaway |
|:---:|---|---|
| 1 | QSVM | Strong advantage on map-aligned data; classical wins on real data without alignment |
| 2 | QK-SVR | Classical preferred for typical operational regression |
| 3 | QAOA assignment (**hybrid**) | −30.3% cost vs greedy (fair protocol) |
| 4 | VQE portfolio (**hybrid**) | Tie / slight improvement at small scale |
| 5 | QAOA TSP (**hybrid**) | Optimal tie vs NN+2-opt — modeling value, not speedup |
| 6 | QSVM (threshold) | Conditional on native structure / threshold |
| 7 | QAOA dominating set (**hybrid**) | Hybrid pattern on an NP-hard coverage problem |
| 8 | OCSVM-ZZ | Wins only when novelty is map-aligned (`zz_phase`) |
| 9 | QK-SVR lags | Wins only when the target is map-aligned (`zz_lag`) |

---

## Deliverables

**Thesis PDF:** [`Documento/main.pdf`](Documento/main.pdf)  
**Reproducible code:** [`POCs/qml_v2/`](POCs/qml_v2/)  
**Figures:** [`POCs/data/output/figures/`](POCs/data/output/figures/)  
**Tests:** `POCs/tests/` (run with `PYTHONPATH=. pytest tests/ -q`)

```
Documento/main.pdf          # Compiled thesis
Documento/Capitulos/        # LaTeX sources
POCs/qml_v2/                # Fair experimental protocols (Python / Qiskit)
POCs/scripts/               # Figure regeneration
POCs/data/output/figures/   # PNGs tracked in git
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

*What was studied, how it was compared, and which research proposals follow — conditional evidence, not hype.*
