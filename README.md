# When does quantum computing actually beat classical ML?

**Master's thesis (TFM) · UNIR · MSc in Artificial Intelligence**  
Juan David Prada Malagón · Advisor: Ricardo Rafael Domínguez Reyes · Applied context: SKG Tecnología

---

## The pitch

Everyone talks about quantum advantage. Few measure it on the problems industry already solves every day.

This thesis studies **quantum advantage empirically** across **nine use cases** built around the workhorses of applied data science — **classification, regression, combinatorial optimization, and clustering-related tasks** (including one-class anomalies and lag forecasting). The same questions that drive churn models, forecasting, assignment, routing, coverage, and outlier detection.

We compare quantum methods (quantum kernels, QAOA, VQE) with strong classical baselines on shared data and metrics — including **hybrid designs** where a quantum module proposes candidates and a classical optimizer drives parameters and final decisions.

From those observations we do not stop at “it depends.” We propose:

1. A **quantumness score** — a diagnostic number that estimates whether a dataset’s structure is likely to favor a quantum kernel *before* you pay for the experiment.  
2. A **representation-space framework** — a formal proposal for remapping classical data into a geometry where quantum methods have a fairer chance (open research line; posed in this TFM, not claimed as proven).

**Read the thesis:** [`Documento/main.pdf`](Documento/main.pdf) · **Run the code:** [`POCs/qml_v2/`](POCs/qml_v2/)

---

## Research question

> *Under what conditions, and for which classes of data-science problems, do NISQ-era quantum algorithms offer a real, measurable advantage over classical algorithms?*

---

## What was studied

### Nine industry-shaped use cases

| Case | Problem family | Quantum method | What we learned |
|:---:|---|---|---|
| 1 | Classification | Quantum kernel SVM | Large gains when data match the feature map; classical wins otherwise |
| 2 | Regression | Quantum-kernel SVR | Classical usually preferred on typical operational data |
| 3 | Optimization (assignment) | Hybrid QAOA | −30.3% cost vs greedy under a fair protocol |
| 4 | Optimization (portfolio) | Hybrid VQE | Tie / slight edge at small scale |
| 5 | Optimization (TSP) | Hybrid QAOA | Matches strong classical heuristics — modeling value, not speedup |
| 6 | Classification (threshold) | Quantum kernel SVM | Advantage conditional on native structure |
| 7 | Optimization (coverage / MDS) | Hybrid QAOA | Clearest hybrid quantum–classical pattern on an NP-hard task |
| 8 | Clustering-related (one-class anomalies) | OCSVM with ZZ kernel | Wins only when novelty matches the map |
| 9 | Forecasting (lags) | Quantum-kernel SVR | Wins only when the target matches the map |

Data layers (when applicable): synthetic sets with **native quantum structure**, **public benchmarks**, and **SKG Tecnología** practice data.

### Hybrid quantum–classical methods

We analyze how both sides work **together**:

- Quantum side explores candidate solutions (especially for QUBO-style combinatorial problems).  
- Classical side tunes parameters, handles preprocessing, and keeps the final decision.  
- Use the hybrid path when the problem fits and the instance is feasible; otherwise stay classical.

### From results to proposals

| Observation → | Proposal |
|---|---|
| Advantage shows up only when the **feature map aligns** with how the data were generated | Measure that fit with a **quantumness score** before committing compute |
| Some classical datasets look “flat” to quantum kernels | Explore a **representation space** (remappings) that could raise that score — formally posed here as an open research line |

In the document these proposals appear as the score \(\mathcal{Q}(D)\) (with a robust variant) and the QCR framework. You do not need those names to get the idea: **diagnose first, remap if needed, activate quantum selectively**.

---

## Core finding (one sentence)

Quantum advantage is **real but conditional** — and the useful engineering response is a **hybrid, selective** stack guided by a quantumness diagnosis, not a blanket replacement of classical ML.

---

## Repository

```
Documento/main.pdf          # Compiled thesis
Documento/Capitulos/        # LaTeX sources
POCs/qml_v2/                # Experimental protocols (Python / Qiskit)
POCs/data/output/figures/   # Versioned figures
POCs/tests/                 # Regression suite
BUSINESS/                   # Technical notes by family
```

```bash
cd POCs
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -q
```

---

## Keywords

quantum advantage · quantum machine learning · hybrid quantum–classical · classification · regression · combinatorial optimization · clustering · quantum kernels · QAOA · VQE · NISQ · quantumness score · representation learning for QML

---

*Measured conditions for advantage — plus tools to decide when to try quantum next.*
