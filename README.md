# Empirical conditions for quantum advantage in data science

**Master's thesis (TFM) · Universidad Internacional de La Rioja (UNIR)**  
**MSc in Artificial Intelligence**  
Juan David Prada Malagón · Advisor: Ricardo Rafael Domínguez Reyes  
Applied context: SKG Tecnología (practice data)

---

## Abstract

Despite rapid progress in noisy intermediate-scale quantum (NISQ) devices and variational algorithms, a persistent gap remains between theoretical claims of quantum advantage and reproducible evidence on problems that matter for applied data science. This thesis addresses that gap through a **systematic empirical comparison** of quantum and classical methods on **nine use cases** spanning the problem families that dominate industrial analytics — **supervised classification, regression, combinatorial optimization, and clustering-related tasks** (one-class anomaly detection and lag-based forecasting).

Across these settings we evaluate quantum kernel methods and hybrid variational algorithms (QAOA, VQE) against strong classical baselines under matched protocols, shared metrics, and two data classes: (i) public reference datasets and (ii) operational practice data from SKG Tecnología. No data are generated with the quantum kernel's own feature map: labelling data with the observable of the model under test makes the advantage circular, so such datasets are deliberately excluded. A central object of analysis is the **hybrid quantum–classical paradigm**, in which a quantum module explores candidate solutions while a classical optimizer governs parameterization, preprocessing, and final decisions.

The evidence supports a negative but informative conclusion: **no quantum advantage is observed** on the real and reference data evaluated. Classical methods win or tie in classification, regression, anomaly detection and forecasting, and in small-scale combinatorial optimization QAOA/VQE are valid QUBO solvers that at best match exact or greedy baselines.

Building on these observations, the thesis advances two research proposals. First, a **quantumness score** that quantifies, *ex ante*, the extent to which a dataset’s geometry is compatible with a quantum kernel, thereby supporting selective activation of quantum modules. Second, a **representation-space framework** for quantum-compatible remappings of classical data, aimed at increasing that compatibility; the framework is **formally posed** here (protocol and open conjecture) and left as a research line for stronger validation beyond the scope of this TFM.

**Document:** [`Documento/main.pdf`](Documento/main.pdf) · **Code:** [`POCs/qml_v2/`](POCs/qml_v2/) · **Figures:** [`POCs/data/output/figures/`](POCs/data/output/figures/)

---

## Research question

> *Under what conditions, and for which classes of data-science problems, do NISQ-era quantum algorithms offer a real, measurable advantage over classical algorithms?*

---

## Contributions

1. **Empirical map of advantage** over nine industry-relevant use cases, with explicit reporting of both positive and negative outcomes.  
2. **Characterization of hybrid quantum–classical integration** (QAOA / VQE with classical optimization) as the practically viable pattern in the NISQ era, illustrated most clearly on NP-hard coverage (minimum dominating set).  
3. A **quantumness score** $\mathcal{Q}(D)$ (robust variant $\mathcal{Q}_{\mathrm{rob}}$ in the QML v2 protocol) as a diagnostic prior to expensive kernel experiments.  
4. A **representation-space / QCR framework**, formally posed, for investigating whether classical data can be remapped to raise quantum compatibility.

### Original research proposals

| Proposal | What it is | Status in this TFM |
|---|---|---|
| Quantumness score $\mathcal{Q}(D)$ (and robust $\mathcal{Q}_{\mathrm{rob}}$) | Diagnostic score estimating whether a dataset’s geometry favors a quantum kernel before expensive experiments | A-priori diagnostic proposal; the indicators show no quantum structure on the real data (null control ≈ +0.03 ± 0.02) and have not been validated as a predictor of advantage (the α-sweep, ρ_S = 0.833, only calibrates the instrument) |
| QCR Framework | Formal protocol and open conjecture on remapping classical data to raise $\mathcal{Q}$ (quantum-compatible remappings) | **Formally posed** with preliminary evidence — strong validation left as a research line |

**Core finding.** No quantum advantage is observed on real (SKG) or public reference data: classical methods win or tie in every case. In combinatorial optimization (cases 3–5, 7) QAOA/VQE are valid solvers of the QUBO model but show **no advantage** over exact classical solvers at the sizes simulated here. Practical conclusion: quantum methods are an exploratory line, to be activated only if they beat the classical baseline under a fair protocol; none does today.

---

## Experimental scope

| Case | Problem family | Quantum method | Principal finding |
|:---:|---|---|---|
| 1 | Classification | Quantum kernel SVM | Classical superior: −4.5 pp (SKG), −3.8 pp (Telco) |
| 2 | Regression | Quantum-kernel SVR | Classical superior on SKG (ΔR² ×100 = −56.2) and Friedman #1 (−92.8) |
| 3 | Combinatorial optimization (assignment) | Hybrid QAOA | Matches the exact optimum on 5/5 instances (same cost as Hungarian and ILP); no advantage |
| 4 | Combinatorial optimization (portfolio) | Hybrid VQE / QAOA | 3 of 6 assets, 10 seeds, penalty 10: QAOA reaches the exact optimum in 6/10 seeds and VQE in 3/10 (Greedy and brute force always do); with penalty 0.01 no run is feasible; no advantage |
| 5 | Combinatorial optimization (TSP) | Hybrid QAOA | Optimal parity with NN+2-opt — modeling value, not asymptotic speedup |
| 6 | Classification (threshold setting) | Quantum kernel SVM | Classical superior: −4.0 pp (SKG), −3.8 pp (Telco) |
| 7 | Combinatorial optimization (dominating set) | Hybrid QAOA | Ties exact ILP on the 6-node network (mean P(opt) 5.7%, p = 3); on Erdős–Rényi graphs (n = 6–15) QAOA reaches a minimum dominating set in 2 of 5 instances vs 5 of 5 for ILP and Greedy (−60 pp) |
| 8 | Clustering-related (one-class anomalies) | OCSVM with ZZ kernel | Classical superior on every dataset (−29.9 pp AUC on SKG, −29.1 on moons, −50.8 on blobs) |
| 9 | Forecasting (lag windows) | Quantum-kernel SVR | On SKG corridor speeds Ridge reaches MAE 2.80 vs 13.59 for the quantum kernel, which matches a train-mean predictor; classical also wins on sine |

Implementation: Qiskit (exact statevector simulation). Reproducible protocols and tests live under `POCs/`.

---

## Implications

The practical implication is not wholesale replacement of classical machine learning, but **selective, diagnosable adoption**: estimate quantum compatibility (quantumness score) as an a-priori diagnostic, and activate hybrid quantum modules only when they beat the classical baseline under a fair protocol. The scientific implication is a clearer separation between *algorithmic promise*, *data–map geometry*, and *integration architecture* — three factors that must be stated jointly when claims of quantum advantage are made.

---

## Repository

```
Documento/main.pdf          # Compiled thesis
Documento/Capitulos/        # LaTeX sources
POCs/qml_v2/                # Experimental protocols (Python / Qiskit)
POCs/scripts/               # Figure regeneration
POCs/data/output/figures/   # Versioned figures
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

quantum advantage · quantum machine learning · hybrid quantum–classical algorithms · NISQ · quantum kernels · QAOA · VQE · classification · regression · combinatorial optimization · clustering · anomaly detection · quantumness score · representation space · QCR

---

*Conditional evidence, hybrid integration, and open proposals for diagnosing and reshaping quantum compatibility.*
