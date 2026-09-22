# Empirical conditions for quantum advantage in data science

**Master's thesis (TFM) · Universidad Internacional de La Rioja (UNIR)**  
**MSc in Artificial Intelligence**  
Juan David Prada Malagón · Advisor: Ricardo Rafael Domínguez Reyes  
Applied context: SKG Tecnología (practice data)

---

## Abstract

Despite rapid progress in noisy intermediate-scale quantum (NISQ) devices and variational algorithms, a persistent gap remains between theoretical claims of quantum advantage and reproducible evidence on problems that matter for applied data science. This thesis addresses that gap through a **systematic empirical comparison** of quantum and classical methods on **nine use cases** spanning the problem families that dominate industrial analytics — **supervised classification, regression, combinatorial optimization, and clustering-related tasks** (one-class anomaly detection and lag-based forecasting).

Across these settings we evaluate quantum kernel methods and hybrid variational algorithms (QAOA, VQE) against strong classical baselines under matched protocols, shared metrics, and three complementary data regimes: (i) synthetic distributions with **native quantum structure**, (ii) public reference datasets, and (iii) operational practice data from SKG Tecnología. A central object of analysis is the **hybrid quantum–classical paradigm**, in which a quantum module explores candidate solutions while a classical optimizer governs parameterization, preprocessing, and final decisions.

The evidence supports a precise conclusion: quantum advantage is **real but conditional**. It materializes when the chosen **feature map is aligned with the data-generating process**, or when the task admits a sound **QUBO / hybrid** formulation at a feasible scale. In the absence of that alignment, classical methods typically dominate — a negative result that is methodologically informative rather than incidental.

Building on these observations, the thesis advances two research proposals. First, a **quantumness score** that quantifies, *ex ante*, the extent to which a dataset’s geometry is compatible with a quantum kernel, thereby supporting selective activation of quantum modules. Second, a **representation-space framework** for quantum-compatible remappings of classical data, aimed at increasing that compatibility; the framework is **formally posed** here (protocol and open conjecture) and left as a research line for stronger validation beyond the scope of this TFM.

**Document:** [`Documento/main.pdf`](Documento/main.pdf) · **Code:** [`POCs/qml_v2/`](POCs/qml_v2/) · **Figures:** [`POCs/data/output/figures/`](POCs/data/output/figures/)

---

## Research question

> *Under what conditions, and for which classes of data-science problems, do NISQ-era quantum algorithms offer a real, measurable advantage over classical algorithms?*

---

## Contributions

1. **Empirical map of advantage** over nine industry-relevant use cases, with explicit reporting of both positive and negative outcomes.  
2. **Characterization of hybrid quantum–classical integration** (QAOA / VQE with classical optimization) as the practically viable pattern in the NISQ era, illustrated most clearly on NP-hard coverage (minimum dominating set).  
3. A **quantumness score** (denoted \(\mathcal{Q}(D)\), with robust variant \(\mathcal{Q}_{\mathrm{rob}}\) in the QML v2 protocol) as a diagnostic prior to expensive kernel experiments.  
4. A **representation-space / QCR framework**, formally posed, for investigating whether classical data can be remapped to raise quantum compatibility.

---

## Experimental scope

| Case | Problem family | Quantum method | Principal finding |
|:---:|---|---|---|
| 1 | Classification | Quantum kernel SVM | Large gains under map-aligned data; classical superior otherwise |
| 2 | Regression | Quantum-kernel SVR | Classical methods preferred on typical operational regression |
| 3 | Combinatorial optimization (assignment) | Hybrid QAOA | −30.3% cost relative to greedy under a matched protocol |
| 4 | Combinatorial optimization (portfolio) | Hybrid VQE | Near-parity / modest improvement at small instance sizes |
| 5 | Combinatorial optimization (TSP) | Hybrid QAOA | Optimal parity with NN+2-opt — modeling value, not asymptotic speedup |
| 6 | Classification (threshold setting) | Quantum kernel SVM | Advantage contingent on native structure |
| 7 | Combinatorial optimization (dominating set) | Hybrid QAOA | End-to-end hybrid pattern on an NP-hard coverage problem |
| 8 | Clustering-related (one-class anomalies) | OCSVM with ZZ kernel | Advantage only when novelty is map-aligned |
| 9 | Forecasting (lag windows) | Quantum-kernel SVR | Advantage only when the target is map-aligned |

Implementation: Qiskit (exact statevector simulation). Reproducible protocols and tests live under `POCs/`.

---

## Implications

The practical implication is not wholesale replacement of classical machine learning, but **selective, diagnosable adoption**: estimate quantum compatibility (quantumness score), consider representation remapping where justified, and activate hybrid quantum modules only when alignment or QUBO structure supports it. The scientific implication is a clearer separation between *algorithmic promise*, *data–map geometry*, and *integration architecture* — three factors that must be stated jointly when claims of quantum advantage are made.

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
