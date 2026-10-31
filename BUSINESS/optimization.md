# Optimization — same model for PuLP, Hungarian, greedy, QAOA

TFM record: [`POCs/notebooks/02_optimizacion_QAOA_vs_clasico.ipynb`](../POCs/notebooks/02_optimizacion_QAOA_vs_clasico.ipynb)  
Protocol: [`POCs/qml_v2/optimization.py`](../POCs/qml_v2/optimization.py)

## What was wrong in notebook 02

Two different problems were compared as if they were one:

1. **Classical FLP (PuLP):** full binary assignment agent×foco with capacity constraints.  
2. **QAOA:** a *reduced* QuadraticProgram on the top‑N focos — not the same FLP.

## QML v2 (robust)

\(n\) tasks, \(n\) agents. Same \(C\), same constraints for every solver.

| Solver | Role |
|---|---|
| **PuLP/CBC** | Exact ILP optimum |
| **Hungarian** | Exact matching (should match PuLP) |
| **Greedy** | Heuristic ≠ PuLP |
| **QAOA** | Multi-start COBYLA; cost from named-var decoded \(X\) (not QUBO energy) |

```bash
cd POCs
python3 -m qml_v2.optimization --quick
python3 -m qml_v2.optimization --n 3 --n-instances 5 --qaoa-p 1 --restarts 3
```

Results: [`optimization_v2_results.md`](optimization_v2_results.md).  
MaxCut companion: [`maxcutm_optimization.md`](maxcutm_optimization.md).
