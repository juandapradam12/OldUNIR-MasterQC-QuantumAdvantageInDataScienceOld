# TSP optimization (`tsp_optimization`) — QML v2

TFM: [`notebooks/05_tsp_QUBO_vs_clasico.ipynb`](../POCs/notebooks/05_tsp_QUBO_vs_clasico.ipynb)  
Protocol: [`POCs/qml_v2/tsp_optimization.py`](../POCs/qml_v2/tsp_optimization.py)

## What was wrong in notebook 05

- Compared n=12 NN to n=4 QAOA as if they shared an instance
- SA over an already-enumerated tour list labeled “quantum”
- QAOA scored Ising energy / lossy bitstrings (often infeasible)
- 2-opt claimed in anexos, never run

## QML v2 — same model

Closed single-vehicle TSP (not VRP). Every solver minimizes the same
tour length \(L(\pi)\) on the same distance matrix \(D\).

| Solver | Role |
|---|---|
| BruteForce-opt | Exact \((n-1)!\) (depot fixed) |
| NearestNeighbor / NN-MS | Greedy |
| **NN+2-opt** | Strong classical local search |
| Random-tour | Floor |
| SA-QUBO | Classical SA on the **same n² Lucas QUBO** |
| QAOA | QP equalities + named-var decode; cost = raw km |

## Run

```bash
cd POCs
python3 -m qml_v2.tsp_optimization --quick
python3 -m qml_v2.tsp_optimization --n 4 --family colombia
python3 -m qml_v2.tsp_optimization --n 4 --family euclidean --n-instances 3
```

Results: [`tsp_optimization_v2_results.md`](tsp_optimization_v2_results.md).

## Headline (Colombia n=4, depot fixed → 9 qubits)

| Solver | Cost (km) | Gap vs exact |
|--------|-----------|--------------|
| BruteForce-opt | **1867.6** | 0 |
| **NN+2-opt** | **1867.6** | 0 |
| **QAOA-p1** | **1867.6** | 0 (feasible) |
| SA-QUBO | 1867.6 | 0 |
| NearestNeighbor | 2124.6 | +13.8% |

2-opt recovers the optimum that plain NN misses. QAOA matches exact under the
fair protocol — the notebook's infeasible / mismatched-n story is fixed.
Not VRP.
