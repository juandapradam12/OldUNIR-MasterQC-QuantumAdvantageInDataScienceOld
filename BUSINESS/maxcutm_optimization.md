# MaxCut optimization (`maxcutm_optimization`) — QML v2

Protocol: [`POCs/qml_v2/maxcutm_optimization.py`](../POCs/qml_v2/maxcutm_optimization.py)

Defaults push past the previous ceiling (n=8, weak greedy → QAOA always 1.0):
n=10, denser ER graphs, multi-start greedy + random baseline.

| Solver | Role |
|---|---|
| Brute force | Optimum (n ≤ 12) |
| Random partition | Weak classical floor |
| Greedy multi-start | Stronger classical heuristic |
| QAOA | Quantum variational |

```bash
cd POCs
python3 -m qml_v2.maxcutm_optimization --quick
python3 -m qml_v2.maxcutm_optimization --n-nodes 10 --n-graphs 5 --edge-prob 0.55
```

Results: [`maxcutm_optimization_v2_results.md`](maxcutm_optimization_v2_results.md).
