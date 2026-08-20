# Optimization results (QML v2 — same model for all solvers)

minimize sum C[i,j] x[i,j]  with one-agent-per-task and one-task-per-agent.
PuLP/Hungarian = exact optimum. Gap = (cost − pulp) / pulp.
QAOA cost from named-var decoded X (not QUBO energy); multi-start COBYLA.

Config: `{"n": 3, "n_instances": 5, "qaoa_p": 1, "cobyla_maxiter": 80, "qaoa_restarts": 3, "random_state": 42, "cost_low": 1.0, "cost_high": 10.0}`

## Summary
```
    model  cost_mean  cost_std  gap_mean  gap_std  feasible_rate  time_mean_s  n
Hungarian   9.377866  3.022173  0.000000 0.000000            1.0     0.000023  2
 PuLP-CBC   9.377866  3.022173  0.000000 0.000000            1.0     0.003935  2
  QAOA-p1  10.361999  4.413948  0.085466 0.120868            1.0    44.828229  2
   Greedy  11.405632  5.889867  0.176100 0.249043            1.0     0.000013  2
```

## Per instance
```
 instance_id     model      cost  gap_vs_pulp  feasible    time_s               assignment
           0  PuLP-CBC 11.514866     0.000000      True  0.003981 [(0, 2), (1, 1), (2, 0)]
           0 Hungarian 11.514866     0.000000      True  0.000023 [(0, 2), (1, 1), (2, 0)]
           0    Greedy 15.570397     0.352200      True  0.000014 [(0, 0), (1, 2), (2, 1)]
           0   QAOA-p1 13.483132     0.170933      True 44.487031 [(0, 1), (1, 2), (2, 0)]
           1  PuLP-CBC  7.240867     0.000000      True  0.003889 [(0, 0), (1, 1), (2, 2)]
           1 Hungarian  7.240867     0.000000      True  0.000023 [(0, 0), (1, 1), (2, 2)]
           1    Greedy  7.240867     0.000000      True  0.000011 [(0, 0), (1, 1), (2, 2)]
           1   QAOA-p1  7.240867     0.000000      True 45.169427 [(0, 0), (1, 1), (2, 2)]
```
