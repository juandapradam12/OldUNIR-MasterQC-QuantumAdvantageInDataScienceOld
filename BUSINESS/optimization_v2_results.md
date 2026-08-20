# Optimization results (QML v2 — same model for all solvers)

minimize sum C[i,j] x[i,j]  with one-agent-per-task and one-task-per-agent.
PuLP/Hungarian = exact optimum. Gap = (cost − pulp) / pulp.
QAOA cost from named-var decoded X (not QUBO energy); multi-start COBYLA.

Config: `{"n": 3, "n_instances": 5, "qaoa_p": 1, "cobyla_maxiter": 80, "qaoa_restarts": 3, "random_state": 42, "cost_low": 1.0, "cost_high": 10.0}`

## Summary
```
    model  cost_mean  cost_std  gap_mean  gap_std  feasible_rate  time_mean_s  n
Hungarian   9.335190  2.154638   0.00000 0.000000            1.0     0.000025  5
 PuLP-CBC   9.335190  2.154638   0.00000 0.000000            1.0     0.005141  5
  QAOA-p1   9.335190  2.154638   0.00000 0.000000            1.0   150.598235  5
   Greedy  10.146296  3.514513   0.07044 0.157508            1.0     0.000012  5
```

## Per instance
```
 instance_id     model      cost  gap_vs_pulp  feasible     time_s               assignment
           0  PuLP-CBC 11.514866       0.0000      True   0.007423 [(0, 2), (1, 1), (2, 0)]
           0 Hungarian 11.514866       0.0000      True   0.000033 [(0, 2), (1, 1), (2, 0)]
           0    Greedy 15.570397       0.3522      True   0.000013 [(0, 0), (1, 2), (2, 1)]
           0   QAOA-p1 11.514866       0.0000      True 140.914269 [(0, 2), (1, 1), (2, 0)]
           1  PuLP-CBC  7.240867       0.0000      True   0.004006 [(0, 0), (1, 1), (2, 2)]
           1 Hungarian  7.240867       0.0000      True   0.000023 [(0, 0), (1, 1), (2, 2)]
           1    Greedy  7.240867       0.0000      True   0.000012 [(0, 0), (1, 1), (2, 2)]
           1   QAOA-p1  7.240867       0.0000      True 144.186096 [(0, 0), (1, 1), (2, 2)]
           2  PuLP-CBC 11.776794       0.0000      True   0.005402 [(0, 1), (1, 0), (2, 2)]
           2 Hungarian 11.776794       0.0000      True   0.000023 [(0, 1), (1, 0), (2, 2)]
           2    Greedy 11.776794       0.0000      True   0.000011 [(0, 1), (1, 0), (2, 2)]
           2   QAOA-p1 11.776794       0.0000      True 154.755105 [(0, 1), (1, 0), (2, 2)]
           3  PuLP-CBC  7.698552       0.0000      True   0.004466 [(0, 2), (1, 0), (2, 1)]
           3 Hungarian  7.698552       0.0000      True   0.000025 [(0, 2), (1, 0), (2, 1)]
           3    Greedy  7.698552       0.0000      True   0.000013 [(0, 2), (1, 0), (2, 1)]
           3   QAOA-p1  7.698552       0.0000      True 150.045192 [(0, 2), (1, 0), (2, 1)]
           4  PuLP-CBC  8.444869       0.0000      True   0.004409 [(0, 2), (1, 1), (2, 0)]
           4 Hungarian  8.444869       0.0000      True   0.000022 [(0, 2), (1, 1), (2, 0)]
           4    Greedy  8.444869       0.0000      True   0.000011 [(0, 2), (1, 1), (2, 0)]
           4   QAOA-p1  8.444869       0.0000      True 163.090514 [(0, 2), (1, 1), (2, 0)]
```
