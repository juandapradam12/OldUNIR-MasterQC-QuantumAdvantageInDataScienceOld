# TSP optimization results (QML v2)

Same closed-tour length L(π) for brute force, NN, NN+2-opt, SA-QUBO (n²), QAOA.
Not VRP. Cost is raw tour length (km or Euclidean), never QUBO/Ising energy.

Config: `{"n": 4, "n_instances": 3, "family": "euclidean", "qaoa_p": 1, "cobyla_maxiter": 60, "qaoa_restarts": 2, "sa_iters": 5000, "sa_T0": 1.0, "fix_depot": true, "nn_starts": null, "random_state": 42, "run_qaoa": true, "run_sa": true}`

## Summary
```
          model  cost_mean  cost_std      gap_mean  approx_mean  feasible_rate  time_mean_s  n_instances
       NN+2-opt   1.838188  0.405850 -6.513677e-17     1.000000            1.0     0.000031            3
          NN-MS   1.838188  0.405850 -6.513677e-17     1.000000            1.0     0.000018            3
        QAOA-p1   1.838188  0.405850  0.000000e+00     1.000000            1.0    96.785018            3
    Random-tour   1.838188  0.405850  0.000000e+00     1.000000            1.0     0.000136            3
NearestNeighbor   1.855847  0.382199  1.202390e-02     0.988395            1.0     0.000014            3
        SA-QUBO   2.091235  0.651273  1.222455e-01     0.897178            1.0     0.283409            3
```

## Per-instance
```
 instance_id  n    family           model     cost  gap_vs_exact  approx_ratio  feasible     time_s                   tour
           0  4 euclidean  BruteForce-opt 2.272599  0.000000e+00      1.000000      True   0.000021 p0 → p1 → p2 → p3 → p0
           0  4 euclidean NearestNeighbor 2.272599  0.000000e+00      1.000000      True   0.000009 p0 → p3 → p2 → p1 → p0
           0  4 euclidean           NN-MS 2.272599 -1.954103e-16      1.000000      True   0.000018 p1 → p0 → p3 → p2 → p1
           0  4 euclidean        NN+2-opt 2.272599 -1.954103e-16      1.000000      True   0.000026 p1 → p0 → p3 → p2 → p1
           0  4 euclidean     Random-tour 2.272599  0.000000e+00      1.000000      True   0.000127 p0 → p1 → p2 → p3 → p0
           0  4 euclidean         SA-QUBO 2.767898  2.179439e-01      0.821056      True   0.436833 p0 → p1 → p3 → p2 → p0
           0  4 euclidean         QAOA-p1 2.272599  0.000000e+00      1.000000      True 103.769102 p0 → p1 → p2 → p3 → p0
           1  4 euclidean  BruteForce-opt 1.773229  0.000000e+00      1.000000      True   0.000034 p0 → p2 → p3 → p1 → p0
           1  4 euclidean NearestNeighbor 1.773229  0.000000e+00      1.000000      True   0.000015 p0 → p2 → p3 → p1 → p0
           1  4 euclidean           NN-MS 1.773229  0.000000e+00      1.000000      True   0.000018 p0 → p2 → p3 → p1 → p0
           1  4 euclidean        NN+2-opt 1.773229  0.000000e+00      1.000000      True   0.000030 p0 → p2 → p3 → p1 → p0
           1  4 euclidean     Random-tour 1.773229  0.000000e+00      1.000000      True   0.000155 p0 → p2 → p3 → p1 → p0
           1  4 euclidean         SA-QUBO 2.037072  1.487925e-01      0.870479      True   0.207054 p0 → p1 → p2 → p3 → p0
           1  4 euclidean         QAOA-p1 1.773229  0.000000e+00      1.000000      True  96.134772 p0 → p2 → p3 → p1 → p0
           2  4 euclidean  BruteForce-opt 1.468735  0.000000e+00      1.000000      True   0.000027 p0 → p1 → p2 → p3 → p0
           2  4 euclidean NearestNeighbor 1.521714  3.607171e-02      0.965184      True   0.000016 p0 → p1 → p3 → p2 → p0
           2  4 euclidean           NN-MS 1.468735  0.000000e+00      1.000000      True   0.000018 p1 → p0 → p3 → p2 → p1
           2  4 euclidean        NN+2-opt 1.468735  0.000000e+00      1.000000      True   0.000038 p0 → p1 → p2 → p3 → p0
           2  4 euclidean     Random-tour 1.468735  0.000000e+00      1.000000      True   0.000126 p0 → p3 → p2 → p1 → p0
           2  4 euclidean         SA-QUBO 1.468735  0.000000e+00      1.000000      True   0.206340 p0 → p3 → p2 → p1 → p0
           2  4 euclidean         QAOA-p1 1.468735  0.000000e+00      1.000000      True  90.451180 p0 → p3 → p2 → p1 → p0
```

## Advantage (gap vs exact; lower is better)
```
QAOA gap=0.0000  vs best classical NN+2-opt gap=-0.0000  (Δgap=+0.0000; feasible_Q=1.00)
```
