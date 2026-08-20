# MaxCut optimization results (QML v2)

Same objective for QAOA, multi-start greedy, random, and brute-force optimum.
Approx ratio = cut / optimum. Bitstring decodes to a classical partition.

Config: `{"n_nodes": 10, "edge_prob": 0.55, "qaoa_p": 1, "cobyla_maxiter": 80, "n_graphs": 5, "greedy_restarts": 8, "random_state": 42, "graph_family": "erdos_renyi"}`

## Summary
```
    model  cut_mean  cut_std  approx_mean  approx_std  time_mean_s  n_graphs
Greedy-MS       7.5 0.707107     1.000000    0.000000     0.000307         2
  QAOA-p1       7.5 0.707107     1.000000    0.000000     1.264250         2
   Random       4.0 1.414214     0.544643    0.239911     0.000156         2
```

## Per-graph
```
 graph_id  n_nodes  n_edges          model  cut_value  approx_ratio   time_s bitstring
        0        6       10 BruteForce-opt        8.0      1.000000 0.000138    011100
        0        6       10         Random        3.0      0.375000 0.000162    010001
        0        6       10      Greedy-MS        8.0      1.000000 0.000330    011100
        0        6       10        QAOA-p1        8.0      1.000000 0.942840    100011
        1        6       10 BruteForce-opt        7.0      1.000000 0.000122    000110
        1        6       10         Random        5.0      0.714286 0.000149    001110
        1        6       10      Greedy-MS        7.0      1.000000 0.000285    010010
        1        6       10        QAOA-p1        7.0      1.000000 1.585659    111001
```

## Advantage (approx ratio)
```
QAOA approx=1.000±0.000  vs Greedy-MS 1.000±0.000  (Δ=+0.000)
```
