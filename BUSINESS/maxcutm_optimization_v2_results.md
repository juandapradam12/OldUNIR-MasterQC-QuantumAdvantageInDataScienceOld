# MaxCut optimization results (QML v2)

Same objective for QAOA, multi-start greedy, random, and brute-force optimum.
Approx ratio = cut / optimum. Bitstring decodes to a classical partition.

Config: `{"n_nodes": 10, "edge_prob": 0.55, "qaoa_p": 1, "cobyla_maxiter": 80, "n_graphs": 5, "greedy_restarts": 8, "random_state": 42, "graph_family": "erdos_renyi"}`

## Summary
```
    model  cut_mean  cut_std  approx_mean  approx_std  time_mean_s  n_graphs
Greedy-MS      18.2 1.303840     1.000000    0.000000     0.002622         5
  QAOA-p1      18.0 1.224745     0.989474    0.023538   293.455739         5
   Random      14.0 1.581139     0.768748    0.060931     0.000168         5
```

## Per-graph
```
 graph_id  n_nodes  n_edges          model  cut_value  approx_ratio     time_s  bitstring
        0       10       24 BruteForce-opt       18.0      1.000000   0.003154 0001110011
        0       10       24         Random       13.0      0.722222   0.000169 0100010001
        0       10       24      Greedy-MS       18.0      1.000000   0.002888 1100010101
        0       10       24        QAOA-p1       18.0      1.000000 288.028276 0001110011
        1       10       25 BruteForce-opt       17.0      1.000000   0.003225 0000110110
        1       10       25         Random       14.0      0.823529   0.000176 0011100111
        1       10       25      Greedy-MS       17.0      1.000000   0.002431 1011001001
        1       10       25        QAOA-p1       17.0      1.000000 315.669853 1110001001
        2       10       27 BruteForce-opt       20.0      1.000000   0.003404 0000011111
        2       10       27         Random       15.0      0.750000   0.000163 0111110011
        2       10       27      Greedy-MS       20.0      1.000000   0.002172 1111100000
        2       10       27        QAOA-p1       20.0      1.000000 178.621739 1111100000
        3       10       23 BruteForce-opt       17.0      1.000000   0.003163 0110101101
        3       10       23         Random       12.0      0.705882   0.000166 1010011110
        3       10       23      Greedy-MS       17.0      1.000000   0.002684 1001010010
        3       10       23        QAOA-p1       17.0      1.000000 259.635347 0110101101
        4       10       27 BruteForce-opt       19.0      1.000000   0.003469 0011011000
        4       10       27         Random       16.0      0.842105   0.000165 1100010110
        4       10       27      Greedy-MS       19.0      1.000000   0.002936 1100100111
        4       10       27        QAOA-p1       18.0      0.947368 425.323481 0100011101
```

## Advantage (approx ratio)
```
QAOA approx=0.989±0.024  vs Greedy-MS 1.000±0.000  (Δ=-0.011)
```
