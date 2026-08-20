"""MaxCut protocol tests (no QAOA / no Qiskit)."""

import networkx as nx
import numpy as np

from qml_v2.maxcutm_optimization import (
    brute_force_maxcut,
    cut_value,
    greedy_maxcut,
    make_graph,
    random_cut,
)


def test_cut_value_cycle4():
    G = nx.cycle_graph(4)
    # Alternating bits → cut all 4 edges
    bits = np.array([0, 1, 0, 1])
    assert cut_value(G, bits) == 4.0


def test_brute_force_cycle4_is_optimal():
    G = nx.cycle_graph(4)
    opt, bits = brute_force_maxcut(G)
    assert opt == 4.0
    assert cut_value(G, bits) == opt


def test_greedy_and_random_le_opt():
    G = make_graph(6, "erdos_renyi", 0.5, seed=0)
    opt, _ = brute_force_maxcut(G)
    g_val, _ = greedy_maxcut(G, random_state=0, n_restarts=4)
    r_val, _ = random_cut(G, random_state=0)
    assert g_val <= opt + 1e-9
    assert r_val <= opt + 1e-9
