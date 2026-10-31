"""Optimization (assignment) protocol tests (no QAOA / no Qiskit)."""

import numpy as np

from qml_v2.optimization import (
    cost_of,
    format_assignment,
    is_feasible,
    make_cost,
    solve_greedy,
    solve_hungarian,
    solve_pulp,
)


def test_pulp_returns_feasible_permutation():
    C = make_cost(3, seed=0, low=1.0, high=10.0)
    cost, X, _ = solve_pulp(C)
    assert is_feasible(X)
    assert abs(cost - cost_of(X, C)) < 1e-9


def test_hungarian_matches_pulp():
    C = make_cost(4, seed=7, low=1.0, high=20.0)
    pulp_cost, _, _ = solve_pulp(C)
    h_cost, h_X, _ = solve_hungarian(C)
    assert is_feasible(h_X)
    assert abs(h_cost - pulp_cost) < 1e-6
    assert abs(h_cost - cost_of(h_X, C)) < 1e-9


def test_greedy_feasible_and_not_better_than_pulp():
    C = make_cost(3, seed=1, low=1.0, high=10.0)
    pulp_cost, _, _ = solve_pulp(C)
    g_cost, g_X, _ = solve_greedy(C)
    assert is_feasible(g_X)
    assert g_cost + 1e-9 >= pulp_cost


def test_infeasible_matrix_detected():
    X = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]])
    assert not is_feasible(X)
    assert format_assignment(X) == "INFEASIBLE"
