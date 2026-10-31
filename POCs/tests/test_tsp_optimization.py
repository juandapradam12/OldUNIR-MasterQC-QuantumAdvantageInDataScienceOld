"""TSP protocol tests (no QAOA / no Qiskit)."""

import numpy as np

from qml_v2.tsp_optimization import (
    COLOMBIA_CITIES,
    brute_force_tsp,
    distance_matrix_haversine,
    is_valid_tour,
    make_instance,
    matrix_to_tour,
    nearest_neighbor,
    nn_plus_two_opt,
    permutation_to_matrix,
    tour_length,
)


def test_haversine_bogota_medellin_reasonable():
    D = distance_matrix_haversine(COLOMBIA_CITIES[:2])
    # Bogotá–Medellín ~240 km
    assert 200 < D[0, 1] < 300
    assert abs(D[0, 1] - D[1, 0]) < 1e-6


def test_nn_12_city_ballpark():
    D = distance_matrix_haversine(COLOMBIA_CITIES)
    route = nearest_neighbor(D, start=0)
    assert is_valid_tour(route, 12)
    L = tour_length(D, route)
    # Notebook reported ~2564 km
    assert 2000 < L < 3200


def test_exact_n4_matches_notebook():
    D, _ = make_instance("colombia", 4, seed=0)
    route = brute_force_tsp(D, fix_depot=True)
    L = tour_length(D, route)
    # Notebook brute force: 1867.4 km
    assert abs(L - 1867.4) < 1.0


def test_two_opt_le_nn():
    D, _ = make_instance("euclidean", 6, seed=1)
    nn = nearest_neighbor(D, start=0)
    improved = nn_plus_two_opt(D, n_starts=3)
    assert tour_length(D, improved) <= tour_length(D, nn) + 1e-9


def test_permutation_roundtrip():
    route = [0, 2, 1, 3, 0]
    X = permutation_to_matrix(route, 4)
    back = matrix_to_tour(X, fix_depot=True)
    assert back is not None
    assert is_valid_tour(back, 4)
    assert tour_length(np.ones((4, 4)) - np.eye(4), back) > 0


def test_infeasible_matrix():
    X = np.zeros((3, 3), dtype=int)
    X[0, 0] = 1
    X[0, 1] = 1  # city 0 twice
    assert matrix_to_tour(X, fix_depot=True) is None
