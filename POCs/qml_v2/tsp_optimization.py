"""TSP optimization — same closed tour for QAOA vs classical (QML v2).

TFM notebook 05 mixed n=12 NN with n=4 QAOA, labeled SA-over-enumerated-tours
as quantum, and never ran 2-opt. This module uses **one shared problem**:

    minimize  L(π) = sum_k D[π_k, π_{k+1}]   (closed cycle)
    same D, same L for every solver.

Solvers
-------
* BruteForce-opt — exact (n-1)! with depot fixed at position 0
* NearestNeighbor / NN-MS — greedy (single start / all starts)
* NN+2-opt — local improvement from NN seeds
* Random-tour — random permutation floor
* SA-QUBO — classical SA on the **same n² Lucas QUBO** (not a tour list)
* QAOA — QuadraticProgram with tour cost + row/column equalities;
         named-var decode; business cost = raw tour length (not Ising fval)

Not VRP: single vehicle, closed tour. VRP-TW is future work.
Module: ``qml_v2.tsp_optimization``.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from qml_v2.classification import REPO_ROOT, repo_paths

# 12 Colombian capitals from TFM notebook 05 (CSV not in repo).
COLOMBIA_CITIES: list[tuple[str, float, float]] = [
    ("Bogotá", 4.7110, -74.0721),
    ("Medellín", 6.2476, -75.5658),
    ("Cali", 3.4516, -76.5320),
    ("Barranquilla", 10.9685, -74.7813),
    ("Cartagena", 10.3910, -75.4794),
    ("Cúcuta", 7.8891, -72.4967),
    ("Bucaramanga", 7.1193, -73.1227),
    ("Pereira", 4.8133, -75.6961),
    ("Manizales", 5.0700, -75.5138),
    ("Ibagué", 4.4389, -75.2322),
    ("Villavicencio", 4.1420, -73.6266),
    ("Santa Marta", 11.2408, -74.1990),
]


@dataclass
class TSPConfig:
    n: int = 4
    n_instances: int = 3
    family: str = "colombia"  # colombia | euclidean
    qaoa_p: int = 1
    cobyla_maxiter: int = 80
    qaoa_restarts: int = 3
    sa_iters: int = 5000
    sa_T0: float = 1.0
    fix_depot: bool = True
    nn_starts: int | None = None  # None → all n starts
    random_state: int = 42
    run_qaoa: bool = True
    run_sa: bool = True


@dataclass
class TSPResult:
    instance_id: int
    n: int
    family: str
    model: str
    cost: float
    gap_vs_exact: float | None
    approx_ratio: float | None
    feasible: bool
    time_s: float
    tour: str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_matrix_haversine(cities: list[tuple[str, float, float]]) -> np.ndarray:
    n = len(cities)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            D[i, j] = haversine_km(cities[i][1], cities[i][2], cities[j][1], cities[j][2])
    return D


def make_instance(
    family: str, n: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    if family == "colombia":
        if n > len(COLOMBIA_CITIES):
            raise ValueError(f"colombia family has only {len(COLOMBIA_CITIES)} cities")
        cities = COLOMBIA_CITIES[:n]
        names = [c[0] for c in cities]
        return distance_matrix_haversine(cities), names
    if family == "euclidean":
        rng = np.random.RandomState(seed)
        pts = rng.uniform(0.0, 1.0, size=(n, 2))
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        names = [f"p{i}" for i in range(n)]
        return D.astype(float), names
    raise ValueError(f"Unknown family: {family}")


def tour_length(D: np.ndarray, route: list[int]) -> float:
    """Closed or open path; if route[-1] != route[0], closes automatically."""
    if len(route) < 2:
        return float("nan")
    r = list(route)
    if r[0] != r[-1]:
        r = r + [r[0]]
    return float(sum(D[r[i], r[i + 1]] for i in range(len(r) - 1)))


def format_tour(route: list[int] | None, names: list[str] | None = None) -> str:
    if route is None:
        return "INFEASIBLE"
    r = list(route)
    if r[0] != r[-1]:
        r = r + [r[0]]
    if names is not None:
        return " → ".join(names[i] for i in r)
    return " → ".join(str(i) for i in r)


def is_valid_tour(route: list[int] | None, n: int) -> bool:
    if route is None or len(route) < n:
        return False
    body = route[:-1] if route[0] == route[-1] else route
    if len(body) != n:
        return False
    return set(body) == set(range(n))


def nearest_neighbor(D: np.ndarray, start: int = 0) -> list[int]:
    n = D.shape[0]
    unvisited = set(range(n)) - {start}
    route = [start]
    while unvisited:
        j = min(unvisited, key=lambda k: D[route[-1], k])
        route.append(j)
        unvisited.remove(j)
    route.append(start)
    return route


def nearest_neighbor_multistart(D: np.ndarray, n_starts: int | None = None) -> list[int]:
    n = D.shape[0]
    starts = range(n) if n_starts is None else range(min(n_starts, n))
    best: list[int] | None = None
    best_L = float("inf")
    for s in starts:
        r = nearest_neighbor(D, start=s)
        L = tour_length(D, r)
        if L < best_L:
            best_L, best = L, r
    assert best is not None
    return best


def two_opt(D: np.ndarray, route: list[int]) -> list[int]:
    """Improve a closed tour with 2-opt until local optimum."""
    r = list(route)
    if r[0] != r[-1]:
        r = r + [r[0]]
    n = len(r) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = r[i - 1], r[i]
                c, d = r[k], r[k + 1]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-12:
                    r[i : k + 1] = reversed(r[i : k + 1])
                    improved = True
    return r


def nn_plus_two_opt(D: np.ndarray, n_starts: int | None = None) -> list[int]:
    n = D.shape[0]
    starts = range(n) if n_starts is None else range(min(n_starts, n))
    best: list[int] | None = None
    best_L = float("inf")
    for s in starts:
        r = two_opt(D, nearest_neighbor(D, start=s))
        L = tour_length(D, r)
        if L < best_L:
            best_L, best = L, r
    assert best is not None
    return best


def random_tour(D: np.ndarray, random_state: int = 0) -> list[int]:
    rng = np.random.RandomState(random_state)
    body = list(range(1, D.shape[0]))
    rng.shuffle(body)
    return [0] + body + [0]


def brute_force_tsp(D: np.ndarray, fix_depot: bool = True) -> list[int]:
    n = D.shape[0]
    if n > 9:
        raise ValueError("brute force only for n <= 9")
    best: list[int] | None = None
    best_L = float("inf")
    if fix_depot:
        for perm in itertools.permutations(range(1, n)):
            r = [0, *perm, 0]
            L = tour_length(D, r)
            if L < best_L:
                best_L, best = L, r
    else:
        for perm in itertools.permutations(range(n)):
            r = list(perm) + [perm[0]]
            L = tour_length(D, r)
            if L < best_L:
                best_L, best = L, r
    assert best is not None
    return best


def permutation_to_matrix(route: list[int], n: int) -> np.ndarray:
    body = route[:-1] if route and route[0] == route[-1] else list(route)
    X = np.zeros((n, n), dtype=int)
    for p, city in enumerate(body):
        X[city, p] = 1
    return X


def matrix_to_tour(X: np.ndarray, fix_depot: bool = True) -> list[int] | None:
    """Decode permutation matrix city×position → closed tour, or None."""
    X = np.asarray(X, dtype=int)
    n = X.shape[0]
    if X.shape != (n, n):
        return None
    if not np.all((X == 0) | (X == 1)):
        return None
    if not (np.all(X.sum(axis=0) == 1) and np.all(X.sum(axis=1) == 1)):
        return None
    if fix_depot and X[0, 0] != 1:
        return None
    route = []
    for p in range(n):
        cities = np.where(X[:, p] == 1)[0]
        if len(cities) != 1:
            return None
        route.append(int(cities[0]))
    route.append(route[0])
    return route


def build_tsp_qp(D: np.ndarray, fix_depot: bool = True):
    """Lucas TSP QP. With ``fix_depot``, only (n-1)² binaries (city 0 at pos 0)."""
    from qiskit_optimization import QuadraticProgram

    n = D.shape[0]
    qp = QuadraticProgram("TSP")

    if fix_depot:
        # Variables for non-depot cities and positions 1..n-1
        for i in range(1, n):
            for p in range(1, n):
                qp.binary_var(name=f"x_{i}_{p}")

        linear: dict[str, float] = {}
        quadratic: dict[tuple[str, str], float] = {}

        # depot → city at position 1
        for i in range(1, n):
            linear[f"x_{i}_1"] = linear.get(f"x_{i}_1", 0.0) + float(D[0, i])
        # city at position n-1 → depot
        for i in range(1, n):
            linear[f"x_{i}_{n-1}"] = linear.get(f"x_{i}_{n-1}", 0.0) + float(D[i, 0])
        # consecutive non-depot positions
        for u in range(1, n):
            for v in range(1, n):
                if u == v:
                    continue
                for p in range(1, n - 1):
                    a, b = f"x_{u}_{p}", f"x_{v}_{p+1}"
                    key = (a, b) if a <= b else (b, a)
                    quadratic[key] = quadratic.get(key, 0.0) + float(D[u, v])

        qp.minimize(linear=linear, quadratic=quadratic)
        for p in range(1, n):
            qp.linear_constraint(
                linear={f"x_{i}_{p}": 1 for i in range(1, n)},
                sense="==",
                rhs=1,
                name=f"pos_{p}",
            )
        for i in range(1, n):
            qp.linear_constraint(
                linear={f"x_{i}_{p}": 1 for p in range(1, n)},
                sense="==",
                rhs=1,
                name=f"city_{i}",
            )
        return qp

    for i in range(n):
        for p in range(n):
            qp.binary_var(name=f"x_{i}_{p}")
    quadratic = {}
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for p in range(n):
                a, b = f"x_{u}_{p}", f"x_{v}_{(p + 1) % n}"
                key = (a, b) if a <= b else (b, a)
                quadratic[key] = quadratic.get(key, 0.0) + float(D[u, v])
    qp.minimize(quadratic=quadratic)
    for p in range(n):
        qp.linear_constraint(
            linear={f"x_{i}_{p}": 1 for i in range(n)},
            sense="==",
            rhs=1,
            name=f"pos_{p}",
        )
    for i in range(n):
        qp.linear_constraint(
            linear={f"x_{i}_{p}": 1 for p in range(n)},
            sense="==",
            rhs=1,
            name=f"city_{i}",
        )
    return qp


def decode_tsp_from_result(result, n: int, qp=None, fix_depot: bool = True) -> np.ndarray:
    X = np.zeros((n, n), dtype=int)
    if fix_depot:
        X[0, 0] = 1
    values = np.asarray(result.x, dtype=float)
    names = list(getattr(result, "variable_names", None) or [])
    if not names and qp is not None:
        names = [v.name for v in qp.variables]
    if names and len(names) >= len(values):
        names = names[: len(values)]
    if names and len(names) == len(values):
        for name, val in zip(names, values):
            if not str(name).startswith("x_"):
                continue
            parts = str(name).split("_")
            if len(parts) != 3:
                continue
            i, p = int(parts[1]), int(parts[2])
            if 0 <= i < n and 0 <= p < n:
                X[i, p] = int(val > 0.5)
        return X
    # Fallback reshape for full n² only
    bits = values[: n * n]
    if bits.size >= n * n:
        return np.round(bits).astype(int).reshape(n, n)
    return X


def qubo_dict_from_qp(qp) -> dict[tuple[str, str], float]:
    """Flatten QuadraticProgram (after to_ising path) via converter."""
    from qiskit_optimization.converters import QuadraticProgramToQubo

    qubo = QuadraticProgramToQubo().convert(qp)
    Q: dict[tuple[str, str], float] = {}
    # linear → diagonal; quadratic as given
    lin = qubo.objective.linear.to_dict()
    for k, v in lin.items():
        name = qubo.variables[k].name if isinstance(k, int) else str(k)
        Q[(name, name)] = Q.get((name, name), 0.0) + float(v)
    quad = qubo.objective.quadratic.to_dict()
    for (i, j), v in quad.items():
        ni = qubo.variables[i].name if isinstance(i, int) else str(i)
        nj = qubo.variables[j].name if isinstance(j, int) else str(j)
        key = (ni, nj) if ni <= nj else (nj, ni)
        Q[key] = Q.get(key, 0.0) + float(v)
    return Q, qubo


def sa_qubo_tsp(
    D: np.ndarray,
    *,
    fix_depot: bool,
    n_iters: int,
    T0: float,
    random_state: int,
) -> tuple[list[int] | None, float]:
    """Classical SA on the Lucas n² QUBO; score = decoded tour length."""
    qp = build_tsp_qp(D, fix_depot=fix_depot)
    Q, qubo = qubo_dict_from_qp(qp)
    names = [v.name for v in qubo.variables]
    # Keep only decision x_* (ignore slacks if any)
    x_names = [nm for nm in names if nm.startswith("x_")]
    idx = {nm: i for i, nm in enumerate(x_names)}
    n = D.shape[0]
    rng = np.random.RandomState(random_state)
    bits = rng.randint(0, 2, size=len(x_names)).astype(int)

    def energy(b: np.ndarray) -> float:
        e = 0.0
        for (a, c), w in Q.items():
            if a not in idx or c not in idx:
                continue
            if a == c:
                e += w * b[idx[a]]
            else:
                e += w * b[idx[a]] * b[idx[c]]
        return e

    cur_e = energy(bits)
    best_b = bits.copy()
    best_e = cur_e
    for t in range(n_iters):
        T = T0 * (1.0 - t / max(n_iters, 1)) + 1e-9
        j = int(rng.randint(0, len(bits)))
        bits[j] = 1 - bits[j]
        new_e = energy(bits)
        if new_e <= cur_e or rng.rand() < math.exp(-(new_e - cur_e) / T):
            cur_e = new_e
            if new_e < best_e:
                best_e = new_e
                best_b = bits.copy()
        else:
            bits[j] = 1 - bits[j]

    X = np.zeros((n, n), dtype=int)
    if fix_depot:
        X[0, 0] = 1
    for nm, bit in zip(x_names, best_b):
        parts = nm.split("_")
        if len(parts) == 3:
            i, p = int(parts[1]), int(parts[2])
            if 0 <= i < n and 0 <= p < n:
                X[i, p] = int(bit)
    return matrix_to_tour(X, fix_depot=fix_depot), float(best_e)


def qaoa_tsp(
    D: np.ndarray,
    *,
    p: int,
    maxiter: int,
    restarts: int,
    fix_depot: bool,
    random_state: int,
) -> tuple[list[int] | None, float]:
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    n = D.shape[0]
    qp = build_tsp_qp(D, fix_depot=fix_depot)
    best_route: list[int] | None = None
    best_L = float("inf")
    t0 = time.time()
    for r in range(max(1, restarts)):
        seed = random_state + 17 * r
        algorithm_globals.random_seed = seed
        sampler = StatevectorSampler(seed=seed)
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=p)
        result = MinimumEigenOptimizer(qaoa).solve(qp)
        X = decode_tsp_from_result(result, n, qp=qp, fix_depot=fix_depot)
        route = matrix_to_tour(X, fix_depot=fix_depot)
        if route is None:
            continue
        L = tour_length(D, route)
        if L < best_L:
            best_L, best_route = L, route
    return best_route, time.time() - t0


def _append_scored(
    rows: list[TSPResult],
    *,
    instance_id: int,
    n: int,
    family: str,
    model: str,
    D: np.ndarray,
    route: list[int] | None,
    exact_L: float,
    time_s: float,
    names: list[str],
) -> None:
    feasible = is_valid_tour(route, n)
    if feasible and route is not None:
        cost = tour_length(D, route)
        gap = (cost - exact_L) / exact_L if exact_L > 0 else None
        approx = exact_L / cost if cost > 0 else None
        tour_s = format_tour(route, names)
    else:
        cost = float("nan")
        gap = None
        approx = None
        tour_s = "INFEASIBLE"
    rows.append(
        TSPResult(
            instance_id=instance_id,
            n=n,
            family=family,
            model=model,
            cost=cost,
            gap_vs_exact=gap,
            approx_ratio=approx,
            feasible=feasible,
            time_s=time_s,
            tour=tour_s,
        )
    )


def run_suite(cfg: TSPConfig | None = None, quick: bool = False) -> pd.DataFrame:
    cfg = cfg or TSPConfig()
    if quick:
        cfg = TSPConfig(
            n=3,
            n_instances=2 if cfg.family == "euclidean" else 1,
            family=cfg.family,
            qaoa_p=1,
            cobyla_maxiter=40,
            qaoa_restarts=1,
            sa_iters=400,
            fix_depot=cfg.fix_depot,
            nn_starts=1,
            random_state=cfg.random_state,
            run_qaoa=cfg.run_qaoa,
            run_sa=cfg.run_sa,
        )

    n_inst = 1 if cfg.family == "colombia" else cfg.n_instances
    rows: list[TSPResult] = []
    for iid in range(n_inst):
        seed = cfg.random_state + iid
        D, names = make_instance(cfg.family, cfg.n, seed)
        n = cfg.n

        t0 = time.time()
        exact = brute_force_tsp(D, fix_depot=cfg.fix_depot)
        exact_L = tour_length(D, exact)
        _append_scored(
            rows,
            instance_id=iid,
            n=n,
            family=cfg.family,
            model="BruteForce-opt",
            D=D,
            route=exact,
            exact_L=exact_L,
            time_s=time.time() - t0,
            names=names,
        )

        t0 = time.time()
        nn = nearest_neighbor(D, start=0)
        _append_scored(
            rows,
            instance_id=iid,
            n=n,
            family=cfg.family,
            model="NearestNeighbor",
            D=D,
            route=nn,
            exact_L=exact_L,
            time_s=time.time() - t0,
            names=names,
        )

        t0 = time.time()
        nn_ms = nearest_neighbor_multistart(D, n_starts=cfg.nn_starts)
        _append_scored(
            rows,
            instance_id=iid,
            n=n,
            family=cfg.family,
            model="NN-MS",
            D=D,
            route=nn_ms,
            exact_L=exact_L,
            time_s=time.time() - t0,
            names=names,
        )

        t0 = time.time()
        nn2 = nn_plus_two_opt(D, n_starts=cfg.nn_starts)
        _append_scored(
            rows,
            instance_id=iid,
            n=n,
            family=cfg.family,
            model="NN+2-opt",
            D=D,
            route=nn2,
            exact_L=exact_L,
            time_s=time.time() - t0,
            names=names,
        )

        t0 = time.time()
        rnd = random_tour(D, random_state=seed)
        _append_scored(
            rows,
            instance_id=iid,
            n=n,
            family=cfg.family,
            model="Random-tour",
            D=D,
            route=rnd,
            exact_L=exact_L,
            time_s=time.time() - t0,
            names=names,
        )

        if cfg.run_sa:
            t0 = time.time()
            sa_route, _ = sa_qubo_tsp(
                D,
                fix_depot=cfg.fix_depot,
                n_iters=cfg.sa_iters,
                T0=cfg.sa_T0,
                random_state=seed,
            )
            _append_scored(
                rows,
                instance_id=iid,
                n=n,
                family=cfg.family,
                model="SA-QUBO",
                D=D,
                route=sa_route,
                exact_L=exact_L,
                time_s=time.time() - t0,
                names=names,
            )

        if cfg.run_qaoa:
            q_route, q_time = qaoa_tsp(
                D,
                p=cfg.qaoa_p,
                maxiter=cfg.cobyla_maxiter,
                restarts=cfg.qaoa_restarts,
                fix_depot=cfg.fix_depot,
                random_state=seed,
            )
            _append_scored(
                rows,
                instance_id=iid,
                n=n,
                family=cfg.family,
                model=f"QAOA-p{cfg.qaoa_p}",
                D=D,
                route=q_route,
                exact_L=exact_L,
                time_s=q_time,
                names=names,
            )

        q_row = [r for r in rows if r.instance_id == iid and r.model.startswith("QAOA")]
        q_cost = q_row[-1].cost if q_row else float("nan")
        print(
            f"instance {iid}: exact={exact_L:.1f}  "
            f"NN={tour_length(D, nn):.1f}  "
            f"2opt={tour_length(D, nn2):.1f}  "
            f"qaoa={q_cost:.1f}"
        )

    return pd.DataFrame([asdict(r) for r in rows])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df[df["model"] != "BruteForce-opt"]
        .groupby("model", as_index=False)
        .agg(
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std"),
            gap_mean=("gap_vs_exact", "mean"),
            approx_mean=("approx_ratio", "mean"),
            feasible_rate=("feasible", "mean"),
            time_mean_s=("time_s", "mean"),
            n_instances=("instance_id", "nunique"),
        )
        .sort_values("gap_mean", ascending=True)
    )
    for col in ("cost_std", "gap_mean", "approx_mean"):
        if col in g.columns:
            g[col] = g[col].fillna(np.nan)
    return g


def run_all(cfg: TSPConfig | None = None, quick: bool = False) -> dict[str, Any]:
    cfg = cfg or TSPConfig()
    df = run_suite(cfg, quick=quick)
    summary = summarize(df)
    metrics_dir = repo_paths()["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "qml_v2_tsp_optimization_raw.csv", index=False)
    summary.to_csv(metrics_dir / "qml_v2_tsp_optimization_summary.csv", index=False)

    report = [
        "# TSP optimization results (QML v2)",
        "",
        "Same closed-tour length L(π) for brute force, NN, NN+2-opt, SA-QUBO (n²), QAOA.",
        "Not VRP. Cost is raw tour length (km or Euclidean), never QUBO/Ising energy.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
        "## Summary",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Per-instance",
        "```",
        df.to_string(index=False),
        "```",
        "",
    ]
    if not summary.empty:
        q = summary[summary["model"].str.startswith("QAOA")]
        best_cl = summary[~summary["model"].str.startswith("QAOA")]
        if not q.empty and not best_cl.empty:
            # lower gap is better
            cl = best_cl.loc[best_cl["gap_mean"].idxmin()]
            dq = float(q["gap_mean"].iloc[0]) - float(cl["gap_mean"])
            report += [
                "## Advantage (gap vs exact; lower is better)",
                "```",
                f"QAOA gap={q['gap_mean'].iloc[0]:.4f}  "
                f"vs best classical {cl['model']} gap={cl['gap_mean']:.4f}  "
                f"(Δgap={dq:+.4f}; feasible_Q={q['feasible_rate'].iloc[0]:.2f})",
                "```",
                "",
            ]
    path = REPO_ROOT / "BUSINESS" / "tsp_optimization_v2_results.md"
    path.write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(f"\nWrote {path}")
    return {"raw": df, "summary": summary}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="QML v2 TSP: QAOA vs classical (same L)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--n-instances", type=int, default=None)
    p.add_argument("--family", choices=["colombia", "euclidean"], default=None)
    p.add_argument("--qaoa-p", type=int, default=None)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--restarts", type=int, default=None)
    p.add_argument("--sa-iters", type=int, default=None)
    p.add_argument("--no-qaoa", action="store_true")
    p.add_argument("--no-sa", action="store_true")
    p.add_argument("--no-fix-depot", action="store_true")
    p.add_argument("--random-state", type=int, default=None)
    args = p.parse_args()
    cfg = TSPConfig()
    if args.n is not None:
        cfg.n = args.n
    if args.n_instances is not None:
        cfg.n_instances = args.n_instances
    if args.family is not None:
        cfg.family = args.family
    if args.qaoa_p is not None:
        cfg.qaoa_p = args.qaoa_p
    if args.maxiter is not None:
        cfg.cobyla_maxiter = args.maxiter
    if args.restarts is not None:
        cfg.qaoa_restarts = args.restarts
    if args.sa_iters is not None:
        cfg.sa_iters = args.sa_iters
    if args.random_state is not None:
        cfg.random_state = args.random_state
    if args.no_qaoa:
        cfg.run_qaoa = False
    if args.no_sa:
        cfg.run_sa = False
    if args.no_fix_depot:
        cfg.fix_depot = False
    run_all(cfg=cfg, quick=args.quick)


if __name__ == "__main__":
    main()
