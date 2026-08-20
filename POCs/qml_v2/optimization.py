"""Combinatorial optimization — same model for PuLP, greedy, and QAOA (QML v2).

Problem (identical for every solver)
------------------------------------
n tasks (focos), n agents.
Binary x[i,j] = 1 if agent j is assigned to task i.

    minimize   sum_{i,j} C[i,j] * x[i,j]
    s.t.       for each task i:  sum_j x[i,j] == 1   (exactly one agent)
               for each agent j: sum_i x[i,j] == 1   (exactly one task)

Classic bipartite assignment / permutation (the TFM “agent assignment” story).
n=3 → 9 binary variables (QAOA-feasible on a simulator).

Solvers (same C, same constraints, same cost metric)
----------------------------------------------------
* PuLP/CBC   — exact ILP optimum (reference)
* Hungarian  — exact bipartite matching (should match PuLP)
* Greedy     — for each task, cheapest *still free* agent
* QAOA       — QuadraticProgram with the same linear objective + constraints;
               multi-start COBYLA; business cost from named-var decoded X
               (not QUBO fval / penalties)

Decode: named x_i_j bits → n×n matrix → list of (task, agent) pairs.

MaxCut bench lives in `qml_v2.maxcutm_optimization`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pulp
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qml_v2.classification import REPO_ROOT, repo_paths


@dataclass
class AssignConfig:
    n: int = 3
    n_instances: int = 5
    qaoa_p: int = 1
    cobyla_maxiter: int = 80
    qaoa_restarts: int = 3
    random_state: int = 42
    cost_low: float = 1.0
    cost_high: float = 10.0


@dataclass
class AssignResult:
    instance_id: int
    model: str
    cost: float
    gap_vs_pulp: float | None
    feasible: bool
    time_s: float
    assignment: str


def make_cost(n: int, seed: int, low: float, high: float) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(low, high, size=(n, n))


def cost_of(X: np.ndarray, C: np.ndarray) -> float:
    return float(np.sum(C * X))


def is_feasible(X: np.ndarray) -> bool:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        return False
    if not np.all((X == 0) | (X == 1)):
        return False
    return bool(np.all(X.sum(axis=1) == 1) and np.all(X.sum(axis=0) == 1))


def format_assignment(X: np.ndarray) -> str:
    """Format only feasible permutations; otherwise mark infeasible."""
    if not is_feasible(X):
        return "INFEASIBLE"
    pairs = [(i, int(np.where(X[i] == 1)[0][0])) for i in range(X.shape[0])]
    return str(pairs)


def decode_assignment_from_result(
    result, n: int, qp: QuadraticProgram | None = None
) -> np.ndarray:
    """Map QuadraticProgram decision vars by name (ignore QUBO slacks)."""
    X = np.zeros((n, n), dtype=int)
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
            i, j = int(parts[1]), int(parts[2])
            if 0 <= i < n and 0 <= j < n:
                X[i, j] = int(val > 0.5)
        return X
    bits = values[: n * n]
    if bits.size < n * n:
        raise ValueError(f"Expected at least {n*n} decision bits, got {bits.size}")
    return (bits.reshape(n, n) > 0.5).astype(int)


def solve_pulp(C: np.ndarray) -> tuple[float, np.ndarray, float]:
    n = C.shape[0]
    t0 = time.time()
    prob = pulp.LpProblem("Assignment", pulp.LpMinimize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in range(n)
        for j in range(n)
    }
    prob += pulp.lpSum(C[i, j] * x[i, j] for i in range(n) for j in range(n))
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n)) == 1
    for j in range(n):
        prob += pulp.lpSum(x[i, j] for i in range(n)) == 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    X = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            X[i, j] = int(pulp.value(x[i, j]) > 0.5)
    return cost_of(X, C), X, time.time() - t0


def solve_hungarian(C: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Exact bipartite matching via Hungarian algorithm (scipy)."""
    from scipy.optimize import linear_sum_assignment

    t0 = time.time()
    rows, cols = linear_sum_assignment(C)
    X = np.zeros_like(C, dtype=int)
    X[rows, cols] = 1
    return cost_of(X, C), X, time.time() - t0


def solve_greedy(C: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Assign each task (row order) to the cheapest free agent."""
    n = C.shape[0]
    t0 = time.time()
    X = np.zeros((n, n), dtype=int)
    free = set(range(n))
    for i in range(n):
        j_best = min(free, key=lambda j: C[i, j])
        X[i, j_best] = 1
        free.remove(j_best)
    return cost_of(X, C), X, time.time() - t0


def build_qp(C: np.ndarray) -> QuadraticProgram:
    n = C.shape[0]
    qp = QuadraticProgram(name="Assignment")
    for i in range(n):
        for j in range(n):
            qp.binary_var(name=f"x_{i}_{j}")
    qp.minimize(linear={f"x_{i}_{j}": float(C[i, j]) for i in range(n) for j in range(n)})
    for i in range(n):
        qp.linear_constraint(
            linear={f"x_{i}_{j}": 1 for j in range(n)},
            sense="==",
            rhs=1,
            name=f"task_{i}",
        )
    for j in range(n):
        qp.linear_constraint(
            linear={f"x_{i}_{j}": 1 for i in range(n)},
            sense="==",
            rhs=1,
            name=f"agent_{j}",
        )
    return qp


def solve_qaoa(
    C: np.ndarray,
    p: int,
    maxiter: int,
    seed: int,
    n_restarts: int = 1,
) -> tuple[float, np.ndarray, float, bool]:
    """QAOA with multi-start COBYLA; keep best feasible (else best cost)."""
    n = C.shape[0]
    qp = build_qp(C)
    t0 = time.time()
    best: tuple[float, np.ndarray, bool] | None = None

    for r in range(max(1, n_restarts)):
        run_seed = seed + 17 * r
        algorithm_globals.random_seed = run_seed
        qaoa = QAOA(
            sampler=StatevectorSampler(seed=run_seed),
            optimizer=COBYLA(maxiter=maxiter),
            reps=p,
            initial_point=np.random.RandomState(run_seed).uniform(0, 2 * np.pi, size=2 * p),
        )
        result = MinimumEigenOptimizer(qaoa).solve(qp)
        X = decode_assignment_from_result(result, n, qp=qp)
        feasible = is_feasible(X)
        cost = cost_of(X, C) if feasible else float("nan")
        cand = (cost if feasible else float("inf"), X, feasible)
        if best is None:
            best = cand
        else:
            # Prefer feasible; among feasible, lower cost; else keep first
            if cand[2] and (not best[2] or cand[0] < best[0]):
                best = cand
            elif not best[2] and not cand[2]:
                # both infeasible — keep lower raw C·X for diagnostics
                raw = cost_of(X, C)
                if raw < cost_of(best[1], C):
                    best = (float("nan"), X, False)

    elapsed = time.time() - t0
    assert best is not None
    cost, X, feasible = best
    if not feasible:
        cost = float("nan")
    return cost, X, elapsed, feasible


def _gap(cost: float, pulp_cost: float) -> float | None:
    if pulp_cost == 0 or np.isnan(cost):
        return None
    return (cost - pulp_cost) / pulp_cost


def run_suite(cfg: AssignConfig | None = None, quick: bool = False) -> pd.DataFrame:
    cfg = cfg or AssignConfig()
    if quick:
        cfg = AssignConfig(
            n=3,
            n_instances=2,
            qaoa_p=1,
            cobyla_maxiter=40,
            qaoa_restarts=1,
            random_state=cfg.random_state,
        )

    rows: list[AssignResult] = []
    for k in range(cfg.n_instances):
        seed = cfg.random_state + k
        C = make_cost(cfg.n, seed, cfg.cost_low, cfg.cost_high)

        pulp_cost, pulp_X, pulp_t = solve_pulp(C)
        rows.append(
            AssignResult(
                instance_id=k,
                model="PuLP-CBC",
                cost=pulp_cost,
                gap_vs_pulp=0.0,
                feasible=True,
                time_s=pulp_t,
                assignment=format_assignment(pulp_X),
            )
        )

        h_cost, h_X, h_t = solve_hungarian(C)
        rows.append(
            AssignResult(
                instance_id=k,
                model="Hungarian",
                cost=h_cost,
                gap_vs_pulp=_gap(h_cost, pulp_cost),
                feasible=is_feasible(h_X),
                time_s=h_t,
                assignment=format_assignment(h_X),
            )
        )

        g_cost, g_X, g_t = solve_greedy(C)
        rows.append(
            AssignResult(
                instance_id=k,
                model="Greedy",
                cost=g_cost,
                gap_vs_pulp=_gap(g_cost, pulp_cost),
                feasible=is_feasible(g_X),
                time_s=g_t,
                assignment=format_assignment(g_X),
            )
        )

        q_cost, q_X, q_t, q_ok = solve_qaoa(
            C,
            p=cfg.qaoa_p,
            maxiter=cfg.cobyla_maxiter,
            seed=seed,
            n_restarts=cfg.qaoa_restarts,
        )
        rows.append(
            AssignResult(
                instance_id=k,
                model=f"QAOA-p{cfg.qaoa_p}",
                cost=q_cost,
                gap_vs_pulp=_gap(q_cost, pulp_cost),
                feasible=q_ok,
                time_s=q_t,
                assignment=format_assignment(q_X),
            )
        )
        g_gap = 100 * (g_cost - pulp_cost) / pulp_cost if pulp_cost else float("nan")
        q_gap_s = (
            f"{100 * (q_cost - pulp_cost) / pulp_cost:+.1f}%"
            if q_ok and pulp_cost
            else "n/a"
        )
        print(
            f"instance {k}: pulp={pulp_cost:.3f}  hung={h_cost:.3f}  "
            f"greedy={g_cost:.3f} (gap={g_gap:+.1f}%)  "
            f"qaoa={q_cost if q_ok else float('nan'):.3f} feas={q_ok} "
            f"(gap={q_gap_s})  t={q_t:.1f}s"
        )

    return pd.DataFrame([asdict(r) for r in rows])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # Gap only over feasible rows with finite cost
    def _mean_gap(s: pd.Series) -> float:
        v = s.dropna()
        return float(v.mean()) if len(v) else float("nan")

    g = (
        df.groupby("model", as_index=False)
        .agg(
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std"),
            gap_mean=("gap_vs_pulp", _mean_gap),
            gap_std=("gap_vs_pulp", "std"),
            feasible_rate=("feasible", "mean"),
            time_mean_s=("time_s", "mean"),
            n=("instance_id", "nunique"),
        )
        .sort_values("cost_mean")
    )
    for col in ("cost_std", "gap_std"):
        g[col] = g[col].fillna(0.0)
    return g


def run_all(cfg: AssignConfig | None = None, quick: bool = False) -> dict:
    cfg = cfg or AssignConfig()
    df = run_suite(cfg, quick=quick)
    summary = summarize(df)
    metrics = repo_paths()["metrics"]
    metrics.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics / "qml_v2_optimization_raw.csv", index=False)
    summary.to_csv(metrics / "qml_v2_optimization_summary.csv", index=False)

    report = [
        "# Optimization results (QML v2 — same model for all solvers)",
        "",
        "minimize sum C[i,j] x[i,j]  with one-agent-per-task and one-task-per-agent.",
        "PuLP/Hungarian = exact optimum. Gap = (cost − pulp) / pulp.",
        "QAOA cost from named-var decoded X (not QUBO energy); multi-start COBYLA.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
        "## Summary",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Per instance",
        "```",
        df.to_string(index=False),
        "```",
        "",
    ]
    path = REPO_ROOT / "BUSINESS" / "optimization_v2_results.md"
    path.write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(f"\nWrote {path}")
    return {"raw": df, "summary": summary}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="QML v2 optimization: PuLP vs greedy vs QAOA")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=None, help="tasks = agents (default 3)")
    p.add_argument("--n-instances", type=int, default=None)
    p.add_argument("--qaoa-p", type=int, default=None)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--restarts", type=int, default=None, help="QAOA COBYLA restarts")
    p.add_argument("--random-state", type=int, default=None)
    args = p.parse_args()
    cfg = AssignConfig()
    if args.n is not None:
        cfg.n = args.n
    if args.n_instances is not None:
        cfg.n_instances = args.n_instances
    if args.qaoa_p is not None:
        cfg.qaoa_p = args.qaoa_p
    if args.maxiter is not None:
        cfg.cobyla_maxiter = args.maxiter
    if args.restarts is not None:
        cfg.qaoa_restarts = args.restarts
    if args.random_state is not None:
        cfg.random_state = args.random_state
    run_all(cfg=cfg, quick=args.quick)


if __name__ == "__main__":
    main()
