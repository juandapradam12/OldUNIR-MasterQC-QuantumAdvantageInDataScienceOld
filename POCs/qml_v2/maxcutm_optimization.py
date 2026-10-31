"""MaxCut bench — shared cut objective for QAOA vs classical (QML v2).

TFM notebook 02 compared:
  * classical FLP (PuLP) on a full agent×foco assignment, versus
  * QAOA on a *different* reduced QP (select subset of top focos).
Those are not the same objective, so the −30% vs greedy claim is not a
kernel-style matched comparison.

This module uses **one shared problem**: MaxCut on small graphs.
Every solver (QAOA, greedy, brute force) maximizes the same cut value on the
same adjacency matrix. When n is small, brute force gives the optimum so we
report approximation ratios.

Decode: bitstring → partition (S, V\\S) and cut size (classical decision).

The matched assignment ILP (PuLP + greedy + QAOA) lives in `qml_v2.optimization`.
Module name: `qml_v2.maxcutm_optimization`.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from qml_v2.classification import REPO_ROOT, repo_paths


@dataclass
class OptConfig:
    n_nodes: int = 10
    edge_prob: float = 0.55
    qaoa_p: int = 1
    cobyla_maxiter: int = 80
    n_graphs: int = 5
    greedy_restarts: int = 8
    random_state: int = 42
    graph_family: str = "erdos_renyi"  # or "cycle"


@dataclass
class OptResult:
    graph_id: int
    n_nodes: int
    n_edges: int
    model: str
    cut_value: float
    approx_ratio: float | None
    time_s: float
    bitstring: str | None = None


def make_graph(n: int, family: str, p: float, seed: int) -> nx.Graph:
    if family == "cycle":
        return nx.cycle_graph(n)
    if family == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        # Prefer a connected graph so MaxCut is meaningful.
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            G = nx.erdos_renyi_graph(n, max(p, 0.4), seed=seed + 10_000)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            return nx.cycle_graph(n)
        return G
    raise ValueError(f"Unknown graph family: {family}")


def cut_value(G: nx.Graph, bits: np.ndarray) -> float:
    bits = np.asarray(bits, dtype=int)
    val = 0.0
    for u, v in G.edges():
        if bits[u] != bits[v]:
            val += 1.0
    return val


def brute_force_maxcut(G: nx.Graph) -> tuple[float, np.ndarray]:
    n = G.number_of_nodes()
    if n > 12:
        raise ValueError("brute force only for n <= 12")
    best_val = -1.0
    best_bits = np.zeros(n, dtype=int)
    # Fix bit 0 = 0 to remove global flip symmetry
    for combo in itertools.product([0, 1], repeat=n - 1):
        bits = np.array([0, *combo], dtype=int)
        val = cut_value(G, bits)
        if val > best_val:
            best_val = val
            best_bits = bits
    return best_val, best_bits


def random_cut(G: nx.Graph, random_state: int = 0) -> tuple[float, np.ndarray]:
    rng = np.random.RandomState(random_state)
    bits = rng.randint(0, 2, size=G.number_of_nodes())
    return cut_value(G, bits), bits


def greedy_maxcut(G: nx.Graph, random_state: int = 0, n_restarts: int = 1) -> tuple[float, np.ndarray]:
    """Multi-start local greedy: flip nodes that improve the cut."""
    rng = np.random.RandomState(random_state)
    n = G.number_of_nodes()
    best_val = -1.0
    best_bits = np.zeros(n, dtype=int)
    for _r in range(max(1, n_restarts)):
        bits = rng.randint(0, 2, size=n)
        improved = True
        while improved:
            improved = False
            for i in rng.permutation(n):
                cur = cut_value(G, bits)
                bits[i] = 1 - bits[i]
                new = cut_value(G, bits)
                if new < cur:
                    bits[i] = 1 - bits[i]  # revert
                elif new > cur:
                    improved = True
        val = cut_value(G, bits)
        if val > best_val:
            best_val = val
            best_bits = bits.copy()
    return best_val, best_bits


def qaoa_maxcut(G: nx.Graph, p: int, maxiter: int, random_state: int) -> tuple[float, np.ndarray, float]:
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.applications import Maxcut

    algorithm_globals.random_seed = random_state
    w = nx.to_numpy_array(G)
    qp = Maxcut(w).to_quadratic_program()
    t0 = time.time()
    sampler = StatevectorSampler(seed=random_state)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=p)
    result = MinimumEigenOptimizer(qaoa).solve(qp)
    elapsed = time.time() - t0
    bits = np.array(result.x, dtype=int)
    # Prefer cut evaluated on the graph (same metric as baselines)
    return cut_value(G, bits), bits, elapsed


def run_suite(cfg: OptConfig | None = None, quick: bool = False) -> pd.DataFrame:
    cfg = cfg or OptConfig()
    if quick:
        cfg = OptConfig(
            n_nodes=6,
            qaoa_p=1,
            cobyla_maxiter=40,
            n_graphs=2,
            greedy_restarts=2,
            random_state=cfg.random_state,
            graph_family=cfg.graph_family,
            edge_prob=cfg.edge_prob,
        )

    rows: list[OptResult] = []
    for g_id in range(cfg.n_graphs):
        seed = cfg.random_state + g_id
        G = make_graph(cfg.n_nodes, cfg.graph_family, cfg.edge_prob, seed)
        n, m = G.number_of_nodes(), G.number_of_edges()

        t0 = time.time()
        opt_val, opt_bits = brute_force_maxcut(G)
        opt_t = time.time() - t0
        rows.append(
            OptResult(
                graph_id=g_id,
                n_nodes=n,
                n_edges=m,
                model="BruteForce-opt",
                cut_value=opt_val,
                approx_ratio=1.0,
                time_s=opt_t,
                bitstring="".join(map(str, opt_bits)),
            )
        )

        t0 = time.time()
        r_val, r_bits = random_cut(G, random_state=seed)
        rows.append(
            OptResult(
                graph_id=g_id,
                n_nodes=n,
                n_edges=m,
                model="Random",
                cut_value=r_val,
                approx_ratio=r_val / opt_val if opt_val else None,
                time_s=time.time() - t0,
                bitstring="".join(map(str, r_bits)),
            )
        )

        t0 = time.time()
        g_val, g_bits = greedy_maxcut(G, random_state=seed, n_restarts=cfg.greedy_restarts)
        rows.append(
            OptResult(
                graph_id=g_id,
                n_nodes=n,
                n_edges=m,
                model="Greedy-MS",
                cut_value=g_val,
                approx_ratio=g_val / opt_val if opt_val else None,
                time_s=time.time() - t0,
                bitstring="".join(map(str, g_bits)),
            )
        )

        q_val, q_bits, q_time = qaoa_maxcut(G, p=cfg.qaoa_p, maxiter=cfg.cobyla_maxiter, random_state=seed)
        rows.append(
            OptResult(
                graph_id=g_id,
                n_nodes=n,
                n_edges=m,
                model=f"QAOA-p{cfg.qaoa_p}",
                cut_value=q_val,
                approx_ratio=q_val / opt_val if opt_val else None,
                time_s=q_time,
                bitstring="".join(map(str, q_bits)),
            )
        )
        print(
            f"graph {g_id}: edges={m}  opt={opt_val:.0f}  "
            f"rand={r_val:.0f} ({r_val / opt_val:.3f})  "
            f"greedy={g_val:.0f} ({g_val / opt_val:.3f})  "
            f"qaoa={q_val:.0f} ({q_val / opt_val:.3f})  t={q_time:.1f}s"
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df[df["model"] != "BruteForce-opt"]
        .groupby("model", as_index=False)
        .agg(
            cut_mean=("cut_value", "mean"),
            cut_std=("cut_value", "std"),
            approx_mean=("approx_ratio", "mean"),
            approx_std=("approx_ratio", "std"),
            time_mean_s=("time_s", "mean"),
            n_graphs=("graph_id", "nunique"),
        )
        .sort_values("approx_mean", ascending=False)
    )
    for col in ("cut_std", "approx_std"):
        g[col] = g[col].fillna(0.0)
    return g


def run_all(cfg: OptConfig | None = None, quick: bool = False) -> dict[str, Any]:
    cfg = cfg or OptConfig()
    df = run_suite(cfg, quick=quick)
    summary = summarize(df)
    metrics_dir = repo_paths()["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "qml_v2_maxcutm_optimization_raw.csv", index=False)
    summary.to_csv(metrics_dir / "qml_v2_maxcutm_optimization_summary.csv", index=False)

    report = [
        "# MaxCut optimization results (QML v2)",
        "",
        "Same objective for QAOA, multi-start greedy, random, and brute-force optimum.",
        "Approx ratio = cut / optimum. Bitstring decodes to a classical partition.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
        "## Summary",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Per-graph",
        "```",
        df.to_string(index=False),
        "```",
        "",
    ]
    if not summary.empty:
        q = summary[summary["model"].str.startswith("QAOA")]
        g = summary[summary["model"] == "Greedy-MS"]
        if not q.empty and not g.empty:
            dq = float(q["approx_mean"].iloc[0]) - float(g["approx_mean"].iloc[0])
            report += [
                "## Advantage (approx ratio)",
                "```",
                f"QAOA approx={q['approx_mean'].iloc[0]:.3f}±{q['approx_std'].iloc[0]:.3f}  "
                f"vs Greedy-MS {g['approx_mean'].iloc[0]:.3f}±{g['approx_std'].iloc[0]:.3f}  "
                f"(Δ={dq:+.3f})",
                "```",
                "",
            ]
    path = REPO_ROOT / "BUSINESS" / "maxcutm_optimization_v2_results.md"
    path.write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(f"\nWrote {path}")
    return {"raw": df, "summary": summary}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="QML v2 maxcutm_optimization: MaxCut QAOA vs classical")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-nodes", type=int, default=None)
    p.add_argument("--n-graphs", type=int, default=None)
    p.add_argument("--qaoa-p", type=int, default=None)
    p.add_argument("--edge-prob", type=float, default=None)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--greedy-restarts", type=int, default=None)
    p.add_argument("--random-state", type=int, default=None)
    p.add_argument("--family", choices=["erdos_renyi", "cycle"], default=None)
    args = p.parse_args()
    cfg = OptConfig()
    if args.n_nodes is not None:
        cfg.n_nodes = args.n_nodes
    if args.n_graphs is not None:
        cfg.n_graphs = args.n_graphs
    if args.qaoa_p is not None:
        cfg.qaoa_p = args.qaoa_p
    if args.edge_prob is not None:
        cfg.edge_prob = args.edge_prob
    if args.maxiter is not None:
        cfg.cobyla_maxiter = args.maxiter
    if args.greedy_restarts is not None:
        cfg.greedy_restarts = args.greedy_restarts
    if args.random_state is not None:
        cfg.random_state = args.random_state
    if args.family is not None:
        cfg.graph_family = args.family
    run_all(cfg=cfg, quick=args.quick)


if __name__ == "__main__":
    main()
