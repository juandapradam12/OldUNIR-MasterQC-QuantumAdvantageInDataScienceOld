"""Estructura cuántica por caso de uso: mismos indicadores, mismo protocolo.

Indicadores (qml_v2.quantumness, sin medir ventaja supervisada):
  ΔA  = A(K_Q, yyᵀ) − A(K_RBF, yyᵀ)      (Cortes / Huang)
  ΔH  = H(K_Q) − H(K_RBF)                (Thanasilp)
  d_F = ‖K_Q − K_RBF‖_F / n               (proxy propio, cf. Liu)
  g   = diferencia geométrica de Huang (norma espectral, kernels con Tr K = n,
        regularización λ=1e-3; ver quantumness.geometric_difference_huang)
  Q   = 0.4·ΔA + 0.2·d̂_F + 0.2·ΔH + 0.2·ĝ  (compuesto TFM; solo informativo)
        con d̂_F = d_F/max d_F y ĝ = g/max g sobre los casos medidos.

Protocolo único: ZZFeatureMap 2 qubits, reps=2, entanglement full,
StandardScaler → PCA(2) → MinMax[−π, π], n_sample=60, n_repeats=3,
baseline = RBF Frobenius-best.

Optimización (honestidad metodológica)
--------------------------------------
Un problema de optimización no tiene (X, y) como un dataset. Aquí se usa el
**paisaje de energía** del QUBO: X = TODOS los bitstrings del espacio de
búsqueda, y = energía del QUBO (con penalización, exactamente lo que ve QAOA).
Es un test de *estructura del paisaje* con el mismo pipeline de kernel; NO mide
el rendimiento de QAOA ni es comparable con una ventaja de coste.

Control nulo: ΔA con y permutada (misma X) cuantifica el ruido de ΔA.
Solo ΔA depende de y; ΔH, d_F y g solo dependen de X.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qml_v2.classification import encode_for_zz_kernel, subsample_stratified
from qml_v2.quantumness import QuantumnessConfig, indicators_on_matrix

CFG = QuantumnessConfig(
    n_qubits=2,
    zz_reps=2,
    entanglement="full",
    n_sample=60,
    n_repeats=3,
    random_state=42,
)

INDICATORS = ["delta_A", "delta_H", "d_F"]
Q_WEIGHTS = {"delta_A": 0.4, "d_F_hat": 0.2, "delta_H": 0.2, "g_hat": 0.2}


@dataclass
class Job:
    family: str
    name: str
    instances: list[tuple[np.ndarray, np.ndarray]]
    already_zz_space: bool = False
    note: str = ""


def subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Estratificado si y es categórica; muestreo simple si es continua."""
    y = np.asarray(y)
    if len(y) <= n:
        return np.asarray(X, float), y
    n_unique = len(np.unique(y))
    categorical = n_unique <= max(8, len(y) // 10)
    if categorical:
        try:
            return subsample_stratified(X, y, n, seed)
        except ValueError:
            pass
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return np.asarray(X, float)[idx], y[idx]


def _standardize_continuous(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind == "f" or (len(np.unique(y)) > 8 and y.dtype.kind in "iuf"):
        return (y - float(np.mean(y))) / (float(np.std(y)) + 1e-12)
    return y


def measure_job(job: Job, cfg: QuantumnessConfig = CFG, verbose: bool = True) -> dict:
    """Media de indicadores sobre instancias × repeticiones, más control nulo de ΔA."""
    rows, null_dA = [], []
    t0 = time.time()
    for k, (X, y) in enumerate(job.instances):
        y = _standardize_continuous(y)
        for r in range(cfg.n_repeats):
            seed = cfg.random_state + 17 * r + 1000 * k
            Xs, ys = subsample(X, y, cfg.n_sample, seed)
            Xq, _, pca_var = encode_for_zz_kernel(
                Xs,
                Xs,
                n_qubits=cfg.n_qubits,
                random_state=seed,
                already_zz_space=job.already_zz_space,
            )
            ind = indicators_on_matrix(Xq, ys, cfg, seed=seed, compute_advantage=False)
            ind["pca_var"] = float(pca_var)
            rows.append(ind)
            rng = np.random.RandomState(seed + 7)
            ind0 = indicators_on_matrix(Xq, rng.permutation(ys), cfg, seed=seed, compute_advantage=False)
            null_dA.append(ind0["delta_A"])
    df = pd.DataFrame(rows)
    out = {
        "family": job.family,
        "dataset": job.name,
        "n_instances": len(job.instances),
        "n_sample": cfg.n_sample,
        "n_repeats": cfg.n_repeats,
        "n_qubits": cfg.n_qubits,
    }
    for key in INDICATORS:
        out[key] = float(df[key].mean())
        out[f"{key}_std"] = float(df[key].std(ddof=0))
    out["g"] = float(df["g_huang"].mean())
    out["g_std"] = float(df["g_huang"].std(ddof=0))
    out["g_legacy"] = float(df["g"].mean())
    out.update(
        H_Q=float(df["H_Q"].mean()),
        H_cl=float(df["H_cl"].mean()),
        A_Q=float(df["A_Q"].mean()),
        A_cl=float(df["A_cl"].mean()),
        pca_var=float(df["pca_var"].mean()),
        delta_A_null=float(np.mean(null_dA)),
        delta_A_null_std=float(np.std(null_dA)),
        seconds=round(time.time() - t0, 1),
    )
    if verbose:
        print(
            f"  [{job.family:13s}] {job.name:12s} ΔA={out['delta_A']:+.4f} "
            f"(nulo {out['delta_A_null']:+.4f}±{out['delta_A_null_std']:.4f})  "
            f"ΔH={out['delta_H']:.3f}  d_F={out['d_F']:.4f}  g={out['g']:.1f}  "
            f"pcaVar={out['pca_var']:.2f}  ({out['seconds']}s)"
        )
    return out


# --------------------------------------------------------------------------
# Paisajes de optimización
# --------------------------------------------------------------------------
def all_bitstrings(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)


def qubo_landscape(qp) -> tuple[np.ndarray, np.ndarray]:
    """(X, y): todos los bitstrings de las variables de decisión y energía QUBO.

    La conversión QuadraticProgramToQubo penaliza las restricciones igual que
    en la ejecución de QAOA (MinimumEigenOptimizer).
    """
    from qiskit_optimization.converters import QuadraticProgramToQubo

    qubo = QuadraticProgramToQubo().convert(qp)
    n = qubo.get_num_vars()
    X = all_bitstrings(n)
    y = np.array([qubo.objective.evaluate(x) for x in X], dtype=float)
    return X, y


def assignment_instances(n_instances: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    from qml_v2.optimization import AssignConfig, build_qp, make_cost

    c = AssignConfig()
    out = []
    for i in range(n_instances):
        C = make_cost(c.n, seed + i, c.cost_low, c.cost_high)
        out.append(qubo_landscape(build_qp(C)))
    return out


def maxcut_instances(n_instances: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    import networkx as nx
    from qiskit_optimization.applications import Maxcut

    from qml_v2.maxcutm_optimization import OptConfig, make_graph

    c = OptConfig()
    out = []
    for i in range(n_instances):
        G = make_graph(c.n_nodes, c.graph_family, c.edge_prob, seed + i)
        qp = Maxcut(nx.to_numpy_array(G)).to_quadratic_program()
        out.append(qubo_landscape(qp))
    return out


def tsp_instances(n_instances: int = 3, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    from qml_v2.tsp_optimization import TSPConfig, build_tsp_qp, make_instance

    c = TSPConfig()
    out = []
    for i in range(n_instances):
        D, _ = make_instance("euclidean", c.n, seed + i)
        out.append(qubo_landscape(build_tsp_qp(D, fix_depot=True)))
    return out


def optimization_jobs() -> list[Job]:
    return [
        Job(
            "optimizacion",
            "Asignación",
            assignment_instances(5),
            note="asignación QUBO n=3 (9 qubits lógicos), 5 matrices de coste",
        ),
        Job(
            "optimizacion",
            "MaxCut",
            maxcut_instances(5),
            note="MaxCut n=10, 5 grafos Erdős–Rényi (módulo maxcutm; proxy de cobertura/MDS)",
        ),
        Job("optimizacion", "TSP", tsp_instances(3), note="TSP n=4 euclídeo, depósito fijo (9 binarias), 3 instancias"),
    ]


# --------------------------------------------------------------------------
# Datasets supervisados (clasificación, regresión, anomalías)
# --------------------------------------------------------------------------
def supervised_jobs() -> list[Job]:
    from qml_v2.anomaly import (
        inject_feature_outliers,
        make_moons_anomaly,
        make_zz_phase_anomaly,
        subsample_rows,
    )
    from qml_v2.classification import load_skg_incidents, load_telco, make_quantum_native
    from qml_v2.regression import (
        load_friedman,
        load_skg_speeds,
        make_quantum_native_regression,
    )

    jobs: list[Job] = []
    X, y = make_quantum_native(n_samples=200, n_qubits=2, reps=2, random_state=42)
    jobs.append(Job("clasificacion", "Nativo", [(X, y)], True))
    X, y, _ = load_skg_incidents()
    jobs.append(Job("clasificacion", "SKG", [(X, y)]))
    X, y, _ = load_telco()
    jobs.append(Job("clasificacion", "Telco", [(X, y)]))

    X, y = make_quantum_native_regression(n_samples=200, n_qubits=2, reps=2, random_state=42)
    jobs.append(Job("regresion", "Nativo", [(X, y)], True))
    X, y = load_friedman(n_samples=200, n_features=5, random_state=42)
    jobs.append(Job("regresion", "Friedman", [(X, y)]))
    try:
        X, y, _ = load_skg_speeds()
        jobs.append(Job("regresion", "SKG", [(X, y)]))
    except FileNotFoundError as e:
        print(f"  WARN regresión SKG omitido: {e}")

    X, y = make_zz_phase_anomaly(n_samples=400, n_total=160, random_state=42)
    jobs.append(Job("anomalias", "zz_phase", [(X, y)], True))
    try:
        X0, _, _ = load_skg_incidents()
        X0 = subsample_rows(X0, 160, random_state=42)
        X, y = inject_feature_outliers(X0, contamination=0.10, random_state=43)
        jobs.append(Job("anomalias", "SKG", [(X, y)]))
    except Exception as e:  # noqa: BLE001
        print(f"  WARN anomalías SKG omitido: {e}")
    X, y = make_moons_anomaly(n_total=160, random_state=42)
    jobs.append(Job("anomalias", "moons", [(X, y)]))
    return jobs


def add_q_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Añade d̂_F, ĝ y el compuesto Q con g como cuarto término (informativo)."""
    df = df.copy()
    df["d_F_hat"] = df["d_F"] / df["d_F"].max()
    df["g_hat"] = df["g"] / df["g"].max()
    df["Q_comp"] = sum(w * df[c] for c, w in Q_WEIGHTS.items())
    return df


def measure_all(include_optimization: bool = True, verbose: bool = True) -> pd.DataFrame:
    jobs = supervised_jobs()
    if include_optimization:
        jobs += optimization_jobs()
    rows = []
    for job in jobs:
        try:
            rows.append(measure_job(job, verbose=verbose))
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR [{job.family}] {job.name}: {e}")
    return pd.DataFrame(rows)
