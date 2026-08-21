"""Quantumness score Q(D) — detect ZZ / quantum-kernel structure (QML v2).

Ports TFM notebook 08 with the same fair preprocessing as classification:

* Train-style encode via ``encode_for_zz_kernel`` (PCA + [-π, π] only when needed).
* Quantum-native points are **not** re-scaled.
* Indicators compare ZZ fidelity kernel vs best-approximating RBF on the
  **same** ZZ-ready matrix.

Score (TFM proposal)::

    Q(D) = w1 * ΔA + w2 * d̂_F + w3 * ΔH

    ΔA  = A(K_Q, yyᵀ) − A(K_RBF, yyᵀ)
    d̂_F = d_F / d_F_ref
    ΔH  = H(K_Q) − H(K_RBF)

Q > 0 → ZZ kernel looks structurally preferred; Q < 0 → classical kernel enough.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.svm import SVC

from qml_v2.classification import (
    REPO_ROOT,
    encode_for_zz_kernel,
    load_skg_incidents,
    load_telco,
    make_quantum_native,
    repo_paths,
    subsample_stratified,
    zz_map,
)


@dataclass
class QuantumnessConfig:
    n_qubits: int = 2
    zz_reps: int = 2
    entanglement: str = "full"
    n_sample: int = 80
    test_size: float = 0.3
    random_state: int = 42
    # Q(D) weights (TFM default)
    w_delta_a: float = 0.4
    w_d_f: float = 0.3
    w_delta_h: float = 0.3
    d_f_ref: float = 1.0
    svm_C: float = 1.0
    # alpha sweep
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    alpha_n: int = 80


# ---------------------------------------------------------------------------
# Indicators (no Qiskit)
# ---------------------------------------------------------------------------

def labels_pm1(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if set(np.unique(y).tolist()) <= {0, 1}:
        return np.where(y == 0, -1.0, 1.0)
    return y.astype(float)


def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """A(K, yyᵀ) with y ∈ {-1, +1}."""
    y = labels_pm1(y)
    Ky = np.outer(y, y)
    num = float(np.sum(K * Ky))
    den = float(np.linalg.norm(K, "fro") * np.linalg.norm(Ky, "fro"))
    return num / den if den > 1e-12 else 0.0


def spectral_entropy(K: np.ndarray) -> float:
    ev = np.real(np.linalg.eigvalsh(K))
    ev = np.maximum(ev, 0.0)
    s = float(ev.sum())
    if s < 1e-12:
        return 0.0
    lam = ev / s
    lam = lam[lam > 1e-15]
    return float(-np.sum(lam * np.log(lam)))


def best_rbf_kernel(X: np.ndarray, K_Q: np.ndarray) -> tuple[np.ndarray, float]:
    """RBF σ* that best approximates K_Q in Frobenius norm (on the same X)."""
    dists_sq = cdist(X, X, "sqeuclidean")

    def objective(log_sigma: float) -> float:
        sigma = float(np.exp(log_sigma))
        K = np.exp(-dists_sq / (2.0 * sigma**2))
        return float(np.linalg.norm(K_Q - K, "fro"))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    sigma = float(np.exp(result.x))
    K_rbf = np.exp(-dists_sq / (2.0 * sigma**2))
    return K_rbf, sigma


def frobenius_distance(K_Q: np.ndarray, K_cl: np.ndarray) -> float:
    n = K_Q.shape[0]
    return float(np.linalg.norm(K_Q - K_cl, "fro") / n)


def geometric_difference(K_Q: np.ndarray, K_cl: np.ndarray, eps: float = 1e-6) -> float:
    n = K_Q.shape[0]
    K_cl_reg = K_cl + eps * np.eye(n)
    K_Q_reg = K_Q + eps * np.eye(n)
    eigvals, eigvecs = np.linalg.eigh(K_Q_reg)
    eigvals = np.maximum(eigvals, 0.0)
    K_Q_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    K_cl_inv = np.linalg.inv(K_cl_reg)
    M = K_Q_sqrt @ K_cl_inv @ K_Q_sqrt
    return float(np.sqrt(np.linalg.norm(M, "fro")))


def quantumness_score(
    A_Q: float,
    A_cl: float,
    d_F: float,
    H_Q: float,
    H_cl: float,
    *,
    d_F_ref: float = 1.0,
    n: int | None = None,
    w: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> dict[str, float]:
    """Return Q and components. ΔH is scaled by log(n) when n is given."""
    dA = A_Q - A_cl
    d_F_norm = d_F / max(d_F_ref, 1e-12)
    dH = H_Q - H_cl
    dH_norm = dH / np.log(max(n, 2)) if n is not None else dH
    Q = float(w[0] * dA + w[1] * d_F_norm + w[2] * dH_norm)
    # Secondary score without entropy (ΔH is the noisiest term in the TFM)
    w_ah = (0.5, 0.5)
    Q_ah = float(w_ah[0] * dA + w_ah[1] * d_F_norm)
    return {
        "Q_score": Q,
        "Q_AH": Q_ah,
        "delta_A": dA,
        "d_F_norm": d_F_norm,
        "delta_H": dH,
        "delta_H_norm": float(dH_norm),
    }


# ---------------------------------------------------------------------------
# Quantum kernel
# ---------------------------------------------------------------------------

def zz_fidelity_kernel(
    X: np.ndarray, n_qubits: int, reps: int, entanglement: str, seed: int
) -> np.ndarray:
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    algorithm_globals.random_seed = seed
    fmap = zz_map(n_qubits, reps, entanglement)
    return np.asarray(FidelityQuantumKernel(feature_map=fmap).evaluate(X), dtype=float)


def prepare_zz_matrix(
    X: np.ndarray,
    y: np.ndarray,
    cfg: QuantumnessConfig,
    already_zz_space: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Subsample + encode to ZZ-ready angles; return (Xq, y, pca_var)."""
    X, y = subsample_stratified(X, y, cfg.n_sample, cfg.random_state)
    Xq, _, pca_var = encode_for_zz_kernel(
        X,
        X,
        n_qubits=cfg.n_qubits,
        random_state=cfg.random_state,
        already_zz_space=already_zz_space,
    )
    return Xq, y, (1.0 if already_zz_space else pca_var)


def compute_indicators(
    X: np.ndarray,
    y: np.ndarray,
    cfg: QuantumnessConfig,
    *,
    already_zz_space: bool,
    dataset: str,
    compute_advantage: bool = True,
) -> dict[str, Any]:
    Xq, y_s, pca_var = prepare_zz_matrix(X, y, cfg, already_zz_space)
    K_Q = zz_fidelity_kernel(
        Xq, cfg.n_qubits, cfg.zz_reps, cfg.entanglement, cfg.random_state
    )
    K_rbf, sigma = best_rbf_kernel(Xq, K_Q)

    A_Q = kernel_alignment(K_Q, y_s)
    A_cl = kernel_alignment(K_rbf, y_s)
    H_Q = spectral_entropy(K_Q)
    H_cl = spectral_entropy(K_rbf)
    d_F = frobenius_distance(K_Q, K_rbf)
    g = geometric_difference(K_Q, K_rbf)
    w = (cfg.w_delta_a, cfg.w_d_f, cfg.w_delta_h)
    scored = quantumness_score(
        A_Q, A_cl, d_F, H_Q, H_cl, d_F_ref=cfg.d_f_ref, n=len(y_s), w=w
    )
    Q = scored["Q_score"]

    # Decision rule: Q>0 alone can be driven by d_F (kernels differ) without
    # label alignment. Prefer ZZ only when ΔA>0 as well; else flag geometry-only.
    if Q > 0 and scored["delta_A"] > 0:
        verdict = "try-ZZ"
    elif Q > 0:
        verdict = "ZZ-geometry-only"
    else:
        verdict = "classical"
    if scored["Q_AH"] > 0 and scored["delta_A"] > 0:
        verdict_ah = "try-ZZ"
    elif scored["Q_AH"] > 0:
        verdict_ah = "ZZ-geometry-only"
    else:
        verdict_ah = "classical"

    out: dict[str, Any] = {
        "dataset": dataset,
        "n": int(len(y_s)),
        "n_qubits": cfg.n_qubits,
        "pca_var": float(pca_var),
        "sigma_rbf": float(sigma),
        "A_Q": A_Q,
        "A_cl": A_cl,
        "delta_A": scored["delta_A"],
        "H_Q": H_Q,
        "H_cl": H_cl,
        "delta_H": scored["delta_H"],
        "delta_H_norm": scored["delta_H_norm"],
        "d_F": d_F,
        "d_F_norm": scored["d_F_norm"],
        "g": g,
        "Q_score": Q,
        "Q_AH": scored["Q_AH"],
        "verdict": verdict,
        "verdict_AH": verdict_ah,
    }

    if compute_advantage and len(np.unique(y_s)) > 1:
        y_pm = labels_pm1(y_s)
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xq,
            y_pm,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y_s,
        )
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        algorithm_globals.random_seed = cfg.random_state
        fmap = zz_map(cfg.n_qubits, cfg.zz_reps, cfg.entanglement)
        qk = FidelityQuantumKernel(feature_map=fmap)
        Kq_tr = qk.evaluate(X_tr)
        Kq_te = qk.evaluate(X_te, X_tr)
        gamma = 1.0 / (2.0 * sigma**2)
        Kr_tr = rbf_kernel(X_tr, gamma=gamma)
        Kr_te = rbf_kernel(X_te, X_tr, gamma=gamma)
        acc_q = float(SVC(kernel="precomputed", C=cfg.svm_C).fit(Kq_tr, y_tr).score(Kq_te, y_te))
        acc_cl = float(SVC(kernel="precomputed", C=cfg.svm_C).fit(Kr_tr, y_tr).score(Kr_te, y_te))
        out["acc_QSVM"] = acc_q
        out["acc_SVM_RBF"] = acc_cl
        out["advantage_pp"] = (acc_q - acc_cl) * 100.0
    return out


# ---------------------------------------------------------------------------
# Alpha sweep (controlled ZZ structure)
# ---------------------------------------------------------------------------

def make_alpha_dataset(
    alpha: float,
    n: int,
    n_qubits: int,
    reps: int,
    entanglement: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix classical-linear labels with ZZ-native labels.

    α=0 → y = 1{x₀ > 0}; α=1 → y = sign(<Z…Z>); α∈(0,1) stochastic mix.
    """
    X, y_q = make_quantum_native(
        n_samples=n,
        n_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
        random_state=seed,
    )
    y_c = (X[:, 0] > 0).astype(int)
    rng = np.random.RandomState(seed + 7)
    use_q = rng.rand(n) < float(alpha)
    y = np.where(use_q, y_q, y_c).astype(int)
    if len(np.unique(y)) < 2:
        y[: n // 2] = 0
        y[n // 2 :] = 1
    return X, y


def run_alpha_sweep(cfg: QuantumnessConfig) -> pd.DataFrame:
    rows = []
    for alpha in cfg.alpha_grid:
        X, y = make_alpha_dataset(
            alpha,
            n=cfg.alpha_n,
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            seed=cfg.random_state,
        )
        # Already in ZZ angle space
        local = QuantumnessConfig(**{**asdict(cfg), "n_sample": min(cfg.n_sample, cfg.alpha_n)})
        r = compute_indicators(
            X, y, local, already_zz_space=True, dataset=f"alpha_{alpha:.2f}"
        )
        r["alpha"] = float(alpha)
        rows.append(r)
        print(
            f"α={alpha:.2f}: Q={r['Q_score']:+.3f}  Q_AH={r['Q_AH']:+.3f}  "
            f"ΔA={r['delta_A']:+.3f}  d_F={r['d_F']:.3f}  "
            f"adv={r.get('advantage_pp', float('nan')):+.1f}pp"
        )
    return pd.DataFrame(rows)


def spearman_alpha_vs_Q(df: pd.DataFrame) -> float:
    if "alpha" not in df.columns or len(df) < 3:
        return float("nan")
    from scipy.stats import spearmanr

    rho, _ = spearmanr(df["alpha"], df["Q_score"])
    return float(rho)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(
    cfg: QuantumnessConfig | None = None,
    datasets: tuple[str, ...] = ("quantum_native", "telco", "skg", "alpha_sweep"),
    quick: bool = False,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or QuantumnessConfig()
    if quick:
        cfg = QuantumnessConfig(
            n_qubits=cfg.n_qubits,
            zz_reps=1,
            n_sample=40,
            alpha_n=40,
            alpha_grid=(0.0, 0.5, 1.0),
            random_state=cfg.random_state,
            w_delta_a=cfg.w_delta_a,
            w_d_f=cfg.w_d_f,
            w_delta_h=cfg.w_delta_h,
        )

    metrics = repo_paths()["metrics"]
    metrics.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    real_rows: list[dict[str, Any]] = []

    report = [
        "# Quantumness Q(D) results (QML v2)",
        "",
        "ZZ fidelity kernel vs best RBF approximant on the same ZZ-ready features.",
        "Q = w1·ΔA + w2·d̂_F + w3·ΔH.  Q>0 → ZZ-preferred.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
    ]

    jobs: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    if "quantum_native" in datasets:
        X, y = make_quantum_native(
            n_samples=max(cfg.n_sample, 120),
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            random_state=cfg.random_state,
        )
        jobs.append(("quantum_native", X, y, True))
    if "telco" in datasets:
        X, y, _ = load_telco()
        jobs.append(("telco", X, y, False))
    if "skg" in datasets:
        X, y, _ = load_skg_incidents()
        jobs.append(("skg", X, y, False))

    for name, X, y, native in jobs:
        print(f"\n=== {name}  n={len(y)}  feats={X.shape[1]}  native={native} ===")
        r = compute_indicators(X, y, cfg, already_zz_space=native, dataset=name)
        real_rows.append(r)
        print(
            f"Q={r['Q_score']:+.3f}/{r['verdict']}  Q_AH={r['Q_AH']:+.3f}/{r['verdict_AH']}  "
            f"ΔA={r['delta_A']:+.3f}  d_F={r['d_F']:.3f}  ΔH_n={r['delta_H_norm']:+.3f}  "
            f"adv={r.get('advantage_pp', float('nan')):+.1f}pp"
        )
        report += [
            f"## {name}",
            "",
            "```",
            f"Q={r['Q_score']:+.4f} ({r['verdict']})  Q_AH={r['Q_AH']:+.4f} ({r['verdict_AH']})",
            f"ΔA={r['delta_A']:+.4f}  d_F={r['d_F']:.4f}  ΔH_norm={r['delta_H_norm']:+.4f}  g={r['g']:.4f}",
            f"acc_QSVM={r.get('acc_QSVM', float('nan')):.3f}  "
            f"acc_RBF={r.get('acc_SVM_RBF', float('nan')):.3f}  "
            f"advantage_pp={r.get('advantage_pp', float('nan')):+.2f}",
            "```",
            "",
        ]

    if real_rows:
        df_real = pd.DataFrame(real_rows)
        out["datasets"] = df_real
        df_real.to_csv(metrics / "qml_v2_quantumness_datasets.csv", index=False)

    if "alpha_sweep" in datasets:
        print("\n=== alpha sweep ===")
        df_a = run_alpha_sweep(cfg)
        rho = spearman_alpha_vs_Q(df_a)
        out["alpha_sweep"] = df_a
        df_a.to_csv(metrics / "qml_v2_quantumness_alpha.csv", index=False)
        report += [
            "## alpha sweep",
            "",
            f"Spearman(α, Q) = `{rho:.3f}`",
            "",
            "```",
            df_a[
                ["alpha", "Q_score", "Q_AH", "delta_A", "d_F", "delta_H_norm", "advantage_pp"]
            ].to_string(index=False),
            "```",
            "",
        ]
        print(f"Spearman(α, Q) = {rho:.3f}")

    (metrics / "qml_v2_quantumness_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    path = REPO_ROOT / "BUSINESS" / "quantumness_v2_results.md"
    path.write_text("\n".join(report) + "\n")
    print(f"\nWrote {path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="QML v2 quantumness Q(D) / ZZ structure")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-sample", type=int, default=None)
    p.add_argument("--n-qubits", type=int, default=None)
    p.add_argument("--zz-reps", type=int, default=None)
    p.add_argument(
        "--datasets",
        default="quantum_native,telco,skg,alpha_sweep",
        help="quantum_native,telco,skg,alpha_sweep",
    )
    args = p.parse_args()
    cfg = QuantumnessConfig()
    if args.n_sample is not None:
        cfg.n_sample = args.n_sample
    if args.n_qubits is not None:
        cfg.n_qubits = args.n_qubits
    if args.zz_reps is not None:
        cfg.zz_reps = args.zz_reps
    names = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    run_all(cfg=cfg, datasets=names, quick=args.quick)


if __name__ == "__main__":
    main()
