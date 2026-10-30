"""Quantumness score Q(D) — robust ZZ-structure detection (QML v2).

Hardening vs TFM notebook 08 / first qml_v2 pass
-------------------------------------------------
1. **Centered kernel alignment** (Cristianini / Cortes).
2. **Classical baseline = Frobenius-best RBF** (approximates K_Q; no label peeking).
   Label-tuned classical is reported only as a diagnostic.
3. **Gate geometry by alignment**: d_F only boosts Q when ΔA>0.
4. **Scale-free Frobenius** on Frobenius-normalized kernels.
5. **Entropy is diagnostic only** (not in the primary score).
6. **Multi-subsample CI** + optional **advantage guardrail** for the verdict.

Primary robust score::

    Q_rob = ΔA_c + λ · σ(κ·ΔA_c) · d̂_F
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
    n_repeats: int = 5
    # Robust score: Q_rob = ΔA_c + λ * gate(ΔA_c) * d̂_F
    lambda_geom: float = 0.35
    gate_kappa: float = 20.0
    d_f_ref: float = 1.0
    svm_C: float = 1.0
    # Legacy TFM weights kept for reporting only
    w_delta_a: float = 0.4
    w_d_f: float = 0.3
    w_delta_h: float = 0.3
    # CI / decision
    ci_low: float = 0.10
    ci_high: float = 0.90
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


def center_kernel(K: np.ndarray) -> np.ndarray:
    """HKC centering: H K H with H = I − 11ᵀ/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def frobenius_normalize(K: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(K, "fro"))
    return K / nrm if nrm > 1e-12 else K


def kernel_alignment(K: np.ndarray, y: np.ndarray, *, centered: bool = True) -> float:
    """A(K, yyᵀ). Default: centered Gram (robust alignment)."""
    y = labels_pm1(y)
    Ky = np.outer(y, y)
    K_use = center_kernel(K) if centered else K
    Ky_use = center_kernel(Ky) if centered else Ky
    num = float(np.sum(K_use * Ky_use))
    den = float(np.linalg.norm(K_use, "fro") * np.linalg.norm(Ky_use, "fro"))
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


def _rbf_from_dists(dists_sq: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-dists_sq / (2.0 * sigma**2))


def best_rbf_kernel(X: np.ndarray, K_Q: np.ndarray) -> tuple[np.ndarray, float]:
    """RBF σ* minimizing Frobenius distance to K_Q (scale-free)."""
    dists_sq = cdist(X, X, "sqeuclidean")
    K_Qn = frobenius_normalize(K_Q)

    def objective(log_sigma: float) -> float:
        sigma = float(np.exp(log_sigma))
        K = frobenius_normalize(_rbf_from_dists(dists_sq, sigma))
        return float(np.linalg.norm(K_Qn - K, "fro"))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    sigma = float(np.exp(result.x))
    K_rbf = _rbf_from_dists(dists_sq, sigma)
    return K_rbf, sigma


def best_classical_for_labels(
    X: np.ndarray, y: np.ndarray, *, centered: bool = True
) -> tuple[np.ndarray, float, str]:
    """Classical kernel with highest (centered) alignment to y.

    Searches RBF σ on a log-grid plus a linear kernel.
    """
    y = labels_pm1(y)
    dists_sq = cdist(X, X, "sqeuclidean")
    best_A = -np.inf
    best_K = None
    best_name = "rbf"
    best_sigma = 1.0

    for log_s in np.linspace(-3.0, 3.0, 13):
        sigma = float(np.exp(log_s))
        K = _rbf_from_dists(dists_sq, sigma)
        A = kernel_alignment(K, y, centered=centered)
        if A > best_A:
            best_A, best_K, best_sigma, best_name = A, K, sigma, f"rbfσ={sigma:.3f}"

    # Linear kernel on features (already in ZZ-ready space)
    K_lin = X @ X.T
    A_lin = kernel_alignment(K_lin, y, centered=centered)
    if A_lin > best_A:
        best_A, best_K, best_name = A_lin, K_lin, "linear"

    assert best_K is not None
    return best_K, float(best_sigma if best_name.startswith("rbf") else best_A), best_name


def frobenius_distance(K_Q: np.ndarray, K_cl: np.ndarray, *, normalize: bool = True) -> float:
    if normalize:
        K_Q = frobenius_normalize(K_Q)
        K_cl = frobenius_normalize(K_cl)
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


def geometric_difference_huang(
    K_Q: np.ndarray, K_cl: np.ndarray, lam: float = 1e-3
) -> float:
    """Diferencia geométrica g_CQ de Huang et al. (2021), con norma espectral.

    g = sqrt( || sqrt(K_Q) (K_cl + lam I)^{-1} sqrt(K_Q) ||_2 ), con ambos kernels
    normalizados a Tr(K) = n. A diferencia de ``geometric_difference`` (norma de
    Frobenius, eps=1e-6), la regularización lam acota la inversa y evita que g
    refleje solo el condicionamiento del RBF.
    """
    n = K_Q.shape[0]
    K_Q = K_Q * (n / max(float(np.trace(K_Q)), 1e-12))
    K_cl = K_cl * (n / max(float(np.trace(K_cl)), 1e-12))
    w, v = np.linalg.eigh((K_Q + K_Q.T) / 2.0)
    K_Q_sqrt = (v * np.sqrt(np.maximum(w, 0.0))) @ v.T
    inv = np.linalg.inv((K_cl + K_cl.T) / 2.0 + lam * np.eye(n))
    M = K_Q_sqrt @ inv @ K_Q_sqrt
    return float(np.sqrt(np.linalg.norm((M + M.T) / 2.0, 2)))


def logistic_gate(x: float, kappa: float = 20.0) -> float:
    """σ(κx) ∈ (0,1); ≈0 for x≪0, ≈1 for x≫0."""
    z = float(np.clip(kappa * x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def quantumness_score_robust(
    delta_A: float,
    d_F: float,
    *,
    lambda_geom: float = 0.35,
    gate_kappa: float = 20.0,
    d_F_ref: float = 1.0,
) -> dict[str, float]:
    """Alignment-gated score: geometry cannot flip a negative ΔA to try-ZZ."""
    d_F_norm = d_F / max(d_F_ref, 1e-12)
    gate = logistic_gate(delta_A, gate_kappa)
    Q_rob = float(delta_A + lambda_geom * gate * d_F_norm)
    # Diagnostics: ungated legacy-style mix (not used for verdict)
    Q_ungated = float(0.6 * delta_A + 0.4 * d_F_norm)
    return {
        "Q_rob": Q_rob,
        "Q_ungated": Q_ungated,
        "delta_A": float(delta_A),
        "d_F_norm": float(d_F_norm),
        "geom_gate": float(gate),
        "geom_term": float(lambda_geom * gate * d_F_norm),
    }


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
    """Legacy TFM composite (reported alongside Q_rob; not used for verdict)."""
    dA = A_Q - A_cl
    d_F_norm = d_F / max(d_F_ref, 1e-12)
    dH = H_Q - H_cl
    dH_norm = dH / np.log(max(n, 2)) if n is not None else dH
    Q = float(w[0] * dA + w[1] * d_F_norm + w[2] * dH_norm)
    Q_ah = float(0.5 * dA + 0.5 * d_F_norm)
    return {
        "Q_score": Q,
        "Q_AH": Q_ah,
        "delta_A": dA,
        "d_F_norm": d_F_norm,
        "delta_H": dH,
        "delta_H_norm": float(dH_norm),
    }


def _indicators_on_matrix(
    Xq: np.ndarray,
    y_s: np.ndarray,
    cfg: QuantumnessConfig,
    *,
    seed: int,
    compute_advantage: bool,
) -> dict[str, Any]:
    """Indicators on one ZZ-ready subsample.

    Classical baseline for ΔA is the **Frobenius-best RBF** (approximates K_Q
    without peeking at labels). A separate label-tuned classical is reported
    only as a diagnostic.
    """
    y_pm = labels_pm1(y_s)
    K_Q = zz_fidelity_kernel(
        Xq, cfg.n_qubits, cfg.zz_reps, cfg.entanglement, seed
    )
    K_rbf, sigma_frob = best_rbf_kernel(Xq, K_Q)
    _, _, cl_label_name = best_classical_for_labels(Xq, y_s, centered=True)

    A_Q = kernel_alignment(K_Q, y_s, centered=True)
    A_cl = kernel_alignment(K_rbf, y_s, centered=True)
    delta_A = A_Q - A_cl

    A_Q_raw = kernel_alignment(K_Q, y_s, centered=False)
    A_cl_raw = kernel_alignment(K_rbf, y_s, centered=False)

    d_F = frobenius_distance(K_Q, K_rbf, normalize=True)
    g = geometric_difference(K_Q, K_rbf)
    g_huang = geometric_difference_huang(K_Q, K_rbf)
    H_Q = spectral_entropy(K_Q)
    H_cl = spectral_entropy(K_rbf)

    rob = quantumness_score_robust(
        delta_A,
        d_F,
        lambda_geom=cfg.lambda_geom,
        gate_kappa=cfg.gate_kappa,
        d_F_ref=cfg.d_f_ref,
    )
    legacy = quantumness_score(
        A_Q_raw,
        A_cl_raw,
        d_F,
        H_Q,
        H_cl,
        d_F_ref=cfg.d_f_ref,
        n=len(y_s),
        w=(cfg.w_delta_a, cfg.w_d_f, cfg.w_delta_h),
    )

    out: dict[str, Any] = {
        "n": int(len(y_s)),
        "sigma_rbf_frob": float(sigma_frob),
        "classical_best": f"rbfσ={sigma_frob:.3f}",
        "classical_label_tuned": cl_label_name,
        "A_Q": A_Q,
        "A_cl": A_cl,
        "delta_A": delta_A,
        "A_Q_raw": A_Q_raw,
        "A_cl_raw": A_cl_raw,
        "H_Q": H_Q,
        "H_cl": H_cl,
        "delta_H": H_Q - H_cl,
        "delta_H_norm": legacy["delta_H_norm"],
        "d_F": d_F,
        "d_F_norm": rob["d_F_norm"],
        "g": g,
        "g_huang": g_huang,
        "geom_gate": rob["geom_gate"],
        "geom_term": rob["geom_term"],
        "Q_rob": rob["Q_rob"],
        "Q_ungated": rob["Q_ungated"],
        "Q_score": legacy["Q_score"],
        "Q_AH": legacy["Q_AH"],
    }

    if compute_advantage and len(np.unique(y_s)) > 1:
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xq,
            y_pm,
            test_size=cfg.test_size,
            random_state=seed,
            stratify=y_s,
        )
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        algorithm_globals.random_seed = seed
        fmap = zz_map(cfg.n_qubits, cfg.zz_reps, cfg.entanglement)
        qk = FidelityQuantumKernel(feature_map=fmap)
        Kq_tr = qk.evaluate(X_tr)
        Kq_te = qk.evaluate(X_te, X_tr)
        gamma = 1.0 / (2.0 * max(sigma_frob, 1e-6) ** 2)
        Kr_tr = rbf_kernel(X_tr, gamma=gamma)
        Kr_te = rbf_kernel(X_te, X_tr, gamma=gamma)
        acc_q = float(
            SVC(kernel="precomputed", C=cfg.svm_C).fit(Kq_tr, y_tr).score(Kq_te, y_te)
        )
        acc_cl = float(
            SVC(kernel="precomputed", C=cfg.svm_C).fit(Kr_tr, y_tr).score(Kr_te, y_te)
        )
        out["acc_QSVM"] = acc_q
        out["acc_SVM_RBF"] = acc_cl
        out["advantage_pp"] = (acc_q - acc_cl) * 100.0
    return out


def decide_verdict(
    Q_rob: float,
    delta_A: float,
    Q_lo: float | None = None,
    advantage_pp: float | None = None,
) -> str:
    """CI + alignment + optional empirical advantage guardrail."""
    adv_ok = advantage_pp is None or advantage_pp >= -2.0  # allow tiny noise
    adv_strong_neg = advantage_pp is not None and advantage_pp < -5.0

    if Q_lo is not None:
        if Q_lo > 0 and delta_A > 0 and adv_ok:
            return "try-ZZ"
        if Q_rob > 0 and delta_A > 0 and (Q_lo <= 0 or adv_strong_neg):
            return "ZZ-unstable"
        if Q_rob > 0 and delta_A <= 0:
            return "ZZ-geometry-only"
        return "classical"

    if Q_rob > 0 and delta_A > 0 and adv_ok:
        return "try-ZZ"
    if Q_rob > 0 and delta_A > 0:
        return "ZZ-unstable"
    if Q_rob > 0:
        return "ZZ-geometry-only"
    return "classical"


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


def _encode_subsample(
    X: np.ndarray,
    y: np.ndarray,
    cfg: QuantumnessConfig,
    already_zz_space: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    X_s, y_s = subsample_stratified(X, y, cfg.n_sample, seed)
    Xq, _, pca_var = encode_for_zz_kernel(
        X_s,
        X_s,
        n_qubits=cfg.n_qubits,
        random_state=seed,
        already_zz_space=already_zz_space,
    )
    return Xq, y_s, (1.0 if already_zz_space else float(pca_var))



def compute_indicators(
    X: np.ndarray,
    y: np.ndarray,
    cfg: QuantumnessConfig,
    *,
    already_zz_space: bool,
    dataset: str,
    compute_advantage: bool = True,
) -> dict[str, Any]:
    """Multi-repeat robust quantumness for one dataset."""
    repeats: list[dict[str, Any]] = []
    pca_vars: list[float] = []
    for r in range(cfg.n_repeats):
        seed = cfg.random_state + 17 * r
        Xq, y_s, pca_var = _encode_subsample(X, y, cfg, already_zz_space, seed)
        pca_vars.append(pca_var)
        row = _indicators_on_matrix(
            Xq, y_s, cfg, seed=seed, compute_advantage=compute_advantage
        )
        row["repeat"] = r
        repeats.append(row)

    df = pd.DataFrame(repeats)
    Q_vals = df["Q_rob"].to_numpy(dtype=float)
    dA_vals = df["delta_A"].to_numpy(dtype=float)
    Q_lo = float(np.quantile(Q_vals, cfg.ci_low))
    Q_hi = float(np.quantile(Q_vals, cfg.ci_high))
    Q_mean = float(Q_vals.mean())
    dA_mean = float(dA_vals.mean())
    adv_mean = float(df["advantage_pp"].mean()) if "advantage_pp" in df.columns else None
    verdict = decide_verdict(Q_mean, dA_mean, Q_lo=Q_lo, advantage_pp=adv_mean)

    out: dict[str, Any] = {
        "dataset": dataset,
        "n": int(df["n"].iloc[0]),
        "n_qubits": cfg.n_qubits,
        "n_repeats": cfg.n_repeats,
        "pca_var": float(np.mean(pca_vars)),
        "classical_best": str(df["classical_best"].mode().iloc[0]),
        "A_Q": float(df["A_Q"].mean()),
        "A_cl": float(df["A_cl"].mean()),
        "delta_A": dA_mean,
        "delta_A_std": float(dA_vals.std(ddof=0)),
        "d_F": float(df["d_F"].mean()),
        "d_F_norm": float(df["d_F_norm"].mean()),
        "g": float(df["g"].mean()),
        "geom_gate": float(df["geom_gate"].mean()),
        "geom_term": float(df["geom_term"].mean()),
        "delta_H_norm": float(df["delta_H_norm"].mean()),
        "Q_rob": Q_mean,
        "Q_rob_std": float(Q_vals.std(ddof=0)),
        "Q_rob_lo": Q_lo,
        "Q_rob_hi": Q_hi,
        "Q_ungated": float(df["Q_ungated"].mean()),
        "Q_score": float(df["Q_score"].mean()),
        "Q_AH": float(df["Q_AH"].mean()),
        "verdict": verdict,
    }
    if "advantage_pp" in df.columns:
        out["acc_QSVM"] = float(df["acc_QSVM"].mean())
        out["acc_SVM_RBF"] = float(df["acc_SVM_RBF"].mean())
        out["advantage_pp"] = float(df["advantage_pp"].mean())
        out["advantage_pp_std"] = float(df["advantage_pp"].std(ddof=0))
    out["_repeats"] = repeats
    return out


# ---------------------------------------------------------------------------
# Alpha sweep
# ---------------------------------------------------------------------------

def make_alpha_dataset(
    alpha: float,
    n: int,
    n_qubits: int,
    reps: int,
    entanglement: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        local = QuantumnessConfig(
            **{
                **asdict(cfg),
                "n_sample": min(cfg.n_sample, cfg.alpha_n),
            }
        )
        r = compute_indicators(
            X, y, local, already_zz_space=True, dataset=f"alpha_{alpha:.2f}"
        )
        r.pop("_repeats", None)
        r["alpha"] = float(alpha)
        rows.append(r)
        print(
            f"α={alpha:.2f}: Q_rob={r['Q_rob']:+.3f} "
            f"[{r['Q_rob_lo']:+.3f},{r['Q_rob_hi']:+.3f}]  "
            f"ΔA={r['delta_A']:+.3f}  gate={r['geom_gate']:.2f}  "
            f"adv={r.get('advantage_pp', float('nan')):+.1f}pp  → {r['verdict']}"
        )
    return pd.DataFrame(rows)


def spearman_alpha_vs_Q(df: pd.DataFrame, col: str = "Q_rob") -> float:
    if "alpha" not in df.columns or col not in df.columns or len(df) < 3:
        return float("nan")
    from scipy.stats import spearmanr

    rho, _ = spearmanr(df["alpha"], df[col])
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
            zz_reps=cfg.zz_reps,  # keep reps matched to label generator
            n_sample=48,
            n_repeats=2,
            alpha_n=48,
            alpha_grid=(0.0, 0.5, 1.0),
            random_state=cfg.random_state,
            lambda_geom=cfg.lambda_geom,
            gate_kappa=cfg.gate_kappa,
        )

    metrics = repo_paths()["metrics"]
    metrics.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    real_rows: list[dict[str, Any]] = []

    report = [
        "# Quantumness Q(D) results (QML v2 — robust)",
        "",
        "Primary score: `Q_rob = ΔA_c + λ · gate(ΔA_c) · d̂_F`",
        "(centered alignment vs best classical; geometry gated by ΔA).",
        "Verdict uses CI lower bound over stratified repeats.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
    ]

    jobs: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    if "quantum_native" in datasets:
        X, y = make_quantum_native(
            n_samples=max(cfg.n_sample * 2, 120),
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
        rep = r.pop("_repeats", None)
        real_rows.append(r)
        if rep is not None:
            pd.DataFrame(rep).to_csv(
                metrics / f"qml_v2_quantumness_{name}_repeats.csv", index=False
            )
        print(
            f"Q_rob={r['Q_rob']:+.3f}±{r['Q_rob_std']:.3f} "
            f"[{r['Q_rob_lo']:+.3f},{r['Q_rob_hi']:+.3f}]  "
            f"ΔA={r['delta_A']:+.3f}  gate={r['geom_gate']:.2f}  "
            f"adv={r.get('advantage_pp', float('nan')):+.1f}pp  → {r['verdict']}"
        )
        report += [
            f"## {name}",
            "",
            "```",
            f"Q_rob={r['Q_rob']:+.4f}±{r['Q_rob_std']:.4f}  "
            f"CI[{r['Q_rob_lo']:+.4f},{r['Q_rob_hi']:+.4f}]  → {r['verdict']}",
            f"ΔA={r['delta_A']:+.4f}±{r['delta_A_std']:.4f}  "
            f"d_F={r['d_F']:.4f}  gate={r['geom_gate']:.3f}  "
            f"geom_term={r['geom_term']:+.4f}  g={r['g']:.4f}",
            f"classical_best={r['classical_best']}  "
            f"acc_QSVM={r.get('acc_QSVM', float('nan')):.3f}  "
            f"acc_RBF={r.get('acc_SVM_RBF', float('nan')):.3f}  "
            f"advantage_pp={r.get('advantage_pp', float('nan')):+.2f}"
            f"±{r.get('advantage_pp_std', 0.0):.2f}",
            f"(legacy Q_TFM={r['Q_score']:+.4f}, Q_ungated={r['Q_ungated']:+.4f})",
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
        rho = spearman_alpha_vs_Q(df_a, "Q_rob")
        out["alpha_sweep"] = df_a
        df_a.to_csv(metrics / "qml_v2_quantumness_alpha.csv", index=False)
        report += [
            "## alpha sweep",
            "",
            f"Spearman(α, Q_rob) = `{rho:.3f}`",
            "",
            "```",
            df_a[
                [
                    "alpha",
                    "Q_rob",
                    "Q_rob_lo",
                    "Q_rob_hi",
                    "delta_A",
                    "geom_gate",
                    "advantage_pp",
                    "verdict",
                ]
            ].to_string(index=False),
            "```",
            "",
        ]
        print(f"Spearman(α, Q_rob) = {rho:.3f}")

    (metrics / "qml_v2_quantumness_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    path = REPO_ROOT / "BUSINESS" / "quantumness_v2_results.md"
    path.write_text("\n".join(report) + "\n")
    print(f"\nWrote {path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="QML v2 robust quantumness Q_rob / ZZ structure")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-sample", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
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
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    if args.n_qubits is not None:
        cfg.n_qubits = args.n_qubits
    if args.zz_reps is not None:
        cfg.zz_reps = args.zz_reps
    names = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    run_all(cfg=cfg, datasets=names, quick=args.quick)


if __name__ == "__main__":
    main()
