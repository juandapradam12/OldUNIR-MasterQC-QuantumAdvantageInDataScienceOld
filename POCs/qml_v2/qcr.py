"""QCR — Quantum-Compatible Representation pre-maps (QML v2).

Ports TFM notebook 09 onto the **robust** quantumness score ``Q_rob``.

Goal: when baseline verdict is ``classical``, try pre-maps ψ that raise
``Q_rob`` / ΔA (and preferably empirical QSVM advantage) before training.

Strategies
----------
1. **ZZ-interaction features** — add z_{jk}=x_j(π−x_k), PCA back to n_qubits.
2. **Orthogonal rotation** — choose θ maximizing Q_rob on a *selection* split.
3. **Re-uploading** — choose ZZFeatureMap ``reps`` maximizing Q_rob.

Framework: diagnose bottleneck → apply recommended ψ → re-score with
``quantumness.compute_indicators`` (multi-repeat CI).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from qml_v2.classification import (
    REPO_ROOT,
    encode_for_zz_kernel,
    load_skg_incidents,
    load_telco,
    make_quantum_native,
    repo_paths,
    subsample_stratified,
)
from qml_v2.quantumness import (
    QuantumnessConfig,
    best_rbf_kernel,
    frobenius_distance,
    kernel_alignment,
    quantumness_score_robust,
    zz_fidelity_kernel,
)


@dataclass
class QCRConfig:
    n_qubits: int = 2
    zz_reps: int = 2
    entanglement: str = "full"
    n_sample: int = 80
    n_repeats: int = 3
    random_state: int = 42
    # rotation sweep
    n_angles: int = 18
    # re-uploading sweep
    reps_grid: tuple[int, ...] = (1, 2, 3, 4)
    # selection subsample (hyperparams); evaluation uses QuantumnessConfig repeats
    select_n: int = 60
    lambda_geom: float = 0.35
    gate_kappa: float = 20.0


def _to_zz(
    X: np.ndarray,
    y: np.ndarray,
    cfg: QCRConfig,
    already_zz_space: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = subsample_stratified(X, y, cfg.n_sample, seed)
    Xq, _, _ = encode_for_zz_kernel(
        Xs, Xs, cfg.n_qubits, seed, already_zz_space=already_zz_space
    )
    return Xq, ys


def quick_Q_rob(
    Xq: np.ndarray,
    y: np.ndarray,
    *,
    n_qubits: int,
    reps: int,
    entanglement: str,
    seed: int,
    lambda_geom: float,
    gate_kappa: float,
) -> dict[str, float]:
    """Fast single-shot Q_rob (no advantage) for hyperparameter selection."""
    K_Q = zz_fidelity_kernel(Xq, n_qubits, reps, entanglement, seed)
    K_rbf, _ = best_rbf_kernel(Xq, K_Q)
    A_Q = kernel_alignment(K_Q, y, centered=True)
    A_cl = kernel_alignment(K_rbf, y, centered=True)
    d_F = frobenius_distance(K_Q, K_rbf, normalize=True)
    delta_A = A_Q - A_cl
    rob = quantumness_score_robust(
        delta_A, d_F, lambda_geom=lambda_geom, gate_kappa=gate_kappa
    )
    return {
        "Q_rob": rob["Q_rob"],
        "delta_A": delta_A,
        "d_F": d_F,
        "geom_gate": rob["geom_gate"],
    }


# ---------------------------------------------------------------------------
# Pre-maps ψ
# ---------------------------------------------------------------------------

def apply_zz_interactions(Xq: np.ndarray, n_qubits: int | None = None) -> np.ndarray:
    """Add ZZ-style interaction features, project back to n_qubits angles."""
    n_qubits = n_qubits or Xq.shape[1]
    cols = [Xq]
    d = Xq.shape[1]
    for j in range(d):
        for k in range(d):
            if j == k:
                continue
            cols.append((Xq[:, j] * (np.pi - Xq[:, k])).reshape(-1, 1))
    X_aug = np.hstack(cols)
    if X_aug.shape[1] > n_qubits:
        X_back = PCA(n_components=n_qubits, random_state=0).fit_transform(X_aug)
    else:
        X_back = X_aug
    return MinMaxScaler(feature_range=(-np.pi, np.pi)).fit_transform(X_back)


def apply_rotation(Xq: np.ndarray, theta: float) -> np.ndarray:
    """2D orthogonal rotation (n_qubits must be 2)."""
    if Xq.shape[1] != 2:
        raise ValueError("Rotation strategy currently supports n_qubits=2 only.")
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    X_rot = Xq @ R.T
    return MinMaxScaler(feature_range=(-np.pi, np.pi)).fit_transform(X_rot)


def select_best_rotation(
    Xq: np.ndarray, y: np.ndarray, cfg: QCRConfig, seed: int
) -> tuple[float, pd.DataFrame]:
    """Pick θ maximizing Q_rob on a selection subset (not the eval repeats)."""
    if len(y) > cfg.select_n:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=cfg.select_n, random_state=seed
        )
        idx, _ = next(sss.split(Xq, y))
        X_sel, y_sel = Xq[idx], y[idx]
    else:
        X_sel, y_sel = Xq, y

    rows = []
    thetas = np.linspace(0.0, np.pi, cfg.n_angles)
    best_theta, best_Q = 0.0, -np.inf
    for theta in thetas:
        X_rot = apply_rotation(X_sel, float(theta))
        m = quick_Q_rob(
            X_rot,
            y_sel,
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            seed=seed,
            lambda_geom=cfg.lambda_geom,
            gate_kappa=cfg.gate_kappa,
        )
        rows.append({"theta": float(theta), **m})
        if m["Q_rob"] > best_Q:
            best_Q, best_theta = m["Q_rob"], float(theta)
    return best_theta, pd.DataFrame(rows)


def select_best_reps(
    Xq: np.ndarray, y: np.ndarray, cfg: QCRConfig, seed: int
) -> tuple[int, pd.DataFrame]:
    if len(y) > cfg.select_n:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=cfg.select_n, random_state=seed
        )
        idx, _ = next(sss.split(Xq, y))
        X_sel, y_sel = Xq[idx], y[idx]
    else:
        X_sel, y_sel = Xq, y

    rows = []
    best_reps, best_Q = cfg.zz_reps, -np.inf
    for reps in cfg.reps_grid:
        m = quick_Q_rob(
            X_sel,
            y_sel,
            n_qubits=cfg.n_qubits,
            reps=int(reps),
            entanglement=cfg.entanglement,
            seed=seed,
            lambda_geom=cfg.lambda_geom,
            gate_kappa=cfg.gate_kappa,
        )
        rows.append({"reps": int(reps), **m})
        if m["Q_rob"] > best_Q:
            best_Q, best_reps = m["Q_rob"], int(reps)
    return best_reps, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluate with robust quantumness
# ---------------------------------------------------------------------------

def _qn_cfg(cfg: QCRConfig, zz_reps: int | None = None) -> QuantumnessConfig:
    return QuantumnessConfig(
        n_qubits=cfg.n_qubits,
        zz_reps=cfg.zz_reps if zz_reps is None else zz_reps,
        entanglement=cfg.entanglement,
        n_sample=cfg.n_sample,
        n_repeats=cfg.n_repeats,
        random_state=cfg.random_state,
        lambda_geom=cfg.lambda_geom,
        gate_kappa=cfg.gate_kappa,
    )


def score_transformed(
    Xq: np.ndarray,
    y: np.ndarray,
    cfg: QCRConfig,
    *,
    dataset: str,
    strategy: str,
    zz_reps: int | None = None,
) -> dict[str, Any]:
    from qml_v2.quantumness import compute_indicators

    # Xq already ZZ-ready; feed as native so encode does not re-scale
    qcfg = _qn_cfg(cfg, zz_reps=zz_reps)
    # compute_indicators will subsample again from Xq — OK if len≈n_sample
    r = compute_indicators(
        Xq, y, qcfg, already_zz_space=True, dataset=f"{dataset}:{strategy}"
    )
    r.pop("_repeats", None)
    r["strategy"] = strategy
    r["dataset"] = dataset
    r["zz_reps_used"] = qcfg.zz_reps
    return r


def diagnose_bottleneck(base: dict[str, Any]) -> str:
    """Recommend strategy from baseline robust components."""
    dA = float(base.get("delta_A", 0.0))
    dF = float(base.get("d_F", 0.0))
    # Prefer raising ΔA if alignment is the problem; else geometry; else reps.
    if dA <= 0:
        return "zz_interactions"
    if dF < 0.25:
        return "rotation"
    return "reuploading"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_dataset_qcr(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: QCRConfig,
    already_zz_space: bool,
) -> list[dict[str, Any]]:
    from qml_v2.quantumness import compute_indicators

    print(f"\n=== QCR {name} ===")
    qcfg = _qn_cfg(cfg)
    base = compute_indicators(
        X, y, qcfg, already_zz_space=already_zz_space, dataset=f"{name}:baseline"
    )
    base.pop("_repeats", None)
    base["strategy"] = "baseline"
    base["dataset"] = name
    base["zz_reps_used"] = qcfg.zz_reps
    print(
        f"baseline: Q_rob={base['Q_rob']:+.3f} [{base['Q_rob_lo']:+.3f},{base['Q_rob_hi']:+.3f}] "
        f"ΔA={base['delta_A']:+.3f} adv={base.get('advantage_pp', float('nan')):+.1f}pp "
        f"→ {base['verdict']}"
    )

    # Fixed ZZ matrix for pre-maps (same subsample seed as first repeat spirit)
    Xq, yq = _to_zz(X, y, cfg, already_zz_space, cfg.random_state)
    rows = [base]

    # Strategy 1: ZZ interactions
    X_zz = apply_zz_interactions(Xq, cfg.n_qubits)
    r1 = score_transformed(X_zz, yq, cfg, dataset=name, strategy="zz_interactions")
    rows.append(r1)
    print(
        f"zz_interactions: Q_rob={r1['Q_rob']:+.3f} ΔA={r1['delta_A']:+.3f} "
        f"adv={r1.get('advantage_pp', float('nan')):+.1f}pp → {r1['verdict']}  "
        f"(ΔQ={r1['Q_rob']-base['Q_rob']:+.3f})"
    )

    # Strategy 2: rotation (2D only)
    if cfg.n_qubits == 2:
        theta, sweep = select_best_rotation(Xq, yq, cfg, cfg.random_state + 1)
        X_rot = apply_rotation(Xq, theta)
        r2 = score_transformed(
            X_rot, yq, cfg, dataset=name, strategy=f"rotation_θ={theta:.3f}"
        )
        r2["theta"] = theta
        rows.append(r2)
        print(
            f"rotation θ={np.degrees(theta):.1f}°: Q_rob={r2['Q_rob']:+.3f} "
            f"ΔA={r2['delta_A']:+.3f} adv={r2.get('advantage_pp', float('nan')):+.1f}pp "
            f"→ {r2['verdict']}  (ΔQ={r2['Q_rob']-base['Q_rob']:+.3f})"
        )
        metrics = repo_paths()["metrics"]
        metrics.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(metrics / f"qml_v2_qcr_{name}_rotation_sweep.csv", index=False)

    # Strategy 3: re-uploading
    best_reps, reps_sweep = select_best_reps(Xq, yq, cfg, cfg.random_state + 2)
    r3 = score_transformed(
        Xq, yq, cfg, dataset=name, strategy=f"reuploading_reps={best_reps}", zz_reps=best_reps
    )
    r3["reps"] = best_reps
    rows.append(r3)
    print(
        f"reuploading reps={best_reps}: Q_rob={r3['Q_rob']:+.3f} ΔA={r3['delta_A']:+.3f} "
        f"adv={r3.get('advantage_pp', float('nan')):+.1f}pp → {r3['verdict']}  "
        f"(ΔQ={r3['Q_rob']-base['Q_rob']:+.3f})"
    )
    repo_paths()["metrics"].mkdir(parents=True, exist_ok=True)
    reps_sweep.to_csv(
        repo_paths()["metrics"] / f"qml_v2_qcr_{name}_reps_sweep.csv", index=False
    )

    # Auto: heuristic recommendation + best empirical ΔQ among tried strategies
    rec = diagnose_bottleneck(base)
    non_base = [r for r in rows if r["strategy"] != "baseline"]
    best_emp = max(non_base, key=lambda r: r["Q_rob"]) if non_base else base
    auto = dict(best_emp)
    auto["strategy"] = f"qcr_auto_best:{best_emp['strategy']}"
    auto["recommended_heuristic"] = rec
    rows.append(auto)
    # Also record heuristic pick
    key = {
        "zz_interactions": "zz_interactions",
        "rotation": "rotation",
        "reuploading": "reuploading",
    }[rec]
    cands = [r for r in rows if key in r["strategy"] and not r["strategy"].startswith("qcr_")]
    heur = dict(cands[0] if cands else base)
    heur["strategy"] = f"qcr_heuristic:{rec}"
    rows.append(heur)
    print(
        f"qcr_auto_best→{best_emp['strategy']}: Q_rob={auto['Q_rob']:+.3f} "
        f"(ΔQ={auto['Q_rob']-base['Q_rob']:+.3f}) → {auto['verdict']}"
    )
    print(
        f"qcr_heuristic→{rec}: Q_rob={heur['Q_rob']:+.3f} "
        f"(ΔQ={heur['Q_rob']-base['Q_rob']:+.3f}) → {heur['verdict']}"
    )
    return rows


def run_all(
    cfg: QCRConfig | None = None,
    datasets: tuple[str, ...] = ("quantum_native", "telco", "skg"),
    quick: bool = False,
) -> pd.DataFrame:
    cfg = cfg or QCRConfig()
    if quick:
        cfg = QCRConfig(
            n_qubits=cfg.n_qubits,
            zz_reps=cfg.zz_reps,
            n_sample=48,
            n_repeats=2,
            n_angles=9,
            reps_grid=(1, 2, 3),
            select_n=40,
            random_state=cfg.random_state,
        )

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

    all_rows: list[dict[str, Any]] = []
    report = [
        "# QCR results (QML v2 — robust Q_rob)",
        "",
        "Pre-maps evaluated with gated/centered quantumness.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
    ]

    for name, X, y, native in jobs:
        rows = run_dataset_qcr(name, X, y, cfg, already_zz_space=native)
        all_rows.extend(rows)
        base = rows[0]
        report.append(f"## {name}")
        report.append("")
        report.append(
            f"Baseline: Q_rob={base['Q_rob']:+.4f} → {base['verdict']}"
        )
        report.append("")
        report.append("```")
        df = pd.DataFrame(rows)
        cols = [
            c
            for c in (
                "strategy",
                "Q_rob",
                "Q_rob_lo",
                "delta_A",
                "d_F",
                "advantage_pp",
                "verdict",
                "zz_reps_used",
            )
            if c in df.columns
        ]
        report.append(df[cols].to_string(index=False))
        report.append("```")
        report.append("")

    out = pd.DataFrame(all_rows)
    metrics = repo_paths()["metrics"]
    metrics.mkdir(parents=True, exist_ok=True)
    out.to_csv(metrics / "qml_v2_qcr_all.csv", index=False)
    (metrics / "qml_v2_qcr_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    path = REPO_ROOT / "BUSINESS" / "qcr_v2_results.md"
    path.write_text("\n".join(report) + "\n")
    print(f"\nWrote {path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="QML v2 QCR pre-maps for robust Q_rob")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-sample", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument(
        "--datasets",
        default="quantum_native,telco,skg",
        help="quantum_native,telco,skg",
    )
    args = p.parse_args()
    cfg = QCRConfig()
    if args.n_sample is not None:
        cfg.n_sample = args.n_sample
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    names = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    run_all(cfg=cfg, datasets=names, quick=args.quick)


if __name__ == "__main__":
    main()
