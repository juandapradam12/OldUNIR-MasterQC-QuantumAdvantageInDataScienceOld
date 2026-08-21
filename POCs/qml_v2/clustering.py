"""Quantum vs classical clustering — QML v2 fair protocol.

Unsupervised fit; ground-truth labels used **only** for ARI / NMI evaluation.

Two tables (never mix them):

* **Table A — matched features.** Same ZZ-ready matrix (``n_qubits`` angles in
  ``[-π, π]``). Classical: KMeans, Spectral-RBF, Agglomerative. Quantum:
  Spectral clustering with the ZZ fidelity kernel as precomputed affinity.

* **Table B — matched n, full classical features.** Classical algorithms on
  the original feature matrix (same subsample). Answers whether a ZZ spectral
  method should replace KMeans on the business table.

Repeats = independent stratified subsamples (labels only for balance / eval).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from qml_v2.classification import (
    REPO_ROOT,
    encode_for_zz_kernel,
    load_skg_incidents,
    load_telco,
    make_quantum_native,
    repo_paths,
    subsample_stratified,
)
from qml_v2.quantumness import zz_fidelity_kernel


@dataclass
class ClusterConfig:
    n_qubits: int = 2
    zz_reps: int = 2
    entanglement: str = "full"
    n_clusters: int = 2
    n_total: int = 120
    n_repeats: int = 5
    random_state: int = 42
    spectral_rbf_gamma: float = 1.0


@dataclass
class ClusterRepeatResult:
    repeat: int
    model: str
    table: str
    ari: float
    nmi: float
    n_samples: int
    n_features: int
    pca_var_explained: float | None = None


def make_moons_dataset(
    n_samples: int = 300, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=n_samples, noise=0.08, random_state=random_state)
    return X.astype(float), y.astype(int)


def make_blobs_dataset(
    n_samples: int = 300, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        n_features=4,
        cluster_std=1.2,
        random_state=random_state,
    )
    return X.astype(float), y.astype(int)


def _relabel_binary(pred: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster ids to {0,1} when possible (does not affect ARI)."""
    return np.asarray(pred, dtype=int)


def _fit_predict_models(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    gamma: float,
    K_zz: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["KMeans"] = _relabel_binary(
        KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(
            X
        )
    )
    out["Spectral-RBF"] = _relabel_binary(
        SpectralClustering(
            n_clusters=n_clusters,
            affinity="rbf",
            gamma=gamma,
            random_state=random_state,
            assign_labels="kmeans",
        ).fit_predict(X)
    )
    out["Agglomerative"] = _relabel_binary(
        AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
    )
    if K_zz is not None:
        # Fidelity kernel ∈ [0,1]; SpectralClustering expects similarity.
        K = np.clip((K_zz + K_zz.T) / 2.0, 0.0, 1.0)
        np.fill_diagonal(K, 1.0)
        out["Spectral-ZZ"] = _relabel_binary(
            SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=random_state,
                assign_labels="kmeans",
            ).fit_predict(K)
        )
    return out


def run_fair_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: ClusterConfig,
    already_zz_space: bool,
    include_table_b: bool = True,
) -> list[ClusterRepeatResult]:
    rows: list[ClusterRepeatResult] = []
    for repeat in range(cfg.n_repeats):
        seed = cfg.random_state + repeat
        Xs, ys = subsample_stratified(X, y, cfg.n_total, seed)
        Xq, _, pca_var = encode_for_zz_kernel(
            Xs, Xs, cfg.n_qubits, seed, already_zz_space=already_zz_space
        )
        K_zz = zz_fidelity_kernel(
            Xq, cfg.n_qubits, cfg.zz_reps, cfg.entanglement, seed
        )
        preds_a = _fit_predict_models(
            Xq,
            n_clusters=cfg.n_clusters,
            random_state=seed,
            gamma=cfg.spectral_rbf_gamma,
            K_zz=K_zz,
        )
        for model, pred in preds_a.items():
            rows.append(
                ClusterRepeatResult(
                    repeat=repeat,
                    model=model,
                    table="A_matched_features",
                    ari=float(adjusted_rand_score(ys, pred)),
                    nmi=float(normalized_mutual_info_score(ys, pred)),
                    n_samples=len(ys),
                    n_features=Xq.shape[1],
                    pca_var_explained=None if already_zz_space else float(pca_var),
                )
            )

        if include_table_b and not already_zz_space:
            Xb = StandardScaler().fit_transform(Xs)
            preds_b = _fit_predict_models(
                Xb,
                n_clusters=cfg.n_clusters,
                random_state=seed,
                gamma=cfg.spectral_rbf_gamma,
                K_zz=None,
            )
            for model, pred in preds_b.items():
                rows.append(
                    ClusterRepeatResult(
                        repeat=repeat,
                        model=f"{model} [full features]",
                        table="B_matched_n_full_features",
                        ari=float(adjusted_rand_score(ys, pred)),
                        nmi=float(normalized_mutual_info_score(ys, pred)),
                        n_samples=len(ys),
                        n_features=Xb.shape[1],
                        pca_var_explained=None,
                    )
                )
    return rows


def summarize(rows: list[ClusterRepeatResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    g = (
        df.groupby(["table", "model"], as_index=False)
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            n_samples=("n_samples", "first"),
            n_features=("n_features", "first"),
            pca_var=("pca_var_explained", "first"),
            n_repeats=("repeat", "nunique"),
        )
        .sort_values(["table", "ari_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    g["delta_ari_vs_best_classical"] = np.nan
    for table, block in g.groupby("table"):
        classical = block[~block["model"].str.contains("ZZ", regex=False)]
        quantum = block[block["model"].str.contains("ZZ", regex=False)]
        if classical.empty or quantum.empty:
            continue
        best_cl = float(classical["ari_mean"].max())
        g.loc[quantum.index, "delta_ari_vs_best_classical"] = (
            quantum["ari_mean"].astype(float) - best_cl
        )
    return g


def advantage_report(summary: pd.DataFrame) -> str:
    lines: list[str] = []
    for table, block in summary.groupby("table", sort=True):
        q = block[block["model"].str.contains("ZZ", regex=False)]
        cl = block[~block["model"].str.contains("ZZ", regex=False)]
        if q.empty or cl.empty:
            continue
        best = cl.loc[cl["ari_mean"].idxmax()]
        qrow = q.iloc[0]
        delta = float(qrow["ari_mean"]) - float(best["ari_mean"])
        q_std = float(qrow["ari_std"]) if not pd.isna(qrow["ari_std"]) else 0.0
        if delta > max(0.02, q_std):
            winner = "Spectral-ZZ"
        elif delta < -max(0.02, q_std):
            winner = "classical"
        else:
            winner = "tie (within noise)"
        lines.append(
            f"{table}: Spectral-ZZ ARI {qrow['ari_mean']:.3f}±{qrow['ari_std']:.3f} "
            f"(NMI={qrow['nmi_mean']:.3f}) vs best classical {best['model']} "
            f"{best['ari_mean']:.3f}±{best['ari_std']:.3f} (Δ={delta:+.3f}) → {winner}"
        )

    q_a = summary[
        (summary["table"] == "A_matched_features")
        & (summary["model"].str.contains("ZZ", regex=False))
    ]
    cl_b = summary[
        (summary["table"] == "B_matched_n_full_features")
        & (~summary["model"].str.contains("ZZ", regex=False))
    ]
    if not q_a.empty and not cl_b.empty:
        best_b = cl_b.loc[cl_b["ari_mean"].idxmax()]
        qa = q_a.iloc[0]
        d_biz = float(qa["ari_mean"]) - float(best_b["ari_mean"])
        lines.append(
            f"business (A-ZZ vs B-classical): Spectral-ZZ {qa['ari_mean']:.3f} vs "
            f"{best_b['model']} {best_b['ari_mean']:.3f} (Δ={d_biz:+.3f})"
        )
    return "\n".join(lines)


def run_all(
    cfg: ClusterConfig | None = None,
    datasets: tuple[str, ...] = ("quantum_native", "moons", "blobs", "telco", "skg"),
    quick: bool = False,
) -> pd.DataFrame:
    cfg = cfg or ClusterConfig()
    if quick:
        cfg = ClusterConfig(
            n_total=48,
            n_repeats=2,
            random_state=cfg.random_state,
            n_qubits=cfg.n_qubits,
            zz_reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            n_clusters=cfg.n_clusters,
        )

    jobs: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    if "quantum_native" in datasets:
        X, y = make_quantum_native(
            n_samples=max(cfg.n_total * 2, 200),
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            random_state=cfg.random_state,
        )
        jobs.append(("quantum_native", X, y, True))
    if "moons" in datasets:
        X, y = make_moons_dataset(n_samples=max(cfg.n_total * 2, 300), random_state=cfg.random_state)
        jobs.append(("moons", X, y, False))
    if "blobs" in datasets:
        X, y = make_blobs_dataset(n_samples=max(cfg.n_total * 2, 300), random_state=cfg.random_state)
        jobs.append(("blobs", X, y, False))
    if "telco" in datasets:
        X, y, _ = load_telco()
        jobs.append(("telco", X, y, False))
    if "skg" in datasets:
        X, y, _ = load_skg_incidents()
        jobs.append(("skg", X, y, False))

    all_rows: list[ClusterRepeatResult] = []
    report = [
        "# Fair clustering results",
        "",
        "Generated by `python -m qml_v2.clustering`.",
        "Table A = same ZZ features. Table B = classical full features (matched n).",
        "Labels used only for ARI/NMI evaluation.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
    ]

    for name, X, y, native in jobs:
        print(f"\n=== {name} ===")
        rows = run_fair_dataset(name, X, y, cfg, already_zz_space=native)
        all_rows.extend(rows)
        summary = summarize(rows)
        summary.insert(0, "dataset", name)
        print(advantage_report(summary))
        report.append(f"## {name}")
        report.append("")
        report.append("```")
        report.append(advantage_report(summary))
        report.append("```")
        report.append("")
        report.append("```")
        report.append(summary.to_string(index=False))
        report.append("```")
        report.append("")

        metrics = repo_paths()["metrics"]
        metrics.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(r) for r in rows]).to_csv(
            metrics / f"fair_clustering_{name}_raw.csv", index=False
        )
        summary.to_csv(metrics / f"fair_clustering_{name}.csv", index=False)

    out = pd.DataFrame([asdict(r) for r in all_rows])
    path = REPO_ROOT / "BUSINESS" / "clustering_v2_results.md"
    path.write_text("\n".join(report) + "\n")
    (repo_paths()["metrics"] / "qml_v2_clustering_config.json").write_text(
        json.dumps(asdict(cfg), indent=2)
    )
    print(f"\nWrote {path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fair Spectral-ZZ vs classical clustering")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-total", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument(
        "--datasets",
        default="quantum_native,moons,blobs,telco,skg",
        help="Comma-separated dataset names",
    )
    args = p.parse_args()
    cfg = ClusterConfig()
    if args.n_total is not None:
        cfg.n_total = args.n_total
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    names = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    run_all(cfg=cfg, datasets=names, quick=args.quick)


if __name__ == "__main__":
    main()
