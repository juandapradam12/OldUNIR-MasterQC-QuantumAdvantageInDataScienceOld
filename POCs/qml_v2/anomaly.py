"""Quantum vs classical anomaly detection — QML v2 fair protocol.

This is **not** fraud detection. Fraud is a labelled rare-class classification
problem (use ``classification.py``). Anomaly detection here is **one-class /
novelty detection**:

* Fit on **inliers only**. Outlier labels never enter training.
* Labels are used **only** for ROC-AUC / average precision on a held-out test
  set that contains both inliers and outliers.

Two tables (never mix them):

* **Table A — matched features.** Same ZZ-ready matrix (``n_qubits`` angles).
  Classical: IsolationForest, OneClassSVM-RBF, LOF. Quantum: OneClassSVM with
  the ZZ fidelity kernel (precomputed).

* **Table B — matched n, full classical features.** Same split; classical
  detectors see the original columns. Answers whether OCSVM-ZZ should replace
  IsolationForest on the business table.

Scaler / PCA are fit on **train inliers only** (no leakage from test or from
known outliers).

SKG / Telco are practice tables with **injected** feature-space outliers.
Churn / gravity labels are discarded — they would turn this into the fraud
special case.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

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
class AnomalyConfig:
    n_qubits: int = 2
    zz_reps: int = 2
    entanglement: str = "full"
    n_total: int = 160
    test_size: float = 0.3
    n_repeats: int = 5
    random_state: int = 42
    contamination: float = 0.10
    nu: float = 0.05
    n_estimators_if: int = 200
    n_neighbors_lof: int = 20


@dataclass
class AnomalyRepeatResult:
    repeat: int
    model: str
    table: str
    roc_auc: float
    avg_precision: float
    f1_anomaly: float
    n_train_inliers: int
    n_test: int
    n_test_outliers: int
    n_features: int
    pca_var_explained: float | None = None


class QuantumKernelOCSVM:
    """Fidelity ZZ kernel + classical One-Class SVM (precomputed).

    ``fit`` receives inliers only. ``decision_function`` follows sklearn:
    larger means more inlier; callers should negate for anomaly scores.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        entanglement: str = "full",
        nu: float = 0.05,
        random_state: int = 42,
    ):
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        algorithm_globals.random_seed = random_state
        fmap = zz_map(n_qubits, reps, entanglement)
        self.kernel = FidelityQuantumKernel(feature_map=fmap)
        self.nu = nu
        self.X_train_: np.ndarray | None = None
        self.ocsvm_: OneClassSVM | None = None

    def fit(self, X: np.ndarray) -> QuantumKernelOCSVM:
        self.X_train_ = np.asarray(X, dtype=float)
        K = self.kernel.evaluate(x_vec=self.X_train_)
        self.ocsvm_ = OneClassSVM(kernel="precomputed", nu=self.nu)
        self.ocsvm_.fit(K)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.ocsvm_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        K = self.kernel.evaluate(x_vec=np.asarray(X, dtype=float), y_vec=self.X_train_)
        return np.asarray(self.ocsvm_.decision_function(K), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.ocsvm_ is None:
            raise RuntimeError("Call fit() before predict().")
        K = self.kernel.evaluate(x_vec=np.asarray(X, dtype=float), y_vec=self.X_train_)
        return np.asarray(self.ocsvm_.predict(K), dtype=int)


# ---------------------------------------------------------------------------
# Generators (y=1 is outlier; used only for splits + evaluation)
# ---------------------------------------------------------------------------


def set_contamination(
    X: np.ndarray,
    y: np.ndarray,
    contamination: float,
    n_total: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep inliers (y=0) and downsample outliers so the rate is ``contamination``."""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y, dtype=int)
    in_idx = np.where(y == 0)[0]
    out_idx = np.where(y == 1)[0]
    if len(in_idx) == 0 or len(out_idx) == 0:
        raise ValueError("Need both inliers (y=0) and outliers (y=1).")
    n_out = max(int(round(n_total * contamination)), 2)
    n_in = n_total - n_out
    if n_in < 8:
        raise ValueError("Too few inliers after contamination split.")
    n_in = min(n_in, len(in_idx))
    n_out = min(n_out, len(out_idx))
    sel = np.concatenate(
        [
            rng.choice(in_idx, n_in, replace=False),
            rng.choice(out_idx, n_out, replace=False),
        ]
    )
    rng.shuffle(sel)
    return X[sel], y[sel]


def make_zz_phase_anomaly(
    n_samples: int = 400,
    n_qubits: int = 2,
    reps: int = 2,
    entanglement: str = "full",
    contamination: float = 0.10,
    n_total: int = 160,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Inliers = majority ZZ phase; outliers = the other phase (same support).

    This is the ZZ-structure cell: IsolationForest cannot separate the two
    phases by geometry in angle space; OCSVM-ZZ can if the kernel sees it.
    """
    X, y = make_quantum_native(
        n_samples=n_samples,
        n_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
        random_state=random_state,
    )
    if int((y == 1).sum()) > int((y == 0).sum()):
        y = 1 - y
    return set_contamination(X, y, contamination, n_total, random_state)


def make_blob_anomaly(
    n_total: int = 160,
    n_features: int = 4,
    contamination: float = 0.10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Single Gaussian inlier blob + far uniform outliers (classical should win)."""
    rng = np.random.RandomState(random_state)
    n_out = max(int(round(n_total * contamination)), 2)
    n_in = n_total - n_out
    X_in, _ = make_blobs(
        n_samples=n_in,
        centers=1,
        n_features=n_features,
        cluster_std=0.7,
        random_state=random_state,
    )
    X_out = rng.uniform(low=-8.0, high=8.0, size=(n_out, n_features))
    X = np.vstack([X_in, X_out]).astype(float)
    y = np.array([0] * n_in + [1] * n_out, dtype=int)
    perm = rng.permutation(n_total)
    return X[perm], y[perm]


def make_moons_anomaly(
    n_total: int = 160,
    contamination: float = 0.10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-moons manifold as inliers; uniform box outliers."""
    rng = np.random.RandomState(random_state)
    n_out = max(int(round(n_total * contamination)), 2)
    n_in = n_total - n_out
    X_in, _ = make_moons(n_samples=n_in, noise=0.08, random_state=random_state)
    lo = X_in.min(axis=0) - 0.5
    hi = X_in.max(axis=0) + 0.5
    X_out = rng.uniform(lo, hi, size=(n_out, 2))
    X = np.vstack([X_in, X_out]).astype(float)
    y = np.array([0] * n_in + [1] * n_out, dtype=int)
    perm = rng.permutation(n_total)
    return X[perm], y[perm]


def inject_feature_outliers(
    X: np.ndarray,
    contamination: float,
    scale: float = 8.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Treat every row as inlier; replace a fraction with far Gaussian spikes.

    Does **not** use any business label (churn, gravity, fraud). The anomaly
    target is synthetic contamination in feature space.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float).copy()
    n = len(X)
    n_out = max(int(round(n * contamination)), 2)
    n_out = min(n_out, n - 8)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    idx = rng.choice(n, n_out, replace=False)
    X[idx] = mu + scale * sd * rng.normal(size=(n_out, X.shape[1]))
    y = np.zeros(n, dtype=int)
    y[idx] = 1
    return X, y


def subsample_rows(X: np.ndarray, n_total: int, random_state: int) -> np.ndarray:
    if len(X) <= n_total:
        return np.asarray(X, dtype=float)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), n_total, replace=False)
    return np.asarray(X[idx], dtype=float)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _roc_ap_f1(y_true: np.ndarray, scores: np.ndarray, pred_outlier: np.ndarray) -> tuple[float, float, float]:
    """scores: higher = more anomalous. pred_outlier: 1 = predicted outlier."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = np.asarray(pred_outlier, dtype=int)
    auc = float("nan")
    ap = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_true, scores))
        except ValueError:
            ap = float("nan")
    f1 = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
    return auc, ap, f1


def _sklearn_outlier_to_binary(pred: np.ndarray) -> np.ndarray:
    """Map sklearn {+1 inlier, -1 outlier} to {0, 1}."""
    return (np.asarray(pred, dtype=int) == -1).astype(int)


def _lof_n_neighbors(n_train: int, requested: int) -> int:
    return max(2, min(requested, n_train - 1))


def fit_score_classical(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    contamination: float,
    nu: float,
    n_estimators: int,
    n_neighbors: int,
    random_state: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return model → (anomaly_scores, pred_outlier_binary)."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    ifor = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    ifor.fit(X_tr)
    out["IsolationForest"] = (
        -np.asarray(ifor.decision_function(X_te), dtype=float),
        _sklearn_outlier_to_binary(ifor.predict(X_te)),
    )

    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(X_tr)
    out["OCSVM-RBF"] = (
        -np.asarray(ocsvm.decision_function(X_te), dtype=float),
        _sklearn_outlier_to_binary(ocsvm.predict(X_te)),
    )

    n_nei = _lof_n_neighbors(len(X_tr), n_neighbors)
    lof = LocalOutlierFactor(n_neighbors=n_nei, novelty=True, contamination=contamination)
    lof.fit(X_tr)
    out["LOF"] = (
        -np.asarray(lof.decision_function(X_te), dtype=float),
        _sklearn_outlier_to_binary(lof.predict(X_te)),
    )
    return out


def _append_result(
    rows: list[AnomalyRepeatResult],
    *,
    repeat: int,
    model: str,
    table: str,
    y_te: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
    n_train_inliers: int,
    n_features: int,
    pca_var: float | None,
) -> None:
    auc, ap, f1 = _roc_ap_f1(y_te, scores, pred)
    rows.append(
        AnomalyRepeatResult(
            repeat=repeat,
            model=model,
            table=table,
            roc_auc=auc,
            avg_precision=ap,
            f1_anomaly=f1,
            n_train_inliers=n_train_inliers,
            n_test=len(y_te),
            n_test_outliers=int(np.sum(y_te == 1)),
            n_features=n_features,
            pca_var_explained=pca_var,
        )
    )


def run_fair_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: AnomalyConfig,
    already_zz_space: bool,
    include_table_b: bool = True,
) -> list[AnomalyRepeatResult]:
    X, y = subsample_stratified(X, y, cfg.n_total, cfg.random_state)
    splitter = StratifiedShuffleSplit(
        n_splits=cfg.n_repeats,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    rows: list[AnomalyRepeatResult] = []

    for repeat, (tr, te) in enumerate(splitter.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        in_tr = y_tr == 0
        X_in = X_tr[in_tr]
        if X_in.shape[0] < 8:
            continue

        Xq_tr, Xq_te, pca_var = encode_for_zz_kernel(X_in, X_te, cfg.n_qubits, cfg.random_state, already_zz_space)
        pca_a = None if already_zz_space else float(pca_var)

        classical_a = fit_score_classical(
            Xq_tr,
            Xq_te,
            contamination=cfg.contamination,
            nu=cfg.nu,
            n_estimators=cfg.n_estimators_if,
            n_neighbors=cfg.n_neighbors_lof,
            random_state=cfg.random_state + repeat,
        )
        for model, (scores, pred) in classical_a.items():
            _append_result(
                rows,
                repeat=repeat,
                model=model,
                table="A_matched_features",
                y_te=y_te,
                scores=scores,
                pred=pred,
                n_train_inliers=len(Xq_tr),
                n_features=Xq_tr.shape[1],
                pca_var=pca_a,
            )

        q = QuantumKernelOCSVM(
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            nu=cfg.nu,
            random_state=cfg.random_state + repeat,
        )
        q.fit(Xq_tr)
        q_scores = -np.asarray(q.decision_function(Xq_te), dtype=float)
        q_pred = _sklearn_outlier_to_binary(q.predict(Xq_te))
        _append_result(
            rows,
            repeat=repeat,
            model="OCSVM-ZZ",
            table="A_matched_features",
            y_te=y_te,
            scores=q_scores,
            pred=q_pred,
            n_train_inliers=len(Xq_tr),
            n_features=Xq_tr.shape[1],
            pca_var=pca_a,
        )

        if include_table_b and not already_zz_space:
            scaler = StandardScaler()
            Xb_tr = scaler.fit_transform(X_in)
            Xb_te = scaler.transform(X_te)
            classical_b = fit_score_classical(
                Xb_tr,
                Xb_te,
                contamination=cfg.contamination,
                nu=cfg.nu,
                n_estimators=cfg.n_estimators_if,
                n_neighbors=cfg.n_neighbors_lof,
                random_state=cfg.random_state + repeat,
            )
            for model, (scores, pred) in classical_b.items():
                _append_result(
                    rows,
                    repeat=repeat,
                    model=f"{model} [full features]",
                    table="B_matched_n_full_features",
                    y_te=y_te,
                    scores=scores,
                    pred=pred,
                    n_train_inliers=len(Xb_tr),
                    n_features=Xb_tr.shape[1],
                    pca_var=None,
                )
    return rows


def summarize(rows: list[AnomalyRepeatResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    g = (
        df.groupby(["table", "model"], as_index=False)
        .agg(
            auc_mean=("roc_auc", "mean"),
            auc_std=("roc_auc", "std"),
            ap_mean=("avg_precision", "mean"),
            ap_std=("avg_precision", "std"),
            f1_mean=("f1_anomaly", "mean"),
            f1_std=("f1_anomaly", "std"),
            n_train=("n_train_inliers", "first"),
            n_test=("n_test", "first"),
            n_features=("n_features", "first"),
            pca_var=("pca_var_explained", "first"),
            n_repeats=("repeat", "nunique"),
        )
        .sort_values(["table", "auc_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    g["delta_auc_vs_best_classical"] = np.nan
    for _table, block in g.groupby("table"):
        classical = block[~block["model"].str.contains("ZZ", regex=False)]
        quantum = block[block["model"].str.contains("ZZ", regex=False)]
        if classical.empty or quantum.empty:
            continue
        best_cl = float(classical["auc_mean"].max())
        g.loc[quantum.index, "delta_auc_vs_best_classical"] = quantum["auc_mean"].astype(float) - best_cl
    return g


def advantage_report(summary: pd.DataFrame) -> str:
    lines: list[str] = []
    for table, block in summary.groupby("table", sort=True):
        q = block[block["model"].str.contains("ZZ", regex=False)]
        cl = block[~block["model"].str.contains("ZZ", regex=False)]
        if q.empty or cl.empty:
            continue
        best = cl.loc[cl["auc_mean"].idxmax()]
        qrow = q.iloc[0]
        delta = float(qrow["auc_mean"]) - float(best["auc_mean"])
        q_std = float(qrow["auc_std"]) if not pd.isna(qrow["auc_std"]) else 0.0
        if delta > max(0.02, q_std):
            winner = "OCSVM-ZZ"
        elif delta < -max(0.02, q_std):
            winner = "classical"
        else:
            winner = "tie (within noise)"
        lines.append(
            f"{table}: OCSVM-ZZ AUC {qrow['auc_mean']:.3f}±{qrow['auc_std']:.3f} "
            f"(AP={qrow['ap_mean']:.3f}) vs best classical {best['model']} "
            f"{best['auc_mean']:.3f}±{best['auc_std']:.3f} (Δ={delta:+.3f}) → {winner}"
        )

    q_a = summary[(summary["table"] == "A_matched_features") & (summary["model"].str.contains("ZZ", regex=False))]
    cl_b = summary[
        (summary["table"] == "B_matched_n_full_features") & (~summary["model"].str.contains("ZZ", regex=False))
    ]
    if not q_a.empty and not cl_b.empty:
        best_b = cl_b.loc[cl_b["auc_mean"].idxmax()]
        qa = q_a.iloc[0]
        d_biz = float(qa["auc_mean"]) - float(best_b["auc_mean"])
        lines.append(
            f"business (A-ZZ vs B-classical): OCSVM-ZZ {qa['auc_mean']:.3f} vs "
            f"{best_b['model']} {best_b['auc_mean']:.3f} (Δ={d_biz:+.3f})"
        )
    return "\n".join(lines)


def _load_injected(loader, n_pool: int, contamination: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    X, _y_discarded, _ = loader()
    del _y_discarded
    Xp = subsample_rows(X, n_pool, random_state)
    return inject_feature_outliers(Xp, contamination, random_state=random_state)


def run_all(
    cfg: AnomalyConfig | None = None,
    datasets: tuple[str, ...] = ("zz_phase", "moons", "blobs", "telco_injected", "skg_injected"),
    quick: bool = False,
) -> pd.DataFrame:
    cfg = cfg or AnomalyConfig()
    if quick:
        cfg = AnomalyConfig(
            n_total=48,
            n_repeats=2,
            random_state=cfg.random_state,
            n_qubits=cfg.n_qubits,
            zz_reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            contamination=cfg.contamination,
            nu=cfg.nu,
        )

    jobs: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    n_pool = max(cfg.n_total * 2, 240)

    if "zz_phase" in datasets:
        X, y = make_zz_phase_anomaly(
            n_samples=max(cfg.n_total * 3, 300),
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            contamination=cfg.contamination,
            n_total=max(cfg.n_total, 48),
            random_state=cfg.random_state,
        )
        jobs.append(("zz_phase", X, y, True))
    if "moons" in datasets:
        X, y = make_moons_anomaly(n_total=n_pool, contamination=cfg.contamination, random_state=cfg.random_state)
        jobs.append(("moons", X, y, False))
    if "blobs" in datasets:
        X, y = make_blob_anomaly(n_total=n_pool, contamination=cfg.contamination, random_state=cfg.random_state)
        jobs.append(("blobs", X, y, False))
    if "telco_injected" in datasets:
        X, y = _load_injected(load_telco, n_pool, cfg.contamination, cfg.random_state)
        jobs.append(("telco_injected", X, y, False))
    if "skg_injected" in datasets:
        X, y = _load_injected(load_skg_incidents, n_pool, cfg.contamination, cfg.random_state)
        jobs.append(("skg_injected", X, y, False))

    all_rows: list[AnomalyRepeatResult] = []
    report = [
        "# Fair anomaly-detection results",
        "",
        "Generated by `python -m qml_v2.anomaly`.",
        "One-class protocol: fit on inliers only. Not fraud (no rare-class labels).",
        "Table A = same ZZ features. Table B = classical full features (matched n).",
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
        pd.DataFrame([asdict(r) for r in rows]).to_csv(metrics / f"fair_anomaly_{name}_raw.csv", index=False)
        summary.to_csv(metrics / f"fair_anomaly_{name}.csv", index=False)

    out = pd.DataFrame([asdict(r) for r in all_rows])
    path = REPO_ROOT / "BUSINESS" / "anomaly_v2_results.md"
    path.write_text("\n".join(report) + "\n")
    (repo_paths()["metrics"] / "qml_v2_anomaly_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"\nWrote {path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fair OCSVM-ZZ vs classical one-class anomaly detection")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-total", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument(
        "--datasets",
        default="zz_phase,moons,blobs,telco_injected,skg_injected",
        help="Comma-separated dataset names",
    )
    args = p.parse_args()
    cfg = AnomalyConfig()
    if args.n_total is not None:
        cfg.n_total = args.n_total
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    names = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    run_all(cfg=cfg, datasets=names, quick=args.quick)


if __name__ == "__main__":
    main()
