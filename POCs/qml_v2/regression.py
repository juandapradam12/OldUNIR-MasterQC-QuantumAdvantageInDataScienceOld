"""Quantum vs classical regression — QML v2 protocol.

Same rigor as ``qml_v2.classification``:

* **Table A — matched features.** QK-SVR (ZZ fidelity) vs SVR-RBF / Ridge /
  GradientBoosting / RandomForest on the same ZZ-ready matrix.
* **Table B — matched n, full classical features.** Classical models keep every
  original column. QK-SVR stays at ``n_qubits`` (one qubit per ZZ feature).

TFM notebook 03 issues this fixes: SKG absolute path; classical on full table
vs quantum on a PCA subsample; quantum-native X re-scaled after generation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from qml_v2.classification import (
    REPO_ROOT,
    encode_for_zz_kernel,
    repo_paths,
    zz_map,
)


@dataclass
class RegConfig:
    n_qubits: int = 2
    zz_reps: int = 2
    entanglement: str = "full"
    svr_C: float = 10.0
    svr_epsilon: float = 0.1
    n_total: int = 200
    test_size: float = 0.3
    n_repeats: int = 10
    random_state: int = 42
    n_estimators: int = 100


@dataclass
class RegRepeatResult:
    repeat: int
    model: str
    table: str
    r2: float
    rmse: float
    mae: float
    n_train: int
    n_test: int
    n_features: int
    pca_var_explained: float | None = None


def make_quantum_native_regression(
    n_samples: int,
    n_qubits: int = 2,
    reps: int = 2,
    entanglement: str = "full",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """X in [-pi, pi]^{n_qubits}; y = <Z...Z> (continuous in [-1, 1])."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    rng = np.random.RandomState(random_state)
    X = rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits))
    fmap = zz_map(n_qubits, reps, entanglement)
    pauli = "Z" * n_qubits if n_qubits != 2 else "ZZ"
    obs = SparsePauliOp.from_list([(pauli, 1.0)])
    y = np.empty(n_samples, dtype=float)
    for i, x in enumerate(X):
        bound = fmap.assign_parameters(dict(zip(fmap.parameters, x)))
        y[i] = float(Statevector(bound).expectation_value(obs).real)
    return X, y


class QuantumKernelSVR:
    """Fidelity quantum kernel + classical SVR (precomputed)."""

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        entanglement: str = "full",
        C: float = 10.0,
        epsilon: float = 0.1,
        random_state: int = 42,
    ):
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        algorithm_globals.random_seed = random_state
        fmap = zz_map(n_qubits, reps, entanglement)
        self.kernel = FidelityQuantumKernel(feature_map=fmap)
        self.C = C
        self.epsilon = epsilon
        self.X_train_: np.ndarray | None = None
        self.svr_: SVR | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantumKernelSVR:
        self.X_train_ = np.asarray(X, dtype=float)
        K = self.kernel.evaluate(x_vec=self.X_train_)
        self.svr_ = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
        self.svr_.fit(K, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.svr_ is None:
            raise RuntimeError("Call fit() before predict().")
        K = self.kernel.evaluate(x_vec=np.asarray(X, dtype=float), y_vec=self.X_train_)
        return self.svr_.predict(K)


def load_friedman(n_samples: int, n_features: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=1.0, random_state=random_state)
    return X, y


def load_skg_speeds(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """SKG travel speeds without near-target leakage features.

    Dropped: ``los_enc`` (level-of-service strongly correlates with speed) and
    ``elapsed_time`` (near-definitional with speed on many corridors).
    Kept: ``readings``, ``hour`` (+ optional corridor id if present).
    """
    path = path or (REPO_ROOT / "Datos" / "bctraveltimes_history.csv")
    if not path.exists():
        raise FileNotFoundError(f"SKG travel-times CSV not found: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["hour"] = pd.to_datetime(df["read_time"]).dt.hour
    cols = ["readings", "hour"]
    if "tid" in df.columns:
        cols = ["tid", *cols]
    use = [c for c in cols if c in df.columns]
    df = df[use + ["speed"]].dropna()
    return df[use].to_numpy(dtype=float), df["speed"].to_numpy(dtype=float), use


def subsample(X: np.ndarray, y: np.ndarray, n_total: int, random_state: int):
    if len(y) <= n_total:
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(y), size=n_total, replace=False)
    return X[idx], y[idx]


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
    )


def _classical_models(C: float, epsilon: float, n_estimators: int, random_state: int) -> dict[str, Any]:
    return {
        "SVR-RBF": SVR(kernel="rbf", C=C, epsilon=epsilon),
        "Ridge": Ridge(alpha=1.0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
    }


def run_reg_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: RegConfig,
    already_zz_space: bool,
    include_table_b: bool = True,
) -> list[RegRepeatResult]:
    X, y = subsample(X, y, cfg.n_total, cfg.random_state)
    splitter = ShuffleSplit(
        n_splits=cfg.n_repeats,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    rows: list[RegRepeatResult] = []

    for repeat, (tr, te) in enumerate(splitter.split(X)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        Xq_tr, Xq_te, pca_var = encode_for_zz_kernel(
            X_tr,
            X_te,
            n_qubits=cfg.n_qubits,
            random_state=cfg.random_state,
            already_zz_space=already_zz_space,
        )

        qmodel = QuantumKernelSVR(
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            C=cfg.svr_C,
            epsilon=cfg.svr_epsilon,
            random_state=cfg.random_state,
        )
        qmodel.fit(Xq_tr, y_tr)
        r2, rmse, mae = _scores(y_te, qmodel.predict(Xq_te))
        rows.append(
            RegRepeatResult(
                repeat=repeat,
                model="QK-SVR-ZZ",
                table="A_matched_features",
                r2=r2,
                rmse=rmse,
                mae=mae,
                n_train=len(y_tr),
                n_test=len(y_te),
                n_features=Xq_tr.shape[1],
                pca_var_explained=None if already_zz_space else pca_var,
            )
        )

        for model_name, clf in _classical_models(
            cfg.svr_C, cfg.svr_epsilon, cfg.n_estimators, cfg.random_state
        ).items():
            clf.fit(Xq_tr, y_tr)
            r2, rmse, mae = _scores(y_te, clf.predict(Xq_te))
            rows.append(
                RegRepeatResult(
                    repeat=repeat,
                    model=model_name,
                    table="A_matched_features",
                    r2=r2,
                    rmse=rmse,
                    mae=mae,
                    n_train=len(y_tr),
                    n_test=len(y_te),
                    n_features=Xq_tr.shape[1],
                    pca_var_explained=None if already_zz_space else pca_var,
                )
            )

        if include_table_b and not already_zz_space:
            scaler = StandardScaler()
            Xb_tr = scaler.fit_transform(X_tr)
            Xb_te = scaler.transform(X_te)
            for model_name, clf in _classical_models(
                cfg.svr_C, cfg.svr_epsilon, cfg.n_estimators, cfg.random_state
            ).items():
                clf.fit(Xb_tr, y_tr)
                r2, rmse, mae = _scores(y_te, clf.predict(Xb_te))
                rows.append(
                    RegRepeatResult(
                        repeat=repeat,
                        model=f"{model_name} [full features]",
                        table="B_matched_n_full_features",
                        r2=r2,
                        rmse=rmse,
                        mae=mae,
                        n_train=len(y_tr),
                        n_test=len(y_te),
                        n_features=Xb_tr.shape[1],
                        pca_var_explained=None,
                    )
                )
    return rows


def summarize(rows: list[RegRepeatResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    g = (
        df.groupby(["table", "model"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
            n_features=("n_features", "first"),
            pca_var=("pca_var_explained", "mean"),
            n_repeats=("repeat", "nunique"),
        )
        .sort_values(["table", "r2_mean"], ascending=[True, False])
    )
    for col in ("r2_std", "rmse_std", "mae_std"):
        g[col] = g[col].fillna(0.0)

    g["delta_r2_vs_best_classical"] = np.nan
    for _table, idx in g.groupby("table").groups.items():
        block = g.loc[idx]
        classical = block[~block["model"].str.startswith("QK-SVR")]
        q = block[block["model"] == "QK-SVR-ZZ"]
        if classical.empty or q.empty:
            continue
        best_cl = float(classical["r2_mean"].max())
        g.loc[q.index, "delta_r2_vs_best_classical"] = float(q["r2_mean"].iloc[0]) - best_cl
    return g


def advantage_report(summary: pd.DataFrame) -> str:
    lines: list[str] = []
    for table, block in summary.groupby("table", sort=True):
        q = block[block["model"] == "QK-SVR-ZZ"]
        cl = block[~block["model"].str.startswith("QK-SVR")]
        if q.empty or cl.empty:
            continue
        best = cl.loc[cl["r2_mean"].idxmax()]
        qrow = q.iloc[0]
        delta = float(qrow["r2_mean"]) - float(best["r2_mean"])
        q_std = float(qrow["r2_std"]) if not pd.isna(qrow["r2_std"]) else 0.0
        if delta > max(0.02, q_std):
            winner = "QK-SVR-ZZ"
        elif delta < -max(0.02, q_std):
            winner = "classical"
        else:
            winner = "tie (within noise)"
        lines.append(
            f"{table}: QK-SVR R²={qrow['r2_mean']:.3f}±{qrow['r2_std']:.3f} "
            f"(MAE={qrow['mae_mean']:.3f}) vs "
            f"best classical {best['model']} R²={best['r2_mean']:.3f}±{best['r2_std']:.3f} "
            f"(Δ={delta:+.3f}) → {winner}"
        )
    q_a = summary[(summary["table"] == "A_matched_features") & (summary["model"] == "QK-SVR-ZZ")]
    cl_b = summary[(summary["table"] == "B_matched_n_full_features") & (~summary["model"].str.startswith("QK-SVR"))]
    if not q_a.empty and not cl_b.empty:
        best_b = cl_b.loc[cl_b["r2_mean"].idxmax()]
        qa = q_a.iloc[0]
        d_biz = float(qa["r2_mean"]) - float(best_b["r2_mean"])
        lines.append(
            f"business (A-QK vs B-classical): QK-SVR R²={qa['r2_mean']:.3f} vs "
            f"{best_b['model']} R²={best_b['r2_mean']:.3f} (Δ={d_biz:+.3f})"
        )
    return "\n".join(lines)


def run_all(
    cfg: RegConfig | None = None,
    datasets: tuple[str, ...] = ("quantum_native", "friedman", "friedman_full", "skg"),
    quick: bool = False,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or RegConfig()
    if quick:
        cfg = RegConfig(
            n_qubits=cfg.n_qubits,
            zz_reps=1,
            n_total=40,
            n_repeats=1,
            random_state=cfg.random_state,
            svr_C=cfg.svr_C,
            svr_epsilon=cfg.svr_epsilon,
        )

    jobs: list[tuple[str, np.ndarray, np.ndarray, bool, int]] = []
    # (name, X, y, already_zz, n_qubits_override)
    if "quantum_native" in datasets:
        X, y = make_quantum_native_regression(
            n_samples=cfg.n_total,
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            random_state=cfg.random_state,
        )
        jobs.append(("quantum_native", X, y, True, cfg.n_qubits))
    if "friedman" in datasets:
        # Table A with PCA down to cfg.n_qubits (often 2) — hard for any kernel.
        n_feat = 5
        X, y = load_friedman(max(cfg.n_total, 120), n_feat, cfg.random_state)
        jobs.append(("friedman", X, y, False, cfg.n_qubits))
    if "friedman_full" in datasets:
        # Matched-width arm: 5 features → 5 qubits, no PCA.
        n_feat = 5
        X, y = load_friedman(max(cfg.n_total, 120), n_feat, cfg.random_state + 1)
        jobs.append(("friedman_full", X, y, False, n_feat))
    if "skg" in datasets:
        X, y, _ = load_skg_speeds()
        jobs.append(("skg", X, y, False, cfg.n_qubits))

    metrics_dir = repo_paths()["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    report_lines = [
        "# Regression results (QML v2)",
        "",
        "Generated by `python -m qml_v2.regression`.",
        "Table A = same features. Table B = classical full features, same n.",
        "SKG features exclude los_enc / elapsed_time (near-target leakage).",
        "friedman_full = 5 qubits, no PCA (matched width).",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
    ]

    for name, X, y, native, n_q in jobs:
        local = RegConfig(**{**asdict(cfg), "n_qubits": n_q})
        print(f"\n=== {name}  n={len(y)}  features={X.shape[1]}  native={native}  qubits={n_q} ===")
        rows = run_reg_dataset(name, X, y, local, already_zz_space=native, include_table_b=not native)
        raw = pd.DataFrame([asdict(r) for r in rows])
        raw.insert(0, "dataset", name)
        raw.to_csv(metrics_dir / f"qml_v2_regression_{name}_raw.csv", index=False)
        summary = summarize(rows)
        summary.insert(0, "dataset", name)
        out[name] = summary
        summary.to_csv(metrics_dir / f"qml_v2_regression_{name}.csv", index=False)
        print(summary.to_string(index=False))
        adv = advantage_report(summary)
        print(adv)
        report_lines += [f"## {name}", "", "```", adv, "```", "", "```", summary.to_string(index=False), "```", ""]

    if out:
        pd.concat(out.values(), ignore_index=True).to_csv(metrics_dir / "qml_v2_regression_all.csv", index=False)
        (metrics_dir / "qml_v2_regression_config.json").write_text(json.dumps(asdict(cfg), indent=2))
        report_path = REPO_ROOT / "BUSINESS" / "regression_v2_results.md"
        report_path.write_text("\n".join(report_lines) + "\n")
        print(f"\nWrote {report_path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="QML v2 fair regression")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-total", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument("--n-qubits", type=int, default=None)
    p.add_argument("--zz-reps", type=int, default=None)
    p.add_argument(
        "--datasets",
        default="quantum_native,friedman,friedman_full,skg",
        help="quantum_native,friedman,friedman_full,skg",
    )
    args = p.parse_args()
    cfg = RegConfig()
    if args.n_total is not None:
        cfg.n_total = args.n_total
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
