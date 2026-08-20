"""Quantum vs classical classification — QML v2 protocol.

Version 2 fixes the TFM notebook comparison: QSVM was trained on a PCA
subset while Gradient Boosting saw the full table.

Two explicit tables (never mix them):

* **Table A — matched features (kernel test).** Same rows, same features,
  same split. Everyone sees the ZZ-ready matrix with ``n_qubits`` columns
  in ``[-pi, pi]``. This isolates ZZ fidelity vs RBF / linear / GB / logistic.

* **Table B — matched sample size, full classical features.** Same rows and
  split as Table A. Classical models use **all original columns** (e.g. Telco
  19 features, SKG 7). QSVM is **not** re-run here with full features: a
  ZZFeatureMap needs one qubit per feature, so 19 qubits is outside this
  protocol. Compare Table A QSVM to Table B classical to answer “should I
  replace GB on this business table?”.

Preprocessing (scaler, PCA) is always fit on the training fold only.
Quantum-native labels use the same ZZ map and sign(<ZZ>); those points are
not re-scaled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
POCS_DIR = Path(__file__).resolve().parents[1]


@dataclass
class FairConfig:
    n_qubits: int = 2
    zz_reps: int = 2
    entanglement: str = "full"
    svm_C: float = 1.0
    n_total: int = 200
    test_size: float = 0.3
    n_repeats: int = 10
    random_state: int = 42
    n_estimators_gb: int = 100
    class_weight: str | None = "balanced"


@dataclass
class RepeatResult:
    repeat: int
    model: str
    table: str
    accuracy: float
    f1_macro: float
    roc_auc: float
    pos_rate_pred: float
    n_train: int
    n_test: int
    n_features: int
    pca_var_explained: float | None = None


def repo_paths() -> dict[str, Path]:
    return {
        "telco": POCS_DIR / "data" / "input" / "telco_customer_churn.csv",
        "skg": REPO_ROOT / "Datos" / "incidents_movility.csv",
        "metrics": POCS_DIR / "data" / "output" / "metrics",
    }


def zz_map(n_qubits: int, reps: int, entanglement: str):
    """ZZ feature map as a plain QuantumCircuit (Qiskit 2.1+)."""
    try:
        from qiskit.circuit.library import zz_feature_map as _zz

        return _zz(n_qubits, reps=reps, entanglement=entanglement)
    except ImportError:
        from qiskit.circuit.library import ZZFeatureMap

        return ZZFeatureMap(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )

def make_quantum_native(
    n_samples: int,
    n_qubits: int = 2,
    reps: int = 2,
    entanglement: str = "full",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """X in [-pi, pi]^{n_qubits}; y = sign(<Z...Z> on ZZFeatureMap|0>)."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    rng = np.random.RandomState(random_state)
    X = rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits))
    fmap = zz_map(n_qubits, reps, entanglement)
    pauli = "Z" * n_qubits if n_qubits != 2 else "ZZ"
    obs = SparsePauliOp.from_list([(pauli, 1.0)])
    y = np.empty(n_samples, dtype=int)
    for i, x in enumerate(X):
        bound = fmap.assign_parameters(dict(zip(fmap.parameters, x)))
        ev = Statevector(bound).expectation_value(obs).real
        y[i] = 1 if ev >= 0 else 0
    return X, y


class QuantumKernelSVC:
    """Fidelity quantum kernel + classical SVC (precomputed).

    The circuit is the model of similarity; the decision is SVM. Train points
    are stored so predict() does not need a second X_train argument.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        entanglement: str = "full",
        C: float = 1.0,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ):
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        algorithm_globals.random_seed = random_state
        fmap = zz_map(n_qubits, reps, entanglement)
        self.kernel = FidelityQuantumKernel(feature_map=fmap)
        self.C = C
        self.random_state = random_state
        self.class_weight = class_weight
        self.X_train_: np.ndarray | None = None
        self.svc_: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVC":
        self.X_train_ = np.asarray(X, dtype=float)
        K = self.kernel.evaluate(x_vec=self.X_train_)
        self.svc_ = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.svc_.fit(K, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.svc_ is None:
            raise RuntimeError("Call fit() before predict().")
        K = self.kernel.evaluate(x_vec=np.asarray(X, dtype=float), y_vec=self.X_train_)
        return self.svc_.predict(K)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.svc_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        K = self.kernel.evaluate(x_vec=np.asarray(X, dtype=float), y_vec=self.X_train_)
        return self.svc_.decision_function(K)


# ---------------------------------------------------------------------------
# Tabular loaders
# ---------------------------------------------------------------------------

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def load_telco(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Telco churn with deterministic category codes (sorted vocab).

    Categories are coded from the full file with a stable sorted mapping so
    fold splits do not depend on LabelEncoder fit order. Unknowns are not an
    issue because every row is encoded once before subsampling.
    """
    path = path or repo_paths()["telco"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(TELCO_URL)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    y = (df["Churn"].astype(str).str.strip() == "Yes").astype(int).to_numpy()
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) in ("bool", "category", "string"):
            cats = sorted(X[col].astype(str).unique())
            mapping = {c: i for i, c in enumerate(cats)}
            X[col] = X[col].astype(str).map(mapping).astype(float)
    X = X.fillna(0)
    return X.to_numpy(dtype=float), y, list(X.columns)


def load_skg_incidents(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = path or repo_paths()["skg"]
    if not path.exists():
        raise FileNotFoundError(f"SKG incidents CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df[df["gravity"].isin([25.0, 26.0, 25, 26])].copy()
    y = (df["gravity"].astype(float) == 26.0).astype(int).to_numpy()
    cols = ["priority", "type", "zone_id", "orient", "implicated", "latitude", "longitude"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"SKG file missing columns {missing}")
    X = df[cols].copy()
    X["implicated"] = X["implicated"].fillna(X["implicated"].median())
    X = X.fillna(0)
    return X.to_numpy(dtype=float), y, cols


def subsample_stratified(
    X: np.ndarray, y: np.ndarray, n_total: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= n_total:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_total, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Preprocess (train-only fit)
# ---------------------------------------------------------------------------

def encode_for_zz_kernel(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_qubits: int,
    random_state: int,
    already_zz_space: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map tabular (or native) features to n_qubits angles in [-pi, pi].

    If already_zz_space, X is assumed generated as ZZFeatureMap angles and is
    only checked/clipped — no StandardScaler (that would break the generator).
    If X already has exactly n_qubits columns, PCA is skipped (matched-width arm).
    """
    if already_zz_space:
        if X_train.shape[1] != n_qubits:
            raise ValueError("Quantum-native X must have n_qubits columns.")
        return np.asarray(X_train, float), np.asarray(X_test, float), 1.0

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    pca_var = 1.0
    if Xtr.shape[1] > n_qubits:
        pca = PCA(n_components=n_qubits, random_state=random_state)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
        pca_var = float(pca.explained_variance_ratio_.sum())
    elif Xtr.shape[1] < n_qubits:
        raise ValueError(
            f"Need at least {n_qubits} features, got {Xtr.shape[1]}."
        )
    mms = MinMaxScaler(feature_range=(-np.pi, np.pi))
    Xtr = mms.fit_transform(Xtr)
    Xte = mms.transform(Xte)
    return Xtr, Xte, pca_var


def _scores(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pos_rate = float(np.mean(y_pred)) if len(y_pred) else float("nan")
    auc = float("nan")
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = float("nan")
    elif len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        # Fallback: treat hard labels as scores (worse than decision_function).
        try:
            auc = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            auc = float("nan")
    return acc, f1, auc, pos_rate


def _decision_or_none(model: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X), dtype=float)
        except Exception:
            return None
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        except Exception:
            return None
    return None


def _classical_models(
    C: float,
    n_estimators: int,
    random_state: int,
    class_weight: str | None = "balanced",
) -> dict[str, Any]:
    return {
        "SVM-RBF": SVC(
            kernel="rbf", C=C, class_weight=class_weight, random_state=random_state
        ),
        "SVM-linear": SVC(
            kernel="linear", C=C, class_weight=class_weight, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=random_state
        ),
    }


def _append_result(
    rows: list[RepeatResult],
    *,
    repeat: int,
    model: str,
    table: str,
    y_te: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    n_train: int,
    n_features: int,
    pca_var: float | None,
) -> None:
    acc, f1, auc, pos_rate = _scores(y_te, y_pred, y_score)
    rows.append(
        RepeatResult(
            repeat=repeat,
            model=model,
            table=table,
            accuracy=acc,
            f1_macro=f1,
            roc_auc=auc,
            pos_rate_pred=pos_rate,
            n_train=n_train,
            n_test=len(y_te),
            n_features=n_features,
            pca_var_explained=pca_var,
        )
    )


def run_fair_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: FairConfig,
    already_zz_space: bool,
    include_table_b: bool = True,
) -> list[RepeatResult]:
    X, y = subsample_stratified(X, y, cfg.n_total, cfg.random_state)
    splitter = StratifiedShuffleSplit(
        n_splits=cfg.n_repeats,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    rows: list[RepeatResult] = []

    for repeat, (tr, te) in enumerate(splitter.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        Xq_tr, Xq_te, pca_var = encode_for_zz_kernel(
            X_tr, X_te, cfg.n_qubits, cfg.random_state, already_zz_space
        )

        qclf = QuantumKernelSVC(
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            C=cfg.svm_C,
            random_state=cfg.random_state,
            class_weight=cfg.class_weight,
        )
        qclf.fit(Xq_tr, y_tr)
        y_pred = qclf.predict(Xq_te)
        _append_result(
            rows,
            repeat=repeat,
            model="QSVM-ZZ",
            table="A_matched_features",
            y_te=y_te,
            y_pred=y_pred,
            y_score=_decision_or_none(qclf, Xq_te),
            n_train=len(y_tr),
            n_features=Xq_tr.shape[1],
            pca_var=None if already_zz_space else pca_var,
        )

        for model_name, clf in _classical_models(
            cfg.svm_C, cfg.n_estimators_gb, cfg.random_state, cfg.class_weight
        ).items():
            clf.fit(Xq_tr, y_tr)
            y_pred = clf.predict(Xq_te)
            _append_result(
                rows,
                repeat=repeat,
                model=model_name,
                table="A_matched_features",
                y_te=y_te,
                y_pred=y_pred,
                y_score=_decision_or_none(clf, Xq_te),
                n_train=len(y_tr),
                n_features=Xq_tr.shape[1],
                pca_var=None if already_zz_space else pca_var,
            )

        if include_table_b and not already_zz_space:
            # Full-width classical baseline on the SAME train/test indices.
            scaler = StandardScaler()
            Xb_tr = scaler.fit_transform(X_tr)
            Xb_te = scaler.transform(X_te)
            for model_name, clf in _classical_models(
                cfg.svm_C, cfg.n_estimators_gb, cfg.random_state, cfg.class_weight
            ).items():
                clf.fit(Xb_tr, y_tr)
                y_pred = clf.predict(Xb_te)
                _append_result(
                    rows,
                    repeat=repeat,
                    model=f"{model_name} [full features]",
                    table="B_matched_n_full_features",
                    y_te=y_te,
                    y_pred=y_pred,
                    y_score=_decision_or_none(clf, Xb_te),
                    n_train=len(y_tr),
                    n_features=Xb_tr.shape[1],
                    pca_var=None,
                )
    return rows


def summarize(rows: list[RepeatResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    g = (
        df.groupby(["table", "model"], as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            auc_mean=("roc_auc", "mean"),
            auc_std=("roc_auc", "std"),
            pos_rate_mean=("pos_rate_pred", "mean"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
            n_features=("n_features", "first"),
            pca_var=("pca_var_explained", "mean"),
            n_repeats=("repeat", "nunique"),
        )
        .sort_values(["table", "acc_mean"], ascending=[True, False])
    )
    for col in ("acc_std", "f1_std", "auc_std"):
        g[col] = g[col].fillna(0.0)

    # Advantage of QSVM-ZZ vs best classical *within the same table*.
    g["delta_acc_vs_best_classical"] = np.nan
    for table, idx in g.groupby("table").groups.items():
        block = g.loc[idx]
        classical = block[~block["model"].str.startswith("QSVM")]
        qsvm = block[block["model"] == "QSVM-ZZ"]
        if classical.empty or qsvm.empty:
            continue
        best_cl = float(classical["acc_mean"].max())
        g.loc[qsvm.index, "delta_acc_vs_best_classical"] = (
            float(qsvm["acc_mean"].iloc[0]) - best_cl
        )
    return g


def advantage_report(summary: pd.DataFrame) -> str:
    """Short human-readable summary for one dataset."""
    lines: list[str] = []
    for table, block in summary.groupby("table", sort=True):
        q = block[block["model"] == "QSVM-ZZ"]
        cl = block[~block["model"].str.startswith("QSVM")]
        if q.empty or cl.empty:
            continue
        best = cl.loc[cl["acc_mean"].idxmax()]
        qrow = q.iloc[0]
        delta = float(qrow["acc_mean"]) - float(best["acc_mean"])
        # Require both mean gap and that it exceeds ~1 std of QSVM noise.
        q_std = float(qrow["acc_std"]) if not pd.isna(qrow["acc_std"]) else 0.0
        if delta > max(0.01, q_std):
            winner = "QSVM-ZZ"
        elif delta < -max(0.01, q_std):
            winner = "classical"
        else:
            winner = "tie (within noise)"
        collapse = ""
        if float(qrow["f1_mean"]) < 0.45 and float(qrow["acc_mean"]) > 0.65:
            collapse = "  [warn: possible majority collapse — check F1/AUC]"
        lines.append(
            f"{table}: QSVM {qrow['acc_mean']:.3f}±{qrow['acc_std']:.3f} "
            f"(F1={qrow['f1_mean']:.3f}, AUC={qrow['auc_mean']:.3f}) vs "
            f"best classical {best['model']} {best['acc_mean']:.3f}±{best['acc_std']:.3f} "
            f"(Δ={delta:+.3f}) → {winner}{collapse}"
        )

    # Business question: Table A QSVM vs Table B best classical
    q_a = summary[
        (summary["table"] == "A_matched_features") & (summary["model"] == "QSVM-ZZ")
    ]
    cl_b = summary[
        (summary["table"] == "B_matched_n_full_features")
        & (~summary["model"].str.startswith("QSVM"))
    ]
    if not q_a.empty and not cl_b.empty:
        best_b = cl_b.loc[cl_b["acc_mean"].idxmax()]
        qa = q_a.iloc[0]
        d_biz = float(qa["acc_mean"]) - float(best_b["acc_mean"])
        lines.append(
            f"business (A-QSVM vs B-classical): QSVM {qa['acc_mean']:.3f} vs "
            f"{best_b['model']} {best_b['acc_mean']:.3f} (Δ={d_biz:+.3f})"
        )
    return "\n".join(lines)


def run_all(
    cfg: FairConfig | None = None,
    datasets: tuple[str, ...] = ("quantum_native", "telco", "skg"),
    quick: bool = False,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or FairConfig()
    if quick:
        cfg = FairConfig(
            n_qubits=cfg.n_qubits,
            zz_reps=1,
            n_total=40,
            n_repeats=1,
            random_state=cfg.random_state,
            class_weight=cfg.class_weight,
        )
    out: dict[str, pd.DataFrame] = {}
    jobs: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    if "quantum_native" in datasets:
        X, y = make_quantum_native(
            n_samples=cfg.n_total,
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

    metrics_dir = repo_paths()["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Fair classification results",
        "",
        "Generated by `python -m qml_v2.classification`.",
        "Table A = same features for QSVM and classical. "
        "Table B = classical keeps full features (matched n only).",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
    ]

    for name, X, y, native in jobs:
        print(f"\n=== {name}  n={len(y)}  features={X.shape[1]}  native={native} ===")
        rows = run_fair_dataset(
            name, X, y, cfg, already_zz_space=native, include_table_b=not native
        )
        raw = pd.DataFrame([asdict(r) for r in rows])
        raw.insert(0, "dataset", name)
        raw.to_csv(metrics_dir / f"fair_classification_{name}_raw.csv", index=False)

        summary = summarize(rows)
        summary.insert(0, "dataset", name)
        out[name] = summary
        summary.to_csv(metrics_dir / f"fair_classification_{name}.csv", index=False)
        print(summary.to_string(index=False))
        adv = advantage_report(summary)
        print(adv)
        report_lines.append(f"## {name}")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append(adv)
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("```")
        report_lines.append(summary.to_string(index=False))
        report_lines.append("```")
        report_lines.append("")

    if out:
        pd.concat(out.values(), ignore_index=True).to_csv(
            metrics_dir / "fair_classification_all.csv", index=False
        )
        (metrics_dir / "fair_classification_config.json").write_text(
            json.dumps(asdict(cfg), indent=2)
        )
        report_path = REPO_ROOT / "BUSINESS" / "classification_fair_results.md"
        report_path.write_text("\n".join(report_lines) + "\n")
        print(f"\nWrote {report_path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fair QSVM vs classical classification")
    p.add_argument("--quick", action="store_true", help="1 repeat, n=40, reps=1")
    p.add_argument("--n-total", type=int, default=None, help="Override sample size")
    p.add_argument("--n-repeats", type=int, default=None, help="Override repeats")
    p.add_argument("--n-qubits", type=int, default=None, help="ZZFeatureMap qubits")
    p.add_argument("--zz-reps", type=int, default=None, help="ZZFeatureMap reps")
    p.add_argument(
        "--class-weight",
        default=None,
        help="SVM/LogReg class_weight (default balanced; use none to disable)",
    )
    p.add_argument(
        "--datasets",
        default="quantum_native,telco,skg",
        help="Comma-separated: quantum_native,telco,skg",
    )
    args = p.parse_args()
    names = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    cfg = FairConfig()
    if args.n_total is not None:
        cfg.n_total = args.n_total
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    if args.n_qubits is not None:
        cfg.n_qubits = args.n_qubits
    if args.zz_reps is not None:
        cfg.zz_reps = args.zz_reps
    if args.class_weight is not None:
        cfg.class_weight = None if args.class_weight.lower() in ("none", "null") else args.class_weight
    run_all(cfg=cfg, datasets=names, quick=args.quick)


if __name__ == "__main__":
    main()
