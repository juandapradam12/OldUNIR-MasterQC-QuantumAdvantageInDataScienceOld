"""Time-series forecasting — QML v2 fair lag protocol.

One-step-ahead prediction from a lag window. This is **not** a full ARIMA /
Prophet bake-off: the matched question is whether a ZZ kernel on the lag
vector beats classical regressors on the **same lags**.

* **Table A — matched lags.** Lag vector of length ``n_qubits`` (ZZ-ready
  angles) for QK-SVR vs Ridge / SVR-RBF / GradientBoosting.
* **Table B — matched n, richer classical features.** Same samples; classical
  models also see calendar / extra lags (hour, lag mean, …).

Splits are **time-ordered** (train = past, test = future) within each repeat
window — no ShuffleSplit leakage.

SKG speeds are practice data only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from qml_v2.classification import REPO_ROOT, encode_for_zz_kernel, repo_paths
from qml_v2.regression import QuantumKernelSVR, make_quantum_native_regression


@dataclass
class TSConfig:
    n_qubits: int = 2
    n_lags: int = 2  # Table A lag width (= n_qubits typically)
    n_extra_lags: int = 4  # Table B additional classical lags
    zz_reps: int = 2
    entanglement: str = "full"
    svr_C: float = 10.0
    svr_epsilon: float = 0.1
    n_total: int = 160
    test_size: float = 0.3
    n_repeats: int = 5
    random_state: int = 42
    n_estimators: int = 100


@dataclass
class TSRepeatResult:
    repeat: int
    model: str
    table: str
    r2: float
    rmse: float
    mae: float
    n_train: int
    n_test: int
    n_features: int


def make_lag_matrix(
    y: np.ndarray, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rows: [y_{t-n_lags}, ..., y_{t-1}] → target y_t."""
    y = np.asarray(y, dtype=float).ravel()
    if len(y) <= n_lags:
        raise ValueError("series shorter than n_lags")
    X, t = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags : i])
        t.append(y[i])
    return np.asarray(X, dtype=float), np.asarray(t, dtype=float)


def make_zz_lag_series(
    n_samples: int,
    n_qubits: int = 2,
    reps: int = 2,
    entanglement: str = "full",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Windows whose target is the ZZ expectation of the lag vector.

    Equivalent to quantum-native regression on lag features already in angle
    space — the series framing is for the forecasting protocol.
    """
    return make_quantum_native_regression(
        n_samples=n_samples,
        n_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
        random_state=random_state,
    )


def make_sine_series(
    n_samples: int = 400, noise: float = 0.15, random_state: int = 42
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    t = np.arange(n_samples)
    return np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 7) + noise * rng.normal(
        size=n_samples
    )


def load_skg_speed_series(
    path: Path | None = None,
    tid: int | None = None,
    max_points: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (speed series, hour series) for one corridor (practice data)."""
    path = path or (REPO_ROOT / "Datos" / "bctraveltimes_history.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["read_time"] = pd.to_datetime(df["read_time"])
    if tid is None:
        # pick a frequent corridor deterministically
        counts = df["tid"].value_counts()
        tid = int(counts.index[0])
    sub = df[df["tid"] == tid].sort_values("read_time")
    if len(sub) > max_points:
        # take a contiguous window starting at a seeded offset
        rng = np.random.RandomState(random_state)
        start = int(rng.randint(0, len(sub) - max_points + 1))
        sub = sub.iloc[start : start + max_points]
    speed = sub["speed"].astype(float).to_numpy()
    hour = sub["read_time"].dt.hour.astype(float).to_numpy()
    return speed, hour


def enrich_classical_features(
    X_lags: np.ndarray, hour: np.ndarray | None = None, extra_lags: np.ndarray | None = None
) -> np.ndarray:
    cols = [X_lags]
    if extra_lags is not None:
        cols.append(extra_lags)
    if hour is not None:
        cols.append(hour.reshape(-1, 1))
        # cyclic hour
        cols.append(np.sin(2 * np.pi * hour / 24).reshape(-1, 1))
        cols.append(np.cos(2 * np.pi * hour / 24).reshape(-1, 1))
    # lag mean / std as cheap classical extras
    cols.append(X_lags.mean(axis=1, keepdims=True))
    cols.append(X_lags.std(axis=1, keepdims=True) + 1e-9)
    return np.hstack(cols)


def time_split_indices(n: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    n_te = max(int(round(n * test_size)), 1)
    n_tr = n - n_te
    if n_tr < 8:
        raise ValueError("not enough train samples for time split")
    return np.arange(0, n_tr), np.arange(n_tr, n)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


def _classical_models(cfg: TSConfig, seed: int) -> dict[str, Any]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "SVR-RBF": SVR(kernel="rbf", C=cfg.svr_C, epsilon=cfg.svr_epsilon),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=cfg.n_estimators, random_state=seed
        ),
    }


def run_fair_dataset(
    name: str,
    X_a: np.ndarray,
    y: np.ndarray,
    cfg: TSConfig,
    *,
    already_zz_space: bool,
    X_b: np.ndarray | None = None,
    include_table_b: bool = True,
) -> list[TSRepeatResult]:
    """``X_a`` = Table A lags; ``X_b`` = richer classical features (optional)."""
    n = len(y)
    # subsample contiguous block for repeats
    rng = np.random.RandomState(cfg.random_state)
    rows: list[TSRepeatResult] = []
    for repeat in range(cfg.n_repeats):
        if n > cfg.n_total:
            start = int(rng.randint(0, n - cfg.n_total + 1))
            sl = slice(start, start + cfg.n_total)
        else:
            sl = slice(0, n)
        Xa = X_a[sl]
        ya = y[sl]
        Xb = X_b[sl] if X_b is not None else None
        tr, te = time_split_indices(len(ya), cfg.test_size)
        Xa_tr, Xa_te = Xa[tr], Xa[te]
        y_tr, y_te = ya[tr], ya[te]

        Xq_tr, Xq_te, _ = encode_for_zz_kernel(
            Xa_tr,
            Xa_te,
            cfg.n_qubits,
            cfg.random_state + repeat,
            already_zz_space=already_zz_space,
        )

        q = QuantumKernelSVR(
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            C=cfg.svr_C,
            epsilon=cfg.svr_epsilon,
            random_state=cfg.random_state + repeat,
        )
        q.fit(Xq_tr, y_tr)
        yp = q.predict(Xq_te)
        r2, rmse, mae = _scores(y_te, yp)
        rows.append(
            TSRepeatResult(
                repeat=repeat,
                model="QK-SVR-ZZ",
                table="A_matched_lags",
                r2=r2,
                rmse=rmse,
                mae=mae,
                n_train=len(y_tr),
                n_test=len(y_te),
                n_features=Xq_tr.shape[1],
            )
        )

        for model_name, model in _classical_models(cfg, cfg.random_state + repeat).items():
            model.fit(Xq_tr, y_tr)
            yp = model.predict(Xq_te)
            r2, rmse, mae = _scores(y_te, yp)
            rows.append(
                TSRepeatResult(
                    repeat=repeat,
                    model=model_name,
                    table="A_matched_lags",
                    r2=r2,
                    rmse=rmse,
                    mae=mae,
                    n_train=len(y_tr),
                    n_test=len(y_te),
                    n_features=Xq_tr.shape[1],
                )
            )

        if include_table_b and Xb is not None and not already_zz_space:
            scaler = StandardScaler()
            Xb_tr = scaler.fit_transform(Xb[tr])
            Xb_te = scaler.transform(Xb[te])
            for model_name, model in _classical_models(
                cfg, cfg.random_state + repeat
            ).items():
                model.fit(Xb_tr, y_tr)
                yp = model.predict(Xb_te)
                r2, rmse, mae = _scores(y_te, yp)
                rows.append(
                    TSRepeatResult(
                        repeat=repeat,
                        model=f"{model_name} [full features]",
                        table="B_matched_n_rich_features",
                        r2=r2,
                        rmse=rmse,
                        mae=mae,
                        n_train=len(y_tr),
                        n_test=len(y_te),
                        n_features=Xb_tr.shape[1],
                    )
                )
    return rows


def summarize(rows: list[TSRepeatResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    g = (
        df.groupby(["table", "model"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
            n_features=("n_features", "first"),
            n_repeats=("repeat", "nunique"),
        )
        .sort_values(["table", "mae_mean"], ascending=[True, True])
        .reset_index(drop=True)
    )
    g["delta_mae_vs_best_classical"] = np.nan
    for table, block in g.groupby("table"):
        classical = block[~block["model"].str.contains("ZZ", regex=False)]
        quantum = block[block["model"].str.contains("ZZ", regex=False)]
        if classical.empty or quantum.empty:
            continue
        best_cl = float(classical["mae_mean"].min())
        g.loc[quantum.index, "delta_mae_vs_best_classical"] = (
            quantum["mae_mean"].astype(float) - best_cl
        )
    return g


def advantage_report(summary: pd.DataFrame) -> str:
    lines: list[str] = []
    for table, block in summary.groupby("table", sort=True):
        q = block[block["model"].str.contains("ZZ", regex=False)]
        cl = block[~block["model"].str.contains("ZZ", regex=False)]
        if q.empty or cl.empty:
            continue
        best = cl.loc[cl["mae_mean"].idxmin()]
        qrow = q.iloc[0]
        delta = float(qrow["mae_mean"]) - float(best["mae_mean"])
        q_std = float(qrow["mae_std"]) if not pd.isna(qrow["mae_std"]) else 0.0
        # lower MAE is better
        if delta < -max(0.01, q_std):
            winner = "QK-SVR-ZZ"
        elif delta > max(0.01, q_std):
            winner = "classical"
        else:
            winner = "tie (within noise)"
        lines.append(
            f"{table}: QK-SVR MAE {qrow['mae_mean']:.3f}±{qrow['mae_std']:.3f} "
            f"(R²={qrow['r2_mean']:.3f}) vs best classical {best['model']} "
            f"{best['mae_mean']:.3f}±{best['mae_std']:.3f} (ΔMAE={delta:+.3f}) → {winner}"
        )
    return "\n".join(lines)


def _prepare_jobs(cfg: TSConfig, datasets: tuple[str, ...]):
    jobs = []
    if "zz_lag" in datasets:
        X, y = make_zz_lag_series(
            n_samples=max(cfg.n_total * 2, 240),
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            random_state=cfg.random_state,
        )
        # already angle features → Table A is X itself; no Table B
        jobs.append(("zz_lag", X, y, True, None))
    if "sine" in datasets:
        series = make_sine_series(
            n_samples=max(cfg.n_total * 3, 400), random_state=cfg.random_state
        )
        total_lags = cfg.n_lags + cfg.n_extra_lags
        X_full, y = make_lag_matrix(series, total_lags)
        X_a = X_full[:, -cfg.n_lags :]
        X_b = enrich_classical_features(X_a, hour=None, extra_lags=X_full[:, : cfg.n_extra_lags])
        jobs.append(("sine", X_a, y, False, X_b))
    if "skg_speed" in datasets:
        speed, hour = load_skg_speed_series(random_state=cfg.random_state)
        total_lags = cfg.n_lags + cfg.n_extra_lags
        # align hour with targets (drop first total_lags)
        X_full, y = make_lag_matrix(speed, total_lags)
        hour_t = hour[total_lags:]
        X_a = X_full[:, -cfg.n_lags :]
        X_b = enrich_classical_features(
            X_a, hour=hour_t, extra_lags=X_full[:, : cfg.n_extra_lags]
        )
        jobs.append(("skg_speed", X_a, y, False, X_b))
    return jobs


def run_all(
    cfg: TSConfig | None = None,
    datasets: tuple[str, ...] = ("zz_lag", "sine", "skg_speed"),
    quick: bool = False,
) -> pd.DataFrame:
    cfg = cfg or TSConfig()
    if quick:
        cfg = TSConfig(
            n_total=48,
            n_repeats=2,
            random_state=cfg.random_state,
            n_qubits=cfg.n_qubits,
            n_lags=cfg.n_lags,
            zz_reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
        )

    all_rows: list[TSRepeatResult] = []
    report = [
        "# Fair time-series forecasting results",
        "",
        "Generated by `python -m qml_v2.timeseries`.",
        "One-step ahead from lag windows; time-ordered splits. SKG = practice data.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
    ]

    for name, X_a, y, native, X_b in _prepare_jobs(cfg, datasets):
        print(f"\n=== {name} ===")
        rows = run_fair_dataset(
            name,
            X_a,
            y,
            cfg,
            already_zz_space=native,
            X_b=X_b,
            include_table_b=X_b is not None,
        )
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
            metrics / f"fair_timeseries_{name}_raw.csv", index=False
        )
        summary.to_csv(metrics / f"fair_timeseries_{name}.csv", index=False)

    out = pd.DataFrame([asdict(r) for r in all_rows])
    path = REPO_ROOT / "BUSINESS" / "timeseries_v2_results.md"
    path.write_text("\n".join(report) + "\n")
    (repo_paths()["metrics"] / "qml_v2_timeseries_config.json").write_text(
        json.dumps(asdict(cfg), indent=2)
    )
    print(f"\nWrote {path}")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fair lag-based time-series forecasting")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-total", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument(
        "--datasets",
        default="zz_lag,sine,skg_speed",
        help="Comma-separated: zz_lag,sine,skg_speed",
    )
    args = p.parse_args()
    cfg = TSConfig()
    if args.n_total is not None:
        cfg.n_total = args.n_total
    if args.n_repeats is not None:
        cfg.n_repeats = args.n_repeats
    names = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    run_all(cfg=cfg, datasets=names, quick=args.quick)


if __name__ == "__main__":
    main()
