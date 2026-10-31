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

SKG speeds are practice data only: each repeat uses a different corridor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from qml_v2.classification import encode_for_zz_kernel
from qml_v2.outputs import (
    REPO_ROOT,
    apply_overrides,
    dataset_names,
    finish_run,
    make_parser,
    metrics_dir,
    record_dataset,
    report_header,
)
from qml_v2.regression import QuantumKernelSVR, RepeatScores, make_quantum_native_regression, regression_scores


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


def make_lag_matrix(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows: [y_{t-n_lags}, ..., y_{t-1}] → target y_t."""
    y = np.asarray(y, dtype=float).ravel()
    if len(y) <= n_lags:
        raise ValueError("series shorter than n_lags")
    X, t = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags : i])
        t.append(y[i])
    return np.asarray(X, dtype=float), np.asarray(t, dtype=float)


def make_sine_series(n_samples: int = 400, noise: float = 0.15, random_state: int = 42) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    t = np.arange(n_samples)
    return np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 7) + noise * rng.normal(size=n_samples)


def load_skg_corridors(n: int, path: Path | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """(speed, hour) series of the ``n`` corridors with most readings (practice data).

    Each corridor has a single 5-minute series (144 readings, 12 hours), so repeats
    use one corridor each instead of resampling windows of the same series.
    """
    path = path or (REPO_ROOT / "Datos" / "bctraveltimes_history.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["read_time"] = pd.to_datetime(df["read_time"])
    counts = df["tid"].value_counts().sort_index().sort_values(ascending=False, kind="stable")
    series = []
    for tid in counts.index[:n]:
        sub = df[df["tid"] == tid].sort_values("read_time")
        series.append((sub["speed"].astype(float).to_numpy(), sub["read_time"].dt.hour.astype(float).to_numpy()))
    return series


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


def _classical_models(cfg: TSConfig, seed: int) -> dict[str, Any]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "SVR-RBF": SVR(kernel="rbf", C=cfg.svr_C, epsilon=cfg.svr_epsilon),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=cfg.n_estimators, random_state=seed),
    }


Block = tuple[np.ndarray, np.ndarray, np.ndarray | None]


def repeat_windows(blocks: list[Block], cfg: TSConfig):
    """Yield ``(repeat, Table A lags, target, Table B features or None)`` for every repeat.

    Repeat ``r`` uses ``blocks[r % len(blocks)]``; a block longer than ``n_total``
    contributes a seeded contiguous window.
    """
    rng = np.random.RandomState(cfg.random_state)
    for repeat in range(cfg.n_repeats):
        Xa, ya, Xb = blocks[repeat % len(blocks)]
        if len(ya) > cfg.n_total:
            start = int(rng.randint(0, len(ya) - cfg.n_total + 1))
            sl = slice(start, start + cfg.n_total)
            Xa, ya = Xa[sl], ya[sl]
            Xb = Xb[sl] if Xb is not None else None
        yield repeat, Xa, ya, Xb


def naive_references(blocks: list[Block], cfg: TSConfig) -> pd.DataFrame:
    """MAE of persistence (``y_{t-1}``) and of the training mean on the same test windows."""
    rows = []
    for repeat, Xa, ya, _ in repeat_windows(blocks, cfg):
        tr, te = time_split_indices(len(ya), cfg.test_size)
        rows.append(
            {
                "repeat": repeat,
                "persistence": float(np.mean(np.abs(ya[te] - Xa[te, -1]))),
                "train_mean": float(np.mean(np.abs(ya[te] - ya[tr].mean()))),
            }
        )
    return pd.DataFrame(rows)


def run_fair_dataset(blocks: list[Block], cfg: TSConfig, *, already_zz_space: bool) -> list[RepeatScores]:
    rows: list[RepeatScores] = []
    for repeat, Xa, ya, Xb in repeat_windows(blocks, cfg):
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
        r2, rmse, mae = regression_scores(y_te, yp)
        rows.append(
            RepeatScores(
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
            r2, rmse, mae = regression_scores(y_te, yp)
            rows.append(
                RepeatScores(
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

        if Xb is not None:
            scaler = StandardScaler()
            Xb_tr = scaler.fit_transform(Xb[tr])
            Xb_te = scaler.transform(Xb[te])
            for model_name, model in _classical_models(cfg, cfg.random_state + repeat).items():
                model.fit(Xb_tr, y_tr)
                yp = model.predict(Xb_te)
                r2, rmse, mae = regression_scores(y_te, yp)
                rows.append(
                    RepeatScores(
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


def summarize(rows: list[RepeatScores]) -> pd.DataFrame:
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
    for _table, block in g.groupby("table"):
        classical = block[~block["model"].str.contains("ZZ", regex=False)]
        quantum = block[block["model"].str.contains("ZZ", regex=False)]
        if classical.empty or quantum.empty:
            continue
        best_cl = float(classical["mae_mean"].min())
        g.loc[quantum.index, "delta_mae_vs_best_classical"] = quantum["mae_mean"].astype(float) - best_cl
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


def lag_block(series: np.ndarray, cfg: TSConfig, hour: np.ndarray | None = None) -> Block:
    """Table A lags, one-step target and Table B classical features for one series."""
    total_lags = cfg.n_lags + cfg.n_extra_lags
    X_full, y = make_lag_matrix(series, total_lags)
    X_a = X_full[:, -cfg.n_lags :]
    hour_t = None if hour is None else hour[total_lags:]
    X_b = enrich_classical_features(X_a, hour=hour_t, extra_lags=X_full[:, : cfg.n_extra_lags])
    return X_a, y, X_b


def prepare_jobs(cfg: TSConfig, datasets: tuple[str, ...]):
    """(name, blocks, already_zz_space); the native ZZ series has no Table B."""
    jobs = []
    if "zz_lag" in datasets:
        X, y = make_quantum_native_regression(
            n_samples=max(cfg.n_total * 2, 240),
            n_qubits=cfg.n_qubits,
            reps=cfg.zz_reps,
            entanglement=cfg.entanglement,
            random_state=cfg.random_state,
        )
        jobs.append(("zz_lag", [(X, y, None)], True))
    if "sine" in datasets:
        series = make_sine_series(n_samples=max(cfg.n_total * 3, 400), random_state=cfg.random_state)
        jobs.append(("sine", [lag_block(series, cfg)], False))
    if "skg_speed" in datasets:
        corridors = load_skg_corridors(cfg.n_repeats)
        jobs.append(("skg_speed", [lag_block(speed, cfg, hour) for speed, hour in corridors], False))
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

    report = report_header(
        "Fair time-series forecasting results",
        "timeseries",
        [
            "One-step ahead from lag windows; time-ordered splits. SKG = practice data.",
        ],
        cfg,
    )
    all_rows: list[RepeatScores] = []
    references = []
    for name, blocks, native in prepare_jobs(cfg, datasets):
        print(f"\n=== {name} ===")
        rows = run_fair_dataset(blocks, cfg, already_zz_space=native)
        all_rows.extend(rows)
        record_dataset(name, rows, summarize, advantage_report, stem="fair_timeseries", report=report)
        stats = naive_references(blocks, cfg).drop(columns="repeat").agg(["mean", "std"]).T
        stats.columns = ["mae_mean", "mae_std"]
        stats.index.name = "reference"
        references.append(stats.reset_index().assign(dataset=name)[["dataset", "reference", "mae_mean", "mae_std"]])

    references_df = pd.concat(references, ignore_index=True)
    references_df.to_csv(metrics_dir() / "fair_timeseries_references.csv", index=False)
    report += [
        "## naive references (MAE, same test windows)",
        "",
        "```",
        references_df.to_string(index=False),
        "```",
        "",
    ]
    finish_run(report, cfg, report_file="timeseries_v2_results.md", config_file="qml_v2_timeseries_config.json")
    return pd.DataFrame([asdict(r) for r in all_rows])


FLAGS = ("n_total", "n_repeats")


def main() -> None:
    parser = make_parser("Fair lag-based time-series forecasting", "zz_lag,sine,skg_speed", FLAGS)
    args = parser.parse_args()
    run_all(cfg=apply_overrides(TSConfig(), args, FLAGS), datasets=dataset_names(args), quick=args.quick)


if __name__ == "__main__":
    main()
