#!/usr/bin/env python3
"""Mide las MISMAS métricas de estructura cuántica (ΔA, d_F, ΔH) por caso de uso.

Protocolo único (qml_v2.quantumness, sin ventaja supervisada):
  - ZZFeatureMap 2 qubits, reps=2, entanglement=full
  - n_sample=60, n_repeats=3
  - baseline clásico = RBF Frobenius-best vs K_Q
  - ΔA = A(K_Q, yyᵀ) − A(K_RBF, yyᵀ)  (alineación centrada)
  - d_F = ‖K_Q − K_RBF‖_F / n  (kernels normalizados Frobenius)
  - ΔH = H(K_Q) − H(K_RBF)

No inventa cifras: si falla un dataset, se omite.
Salida: POCs/data/output/metrics/estructura_por_caso_uso.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

POCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POCS))

from qml_v2.anomaly import (  # noqa: E402
    inject_feature_outliers,
    make_moons_anomaly,
    make_zz_phase_anomaly,
)
from qml_v2.classification import (  # noqa: E402
    encode_for_zz_kernel,
    load_skg_incidents,
    load_telco,
    make_quantum_native,
    subsample_stratified,
)
from qml_v2.quantumness import (  # noqa: E402
    QuantumnessConfig,
    _indicators_on_matrix,
)
from qml_v2.regression import (  # noqa: E402
    load_friedman,
    load_skg_speeds,
    make_quantum_native_regression,
)

OUT = POCS / "data" / "output" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

CFG = QuantumnessConfig(
    n_qubits=2,
    zz_reps=2,
    entanglement="full",
    n_sample=60,
    n_repeats=3,
    random_state=42,
)


def _subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Estratificado si y es categórica; shuffle simple si es continua."""
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


def _structure_one(
    name: str,
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    already_zz_space: bool,
) -> dict:
    """Media de ΔA, d_F, ΔH en n_repeats (sin advantage)."""
    y = np.asarray(y)
    # Alineación con y continua: estandarizar para escala comparable
    if y.dtype.kind == "f" or (len(np.unique(y)) > 8 and y.dtype.kind in "iuf"):
        y = (y - float(np.mean(y))) / (float(np.std(y)) + 1e-12)

    rows = []
    t0 = time.time()
    for r in range(CFG.n_repeats):
        seed = CFG.random_state + 17 * r
        Xs, ys = _subsample(X, y, CFG.n_sample, seed)
        Xq, _, pca_var = encode_for_zz_kernel(
            Xs, Xs,
            n_qubits=CFG.n_qubits,
            random_state=seed,
            already_zz_space=already_zz_space,
        )
        ind = _indicators_on_matrix(
            Xq, ys, CFG, seed=seed, compute_advantage=False
        )
        ind["pca_var"] = float(pca_var)
        rows.append(ind)
    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    out = {
        "family": family,
        "dataset": name,
        "n_sample": CFG.n_sample,
        "n_repeats": CFG.n_repeats,
        "n_qubits": CFG.n_qubits,
        "delta_A": float(df["delta_A"].mean()),
        "delta_A_std": float(df["delta_A"].std(ddof=0)),
        "d_F": float(df["d_F"].mean()),
        "d_F_std": float(df["d_F"].std(ddof=0)),
        "delta_H": float(df["delta_H"].mean()),
        "delta_H_std": float(df["delta_H"].std(ddof=0)),
        "g": float(df["g"].mean()),
        "g_std": float(df["g"].std(ddof=0)),
        "H_Q": float(df["H_Q"].mean()),
        "H_cl": float(df["H_cl"].mean()),
        "A_Q": float(df["A_Q"].mean()),
        "A_cl": float(df["A_cl"].mean()),
        "pca_var": float(df["pca_var"].mean()),
        "seconds": round(elapsed, 1),
    }
    print(
        f"  [{family:12s}] {name:16s}  "
        f"ΔA={out['delta_A']:+.4f}  d_F={out['d_F']:.4f}  ΔH={out['delta_H']:+.4f}  "
        f"({elapsed:.1f}s)"
    )
    return out


def load_jobs() -> list[tuple[str, str, np.ndarray, np.ndarray, bool]]:
    """(family, dataset, X, y, already_zz_space)."""
    jobs: list[tuple[str, str, np.ndarray, np.ndarray, bool]] = []

    # --- Clasificación (mismos tres del TFM) ---
    X, y = make_quantum_native(n_samples=200, n_qubits=2, reps=2, random_state=42)
    jobs.append(("clasificacion", "Nativo", X, y, True))
    X, y, _ = load_skg_incidents()
    jobs.append(("clasificacion", "SKG", X, y, False))
    X, y, _ = load_telco()
    jobs.append(("clasificacion", "Telco", X, y, False))

    # --- Regresión ---
    X, y = make_quantum_native_regression(n_samples=200, n_qubits=2, reps=2, random_state=42)
    jobs.append(("regresion", "Nativo", X, y, True))
    X, y = load_friedman(n_samples=200, n_features=5, random_state=42)
    jobs.append(("regresion", "Friedman", X, y, False))
    try:
        X, y, _ = load_skg_speeds()
        jobs.append(("regresion", "SKG", X, y, False))
    except FileNotFoundError as e:
        print(f"  WARN regresión SKG omitido: {e}")

    # --- Anomalías ---
    X, y = make_zz_phase_anomaly(n_samples=400, n_total=160, random_state=42)
    jobs.append(("anomalias", "zz_phase", X, y, True))
    try:
        X0, _, _ = load_skg_incidents()
        X, y = inject_feature_outliers(X0, contamination=0.10, random_state=42)
        # recortar a ~160
        if len(y) > 160:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y), size=160, replace=False)
            X, y = X[idx], y[idx]
        jobs.append(("anomalias", "SKG", X, y, False))
    except Exception as e:
        print(f"  WARN anomalías SKG omitido: {e}")
    X, y = make_moons_anomaly(n_total=160, random_state=42)
    jobs.append(("anomalias", "moons", X, y, False))

    return jobs


def main() -> None:
    print("Midiendo estructura cuántica (ΔA, d_F, ΔH) — mismo protocolo por familia")
    print(f"  n_sample={CFG.n_sample}, n_repeats={CFG.n_repeats}, qubits={CFG.n_qubits}")
    rows = []
    for family, name, X, y, native in load_jobs():
        try:
            rows.append(_structure_one(name, family, X, y, already_zz_space=native))
        except Exception as e:
            print(f"  ERROR [{family}] {name}: {e}")
    if not rows:
        raise SystemExit("No se pudo medir ningún dataset.")
    df = pd.DataFrame(rows)
    path = OUT / "estructura_por_caso_uso.csv"
    df.to_csv(path, index=False)
    print(f"\nGuardado: {path}")
    print(df[["family", "dataset", "delta_A", "d_F", "delta_H"]].to_string(index=False))


if __name__ == "__main__":
    main()
