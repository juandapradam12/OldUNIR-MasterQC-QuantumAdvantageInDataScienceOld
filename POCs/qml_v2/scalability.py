"""Escalabilidad de la ventaja del QSVM con repeticiones (dataset cuántico-nativo).

El Notebook 01 mide una única partición por tamaño de entrenamiento. Este módulo
repite el experimento con ``N_REPEATS`` semillas: en cada repetición se regenera el
dataset nativo (etiqueta = signo de ⟨ZZ⟩ bajo el ZZFeatureMap de 2 qubits, reps=2),
se reserva el 30 % como test y se entrena con los primeros ``n`` ejemplos del resto.

El kernel cuántico es la fidelidad exacta |⟨φ(x)|φ(x')⟩|², calculada con vectores de
estado de Qiskit; ``verificar_kernel`` lo contrasta con ``FidelityQuantumKernel``.

Salidas en ``POCs/data/output/metrics``:

* ``fair_scalability_native_raw.csv``  una fila por (repetición, tamaño, modelo)
* ``fair_scalability_native.csv``      media, desviación, IC 95 % y recuento de victorias

Uso, desde ``POCs/``::  python -m qml_v2.scalability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qml_v2.classification import make_quantum_native
from qml_v2.kernels import zz_fidelity_kernel, zz_map
from qml_v2.outputs import repo_paths

SIZES = (30, 50, 80, 100, 140, 180)
N_TOTAL = 300
N_REPEATS = 10
BASE_SEED = 42


def estados_zz(X: np.ndarray, n_qubits: int = 2, reps: int = 2) -> np.ndarray:
    """Vectores de estado |φ(x)⟩ (una fila por muestra) del ZZFeatureMap."""
    from qiskit.quantum_info import Statevector

    mapa = zz_map(n_qubits, reps, "full")
    return np.array([Statevector(mapa.assign_parameters(dict(zip(mapa.parameters, x)))).data for x in X])


def kernel_fidelidad(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matriz |⟨a|b⟩|² entre dos conjuntos de estados."""
    return np.abs(A.conj() @ B.T) ** 2


def verificar_kernel(n: int = 6, seed: int = BASE_SEED) -> float:
    """Máxima diferencia con ``FidelityQuantumKernel`` sobre ``n`` puntos."""
    X = np.random.RandomState(seed).uniform(-np.pi, np.pi, (n, 2))
    esperado = zz_fidelity_kernel(X, 2, 2, "full", seed)
    obtenido = kernel_fidelidad(estados_zz(X), estados_zz(X)).T
    return float(np.abs(esperado - obtenido).max())


def una_repeticion(repeticion: int, tamanos=SIZES) -> list[dict]:
    seed = BASE_SEED + repeticion
    X, y = make_quantum_native(N_TOTAL, random_state=seed)
    X = MinMaxScaler((-np.pi, np.pi)).fit_transform(X)
    X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    S_pool, S_te = estados_zz(X_pool), estados_zz(X_te)
    K_pool = kernel_fidelidad(S_pool, S_pool)
    K_te = kernel_fidelidad(S_te, S_pool)

    filas = []
    for n in tamanos:
        modelos = {
            "QSVM-ZZ": (
                SVC(kernel="precomputed", C=1.0, random_state=seed).fit(K_pool[:n, :n], y_pool[:n]),
                K_te[:, :n],
            ),
            "SVM-RBF": (SVC(kernel="rbf", C=1.0, random_state=seed).fit(X_pool[:n], y_pool[:n]), X_te),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=seed).fit(X_pool[:n], y_pool[:n]), X_te),
        }
        for nombre, (modelo, entrada) in modelos.items():
            filas.append(
                {
                    "repeticion": repeticion,
                    "n_train": n,
                    "modelo": nombre,
                    "n_test": len(y_te),
                    "accuracy": float((modelo.predict(entrada) == y_te).mean()),
                }
            )
    return filas


def resumir(raw: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación e IC 95 % (t de Student) de la diferencia QSVM − mejor clásico por tamaño."""
    ancho = raw.pivot_table(index=["n_train", "repeticion"], columns="modelo", values="accuracy")
    filas = []
    for n, bloque in ancho.groupby(level="n_train"):
        clasico = bloque[["SVM-RBF", "Gradient Boosting"]].max(axis=1)
        delta = 100 * (bloque["QSVM-ZZ"] - clasico)
        delta_rbf = 100 * (bloque["QSVM-ZZ"] - bloque["SVM-RBF"])
        k = len(delta)
        semi = stats.t.ppf(0.975, k - 1) * delta.std(ddof=1) / np.sqrt(k)
        filas.append(
            {
                "n_train": int(n),
                "n_repeticiones": k,
                "qsvm_media": 100 * bloque["QSVM-ZZ"].mean(),
                "qsvm_std": 100 * bloque["QSVM-ZZ"].std(ddof=1),
                "rbf_media": 100 * bloque["SVM-RBF"].mean(),
                "rbf_std": 100 * bloque["SVM-RBF"].std(ddof=1),
                "gb_media": 100 * bloque["Gradient Boosting"].mean(),
                "gb_std": 100 * bloque["Gradient Boosting"].std(ddof=1),
                "delta_vs_mejor_clasico_pp": delta.mean(),
                "delta_ic95_inf": delta.mean() - semi,
                "delta_ic95_sup": delta.mean() + semi,
                "delta_vs_rbf_pp": delta_rbf.mean(),
                "repeticiones_qsvm_gana": int((delta > 0).sum()),
            }
        )
    return pd.DataFrame(filas)


def main() -> None:
    metrics = repo_paths()["metrics"]
    print(f"Máx. diferencia kernel exacto vs FidelityQuantumKernel: {verificar_kernel():.2e}")
    filas = []
    for r in range(N_REPEATS):
        filas.extend(una_repeticion(r))
        print(f"repetición {r + 1}/{N_REPEATS}")
    raw = pd.DataFrame(filas)
    raw.to_csv(metrics / "fair_scalability_native_raw.csv", index=False)
    resumen = resumir(raw)
    resumen.to_csv(metrics / "fair_scalability_native.csv", index=False)
    print(resumen.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
