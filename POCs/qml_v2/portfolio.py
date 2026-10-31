"""Caso 4 — Optimización de portafolio con VQE y QAOA (QML v2).

Selecciona exactamente ``B = 3`` de 6 acciones (EC, CIB, AAPL, MSFT, GOOGL, AMZN) minimizando
``q·xᵀΣx − μᵀx`` con ``q = 0.5``. Los precios son cierres ajustados de Yahoo Finance
(``data/output/input/portfolio_prices.csv``, 501 sesiones de 2022-01-03 a 2023-12-29);
``μ`` y ``Σ`` son la media y la covarianza anualizadas (252 días) de los log-retornos.

Protocolo (idéntico para VQE y QAOA, 10 semillas independientes):

* Configuración ``penalizacion_alta``: penalización 10, 5 reinicios COBYLA por semilla, 300 evaluaciones.
* Configuración ``penalizacion_baja``: penalización 0.01 (la del notebook original), 1 inicio, 200 evaluaciones.
* Cada ejecución se decodifica con la cadena más probable de 4096 disparos; solo cuenta como solución
  si es factible (suma = B) y se evalúa en el problema cuadrático original (no en la energía penalizada).
* Referencias clásicas: fuerza bruta (óptimo exacto) y selección voraz hacia delante (Greedy).

Los tiempos (``tiempo_s``) son orientativos: se midieron con tres procesos en paralelo sobre una máquina compartida.

Salidas en ``POCs/data/output/metrics``: ``qml_v2_portfolio_raw.csv`` (una fila por semilla, método y
configuración), ``qml_v2_portfolio_summary.csv`` y ``qml_v2_portfolio_reference.json``.

Uso:  python -m qml_v2.portfolio
"""
from __future__ import annotations

import itertools
import json
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "data" / "output" / "metrics"
PRICES = REPO / "data" / "output" / "input" / "portfolio_prices.csv"

TICKERS = ["EC", "CIB", "AAPL", "MSFT", "GOOGL", "AMZN"]
BUDGET = 3
RISK = 0.5
TRADING_DAYS = 252
SEEDS = list(range(10))
SHOTS = 4096
QAOA_REPS = 3
CONFIGS = {
    "penalizacion_alta": {"penalty": 10.0, "restarts": 5, "maxiter": 300},
    "penalizacion_baja": {"penalty": 0.01, "restarts": 1, "maxiter": 200},
}


def cargar_problema(ruta: Path = PRICES):
    """Devuelve (mu, sigma, QuadraticProgram) a partir de los precios guardados."""
    from qiskit_finance.applications.optimization import PortfolioOptimization

    precios = pd.read_csv(ruta, index_col=0, parse_dates=True)[TICKERS]
    retornos = np.log(precios / precios.shift(1)).dropna()
    mu = retornos.mean().values * TRADING_DAYS
    sigma = retornos.cov().values * TRADING_DAYS
    qp = PortfolioOptimization(
        expected_returns=mu, covariances=sigma, risk_factor=RISK, budget=BUDGET
    ).to_quadratic_program()
    return mu, sigma, qp


def _bits(texto: str) -> np.ndarray:
    """Cadena de Qiskit (qubit 0 a la derecha) -> vector x con x[i] = activo i."""
    return np.array([int(b) for b in texto[::-1]])


def fuerza_bruta(qp) -> tuple[float, np.ndarray]:
    """Óptimo exacto entre las C(6,3) = 20 selecciones factibles."""
    n = len(TICKERS)
    mejor = (np.inf, None)
    for combinacion in itertools.combinations(range(n), BUDGET):
        x = np.zeros(n, dtype=int)
        x[list(combinacion)] = 1
        coste = qp.objective.evaluate(x)
        if coste < mejor[0]:
            mejor = (coste, x)
    return float(mejor[0]), mejor[1]


def voraz(qp) -> tuple[float, np.ndarray]:
    """Selección hacia delante: añade en cada paso el activo que menos aumenta el coste."""
    n = len(TICKERS)
    x = np.zeros(n, dtype=int)
    for _ in range(BUDGET):
        candidatos = []
        for i in np.flatnonzero(x == 0):
            y = x.copy()
            y[i] = 1
            candidatos.append((qp.objective.evaluate(y), i))
        x[min(candidatos)[1]] = 1
    return float(qp.objective.evaluate(x)), x


def estado_fundamental(qp, penalizacion: float) -> tuple[str, float]:
    """Estado fundamental exacto del Hamiltoniano de Ising (diagonalización) y su energía."""
    from qiskit_optimization.converters import QuadraticProgramToQubo

    hamiltoniano, offset = QuadraticProgramToQubo(penalty=penalizacion).convert(qp).to_ising()
    valores, vectores = np.linalg.eigh(hamiltoniano.to_matrix())
    indice = int(np.argmax(np.abs(vectores[:, 0])))
    return format(indice, f"0{len(TICKERS)}b"), float(valores[0] + offset)


def _ejecutar(args: tuple) -> dict:
    """Una semilla, un método y una configuración."""
    metodo, config, semilla = args
    warnings.filterwarnings("ignore")
    from qiskit.circuit.library import QAOAAnsatz, TwoLocal
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    from qiskit_optimization.converters import QuadraticProgramToQubo

    _, _, qp = cargar_problema()
    parametros = CONFIGS[config]
    hamiltoniano, _ = QuadraticProgramToQubo(penalty=parametros["penalty"]).convert(qp).to_ising()
    estimador = StatevectorEstimator()
    muestreador = StatevectorSampler(seed=semilla)
    n = len(TICKERS)

    inicio = time.perf_counter()
    mejor_coste, mejor_bits, factibles = np.inf, "", 0
    for reinicio in range(parametros["restarts"]):
        rng = np.random.default_rng(1000 * semilla + reinicio)
        if metodo == "VQE":
            ansatz = TwoLocal(n, "ry", "cz", reps=3, entanglement="full")
        else:
            ansatz = QAOAAnsatz(cost_operator=hamiltoniano, reps=QAOA_REPS)
        theta0 = rng.random(ansatz.num_parameters)

        def energia(theta, a=ansatz):
            return float(estimador.run([(a, [hamiltoniano], [theta])]).result()[0].data.evs[0])

        resultado = minimize(energia, theta0, method="COBYLA", options={"maxiter": parametros["maxiter"]})
        medido = ansatz.copy()
        medido.measure_all()
        cuentas = muestreador.run([(medido, [resultado.x])], shots=SHOTS).result()[0].data.meas.get_counts()
        modal = max(cuentas, key=cuentas.get)
        x = _bits(modal)
        if qp.is_feasible(x):
            factibles += 1
            coste = float(qp.objective.evaluate(x))
            if coste < mejor_coste:
                mejor_coste, mejor_bits = coste, modal
    return {
        "metodo": metodo, "configuracion": config, "semilla": semilla,
        "reinicios": parametros["restarts"], "reinicios_factibles": factibles,
        "factible": factibles > 0,
        "coste": mejor_coste if factibles else np.nan,
        "activos": ", ".join(TICKERS[i] for i in np.flatnonzero(_bits(mejor_bits))) if factibles else "",
        "tiempo_s": round(time.perf_counter() - inicio, 2),
    }


def ejecutar(workers: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _, _, qp = cargar_problema()
    optimo, x_opt = fuerza_bruta(qp)
    coste_voraz, x_voraz = voraz(qp)
    bits_gs, energia_gs = estado_fundamental(qp, CONFIGS["penalizacion_alta"]["penalty"])
    referencia = {
        "optimo_exacto": optimo,
        "activos_optimo": [TICKERS[i] for i in np.flatnonzero(x_opt)],
        "coste_voraz": coste_voraz,
        "activos_voraz": [TICKERS[i] for i in np.flatnonzero(x_voraz)],
        "estado_fundamental_penalizacion_10": {
            "activos": [TICKERS[i] for i in np.flatnonzero(_bits(bits_gs))],
            "energia_qp": energia_gs,
        },
        "semillas": SEEDS, "disparos": SHOTS, "qaoa_p": QAOA_REPS, "configuraciones": CONFIGS,
    }
    tareas = [(m, c, s) for c in CONFIGS for m in ("VQE", "QAOA") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        filas = list(pool.map(_ejecutar, tareas))
    crudo = pd.DataFrame(filas)
    crudo["brecha_pct"] = 100 * (crudo["coste"] - optimo) / abs(optimo)
    crudo["optimo"] = np.isclose(crudo["coste"], optimo, atol=1e-6)

    resumen = (
        crudo.groupby(["configuracion", "metodo"])
        .agg(
            semillas=("semilla", "count"),
            factibles=("factible", "sum"),
            optimo_encontrado=("optimo", "sum"),
            brecha_media_pct=("brecha_pct", "mean"),
            brecha_mediana_pct=("brecha_pct", "median"),
            brecha_max_pct=("brecha_pct", "max"),
            coste_medio=("coste", "mean"),
            tiempo_medio_s=("tiempo_s", "mean"),
        )
        .reset_index()
    )
    return crudo, resumen, referencia


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    crudo, resumen, referencia = ejecutar()
    crudo.to_csv(METRICS / "qml_v2_portfolio_raw.csv", index=False)
    resumen.to_csv(METRICS / "qml_v2_portfolio_summary.csv", index=False)
    (METRICS / "qml_v2_portfolio_reference.json").write_text(
        json.dumps(referencia, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(referencia, indent=2, ensure_ascii=False))
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
