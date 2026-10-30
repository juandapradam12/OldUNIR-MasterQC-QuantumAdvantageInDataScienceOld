"""Consolida los resultados de los nueve casos de uso en un único CSV trazable.

Cada fila de ``07_sintesis_resultados.csv`` se lee de una salida ejecutada
(notebook, CSV del protocolo QML v2 o JSON); ningún valor se escribe a mano.
Las figuras de síntesis (panel y mapa de calor) leen solo ese CSV.

Convención: ``valor`` es la ventaja del método cuántico sobre el mejor clásico
en la unidad indicada; positivo favorece al cuántico, negativo al clásico.

Uso, desde ``POCs/``::

    python scripts/sintesis_resultados.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

POCS = Path(__file__).resolve().parents[1]
NOTEBOOKS = POCS / "notebooks"
METRICS = POCS / "data" / "output" / "metrics"
SALIDA = METRICS / "07_sintesis_resultados.csv"

CAPAS = ("nativo", "skg", "referencia")
COLUMNAS = ["caso", "capa", "etiqueta", "valor", "unidad", "fuente"]


def _texto_salidas(nombre: str) -> str:
    """Concatena las salidas de texto de todas las celdas de un notebook."""
    nb = nbformat.read(NOTEBOOKS / nombre, as_version=4)
    trozos: list[str] = []
    for celda in nb.cells:
        if celda.cell_type != "code":
            continue
        for salida in celda.get("outputs", []):
            trozos.append(salida.get("text", ""))
    return "\n".join(trozos)


def _fila(caso: int, capa: str, etiqueta: str, valor: float, unidad: str, fuente: str) -> dict:
    return dict(caso=caso, capa=capa, etiqueta=etiqueta, valor=float(valor), unidad=unidad, fuente=fuente)


def _por_capa(tabla: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Empareja las tres filas de un resumen de datasets con sus capas."""
    return [(capa, fila) for capa, (_, fila) in zip(CAPAS, tabla.iterrows())]


def caso1_clasificacion() -> list[dict]:
    """Notebook 01, resumen de tres datasets (pp de accuracy)."""
    tabla = pd.read_csv(METRICS / "01_clasificacion_resumen_3datasets.csv")
    return [
        _fila(1, capa, "QSVM", fila["Ventaja (pp)"], "pp", "NB01 §13.6")
        for capa, fila in _por_capa(tabla)
    ]


def caso2_regresion() -> list[dict]:
    """Notebook 03, resumen de tres datasets (diferencia de R² por 100)."""
    tabla = pd.read_csv(METRICS / "03_regresion_resumen_3datasets.csv")
    return [
        _fila(2, capa, "QK-SVR", 100 * fila["Ventaja R²"], "ΔR²×100", "NB03 §13.5")
        for capa, fila in _por_capa(tabla)
    ]


def caso3_asignacion() -> list[dict]:
    """Protocolo QML v2 de asignación: coste de QAOA frente al mejor clásico exacto."""
    resumen = pd.read_csv(METRICS / "qml_v2_optimization_summary.csv").set_index("model")
    mejor_clasico = min(resumen.loc[m, "cost_mean"] for m in ("Hungarian", "PuLP-CBC"))
    qaoa = resumen.loc[[m for m in resumen.index if m.startswith("QAOA")][0], "cost_mean"]
    valor = 100 * (mejor_clasico - qaoa) / mejor_clasico
    return [_fila(3, "referencia", "QAOA p=1", valor, "% coste", "qml_v2.optimization")]


def caso4_portafolio() -> list[dict]:
    """Notebook 04: brecha de energía de VQE respecto al óptimo exacto (%, de la salida impresa)."""
    texto = _texto_salidas("04_portafolio_VQE_QAOA_vs_clasico.ipynb")
    brecha = re.search(r"Gap vs óptimo\s*:\s*[\d.]+ \(([\d.]+)%\)", texto)
    if brecha is None:
        raise ValueError("El Notebook 04 no imprime la brecha frente al óptimo exacto.")
    return [_fila(4, "referencia", "VQE (5 reinicios)", -float(brecha.group(1)), "% brecha", "NB04 §11")]


def caso5_tsp() -> list[dict]:
    """Protocolo QML v2 de TSP: diferencia de brecha QAOA frente al mejor clásico."""
    resumen = pd.read_csv(METRICS / "qml_v2_tsp_colombia_summary.csv").set_index("model")
    qaoa = [m for m in resumen.index if m.startswith("QAOA")][0]
    clasicos = [m for m in resumen.index if not m.startswith("QAOA") and m != "Brute-force"]
    mejor = resumen.loc[clasicos, "gap_mean"].min()
    valor = 100 * (mejor - resumen.loc[qaoa, "gap_mean"])
    return [_fila(5, "referencia", "QAOA p=1", valor, "pp brecha", "qml_v2.tsp_optimization")]


def caso6_umbral() -> list[dict]:
    """Notebook 06: tres datasets, pp de accuracy (una sola ejecución)."""
    tabla = pd.read_csv(METRICS / "06_qkernel_resumen_3datasets.csv")
    return [
        _fila(6, capa, "QSVC", fila["Ventaja (pp)"], "pp", "NB06 §5")
        for capa, fila in _por_capa(tabla)
    ]


def caso7_mds() -> list[dict]:
    """Experimento propio de MDS: probabilidad exacta de medir el óptimo."""
    datos = json.loads((METRICS / "10_mds_qaoa_resumen.json").read_text())
    por_p = datos["resumen_por_p"]
    return [
        _fila(7, "nativo", f"p={p} media", 100 * r["p_opt_media"], "% P(óptimo)", "qml_v2.mds_qaoa")
        for p, r in por_p.items()
    ] + [_fila(7, "nativo", "p=3 mejor", 100 * por_p["3"]["p_opt_max"], "% P(óptimo)", "qml_v2.mds_qaoa")]


def caso8_anomalias() -> list[dict]:
    """Protocolo QML v2 de anomalías: ΔAUC de OCSVM-ZZ frente al mejor clásico (pp)."""
    capas = {"zz_phase": "nativo", "skg_injected": "skg", "moons": "referencia"}
    filas = []
    for dataset in ("zz_phase", "telco_injected", "skg_injected", "moons", "blobs"):
        tabla = pd.read_csv(METRICS / f"fair_anomaly_{dataset}.csv")
        delta = tabla["delta_auc_vs_best_classical"].dropna().iloc[0]
        filas.append(
            _fila(8, capas.get(dataset, "complementario"), dataset, 100 * delta, "pp AUC", "qml_v2.anomaly")
        )
    return filas


def caso9_series() -> list[dict]:
    """Protocolo QML v2 de series: mejora relativa del MAE de QK-SVR (en %)."""
    capas = {"zz_lag": "nativo", "skg_speed": "skg", "sine": "referencia"}
    filas = []
    for dataset in ("zz_lag", "skg_speed", "sine"):
        tabla = pd.read_csv(METRICS / f"fair_timeseries_{dataset}.csv")
        tabla = tabla[tabla["table"].str.startswith("A")]
        qk = tabla[tabla["model"].str.startswith("QK-SVR")]["mae_mean"].iloc[0]
        clasico = tabla[~tabla["model"].str.startswith("QK-SVR")]["mae_mean"].min()
        filas.append(
            _fila(9, capas[dataset], dataset, 100 * (clasico - qk) / clasico, "% MAE", "qml_v2.timeseries")
        )
    return filas


def consolidar() -> pd.DataFrame:
    generadores = (
        caso1_clasificacion, caso2_regresion, caso3_asignacion, caso4_portafolio, caso5_tsp,
        caso6_umbral, caso7_mds, caso8_anomalias, caso9_series,
    )
    filas: list[dict] = []
    for generar in generadores:
        filas.extend(generar())
    tabla = pd.DataFrame(filas, columns=COLUMNAS)
    tabla["valor"] = tabla["valor"].round(1).replace(-0.0, 0.0)
    return tabla


def cargar() -> pd.DataFrame:
    """Lee el CSV consolidado (las figuras usan esta función)."""
    return pd.read_csv(SALIDA)


def main() -> None:
    tabla = consolidar()
    tabla.to_csv(SALIDA, index=False)
    print(tabla.to_string(index=False))
    print(f"\nGuardado: {SALIDA}")


if __name__ == "__main__":
    main()
