"""Mapa de errores cuántico vs clásico para los cinco casos de uso (protocolo justo, Tabla A).

Cada panel es un caso de uso: filas = conjunto de datos, columnas = modelos, color = error
(0 es perfecto). Todo se lee de ``POCs/data/output/metrics``; no se escribe ningún número a mano.

Error por caso de uso: 1 - exactitud (clasificación), 1 - R² (regresión y series),
1 - AUC (anomalías) y 1 - P(óptimo) en optimización (fracción de instancias o semillas en
las que el método no alcanza el óptimo exacto). El color se recorta en 1; la cifra impresa es el valor real
(un error mayor que 1 es peor que predecir la media).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

POCS = Path(__file__).resolve().parents[1]
METRICS = POCS / "data" / "output" / "metrics"
OUT = POCS / "data" / "output" / "figures" / "13_mapa_errores_q_vs_c.png"

AZUL, NAVY = "#219BC3", "#0E4A75"
CMAP = LinearSegmentedColormap.from_list("unir", ["#F3F8FB", AZUL, NAVY])

Panel = tuple[str, list[str], list[str], np.ndarray, set[str]]


def tabla_a(nombre: str, valor: str, columna: str, error) -> pd.DataFrame:
    """Error por (dataset, modelo) en la Tabla A de un CSV agregado o por dataset."""
    d = pd.read_csv(METRICS / nombre)
    d = d[d["table"].str.startswith("A")]
    d = d.assign(error=error(d[valor]))
    return d.pivot_table(index=columna, columns="model", values="error")


def matriz(pivot: pd.DataFrame, filas: dict[str, str], columnas: dict[str, str]) -> np.ndarray:
    return np.array(
        [[pivot.loc[f, c] if c in pivot.columns else np.nan for c in columnas.values()] for f in filas.values()]
    )


def panel_clasificacion() -> Panel:
    p = tabla_a("fair_classification_all.csv", "acc_mean", "dataset", lambda s: 1 - s)
    filas = {"SKG": "skg", "Telco": "telco"}
    cols = {
        "QSVM": "QSVM-ZZ",
        "SVM-RBF": "SVM-RBF",
        "SVM lineal": "SVM-linear",
        "Reg. log.": "LogisticRegression",
        "GB": "GradientBoosting",
    }
    return "Clasificación: 1 − exactitud", list(filas), list(cols), matriz(p, filas, cols), {"QSVM"}


def panel_regresion() -> Panel:
    p = tabla_a("qml_v2_regression_all.csv", "r2_mean", "dataset", lambda s: 1 - s)
    filas = {"SKG": "skg", "Friedman #1": "friedman"}
    cols = {
        "QK-SVR": "QK-SVR-ZZ",
        "SVR-RBF": "SVR-RBF",
        "Ridge": "Ridge",
        "RF": "RandomForest",
        "GB": "GradientBoosting",
    }
    return "Regresión: 1 − R²", list(filas), list(cols), matriz(p, filas, cols), {"QK-SVR"}


def panel_anomalias() -> Panel:
    filas = {"SKG": "skg_injected", "moons": "moons"}
    cols = {"OCSVM-ZZ": "OCSVM-ZZ", "OCSVM-RBF": "OCSVM-RBF", "IsoForest": "IsolationForest", "LOF": "LOF"}
    frames = []
    for ds in filas.values():
        d = pd.read_csv(METRICS / f"fair_anomaly_{ds}.csv")
        frames.append(d[d["table"].str.startswith("A")].assign(dataset=ds, error=lambda x: 1 - x["auc_mean"]))
    p = pd.concat(frames).pivot_table(index="dataset", columns="model", values="error")
    return "Anomalías: 1 − AUC", list(filas), list(cols), matriz(p, filas, cols), {"OCSVM-ZZ"}


def panel_series() -> Panel:
    frames = []
    filas = {"SKG": "skg_speed", "sine": "sine"}
    for ds in filas.values():
        d = pd.read_csv(METRICS / f"fair_timeseries_{ds}.csv")
        frames.append(d[d["table"].str.startswith("A")].assign(dataset=ds, error=lambda x: 1 - x["r2_mean"]))
    p = pd.concat(frames).pivot_table(index="dataset", columns="model", values="error")
    cols = {"QK-SVR": "QK-SVR-ZZ", "SVR-RBF": "SVR-RBF", "Ridge": "Ridge", "GB": "GradientBoosting"}
    return "Series temporales: 1 − R²", list(filas), list(cols), matriz(p, filas, cols), {"QK-SVR"}


def panel_optimizacion() -> Panel:
    asignacion = (
        pd.read_csv(METRICS / "qml_v2_optimization_raw.csv")
        .groupby("model")["gap_vs_pulp"]
        .apply(lambda s: float((s > 1e-9).mean()))
    )
    portafolio = pd.read_csv(METRICS / "qml_v2_portfolio_summary.csv")
    portafolio = portafolio[portafolio["configuracion"] == "penalizacion_alta"].set_index("metodo")
    ref = json.loads((METRICS / "qml_v2_portfolio_reference.json").read_text(encoding="utf-8"))
    voraz_portafolio = float(not np.isclose(ref["coste_voraz"], ref["optimo_exacto"]))
    tsp = pd.read_csv(METRICS / "qml_v2_tsp_colombia_summary.csv").set_index("model")["gap_mean"].gt(1e-9).astype(float)
    mds = pd.read_csv(METRICS / "10_mds_qaoa_escalabilidad.csv")
    mds_qaoa = 1 - float(mds["qaoa_mejor_muestra_es_dominante"].mean())
    mds_voraz = float((mds["greedy_tamano"] > mds["ilp_tamano"]).mean())
    cols = ["QAOA", "VQE", "Voraz", "Exacto", "SA-QUBO"]
    datos = {
        "Asignación": [asignacion["QAOA-p1"], np.nan, asignacion["Greedy"], asignacion["PuLP-CBC"], np.nan],
        "MDS (ER)": [mds_qaoa, np.nan, mds_voraz, 0.0, np.nan],
        "Portafolio": [
            1 - portafolio.loc["QAOA", "optimo_encontrado"] / portafolio.loc["QAOA", "semillas"],
            1 - portafolio.loc["VQE", "optimo_encontrado"] / portafolio.loc["VQE", "semillas"],
            voraz_portafolio,
            0.0,
            np.nan,
        ],
        "TSP n=4": [tsp["QAOA-p1"], np.nan, tsp["NearestNeighbor"], 0.0, tsp["SA-QUBO"]],
    }
    return (
        "Optimización: 1 − P(óptimo)",
        list(datos),
        cols,
        np.array(list(datos.values()), dtype=float),
        {"QAOA", "VQE"},
    )


def dibujar(ax: plt.Axes, panel: Panel) -> None:
    titulo, filas, cols, valores, cuanticos = panel
    ax.imshow(np.ma.masked_invalid(np.clip(valores, 0, 1)), cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            v = valores[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color="#E4E4E8", lw=0))
                continue
            texto = ">10" if v > 10 else f"{v:.2f}"
            ax.text(
                j, i, texto, ha="center", va="center", fontsize=10, color="white" if min(v, 1) > 0.55 else "#1F2933"
            )
        for mejor in np.flatnonzero(valores[i] <= np.nanmin(valores[i]) + 1e-9):
            ax.add_patch(Rectangle((mejor - 0.5, i - 0.5), 1, 1, fill=False, ec="#1F2933", lw=2.4))
    ax.set_xticks(range(len(cols)), cols, fontsize=9, rotation=25, ha="right")
    ax.set_yticks(range(len(filas)), filas, fontsize=10)
    ax.set_title(titulo, fontsize=11.5, fontweight="bold", color=NAVY, loc="left")
    ax.tick_params(length=0)
    for etiqueta in ax.get_xticklabels():
        if etiqueta.get_text() in cuanticos:
            etiqueta.set_color(AZUL)
            etiqueta.set_fontweight("bold")
    for lado in ax.spines.values():
        lado.set_visible(False)


def main() -> None:
    paneles = [panel_clasificacion(), panel_regresion(), panel_anomalias(), panel_series(), panel_optimizacion()]
    fig = plt.figure(figsize=(14, 5.6))
    rejilla = fig.add_gridspec(2, 4, width_ratios=[5, 5, 5, 0.35], hspace=0.55, wspace=0.5)
    ejes = [fig.add_subplot(rejilla[i // 3, i % 3]) for i in range(5)]
    for ax, panel in zip(ejes, paneles, strict=True):
        dibujar(ax, panel)
    leyenda = fig.add_subplot(rejilla[1, 2])
    leyenda.axis("off")
    leyenda.text(
        0.0,
        0.5,
        "Color: error, de 0 (claro) a 1 (oscuro).\n"
        "Borde negro: mejor modelo de la fila\n(empates incluidos).\n"
        "Etiqueta azul: modelo cuántico.\n"
        "Gris: no aplica.\n"
        "Color recortado en 1; la cifra es el valor real.\n"
        "Aprendizaje: mismas variables de entrada\npara todos los modelos.",
        fontsize=9.5,
        color="#3D3D4A",
        va="center",
        linespacing=1.6,
    )
    barra = fig.add_subplot(rejilla[:, 3])
    fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)), cax=barra, label="error")
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
