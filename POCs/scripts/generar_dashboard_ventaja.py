"""Genera ``07_dashboard_ventaja_cuantica.png``: panel 3x3 con los nueve casos de uso.

Todos los valores se leen de ``07_sintesis_resultados.csv`` (véase
``sintesis_resultados.py``); el script no contiene resultados escritos a mano.
Paleta de la plantilla de UNIR.

Uso, desde ``POCs/``::

    python scripts/sintesis_resultados.py
    python scripts/generar_dashboard_ventaja.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sintesis_resultados

warnings.filterwarnings("ignore")

SALIDA = Path(__file__).resolve().parents[1] / "data" / "output" / "figures" / "07_dashboard_ventaja_cuantica.png"

AZUL = "#219BC3"
AZUL_OSCURO = "#0E4A75"
GRIS = "#858591"
CARBON = "#4A4A57"
GRIS_CLARO = "#C0C0C8"

TITULOS = {
    1: "01 · QSVM\nClasificación",
    2: "02 · QK-SVR\nRegresión",
    3: "03 · QAOA\nAsignación",
    4: "04 · VQE\nPortafolio",
    5: "05 · QAOA\nTSP",
    6: "06 · QSVM\nUmbral de tamaño",
    7: "07 · QAOA MDS\nConjunto dominante",
    8: "08 · OCSVM-ZZ\nAnomalías",
    9: "09 · QK-SVR\nSeries temporales",
}
CAPAS = {
    "nativo": "Cuántico-nativo",
    "skg": "SKG",
    "referencia": "Referencia",
    "complementario": "Complementario",
}
ETIQUETAS_DE_MODELO = {"QSVM", "QK-SVR", "QSVC"}
UMBRAL_EMPATE = 1.0


def color_barra(valor: float) -> str:
    if abs(valor) < UMBRAL_EMPATE:
        return GRIS_CLARO
    return AZUL if valor > 0 else GRIS


def veredicto(valores: np.ndarray) -> str:
    """Etiqueta del panel según el signo de las ventajas (umbral de empate de 1 unidad)."""
    a_favor = (valores >= UMBRAL_EMPATE).any()
    en_contra = (valores <= -UMBRAL_EMPATE).any()
    if a_favor and en_contra:
        return "Condicional"
    if a_favor:
        return "Ventaja cuántica"
    if en_contra:
        return "Clásico superior"
    return "Empate"


def dibujar_panel(ax: plt.Axes, caso: int, filas: pd.DataFrame) -> str:
    etiquetas = [
        CAPAS[capa] if etiqueta in ETIQUETAS_DE_MODELO else etiqueta.replace("_", " ")
        for capa, etiqueta in zip(filas["capa"], filas["etiqueta"])
    ]
    valores = filas["valor"].to_numpy(dtype=float)
    posiciones = np.arange(len(valores))

    barras = ax.barh(posiciones, valores, color=[color_barra(v) for v in valores],
                     edgecolor="white", height=0.55)
    ax.axvline(0, color=CARBON, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(posiciones)
    ax.set_yticklabels(etiquetas, fontsize=8.5, color=CARBON)
    ax.invert_yaxis()
    ax.set_title(TITULOS[caso], fontweight="bold", fontsize=10, pad=6, color=AZUL_OSCURO)
    ax.set_xlabel(filas["unidad"].iloc[0], fontsize=8.5, color=CARBON)

    holgura = max(np.abs(valores).max() * 0.12, 3)
    ax.set_xlim(min(valores.min(), 0) - 2.5 * holgura, max(valores.max(), 0) + 2.5 * holgura)
    for barra, valor in zip(barras, valores):
        ax.text(valor + (holgura * 0.1 if valor >= 0 else -holgura * 0.1),
                barra.get_y() + barra.get_height() / 2, f"{valor:+.1f}",
                va="center", ha="left" if valor >= 0 else "right", fontsize=8, color=CARBON)

    texto = veredicto(valores)
    ax.text(0.98, 0.02, texto, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=AZUL_OSCURO, fontweight="bold")
    return texto


def main() -> None:
    sintesis = sintesis_resultados.cargar()
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("Dashboard de ventaja cuántica: nueve casos de uso\n"
                 "Juan David Prada Malagón  ·  SKG Tecnología  ·  UNIR 2026",
                 fontsize=16, fontweight="bold", y=0.995, color=AZUL_OSCURO)
    cuadricula = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                                   left=0.08, right=0.97, top=0.94, bottom=0.08)

    veredictos = []
    for indice, caso in enumerate(sorted(TITULOS)):
        ax = fig.add_subplot(cuadricula[indice // 3, indice % 3])
        veredictos.append(dibujar_panel(ax, caso, sintesis[sintesis["caso"] == caso]))

    resumen = pd.Series(veredictos).value_counts()
    fig.text(0.5, 0.02,
             "Veredictos: " + "  ·  ".join(f"{n} {v.lower()}" for v, n in resumen.items())
             + "\nCada panel usa su propia unidad; positivo = favorece al método cuántico.",
             ha="center", fontsize=9.5, color=CARBON)
    fig.legend(
        handles=[
            mpatches.Patch(color=AZUL, label="Ventaja cuántica"),
            mpatches.Patch(color=GRIS, label="Clásico superior"),
            mpatches.Patch(color=GRIS_CLARO, label="Empate (|Δ| < 1)"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, fontsize=9,
        framealpha=0.92, edgecolor=GRIS_CLARO,
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SALIDA, dpi=150, bbox_inches="tight", facecolor="white", metadata={"paleta": "unir"})
    plt.close()
    print(f"Dashboard guardado: {SALIDA.name}")
    print(resumen.to_string())


if __name__ == "__main__":
    main()
