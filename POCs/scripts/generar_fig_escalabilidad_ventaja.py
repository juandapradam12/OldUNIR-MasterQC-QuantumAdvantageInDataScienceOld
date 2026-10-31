"""
Figura de escalabilidad de la ventaja del QSVM (Caso 1, dataset cuántico-nativo).

Salida: ``01_escalabilidad_ventaja_clasificacion.png``
  - Panel izquierdo: exactitud media (± desviación) de QSVM, SVM-RBF y Gradient Boosting
    según el tamaño de entrenamiento.
  - Panel derecho: ventaja del QSVM sobre el mejor clásico de cada repetición, con IC 95 %.

Datos: ``fair_scalability_native.csv`` (``python -m qml_v2.scalability``: 10 repeticiones por
tamaño, dataset regenerado con una semilla distinta en cada una; ZZFeatureMap de 2 qubits,
reps=2). Paleta de la plantilla de UNIR.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AZUL, AZUL_OSCURO, GRIS = "#219BC3", "#0E4A75", "#858591"

OUT = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})

datos = pd.read_csv(OUT.parent / "metrics" / "fair_scalability_native.csv")
n_train = datos["n_train"].to_numpy()
n_rep = int(datos["n_repeticiones"].iloc[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})

series = [
    ("qsvm", "Quantum Kernel SVM", AZUL_OSCURO, "o-"),
    ("rbf", "SVM-RBF (clásico)", GRIS, "s--"),
    ("gb", "Gradient Boosting (clásico)", AZUL, "^:"),
]
for clave, etiqueta, color, estilo in series:
    media, std = datos[f"{clave}_media"].to_numpy(), datos[f"{clave}_std"].to_numpy()
    ax1.plot(n_train, media, estilo, color=color, lw=2.2, ms=6, label=etiqueta)
    ax1.fill_between(n_train, media - std, media + std, color=color, alpha=0.12)
ax1.axhline(50, color="#C0C0C8", lw=1, ls=":")
ax1.set_ylim(0, 105)
ax1.set_xticks(n_train)
ax1.set_xlabel("Tamaño del conjunto de entrenamiento ($n$)")
ax1.set_ylabel("Exactitud en test (%)")
ax1.set_title(f"Exactitud media $\\pm$ desviación ({n_rep} repeticiones)", fontsize=11)
ax1.legend(loc="lower center", ncol=2, fontsize=8.5, framealpha=0.95, edgecolor="#C0C0C8")

delta = datos["delta_vs_mejor_clasico_pp"].to_numpy()
error = [delta - datos["delta_ic95_inf"].to_numpy(), datos["delta_ic95_sup"].to_numpy() - delta]
ax2.bar(n_train, delta, width=11, color=AZUL, zorder=3)
ax2.errorbar(n_train, delta, yerr=error, fmt="none", ecolor=AZUL_OSCURO, capsize=4, lw=1.4, zorder=4)
for x, v in zip(n_train, delta):
    ax2.text(
        x,
        v + error[1][list(n_train).index(x)] + 1,
        f"{v:+.1f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=AZUL_OSCURO,
    )
ax2.set_ylim(0, (delta + error[1]).max() * 1.2)
ax2.set_xticks(n_train)
ax2.set_xticklabels([f"{n}\n({k}/{n_rep})" for n, k in zip(n_train, datos["repeticiones_qsvm_gana"])])
ax2.set_xlabel("Tamaño de entrenamiento ($n$) y repeticiones con ventaja")
ax2.set_ylabel("QSVM $-$ mejor clásico (pp)")
ax2.set_title("Ventaja del QSVM sobre el mejor clásico (IC 95 %)", fontsize=11)

fig.suptitle(
    "Escalabilidad de la ventaja cuántica: dataset cuántico-nativo (2 qubits, ZZFeatureMap reps=2)",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()
ruta = OUT / "01_escalabilidad_ventaja_clasificacion.png"
fig.savefig(ruta, dpi=150, bbox_inches="tight", metadata={"paleta": "unir"})
plt.close(fig)
print(f"Guardado: {ruta}")
