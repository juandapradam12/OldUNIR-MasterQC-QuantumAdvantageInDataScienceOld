"""
Figura de escalabilidad de la ventaja del QSVM (Caso 1, dataset cuántico-nativo).

Salida: ``01_escalabilidad_ventaja_clasificacion.png``
  - Panel izquierdo: exactitud de QSVM y SVM-RBF según el tamaño de entrenamiento (eje desde 0).
  - Panel derecho: ventaja QSVM - SVM-RBF en puntos porcentuales.

Datos: ``01_clasificacion_escalabilidad.csv`` (salida del análisis de escalabilidad del
notebook 01; ZZFeatureMap de 2 qubits, reps=2). Paleta de la plantilla de UNIR.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AZUL, AZUL_OSCURO, GRIS = "#219BC3", "#0E4A75", "#858591"

OUT = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})

datos = pd.read_csv(OUT.parent / "metrics" / "01_clasificacion_escalabilidad.csv")
n_train = datos["n_train"].to_numpy()
acc_q = datos["acc_quantum"].to_numpy() * 100
acc_cl = datos["acc_classical"].to_numpy() * 100
ventaja = acc_q - acc_cl

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})

ax1.plot(n_train, acc_q, "o-", color=AZUL_OSCURO, lw=2.4, ms=7, label="Quantum Kernel SVM")
ax1.plot(n_train, acc_cl, "s--", color=GRIS, lw=2, ms=6, label="SVM-RBF (clásico)")
ax1.fill_between(n_train, acc_cl, acc_q, color=AZUL, alpha=0.15)
ax1.axhline(50, color="#C0C0C8", lw=1, ls=":")
ax1.set_ylim(0, 105)
ax1.set_xticks(n_train)
ax1.set_xlabel("Tamaño del conjunto de entrenamiento ($n$)")
ax1.set_ylabel("Exactitud en test (%)")
ax1.set_title("Exactitud según tamaño de entrenamiento", fontsize=11)
ax1.legend(loc="lower center", ncol=2, fontsize=9, framealpha=0.95, edgecolor="#C0C0C8")

barras = ax2.bar(n_train, ventaja, width=11, color=AZUL, zorder=3)
ax2.axhline(ventaja.mean(), color=AZUL_OSCURO, lw=1.3, ls="--", label=f"Media = {ventaja.mean():+.1f} pp")
for x, v in zip(n_train, ventaja):
    ax2.text(x, v + 1, f"{v:+.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=AZUL_OSCURO)
ax2.set_ylim(0, ventaja.max() * 1.3)
ax2.set_xticks(n_train)
ax2.set_xlabel("Tamaño del conjunto de entrenamiento ($n$)")
ax2.set_ylabel("QSVM $-$ SVM-RBF (pp)")
ax2.set_title("Ventaja del QSVM (positivo = cuántico mejor)", fontsize=11)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.95, edgecolor="#C0C0C8")

fig.suptitle(
    "Escalabilidad de la ventaja cuántica: dataset cuántico-nativo (2 qubits, ZZFeatureMap reps=2)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
ruta = OUT / "01_escalabilidad_ventaja_clasificacion.png"
fig.savefig(ruta, dpi=150, bbox_inches="tight", metadata={"paleta": "unir"})
plt.close(fig)
print(f"Guardado: {ruta}")
