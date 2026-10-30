"""
Genera la figura dual-eje para el TFM:
  01_escalabilidad_ventaja_clasificacion.png

Figura 5.1: Evolución de la exactitud de QSVM y SVM-RBF en función del
tamaño del conjunto de entrenamiento. Eje izquierdo: accuracy (desde 0 %).
Eje derecho: ventaja cuántica en puntos porcentuales (barras verdes).

Datos: dataset cuántico-nativo (Liu 2021, ZZFeatureMap 2 qubits, reps=2), leídos de
``01_clasificacion_escalabilidad.csv`` (salida del análisis de escalabilidad del notebook 01).
Paleta de la plantilla de UNIR.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
})

# ── Datos (salida ejecutada del notebook 01, sección de escalabilidad) ─────
escalabilidad = pd.read_csv(OUT.parent / "metrics" / "01_clasificacion_escalabilidad.csv")
n_train = escalabilidad["n_train"].to_numpy()
acc_qsvm = escalabilidad["acc_quantum"].to_numpy()
acc_svm_rbf = escalabilidad["acc_classical"].to_numpy()
advantage_pp = (acc_qsvm - acc_svm_rbf) * 100

# ── Figura con eje dual ────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5.5))

# Eje izquierdo: accuracy (desde 0 % hasta 100 %)
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)

# Barras de ventaja cuántica (eje derecho, fondo)
bar_colors = ["#219BC3" if v > 0 else "#858591" for v in advantage_pp]
bars = ax2.bar(n_train, advantage_pp,
               color=bar_colors, alpha=0.28, width=10,
               label="Ventaja cuántica (pp)", zorder=1)
ax2.axhline(advantage_pp.mean(), color="#219BC3", lw=1.4, ls="--",
            alpha=0.55, zorder=2)
ax2.set_ylabel("Ventaja cuántica (pp) = QSVM − SVM-RBF", fontsize=10,
               color="#219BC3")
ax2.tick_params(axis="y", labelcolor="#219BC3")
ax2.set_ylim(0, advantage_pp.max() * 2.2)   # escala holgada eje derecho

# Líneas de accuracy (eje izquierdo, primer plano)
ax1.plot(n_train, acc_qsvm * 100, "o-", color="#0E4A75", linewidth=2.5,
         markersize=8, label="Quantum Kernel SVM", zorder=5)
ax1.plot(n_train, acc_svm_rbf * 100, "s--", color="#858591", linewidth=2,
         markersize=7, label="SVM-RBF (clásico)", zorder=4)
ax1.fill_between(n_train, acc_svm_rbf * 100, acc_qsvm * 100,
                 alpha=0.12, color="#0E4A75", zorder=3)

# Eje izquierdo: desde 0 % (requisito del profesor)
ax1.set_ylim(0, 110)
ax1.set_ylabel("Exactitud (%)", fontsize=11)
ax1.set_xlabel("Tamaño del conjunto de entrenamiento ($n$)", fontsize=11)
ax1.set_xticks(n_train)
ax1.set_title(
    "Escalabilidad de la ventaja cuántica — dataset cuántico-nativo\n"
    f"QSVM vs SVM-RBF  ·  2 qubits, ZZFeatureMap reps=2  ·  media Δ = {advantage_pp.mean():+.1f} pp",
    fontsize=12, fontweight="bold", pad=10
)

# Anotaciones de ventaja sobre cada barra
for x_pos, adv in zip(n_train, advantage_pp):
    ax2.text(x_pos, adv + 1.5, f"+{adv:.1f}pp",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold",
             color="#0E4A75", zorder=6)

# Leyenda en el hueco central entre SVM-RBF (~43-56%) y QSVM (~86-91%)
# → bbox_to_anchor y=0.57 en coords de ejes (0-110%) ≈ 63% accuracy,
#   que está claramente por encima del SVM-RBF max (55.6%) y
#   por debajo del QSVM min (85.6%).
handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
patch_bars = mpatches.Patch(color="#219BC3", alpha=0.40,
                             label=f"Ventaja cuántica (media +{advantage_pp.mean():.1f} pp)")
all_handles = handles_ax1 + [patch_bars]
ax1.legend(handles=all_handles, fontsize=9,
           loc='center left', bbox_to_anchor=(0.02, 0.57),
           framealpha=0.92, edgecolor="#C0C0C8")

plt.tight_layout()
out_path = OUT / "01_escalabilidad_ventaja_clasificacion.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", metadata={"paleta": "unir"})
plt.close()
print(f"✅  Guardado: {out_path}")
