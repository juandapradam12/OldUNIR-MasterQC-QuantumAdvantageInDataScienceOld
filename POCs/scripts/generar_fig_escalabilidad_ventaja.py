"""
Genera la figura dual-eje para el TFM:
  01_escalabilidad_ventaja_clasificacion.png

Figura 5.1: Evolución de la exactitud de QSVM y SVM-RBF en función del
tamaño del conjunto de entrenamiento. Eje izquierdo: accuracy (desde 0 %).
Eje derecho: ventaja cuántica en puntos porcentuales (barras verdes).

Datos: dataset cuántico-nativo (Liu 2021, ZZFeatureMap 2 qubits, reps=2).
Resultados extraídos del análisis de escalabilidad del notebook 01.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("/Users/juanprada/Documents/TFM_ComputacionCuantica/POCs/data/output/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
})

# ── Datos (del notebook 01, sección de escalabilidad) ─────────────────────
n_train      = np.array([30,    50,    80,    100,   140,   180  ])
acc_qsvm     = np.array([0.889, 0.911, 0.856, 0.856, 0.889, 0.911])
acc_svm_rbf  = np.array([0.467, 0.422, 0.489, 0.478, 0.444, 0.556])
advantage_pp = (acc_qsvm - acc_svm_rbf) * 100   # media ≈ +40.9 pp

# ── Figura con eje dual ────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5.5))

# Eje izquierdo: accuracy (desde 0 % hasta 100 %)
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)

# Barras de ventaja cuántica (eje derecho, fondo)
bar_colors = ["#2E7D32" if v > 0 else "#D32F2F" for v in advantage_pp]
bars = ax2.bar(n_train, advantage_pp,
               color=bar_colors, alpha=0.28, width=10,
               label="Ventaja cuántica (pp)", zorder=1)
ax2.axhline(advantage_pp.mean(), color="#2E7D32", lw=1.4, ls="--",
            alpha=0.55, zorder=2)
ax2.set_ylabel("Ventaja cuántica (pp) = QSVM − SVM-RBF", fontsize=10,
               color="#2E7D32")
ax2.tick_params(axis="y", labelcolor="#2E7D32")
ax2.set_ylim(0, advantage_pp.max() * 2.2)   # escala holgada eje derecho

# Líneas de accuracy (eje izquierdo, primer plano)
ax1.plot(n_train, acc_qsvm * 100, "o-", color="#7B2FBE", linewidth=2.5,
         markersize=8, label="Quantum Kernel SVM", zorder=5)
ax1.plot(n_train, acc_svm_rbf * 100, "s--", color="#1976D2", linewidth=2,
         markersize=7, label="SVM-RBF (clásico)", zorder=4)
ax1.fill_between(n_train, acc_svm_rbf * 100, acc_qsvm * 100,
                 alpha=0.12, color="#7B2FBE", zorder=3)

# Eje izquierdo: desde 0 % (requisito del profesor)
ax1.set_ylim(0, 110)
ax1.set_ylabel("Exactitud (%)", fontsize=11)
ax1.set_xlabel("Tamaño del conjunto de entrenamiento ($n$)", fontsize=11)
ax1.set_xticks(n_train)
ax1.set_title(
    "Escalabilidad de la Ventaja Cuántica — Dataset Cuántico-Nativo\n"
    "QSVM vs SVM-RBF  ·  2 qubits, ZZFeatureMap reps=2  ·  media Δ = +40.9 pp",
    fontsize=12, fontweight="bold", pad=10
)

# Anotaciones de ventaja sobre cada barra
for x_pos, adv in zip(n_train, advantage_pp):
    ax2.text(x_pos, adv + 1.5, f"+{adv:.1f}pp",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold",
             color="#1B5E20", zorder=6)

# Leyenda en el hueco central entre SVM-RBF (~43-56%) y QSVM (~86-91%)
# → bbox_to_anchor y=0.57 en coords de ejes (0-110%) ≈ 63% accuracy,
#   que está claramente por encima del SVM-RBF max (55.6%) y
#   por debajo del QSVM min (85.6%).
handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
patch_bars = mpatches.Patch(color="#2E7D32", alpha=0.40,
                             label=f"Ventaja cuántica (media +{advantage_pp.mean():.1f} pp)")
all_handles = handles_ax1 + [patch_bars]
ax1.legend(handles=all_handles, fontsize=9,
           loc='center left', bbox_to_anchor=(0.02, 0.57),
           framealpha=0.92, edgecolor="#cccccc")

plt.tight_layout()
out_path = OUT / "01_escalabilidad_ventaja_clasificacion.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅  Guardado: {out_path}")
