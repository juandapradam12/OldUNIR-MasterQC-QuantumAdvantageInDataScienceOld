"""
Genera la figura D3 del TFM a partir de resultados REALES de QAOA:
  10_qaoa_histograma_medicion.png — distribución de medición (6 000 shots) del mejor
                                    reinicio de QAOA p=3 sobre el MDS de 6 nodos.

Los datos salen de ``python -m qml_v2.mds_qaoa`` (POCs/data/output/metrics/10_mds_qaoa_*).
No hay probabilidades escritas a mano.
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "output" / "figures"
METRICS = ROOT / "data" / "output" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})

info = json.loads((METRICS / "10_mds_qaoa_resumen.json").read_text())
dist = pd.read_csv(METRICS / "10_mds_qaoa_probabilidades.csv", dtype={"bitstring": str})
dist["bitstring"] = dist["bitstring"].str.zfill(info["n_qubits"])
opt_bs = info["optimo_bitstring"]
SHOTS = info["shots"]

top = dist.sort_values("conteos_6000_shots", ascending=False).head(8).reset_index(drop=True)
labels = top["bitstring"].tolist()
probs = top["frecuencia_shots"].tolist()
costs = top["coste_qubo"].astype(int).tolist()
sets = top["conjunto"].tolist()
colors = ["#0E4A75" if lb == opt_bs else "#219BC3" for lb in labels]
p_opt_shots = float(dist.loc[dist["bitstring"] == opt_bs, "frecuencia_shots"].iloc[0])
p_opt_exact = float(dist.loc[dist["bitstring"] == opt_bs, "prob_exacta"].iloc[0])

fig, ax = plt.subplots(figsize=(11, 6.5))
bars = ax.bar(range(len(labels)), probs, color=colors, edgecolor="white", linewidth=1.2, width=0.7)

for i, (bar, p, c) in enumerate(zip(bars, probs, costs)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.008,
        f"{p:.3f}\n(C={c})",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold" if labels[i] == opt_bs else "normal",
        color="#0E4A75" if labels[i] == opt_bs else "#333333",
    )

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(
    [(f"$\\bf{{{bs}}}$\n" if bs == opt_bs else f"{bs}\n") + s for bs, s in zip(labels, sets)],
    fontsize=8,
    rotation=30,
    ha="right",
)
ax.set_ylabel("Frecuencia de medición (6 000 shots)", fontsize=11)
ax.set_ylim(0, max(probs) * 1.25)
ax.set_title(
    f"Histograma de medición QAOA — MDS red social (p=3, {SHOTS} shots, mejor de {info['n_restarts']} reinicios)\n"
    + "Los 8 estados más frecuentes; el azul oscuro es la solución óptima $D^*=\\{\\text{Ana, Enrique}\\}$ (coste = 2)",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.annotate(
    f"$D^*$: frecuencia {p_opt_shots:.3f}\n(prob. exacta {p_opt_exact:.3f})",
    xy=(labels.index(opt_bs), probs[labels.index(opt_bs)]),
    xytext=(len(labels) - 4.2, max(probs) * 1.02),
    arrowprops={"arrowstyle": "->", "color": "#0E4A75", "lw": 1.6},
    fontsize=9,
    color="#0E4A75",
    fontweight="bold",
    bbox={"boxstyle": "round,pad=0.3", "fc": "#EAF4F8", "ec": "#0E4A75", "alpha": 0.9},
)
ax.legend(
    handles=[
        mpatches.Patch(color="#0E4A75", label="Solución óptima $D^*$ (fuerza bruta)"),
        mpatches.Patch(color="#219BC3", label="Estado sub-óptimo (factible o infactible)"),
    ],
    fontsize=9,
    loc="upper right",
)

plt.tight_layout()
out_path = OUT / "10_qaoa_histograma_medicion.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", metadata={"paleta": "unir"})
plt.close()
print(f"Guardado: {out_path}")
