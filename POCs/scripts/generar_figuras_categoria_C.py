"""
Genera las 3 figuras de Categoría C para el TFM.

  10_mds_grafo_solucion.png      — grafo social MDS + solución óptima (C3)
  10_qaoa_convergencia.png       — convergencia QAOA p=1,2,3 con datos reales de qml_v2.mds_qaoa (C1)
  03_regresion_3datasets.png     — comparativa regresión Q vs C en 3 datasets (C2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ──────────────────────────────────────────────────────────────────────────────
# C3 — Grafo social MDS con solución óptima D*={Ana, Enrique}
# ──────────────────────────────────────────────────────────────────────────────
import json
METRICS = OUT.parent / "metrics"
_info = json.loads((METRICS / "10_mds_qaoa_resumen.json").read_text())
nodes = _info["nodes"]
edges = [tuple(e) for e in _info["edges"]]
solution = set(_info["optimo_conjunto"])

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Layout fijo para reproducibilidad
pos = {
    "Mario":   (-1.5,  1.0),
    "Sarah":   ( 0.0,  1.8),
    "Raúl":   ( 1.5,  1.0),
    "Ana":     (-0.8, -0.2),
    "Enrique": ( 0.8, -0.2),
    "Saúl":   ( 0.0, -1.5),
}

node_colors = []
node_sizes  = []
for n_ in G.nodes():
    if n_ in solution:
        node_colors.append("#0E4A75")    # azul oscuro → influencer
        node_sizes.append(1600)
    else:
        node_colors.append("#219BC3")    # azul → no contratado
        node_sizes.append(1100)

fig, ax = plt.subplots(figsize=(8, 7))

nx.draw_networkx_edges(G, pos, ax=ax,
                       edge_color="#888888", width=2.2, alpha=0.65)

nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors,
                       node_size=node_sizes,
                       linewidths=2.5,
                       edgecolors="white")

nx.draw_networkx_labels(G, pos, ax=ax,
                        font_size=10, font_weight="bold", font_color="white")

# Leyenda
patch_inf = mpatches.Patch(color="#0E4A75", label=r"$D^*$ = {Ana, Enrique}  (influencers contratados, coste = 2)")
patch_std = mpatches.Patch(color="#219BC3", label="Nodo cubierto (≥ 1 vecino en $D^*$)")
ax.legend(handles=[patch_inf, patch_std], loc="lower center",
          bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=9,
          framealpha=0.9, edgecolor="#cccccc")

# Anotación de cobertura
for node in G.nodes():
    if node not in solution:
        # flecha suave desde el nodo hacia su influencer vecino
        nbrs_in_sol = [nb for nb in G.neighbors(node) if nb in solution]
        if nbrs_in_sol:
            ax.annotate("",
                xy=pos[nbrs_in_sol[0]], xytext=pos[node],
                arrowprops=dict(arrowstyle="->", color="#858591",
                                lw=1.5, connectionstyle="arc3,rad=0.12"))

ax.set_title(
    "Conjunto Dominante Mínimo (MDS) — Red Social de 6 nodos\n"
    r"Problema: $\min|D|$ s.t. $\forall v \in V,\; N[v]\cap D \neq \emptyset$",
    fontsize=12, fontweight="bold", pad=14
)
ax.axis("off")
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
out_path = OUT / "10_mds_grafo_solucion.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", metadata={"paleta": "unir"})
plt.close()
print(f"✅  Guardado: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# C1 — Convergencia QAOA p=1, p=2, p=3 (datos reales de qml_v2.mds_qaoa)
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd

reinicios = pd.read_csv(METRICS / "10_mds_qaoa_reinicios.csv")
trayectorias = pd.read_csv(METRICS / "10_mds_qaoa_trayectorias.csv")
e_opt = float(_info["optimo_coste"])
estilo = {1: dict(color="#219BC3", marker="o", label="p = 1  (2 parámetros)"),
          2: dict(color="#858591", marker="s", label="p = 2  (4 parámetros)"),
          3: dict(color="#0E4A75", marker="^", label="p = 3  (6 parámetros)")}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Convergencia del loop híbrido QAOA–COBYLA — MDS (6 nodos, A=1, B=2)\n"
    f"Coste QUBO esperado $\\langle C \\rangle$ — {int(reinicios['restart'].nunique())} reinicios por profundidad, "
    f"COBYLA maxiter={_info['maxiter']}",
    fontsize=12, fontweight="bold")
ax_sort, ax_conv = axes

for p, cfg in estilo.items():
    r = reinicios[reinicios["p"] == p].sort_values("energy_final")
    ax_sort.plot(range(1, len(r) + 1), r["energy_final"].to_numpy(), color=cfg["color"],
                 marker=cfg["marker"], linewidth=2, markersize=6, label=cfg["label"])
    mejor = int(r.iloc[0]["restart"])
    t = trayectorias[(trayectorias["p"] == p) & (trayectorias["restart"] == mejor)]
    ax_conv.plot(t["iteration"], t["energy"], color=cfg["color"], linewidth=1.6, alpha=0.9, label=cfg["label"])

for ax in axes:
    ax.axhline(e_opt, color="black", linestyle="--", linewidth=1.4, alpha=0.65,
               label=f"Coste óptimo $C^*={e_opt:.0f}$ (fuerza bruta)")
ax_sort.set_xlabel("Reinicio (ordenado por coste final)", fontsize=10)
ax_sort.set_ylabel("$\\langle C \\rangle$ final", fontsize=11)
ax_sort.set_title("Coste esperado final por reinicio", fontsize=10)
ax_sort.legend(fontsize=9)
ax_conv.set_xlabel("Evaluación de la función objetivo (COBYLA)", fontsize=10)
ax_conv.set_ylabel("$\\langle C \\rangle_\\theta$", fontsize=11)
ax_conv.set_title("Trayectoria del mejor reinicio por profundidad", fontsize=10)
ax_conv.legend(fontsize=9)

plt.tight_layout()
out_path = OUT / "10_qaoa_convergencia.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", metadata={"paleta": "unir"})
plt.close()
print(f"✅  Guardado: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# C2 — Regresión cuántica vs clásica en 3 datasets
# Todos los valores se leen de las salidas ejecutadas del Notebook 03:
#   03_regresion_resumen_3datasets.csv  (R² cuántico, mejor clásico, ventaja)
#   03_regresion_ventaja_cuantica.csv   (modelos del dataset cuántico-nativo)
#   03_regresion_skg_bc.csv             (modelos del dataset SKG)
#   03_regresion_resultados.csv         (modelos de Friedman #1)
# ──────────────────────────────────────────────────────────────────────────────
AZUL_UNIR = "#219BC3"
GRIS_UNIR = "#858591"
CARBON_UNIR = "#4A4A57"
QUANTUM_COLOR = AZUL_UNIR
CLASSICAL_COLOR = GRIS_UNIR

resumen_reg = pd.read_csv(METRICS / "03_regresion_resumen_3datasets.csv")
nativo_cl = pd.read_csv(METRICS / "03_regresion_ventaja_cuantica.csv").query("type == 'classical'")
skg_cl = pd.read_csv(METRICS / "03_regresion_skg_bc.csv").query("Tipo == 'Clásico'")
friedman_cl = pd.read_csv(METRICS / "03_regresion_resultados.csv").query("type == 'classical'")
labels_classical = [
    nativo_cl.loc[nativo_cl["r2"].idxmax(), "model"],
    skg_cl.loc[skg_cl["R2"].idxmax(), "Modelo"],
    friedman_cl.loc[friedman_cl["r2"].idxmax(), "model"],
]

datasets = [
    "Dataset\ncuántico-nativo\n(Liu 2021)",
    "SKG Bogotá\n(datos reales\nmovilidad)",
    "Friedman #1\n(sintético\nno-cuántico)",
]
r2_quantum = resumen_reg["QK-SVR"].to_numpy()
r2_classical = resumen_reg["Mejor Clásico"].to_numpy()
gap = r2_quantum - r2_classical
veredictos = ["VENTAJA Q" if g > 0 else "GANA CLÁSICO" for g in gap]
gap_colors = [QUANTUM_COLOR if g > 0 else CLASSICAL_COLOR for g in gap]

x = np.arange(len(datasets))
w = 0.32

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Regresión cuántica vs clásica: la ventaja solo aparece con estructura alineada\n"
    "Métrica: $R^2$ (una partición)  |  Modelo cuántico: QK-SVR (ZZFeatureMap)",
    fontsize=12, fontweight="bold", y=1.02, color=CARBON_UNIR
)

ax0 = axes[0]
bq = ax0.bar(x - w / 2, r2_quantum, w,
             color=QUANTUM_COLOR, label="QK-SVR (cuántico)", edgecolor="white", linewidth=1.2)
bc = ax0.bar(x + w / 2, r2_classical, w,
             color=CLASSICAL_COLOR, label="Mejor clásico", edgecolor="white", linewidth=1.2)

ax0.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax0.set_xticks(x)
ax0.set_xticklabels(datasets, fontsize=9)
ax0.set_ylabel("$R^2$", fontsize=11)
techo = max(r2_quantum.max(), r2_classical.max())
suelo = min(r2_quantum.min(), r2_classical.min(), 0)
ax0.set_ylim(suelo - 0.15, techo + 0.3)
ax0.set_title("$R^2$: QK-SVR vs mejor clásico\npor tipo de dataset", fontweight="bold")
ax0.legend(fontsize=8, loc="upper right")

for bar, val in zip(list(bq) + list(bc), list(r2_quantum) + list(r2_classical)):
    ypos = val + 0.022 if val >= 0 else val - 0.05
    ax0.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

for i, (lbl, verd) in enumerate(zip(labels_classical, veredictos)):
    top = max(r2_quantum[i], r2_classical[i])
    color = QUANTUM_COLOR if gap[i] > 0 else CLASSICAL_COLOR
    ax0.text(x[i], top + 0.08, verd, ha="center", va="bottom",
             fontsize=8, fontweight="bold", color=color)
    ax0.text(x[i] + w / 2, suelo - 0.04, lbl, ha="center", va="top",
             fontsize=7, color="#555555", style="italic")

ax1 = axes[1]
bars_gap = ax1.bar(x, gap, color=gap_colors, edgecolor="white", linewidth=1.2, width=0.5)
ax1.axhline(0, color="black", linewidth=1.2)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=9)
ax1.set_ylabel("Ventaja cuántica ($\\Delta R^2$)", fontsize=10)
ax1.set_title(f"Solo el nativo es positivo ({gap[0]:+.3f}, marginal)\n(QK-SVR − mejor clásico)",
              fontweight="bold")
ax1.set_ylim(gap.min() - 0.2, gap.max() + 0.3)

for bar, val, verd in zip(bars_gap, gap, veredictos):
    ypos = val + 0.03 if val >= 0 else val - 0.05
    ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:+.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
    ax1.text(bar.get_x() + bar.get_width() / 2,
             gap.max() + 0.15 if val > 0 else gap.min() - 0.1, verd,
             ha="center", va="center", fontsize=8, fontweight="bold",
             color=QUANTUM_COLOR if val > 0 else CLASSICAL_COLOR)

plt.tight_layout()
out_path = OUT / "03_regresion_3datasets.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", metadata={"paleta": "unir"})
plt.close()
print(f"✅  Guardado: {out_path}")

print("\n🎉  Todas las figuras de Categoría C generadas correctamente.")
