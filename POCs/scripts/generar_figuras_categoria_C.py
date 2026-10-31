"""
Genera las figuras de Categoría C para el TFM.

  10_mds_grafo_solucion.png      — grafo social MDS + solución óptima (C3)
  10_qaoa_convergencia.png       — convergencia QAOA p=1,2,3 con datos reales de qml_v2.mds_qaoa (C1)
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# C3 — Grafo social MDS con solución óptima D*={Ana, Enrique}
# ──────────────────────────────────────────────────────────────────────────────
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
    "Mario": (-1.5, 1.0),
    "Sarah": (0.0, 1.8),
    "Raúl": (1.5, 1.0),
    "Ana": (-0.8, -0.2),
    "Enrique": (0.8, -0.2),
    "Saúl": (0.0, -1.5),
}

node_colors = []
node_sizes = []
for n_ in G.nodes():
    if n_ in solution:
        node_colors.append("#0E4A75")  # azul oscuro → influencer
        node_sizes.append(1600)
    else:
        node_colors.append("#219BC3")  # azul → no contratado
        node_sizes.append(1100)

fig, ax = plt.subplots(figsize=(8, 7))

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", width=2.2, alpha=0.65)

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=2.5, edgecolors="white")

nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold", font_color="white")

# Leyenda
patch_inf = mpatches.Patch(color="#0E4A75", label=r"$D^*$ = {Ana, Enrique}  (influencers contratados, coste = 2)")
patch_std = mpatches.Patch(color="#219BC3", label="Nodo cubierto (≥ 1 vecino en $D^*$)")
ax.legend(
    handles=[patch_inf, patch_std],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=1,
    fontsize=9,
    framealpha=0.9,
    edgecolor="#cccccc",
)

# Anotación de cobertura
for node in G.nodes():
    if node not in solution:
        # flecha suave desde el nodo hacia su influencer vecino
        nbrs_in_sol = [nb for nb in G.neighbors(node) if nb in solution]
        if nbrs_in_sol:
            ax.annotate(
                "",
                xy=pos[nbrs_in_sol[0]],
                xytext=pos[node],
                arrowprops={"arrowstyle": "->", "color": "#858591", "lw": 1.5, "connectionstyle": "arc3,rad=0.12"},
            )

ax.set_title(
    "Conjunto Dominante Mínimo (MDS) — Red Social de 6 nodos\n"
    r"Problema: $\min|D|$ s.t. $\forall v \in V,\; N[v]\cap D \neq \emptyset$",
    fontsize=12,
    fontweight="bold",
    pad=14,
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

reinicios = pd.read_csv(METRICS / "10_mds_qaoa_reinicios.csv")
trayectorias = pd.read_csv(METRICS / "10_mds_qaoa_trayectorias.csv")
e_opt = float(_info["optimo_coste"])
estilo = {
    1: {"color": "#219BC3", "marker": "o", "label": "p = 1  (2 parámetros)"},
    2: {"color": "#858591", "marker": "s", "label": "p = 2  (4 parámetros)"},
    3: {"color": "#0E4A75", "marker": "^", "label": "p = 3  (6 parámetros)"},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Convergencia del loop híbrido QAOA–COBYLA — MDS (6 nodos, A=1, B=2)\n"
    f"Coste QUBO esperado $\\langle C \\rangle$ — {int(reinicios['restart'].nunique())} reinicios por profundidad, "
    f"COBYLA maxiter={_info['maxiter']}",
    fontsize=12,
    fontweight="bold",
)
ax_sort, ax_conv = axes

for p, cfg in estilo.items():
    r = reinicios[reinicios["p"] == p].sort_values("energy_final")
    ax_sort.plot(
        range(1, len(r) + 1),
        r["energy_final"].to_numpy(),
        color=cfg["color"],
        marker=cfg["marker"],
        linewidth=2,
        markersize=6,
        label=cfg["label"],
    )
    mejor = int(r.iloc[0]["restart"])
    t = trayectorias[(trayectorias["p"] == p) & (trayectorias["restart"] == mejor)]
    ax_conv.plot(t["iteration"], t["energy"], color=cfg["color"], linewidth=1.6, alpha=0.9, label=cfg["label"])

for ax in axes:
    ax.axhline(
        e_opt,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.65,
        label=f"Coste óptimo $C^*={e_opt:.0f}$ (fuerza bruta)",
    )
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


print("\n🎉  Todas las figuras de Categoría C generadas correctamente.")
