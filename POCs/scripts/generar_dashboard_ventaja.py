"""
Genera la figura del TFM:
  07_dashboard_ventaja_cuantica.png

Dashboard 3×3 con los nueve casos de uso (Tabla de síntesis del Cap. 5).
"""
import warnings

warnings.filterwarnings("ignore")

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_FIGURES = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"

QUANTUM_COLOR = "#7B2FBE"
CLASSICAL_COLOR = "#1976D2"
ADVANTAGE_COLOR = "#2E7D32"
NEUTRAL_COLOR = "#F57C00"
DISADV_COLOR = "#D32F2F"

# Resultados alineados con Tabla~\ref{tab:sintesis} (evaluacion.tex)
CASES = [
    {
        "id": 1,
        "title": "01 · QSVM\nClasificación",
        "ylabel": "Δ Accuracy (pp)",
        "bars": [
            ("Cuántico-nativo", 38.3),
            ("SKG", -4.5),
            ("Referencia", -3.8),
        ],
    },
    {
        "id": 2,
        "title": "02 · QK-SVR\nRegresión",
        "ylabel": "Δ R² (×100)",
        "bars": [
            ("Cuántico-nativo", -92.8),
            ("SKG", -56.2),
            ("Referencia", 9.3),
        ],
    },
    {
        "id": 3,
        "title": "03 · QAOA\nAsignación SKG",
        "ylabel": "Δ coste vs Greedy (%)",
        "bars": [("SKG / sintético", -30.3)],
    },
    {
        "id": 4,
        "title": "04 · VQE\nPortafolio",
        "ylabel": "Δ Sharpe vs Markowitz (%)",
        "bars": [("Sintético n≤6", 5.3)],
    },
    {
        "id": 5,
        "title": "05 · QAOA TSP\nColombia n=4",
        "ylabel": "Gap vs óptimo (%)",
        "bars": [("NN+2-opt = QAOA", 0.0)],
    },
    {
        "id": 6,
        "title": "06 · QSVM\nUmbral (Liu)",
        "ylabel": "Δ Accuracy (pp)",
        "bars": [
            ("Cuántico-nativo", 8.9),
            ("SKG", -6.7),
            ("Referencia", 6.7),
        ],
    },
    {
        "id": 7,
        "title": "07 · QAOA MDS\nConj. dominante",
        "ylabel": "P(óptimo) QAOA (%)",
        "bars": [("Grafo sintético", 73.0)],
    },
    {
        "id": 8,
        "title": "08 · OCSVM-ZZ\nAnomalías",
        "ylabel": "Δ AUC vs IF (pp)",
        "bars": [
            ("zz_phase", 31.2),
            ("SKG inyectado", -25.0),
            ("moons / blobs", -30.0),
        ],
    },
    {
        "id": 9,
        "title": "09 · QK-SVR\nLags",
        "ylabel": "Mejora rel. MAE (%)",
        "bars": [
            ("zz_lag", 86.5),
            ("sine", -154.6),
            ("skg_speed", -325.8),
        ],
    },
]


def outcome_flag(case: dict) -> tuple[str, str]:
    """Return (label, color) for panel corner badge."""
    vals = [v for _, v in case["bars"]]
    if case["id"] == 5:
        return "≈ Empate ópt.", NEUTRAL_COLOR
    if case["id"] == 7:
        return "✅ P(opt) 73%", ADVANTAGE_COLOR
    if any(v > 5 for v in vals):
        if all(v <= 0 for v in vals):
            return "❌ Clásico", DISADV_COLOR
        if any(v > 0 for v in vals) and any(v < 0 for v in vals):
            return "⚡ Condicional", NEUTRAL_COLOR
        return "✅ Ventaja", ADVANTAGE_COLOR
    if all(v < 0 for v in vals):
        return "❌ Clásico", DISADV_COLOR
    if any(v > 0 for v in vals):
        return "⚡ Condicional", NEUTRAL_COLOR
    return "≈ Empate", NEUTRAL_COLOR


def bar_color(val: float, case_id: int) -> str:
    if case_id == 5 and abs(val) < 0.01:
        return NEUTRAL_COLOR
    if case_id == 7:
        return QUANTUM_COLOR
    if val > 0:
        return ADVANTAGE_COLOR
    if val < 0:
        return DISADV_COLOR
    return NEUTRAL_COLOR


def count_outcomes() -> tuple[int, int, int]:
    """Count favorable / conditional / unfavorable cases."""
    fav = cond = unfav = 0
    for case in CASES:
        vals = [v for _, v in case["bars"]]
        if case["id"] in (5,):
            cond += 1
        elif case["id"] in (7,):
            fav += 1
        elif any(v > 5 for v in vals) and not all(v < 0 for v in vals):
            if any(v < -5 for v in vals):
                cond += 1
            else:
                fav += 1
        elif all(v < -5 for v in vals):
            unfav += 1
        else:
            cond += 1
    return fav, cond, unfav


plt.rcParams.update(
    {"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False}
)

fig = plt.figure(figsize=(22, 20))
fig.suptitle(
    "Dashboard de Ventaja Cuántica — Nueve Casos de Uso\n"
    "Juan David Prada Malagón  ·  SKG Tecnología  ·  UNIR 2026",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

gs = gridspec.GridSpec(
    3, 3, figure=fig, hspace=0.55, wspace=0.38, left=0.08, right=0.97, top=0.94, bottom=0.08
)

for idx, case in enumerate(CASES):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    labels = [b[0] for b in case["bars"]]
    values = [b[1] for b in case["bars"]]
    colors = [bar_color(v, case["id"]) for v in values]
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_title(case["title"], fontweight="bold", fontsize=10, pad=6)
    ax.set_xlabel(case["ylabel"], fontsize=8.5)

    vmax = max(abs(v) for v in values) if values else 1
    pad = max(vmax * 0.12, 3)
    ax.set_xlim(min(values + [0]) - pad, max(values + [0]) + pad)

    for bar, val in zip(bars, values):
        xpos = val + (pad * 0.08 if val >= 0 else -pad * 0.08)
        ha = "left" if val >= 0 else "right"
        fmt = f"{val:+.1f}" if case["id"] != 7 else f"{val:.0f}%"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, fmt, va="center", ha=ha, fontsize=8)

    flag, fcolor = outcome_flag(case)
    ax.text(
        0.98,
        0.02,
        flag,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=fcolor,
        fontweight="bold",
    )

n_fav, n_cond, n_unfav = count_outcomes()
summary = (
    f"Resumen: {n_fav} favorables · {n_cond} condicionales/empate · {n_unfav} desfavorables  |  "
    "Ventaja cuando el feature map está alineado con el generador de datos"
)
fig.text(0.5, 0.02, summary, ha="center", fontsize=9.5, color="#444")

patch_q = mpatches.Patch(color=QUANTUM_COLOR, label="Métrica cuántica / híbrida")
patch_adv = mpatches.Patch(color=ADVANTAGE_COLOR, label="Ventaja cuántica")
patch_neu = mpatches.Patch(color=NEUTRAL_COLOR, label="Empate / condicional")
patch_dis = mpatches.Patch(color=DISADV_COLOR, label="Clásico superior")
fig.legend(
    handles=[patch_q, patch_adv, patch_neu, patch_dis],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.045),
    ncol=4,
    fontsize=9,
    framealpha=0.92,
    edgecolor="#cccccc",
)

OUT_FIGURES.mkdir(parents=True, exist_ok=True)
plt.savefig(
    OUT_FIGURES / "07_dashboard_ventaja_cuantica.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("✅ Dashboard guardado: 07_dashboard_ventaja_cuantica.png")
print(f"   Casos: {len(CASES)}/9  |  Favorables: {n_fav}  Condicionales: {n_cond}  Desfavorables: {n_unfav}")
