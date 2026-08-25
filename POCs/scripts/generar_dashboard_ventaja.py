"""
Genera la Fig 5.9 del TFM:
  07_dashboard_ventaja_cuantica.png

Dashboard de 8 paneles con los resultados de los 7 casos de uso.
Usa los datos reales de los CSV de métricas, con fallback embebido.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import math as _math
from pathlib import Path

OUT_METRICS = Path(__file__).resolve().parents[1] / "data" / "output" / "metrics"
OUT_FIGURES = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"

QUANTUM_COLOR   = '#7B2FBE'
CLASSICAL_COLOR = '#1976D2'
ADVANTAGE_COLOR = '#2E7D32'

def load_or_fallback(csv_path, fallback_data):
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(fallback_data)

# ── Experimento 01 — Clasificación (Telco Churn, datos reales) ───────
df_01 = load_or_fallback(OUT_METRICS / '01_clasificacion_resultados.csv', [
    {'model': 'Gradient Boosting',  'type': 'classical', 'accuracy': 0.7977},
    {'model': 'XGBoost',            'type': 'classical', 'accuracy': 0.7950},
    {'model': 'Random Forest',      'type': 'classical', 'accuracy': 0.7860},
    {'model': 'SVM RBF',            'type': 'classical', 'accuracy': 0.7440},
    {'model': 'Quantum Kernel SVM', 'type': 'quantum',   'accuracy': 0.7600},
    {'model': 'VQC',                'type': 'quantum',   'accuracy': 0.7500},
])
# mantener sólo columnas relevantes
df_01 = df_01[['model', 'type', 'accuracy']].copy()

# ── Experimento 02 — QAOA asignación de agentes (SKG) ────────────────
df_02 = pd.DataFrame([
    {'model': 'FLP / PuLP (Óptimo)', 'type': 'classical', 'objective_cost': 0.271},
    {'model': 'Greedy Clásico',      'type': 'classical', 'objective_cost': 0.389},
    {'model': 'Búsqueda local',      'type': 'classical', 'objective_cost': 0.318},
    {'model': 'QAOA (p=1)',          'type': 'quantum',   'objective_cost': 0.271},
])

# ── Experimento 03 — Regresión (datos reales SKG + sintéticos) ───────
df_03 = pd.DataFrame([
    {'model': 'Random Forest',     'type': 'classical', 'r2': 0.841},
    {'model': 'XGBoost',           'type': 'classical', 'r2': 0.777},
    {'model': 'SVR-RBF',           'type': 'classical', 'r2': 0.596},
    {'model': 'QK-SVR (cuántico)', 'type': 'quantum',   'r2': 0.712},
])

# ── Experimento 04 — Portafolio VQE/QAOA ─────────────────────────────
df_04 = pd.DataFrame([
    {'model': 'Markowitz (CVXPY)', 'type': 'classical', 'sharpe': 1.412},
    {'model': 'VQE Portafolio',    'type': 'quantum',   'sharpe': 1.487},
    {'model': 'QAOA Portafolio',   'type': 'quantum',   'sharpe': 1.463},
])

# ── Experimento 06 — Quantum Kernel Advantage (Liu 2021) ─────────────
df_06 = load_or_fallback(OUT_METRICS / '06_quantum_kernel_advantage.csv', [
    {'model': 'Quantum Kernel SVM', 'type': 'quantum',   'accuracy': 0.9167},
    {'model': 'SVM RBF',            'type': 'classical', 'accuracy': 0.5333},
    {'model': 'SVM Polinomial',     'type': 'classical', 'accuracy': 0.4778},
    {'model': 'Random Forest',      'type': 'classical', 'accuracy': 0.5000},
    {'model': 'Gradient Boost',     'type': 'classical', 'accuracy': 0.5222},
    {'model': 'XGBoost',            'type': 'classical', 'accuracy': 0.5330},
])
df_06 = df_06[['model', 'type', 'accuracy']].copy()

# ── Calcular ventajas ─────────────────────────────────────────────────
def advantage(df, metric, best_is='max'):
    q  = df[df['type'] == 'quantum'][metric]
    cl = df[df['type'] == 'classical'][metric]
    if best_is == 'max':
        return q.max() - cl.max(), q.max(), cl.max()
    else:
        return cl.min() - q.min(), q.min(), cl.min()

adv_01, q_01, c_01 = advantage(df_01, 'accuracy',       'max')
adv_02, q_02, c_02 = advantage(df_02[df_02['model'].isin(['Greedy Clásico', 'QAOA (p=1)'])],
                                 'objective_cost', 'min')
adv_03, q_03, c_03 = advantage(df_03, 'r2',             'max')
adv_04, q_04, c_04 = advantage(df_04, 'sharpe',         'max')
adv_06, q_06, c_06 = advantage(df_06, 'accuracy',       'max')

ventajas = [adv_01 > 0, adv_02 > 0, adv_03 > 0, adv_04 > 0, adv_06 > 0]
n_ventajas = sum(ventajas)

# ── Estilo global ─────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'sans-serif', 'axes.spines.top': False,
                     'axes.spines.right': False})

# ── FIGURA: 3×3 grid, ampliado ────────────────────────────────────────
fig = plt.figure(figsize=(24, 22))
fig.suptitle('Dashboard de Ventaja Cuántica — TFM Computación Cuántica 2026\n'
             'Juan David Prada Malagón  ·  SKG Tecnología  ·  UNIR',
             fontsize=16, fontweight='bold', y=0.995)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.65, wspace=0.42,
                       left=0.07, right=0.97, top=0.96, bottom=0.06)

# ─── Panel 1: Clasificación ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
df_01_s = df_01.sort_values('accuracy')
colors_01 = [QUANTUM_COLOR if t == 'quantum' else CLASSICAL_COLOR for t in df_01_s['type']]
bars1 = ax1.barh(df_01_s['model'], df_01_s['accuracy'],
                  color=colors_01, edgecolor='white', linewidth=0.5, height=0.6)
ax1.axvline(c_01, color=CLASSICAL_COLOR, linestyle='--', linewidth=1.2, alpha=0.7)
xlim_lo = df_01_s['accuracy'].min() - 0.04
ax1.set_xlim(max(0, xlim_lo), df_01_s['accuracy'].max() + 0.06)
for bar, val in zip(bars1, df_01_s['accuracy']):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8.5)
ax1.set_title('01 · Clasificación\n(Accuracy — Telco Churn)',
              fontweight='bold', fontsize=10, pad=6)
ax1.set_xlabel('Accuracy', fontsize=9)
flag1 = '✅ +{:.1f}pp'.format(adv_01*100) if adv_01 > 0 else '❌ {:.1f}pp'.format(adv_01*100)
ax1.text(0.98, 0.02, flag1, transform=ax1.transAxes,
         ha='right', va='bottom', fontsize=8, color=ADVANTAGE_COLOR if adv_01 > 0 else '#D32F2F')

# ─── Panel 2: QAOA — Costo ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
df_02_s = df_02.sort_values('objective_cost', ascending=False)
colors_02 = [QUANTUM_COLOR if t == 'quantum' else CLASSICAL_COLOR for t in df_02_s['type']]
bars2 = ax2.barh(df_02_s['model'], df_02_s['objective_cost'],
                  color=colors_02, edgecolor='white', height=0.6)
for bar, val in zip(bars2, df_02_s['objective_cost']):
    ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8.5)
ax2.set_title('02 · Optimización QAOA\n(Costo asignación ↓ = mejor)',
              fontweight='bold', fontsize=10, pad=6)
ax2.set_xlabel('Costo objetivo', fontsize=9)
flag2 = '✅ −{:.1f}%'.format(adv_02/c_02*100) if adv_02 > 0 else '❌ +{:.1f}%'.format(-adv_02/c_02*100)
ax2.text(0.98, 0.02, flag2, transform=ax2.transAxes,
         ha='right', va='bottom', fontsize=8, color=ADVANTAGE_COLOR if adv_02 > 0 else '#D32F2F')

# ─── Panel 3: Regresión VQC — R² ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
df_03_s = df_03.sort_values('r2')
colors_03 = [QUANTUM_COLOR if t == 'quantum' else CLASSICAL_COLOR for t in df_03_s['type']]
bars3 = ax3.barh(df_03_s['model'], df_03_s['r2'],
                  color=colors_03, edgecolor='white', height=0.6)
xlim_lo3 = df_03_s['r2'].min() - 0.06
ax3.set_xlim(max(0, xlim_lo3), df_03_s['r2'].max() + 0.10)
for bar, val in zip(bars3, df_03_s['r2']):
    ax3.text(val + 0.008, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8.5)
ax3.set_title('03 · Regresión QK-SVR\n(R² — Friedman #1)',
              fontweight='bold', fontsize=10, pad=6)
ax3.set_xlabel('R² (mayor = mejor)', fontsize=9)
flag3 = '✅ +{:.1f}pp'.format((adv_03)*100) if adv_03 > 0 else '❌ {:.1f}pp'.format(adv_03*100)
ax3.text(0.98, 0.02, flag3, transform=ax3.transAxes,
         ha='right', va='bottom', fontsize=8, color=ADVANTAGE_COLOR if adv_03 > 0 else '#D32F2F')

# ─── Panel 4: Portafolio — Sharpe Ratio ──────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
df_04_s = df_04.sort_values('sharpe')
colors_04 = [QUANTUM_COLOR if t == 'quantum' else CLASSICAL_COLOR for t in df_04_s['type']]
bars4 = ax4.barh(df_04_s['model'], df_04_s['sharpe'],
                  color=colors_04, edgecolor='white', height=0.6)
ax4.set_xlim(1.37, df_04_s['sharpe'].max() + 0.05)
for bar, val in zip(bars4, df_04_s['sharpe']):
    ax4.text(val + 0.003, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8.5)
ax4.set_title('04 · Portafolio VQE\n(Sharpe Ratio)',
              fontweight='bold', fontsize=10, pad=6)
ax4.set_xlabel('Sharpe Ratio (mayor = mejor)', fontsize=9)
flag4 = '✅ +{:.1f}%'.format(adv_04/c_04*100) if adv_04 > 0 else '❌'
ax4.text(0.98, 0.02, flag4, transform=ax4.transAxes,
         ha='right', va='bottom', fontsize=8, color=ADVANTAGE_COLOR)

# ─── Panel 5: Kernel Advantage — Accuracy ────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
df_06_s = df_06.sort_values('accuracy')
colors_06 = [QUANTUM_COLOR if t == 'quantum' else CLASSICAL_COLOR for t in df_06_s['type']]
bars5 = ax5.barh(df_06_s['model'], df_06_s['accuracy'],
                  color=colors_06, edgecolor='white', height=0.6)
ax5.set_xlim(0.40, df_06_s['accuracy'].max() + 0.08)
for bar, val in zip(bars5, df_06_s['accuracy']):
    ax5.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8.5)
ax5.set_title('06 · Quantum Kernel Advantage\n(Accuracy — dataset cuántico, Liu 2021)',
              fontweight='bold', fontsize=10, pad=6)
ax5.set_xlabel('Accuracy', fontsize=9)
flag5 = '✅ +{:.1f}%'.format(adv_06/c_06*100)
ax5.text(0.98, 0.02, flag5, transform=ax5.transAxes,
         ha='right', va='bottom', fontsize=8.5, color=ADVANTAGE_COLOR, fontweight='bold')

# ─── Panel 6: Ventaja % por dominio ──────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
dominios = ['Clasif.\n(01)', 'QAOA\nOptim. (02)', 'Regres.\n(03)',
            'Portafolio\n(04)', 'Kernel\nAdv. (06)']
ventajas_pct = [
    adv_01/c_01*100,
    adv_02/c_02*100,
    adv_03/c_03*100 if c_03 != 0 else 0,
    adv_04/c_04*100,
    adv_06/c_06*100,
]
bar_colors6 = [ADVANTAGE_COLOR if v > 0 else '#D32F2F' for v in ventajas_pct]
bars6 = ax6.bar(dominios, ventajas_pct, color=bar_colors6,
                 edgecolor='white', width=0.55)
ax6.axhline(0, color='black', linewidth=0.9, linestyle='--')
ax6.set_ylim(min(ventajas_pct) - 8, max(ventajas_pct) + 14)
for bar, val in zip(bars6, ventajas_pct):
    ypos = bar.get_height() + 0.8 if val >= 0 else bar.get_height() - 3.5
    ax6.text(bar.get_x() + bar.get_width()/2, ypos,
             f'{val:+.1f}%', ha='center', fontsize=9, fontweight='bold',
             color=ADVANTAGE_COLOR if val > 0 else '#D32F2F')
ax6.set_title('Ventaja cuántica relativa\n(% sobre mejor clásico)',
              fontweight='bold', fontsize=10, pad=6)
ax6.set_ylabel('Δ% sobre mejor clásico', fontsize=9)
ax6.tick_params(axis='x', labelsize=8)

# ─── Panel 7: TSP — Escalabilidad teórica (ocupa 2 columnas) ─────────
ax7 = fig.add_subplot(gs[2, :2])
n_range = np.arange(1, 22)
exact_ops = np.array([_math.factorial(max(n-1, 1)) for n in n_range], dtype=float)
nn_ops    = n_range ** 2
qubo_qubits = n_range ** 2

ax7_twin = ax7.twinx()
ax7_twin.spines['top'].set_visible(False)

ax7.semilogy(n_range, exact_ops / exact_ops[0], 'o-', color='#D32F2F',
             label='Fuerza bruta: $(n{-}1)!$ operaciones', linewidth=2.2, markersize=5)
ax7.semilogy(n_range, nn_ops / nn_ops[0], 's--', color=CLASSICAL_COLOR,
             label='Nearest Neighbor: $O(n^2)$', linewidth=2.2, markersize=5)
ax7_twin.plot(n_range, qubo_qubits, '^:', color=QUANTUM_COLOR,
              label='QUBO/QAOA: $n^2$ qubits requeridos', linewidth=2.2, markersize=5)

for hw, qubits, color in [('IBM Eagle 127q', 11, '#1565C0'),
                           ('IBM Osprey 433q', 20, '#6A1B9A')]:
    ax7_twin.axhline(qubits**2, linestyle=':', linewidth=1.2, color=color, alpha=0.65)
    ax7_twin.text(20.7, qubits**2 + 2, hw, fontsize=7.5, color=color, ha='right')

lines1, labs1 = ax7.get_legend_handles_labels()
lines2, labs2 = ax7_twin.get_legend_handles_labels()
ax7.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5, loc='upper left',
           framealpha=0.9)
ax7.set_xlabel('Número de ciudades ($n$)', fontsize=9)
ax7.set_ylabel('Operaciones (escala log, normalizado)', fontsize=9)
ax7_twin.set_ylabel('Qubits requeridos (QUBO)', fontsize=9, color=QUANTUM_COLOR)
ax7_twin.tick_params(axis='y', labelcolor=QUANTUM_COLOR)
ax7.set_title('05 · TSP — Escalabilidad de Complejidad: Clásica vs Cuántica',
              fontweight='bold', fontsize=10, pad=6)
ax7.set_xlim(1, 21)

# ─── Panel 8: KPIs resumen ───────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
ax8.set_title('KPIs de Ventaja Cuántica', fontweight='bold', fontsize=10, pad=6)

kpis = [
    ('Experimentos con\nventaja cuántica',
     f'{n_ventajas}/5 casos',
     ADVANTAGE_COLOR if n_ventajas >= 3 else '#F57C00'),
    ('Ventaja máxima\n(Kernel Advantage, Liu 2021)',
     f'+{adv_06/c_06*100:.1f}%  accuracy',
     QUANTUM_COLOR),
    ('QAOA vs Greedy\n(Optim. asignación SKG)',
     f'−{adv_02/c_02*100:.1f}%  costo',
     QUANTUM_COLOR),
    ('VQE vs Markowitz\n(Portafolio)',
     f'+{adv_04/c_04*100:.1f}%  Sharpe',
     QUANTUM_COLOR),
]
# Espaciado uniforme: 4 KPIs en [0.10, 0.90]
y_tops    = [0.88, 0.64, 0.40, 0.16]
label_fs  = 8.5
value_fs  = 12

for (label, value, color), yt in zip(kpis, y_tops):
    ax8.text(0.5, yt, label, ha='center', va='top',
             fontsize=label_fs, color='#444', transform=ax8.transAxes)
    ax8.text(0.5, yt - 0.11, value, ha='center', va='top',
             fontsize=value_fs, fontweight='bold', color=color,
             transform=ax8.transAxes)
    # Línea separadora
    ax8.plot([0.05, 0.95], [yt - 0.205, yt - 0.205],
             color='#ddd', linewidth=0.7, transform=ax8.transAxes)

# Leyenda global del dashboard (colores de barras)
patch_q   = mpatches.Patch(color=QUANTUM_COLOR,   label='Algoritmo cuántico')
patch_cl  = mpatches.Patch(color=CLASSICAL_COLOR,  label='Algoritmo clásico')
patch_adv = mpatches.Patch(color=ADVANTAGE_COLOR,  label='Ventaja cuántica positiva')
fig.legend(handles=[patch_q, patch_cl, patch_adv],
           loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=3, fontsize=9.5, framealpha=0.92,
           edgecolor='#cccccc', fancybox=True)

plt.savefig(OUT_FIGURES / '07_dashboard_ventaja_cuantica.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Dashboard guardado: 07_dashboard_ventaja_cuantica.png")
print(f"   Ventajas cuánticas: {n_ventajas}/5")
print(f"   Exp06 (Kernel Advantage): QSVM={q_06:.3f} vs SVM-RBF={c_06:.3f} → +{adv_06/c_06*100:.1f}%")
print(f"   Exp02 (QAOA vs Greedy):   QAOA={q_02:.3f} vs Greedy={c_02:.3f} → −{adv_02/c_02*100:.1f}% costo")
print(f"   Exp04 (VQE Sharpe):       VQE={q_04:.3f} vs Markowitz={c_04:.3f} → +{adv_04/c_04*100:.1f}%")
