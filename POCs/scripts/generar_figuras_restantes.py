"""
Genera figuras restantes del TFM extraídas de notebooks 01, 06, 07, 08 y 09.

Criterio de integridad (no negociable):
  - Alterar resultados o inventar datos es inadmisible.
  - Preferir CSV en POCs/data/output/metrics/ (salida real de notebooks).
  - Fallbacks embebidos = transcripción del log del notebook, no “embellecidos”.
  - Si un artefacto falta (p.ej. matrices K, series θ), omitir — no sintetizar.
"""
from __future__ import annotations

import math
import timeit
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parents[1] / "data" / "output" / "figures"
OUT_METRICS = Path(__file__).resolve().parents[1] / "data" / "output" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)
OUT_METRICS.mkdir(parents=True, exist_ok=True)

QUANTUM_COLOR = "#7B2FBE"
CLASSICAL_COLOR = "#1976D2"
ADVANTAGE_COLOR = "#2E7D32"

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

WRITTEN: list[Path] = []
SKIPPED: list[str] = []


def save_fig(name: str) -> Path:
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    WRITTEN.append(path)
    return path


def load_or_fallback(csv_path: Path, fallback: list[dict]) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(fallback)


def quantum_advantage(df: pd.DataFrame, metric: str, best_is: str = "max"):
    q = df[df["type"] == "quantum"][metric]
    cl = df[df["type"] == "classical"][metric]
    if best_is == "max":
        return q.max() - cl.max(), q.max(), cl.max()
    return cl.min() - q.min(), q.min(), cl.min()


def quantumness_score(A_Q, A_cl, d_F, H_Q, H_cl, d_F_ref=1.0, w=(0.4, 0.3, 0.3)):
    w1, w2, w3 = w
    delta_A = A_Q - A_cl
    d_F_norm = d_F / d_F_ref if d_F_ref > 0 else d_F
    delta_H = H_Q - H_cl
    return w1 * delta_A + w2 * d_F_norm + w3 * delta_H


# ── Datos embebidos (salidas de notebooks) ───────────────────────────────────

DF_01_RESULTS = pd.DataFrame([
    {"model": "Gradient Boosting", "type": "classical", "accuracy": 0.797729,
     "precision": 0.786819, "recall": 0.797729, "f1_score": 0.788752, "training_time_s": 1.52},
    {"model": "Random Forest", "type": "classical", "accuracy": 0.770050,
     "precision": 0.798829, "recall": 0.770050, "f1_score": 0.778869, "training_time_s": 0.79},
    {"model": "XGBoost", "type": "classical", "accuracy": 0.757275,
     "precision": 0.797577, "recall": 0.757275, "f1_score": 0.768326, "training_time_s": 0.33},
    {"model": "SVM (RBF)", "type": "classical", "accuracy": 0.743790,
     "precision": 0.794319, "recall": 0.743790, "f1_score": 0.756619, "training_time_s": 0.47},
    {"model": "Quantum Kernel SVM", "type": "quantum", "accuracy": 0.760000,
     "precision": 0.577600, "recall": 0.760000, "f1_score": 0.656364, "training_time_s": 20.55},
    {"model": "VQC", "type": "quantum", "accuracy": 0.620000,
     "precision": 0.688693, "recall": 0.620000, "f1_score": 0.643628, "training_time_s": 15.71},
])

CMP_DF = pd.DataFrame([
    {"dataset": "Cuántico-nativo (Liu 2021)", "acc_qsvm": 0.5333, "acc_svm_rbf": 0.4667,
     "delta_acc": 0.0667, "winner": "QSVM"},
    {"dataset": "SKG Incidentes movilidad", "acc_qsvm": 0.4667, "acc_svm_rbf": 0.5250,
     "delta_acc": -0.0583, "winner": "SVM-RBF"},
    {"dataset": "Telco Customer Churn", "acc_qsvm": 0.7346, "acc_svm_rbf": 0.7473,
     "delta_acc": -0.0128, "winner": "SVM-RBF"},
])

DF_06 = load_or_fallback(OUT_METRICS / "06_quantum_kernel_advantage.csv", [
    {"model": "Quantum Kernel SVM", "type": "quantum", "accuracy": 0.6667, "f1": 0.6626, "time_s": 18.025},
    {"model": "SVM Polinomial", "type": "classical", "accuracy": 0.6000, "f1": 0.5361, "time_s": 0.050},
    {"model": "Random Forest", "type": "classical", "accuracy": 0.5556, "f1": 0.5542, "time_s": 0.146},
    {"model": "XGBoost", "type": "classical", "accuracy": 0.5556, "f1": 0.5564, "time_s": 0.185},
    {"model": "SVM RBF", "type": "classical", "accuracy": 0.5333, "f1": 0.5343, "time_s": 0.005},
    {"model": "Gradient Boost", "type": "classical", "accuracy": 0.4444, "f1": 0.4444, "time_s": 0.053},
])

SCALE_DF_06 = pd.DataFrame([
    {"n_samples": 50, "acc_classical": 0.733, "acc_quantum": 0.667, "advantage": -0.067},
    {"n_samples": 80, "acc_classical": 0.625, "acc_quantum": 0.625, "advantage": 0.000},
    {"n_samples": 100, "acc_classical": 0.500, "acc_quantum": 0.700, "advantage": 0.200},
    {"n_samples": 120, "acc_classical": 0.417, "acc_quantum": 0.722, "advantage": 0.306},
    {"n_samples": 150, "acc_classical": 0.444, "acc_quantum": 0.733, "advantage": 0.289},
])

# Baseline = mejor clásico operativo (GB/RF en datos reales; SVM-poly en Liu).
# Fuente preferente: metrics/06_qkernel_resumen_3datasets.csv
_SUMMARY_06_FALLBACK = [
    {"Dataset": "1. Cuántico-nativo\n(Liu 2021)", "QSVC": 0.733, "Mejor Clásico": 0.600,
     "veredicto": "VENTAJA Q"},
    {"Dataset": "2. SKG\n(Incidentes Bogotá)", "QSVC": 0.547, "Mejor Clásico": 0.607,
     "veredicto": "GANA CLÁSICO"},
    {"Dataset": "3. Vida real\n(Telco Churn)", "QSVC": 0.760, "Mejor Clásico": 0.798,
     "veredicto": "GANA CLÁSICO"},
]
_summary_path = OUT_METRICS / "06_qkernel_resumen_3datasets.csv"
if _summary_path.exists():
    SUMMARY_06 = pd.read_csv(_summary_path)
    if "Ventaja (pp)" not in SUMMARY_06.columns:
        SUMMARY_06["Ventaja (pp)"] = (SUMMARY_06["QSVC"] - SUMMARY_06["Mejor Clásico"]) * 100
else:
    SUMMARY_06 = pd.DataFrame(_SUMMARY_06_FALLBACK)
    SUMMARY_06["Ventaja (pp)"] = (SUMMARY_06["QSVC"] - SUMMARY_06["Mejor Clásico"]) * 100

# NB08 indicadores: CSV preferente (transcripción del log del notebook).
# advantage_pp aquí = Δ vs RBF (NB08). Δ_op vive aparte.
_RESULTS_08_FALLBACK = [
    {"dataset": "Cuántico-nativo", "A_Q": 0.0181, "A_cl": 0.0469, "delta_A": -0.0288,
     "H_Q": 2.2749, "H_cl": 2.2411, "delta_H": 0.0339, "d_F": 0.3532, "g": 1541.657,
     "Q_score": 0.2858, "advantage_pp_rbf": 0.0, "advantage_pp_op": 13.3},
    {"dataset": "SKG Incidentes", "A_Q": 0.0396, "A_cl": 0.0499, "delta_A": -0.0103,
     "H_Q": 2.1545, "H_cl": 1.9666, "delta_H": 0.1880, "d_F": 0.3689, "g": 2080.864,
     "Q_score": 0.3523, "advantage_pp_rbf": 8.3, "advantage_pp_op": -6.0},
    {"dataset": "Telco Churn", "A_Q": 0.1899, "A_cl": 0.1698, "delta_A": 0.0201,
     "H_Q": 2.3332, "H_cl": 3.4646, "delta_H": -1.1313, "d_F": 0.3457, "g": 89.725,
     "Q_score": -0.0503, "advantage_pp_rbf": 0.0, "advantage_pp_op": -3.8},
]
_df08 = load_or_fallback(OUT_METRICS / "08_test_cuanticidad_resultados.csv", _RESULTS_08_FALLBACK)
RESULTS_08 = []
OPERATIONAL_08 = {}
for _, row in _df08.iterrows():
    adv_rbf = float(row.get("advantage_pp_rbf", row.get("advantage_pp", 0.0)))
    adv_op = float(row["advantage_pp_op"]) if "advantage_pp_op" in row and pd.notna(row["advantage_pp_op"]) else None
    RESULTS_08.append({
        "dataset": row["dataset"],
        "A_Q": float(row["A_Q"]), "A_cl": float(row["A_cl"]), "delta_A": float(row["delta_A"]),
        "H_Q": float(row["H_Q"]), "H_cl": float(row["H_cl"]), "delta_H": float(row["delta_H"]),
        "d_F": float(row["d_F"]), "g": float(row["g"]), "Q_score": float(row["Q_score"]),
        "advantage_pp": adv_rbf,  # canónico NB08 = vs RBF
    })
    if adv_op is not None:
        OPERATIONAL_08[row["dataset"]] = adv_op
if not OPERATIONAL_08:
    OPERATIONAL_08 = {"Cuántico-nativo": 13.3, "SKG Incidentes": -6.0, "Telco Churn": -3.8}

# Barrido α REAL del Notebook 08 (CSV / log). X fijo ⇒ d_F/H/g constantes.
# ρ_S(α, ΔA)=+0.833. NO afirmar ρ(Q, ventaja)=0.833.
DF_SWEEP_08 = load_or_fallback(OUT_METRICS / "08_barrido_sintetico_alpha.csv", [
    {"alpha": 0.00, "A_Q": 0.0174, "A_cl": 0.3909, "delta_A": -0.3735,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": -40.0},
    {"alpha": 0.10, "A_Q": 0.0179, "A_cl": 0.3317, "delta_A": -0.3138,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": -30.0},
    {"alpha": 0.25, "A_Q": 0.0749, "A_cl": 0.2436, "delta_A": -0.1688,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": 0.0},
    {"alpha": 0.40, "A_Q": 0.1786, "A_cl": 0.2537, "delta_A": -0.0751,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": 3.3},
    {"alpha": 0.50, "A_Q": 0.2707, "A_cl": 0.2701, "delta_A": 0.0006,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": 0.0},
    {"alpha": 0.65, "A_Q": 0.3009, "A_cl": 0.2620, "delta_A": 0.0389,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": 13.3},
    {"alpha": 0.75, "A_Q": 0.4209, "A_cl": 0.3331, "delta_A": 0.0878,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": 0.0},
    {"alpha": 0.90, "A_Q": 0.6911, "A_cl": 0.5300, "delta_A": 0.1611,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": 0.0},
    {"alpha": 1.00, "A_Q": 0.0101, "A_cl": 0.0104, "delta_A": -0.0003,
     "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246, "advantage_pp": -3.3},
])

BASELINE_09 = {
    "SKG Incidentes": {"Q_score": 0.1629, "advantage_pp": 8.3, "delta_A": -0.0103,
                       "d_F": 0.3689, "delta_H": 0.1880},
    "Telco Churn": {"Q_score": -0.2277, "advantage_pp": 0.0, "delta_A": 0.0201,
                    "d_F": 0.3457, "delta_H": -1.1313},
}

# NB09 mapeos: CSV preferente (tabla resumen del notebook).
_df09 = load_or_fallback(OUT_METRICS / "09_mapeos_resultados.csv", [])
ALL_RESULTS_09: dict[str, list] = {"SKG Incidentes": [], "Telco Churn": []}
if len(_df09):
    for ds in ALL_RESULTS_09:
        sub = _df09[(_df09["dataset"] == ds) & (_df09["strategy"] != "Baseline")]
        ALL_RESULTS_09[ds] = [
            {"strategy": r["strategy"], "Q_score": float(r["Q_score"]),
             "advantage_pp": float(r["advantage_pp"])}
            for _, r in sub.iterrows()
        ]
        base_row = _df09[(_df09["dataset"] == ds) & (_df09["strategy"] == "Baseline")]
        if len(base_row):
            BASELINE_09[ds]["Q_score"] = float(base_row.iloc[0]["Q_score"])
            BASELINE_09[ds]["advantage_pp"] = float(base_row.iloc[0]["advantage_pp"])
else:
    ALL_RESULTS_09 = {
        "SKG Incidentes": [
            {"strategy": "ZZ-compatible", "Q_score": -0.2033, "advantage_pp": 12.5},
            {"strategy": "Rotación max-dF", "Q_score": 0.1698, "advantage_pp": 0.0},
            {"strategy": "Re-uploading reps=3", "Q_score": 0.2118, "advantage_pp": 16.7},
            {"strategy": "Q-UMAP/MDS", "Q_score": 0.0537, "advantage_pp": -8.3},
        ],
        "Telco Churn": [
            {"strategy": "ZZ-compatible", "Q_score": 0.1360, "advantage_pp": 4.2},
            {"strategy": "Rotación max-dF", "Q_score": 0.0956, "advantage_pp": 4.2},
            {"strategy": "Re-uploading reps=3", "Q_score": 0.1674, "advantage_pp": 4.2},
            {"strategy": "Q-UMAP/MDS", "Q_score": -0.2481, "advantage_pp": 4.2},
        ],
    }

# Post-QCR Telco: SOLO Q y Adv están en el log NB09.
# ΔA/d_F/ΔH post NO se imprimieron → NO inventar (antes se mezclaron cifras 4Q).
QCR_TELCO = {
    "baseline": BASELINE_09["Telco Churn"],
    "post_qcr": {"Q_score": 0.1674, "advantage_pp": 4.2},  # re-uploading reps=3
    "bottleneck": "delta_H",
}


# ── Notebook 07 ────────────────────────────────────────────────────────────────

def fig_07_complejidad():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Complejidad Computacional: Cuántico vs Clásico", fontsize=14, fontweight="bold")

    N = np.logspace(1, 8, 300)
    ax = axes[0]
    ax.loglog(N, N, "-", color=CLASSICAL_COLOR, linewidth=2.5, label="Clásico: $O(N)$")
    ax.loglog(N, np.sqrt(N), "-", color=QUANTUM_COLOR, linewidth=2.5, label="Grover: $O(\\sqrt{N})$")
    ax.fill_between(N, np.sqrt(N), N, alpha=0.12, color=QUANTUM_COLOR, label="Speedup cuadrático")
    ax.set_xlabel("Tamaño del espacio de búsqueda (N)")
    ax.set_ylabel("Operaciones")
    ax.set_title("Búsqueda: Grover vs Clásico\nSpeedup cuadrático")
    ax.legend(fontsize=9)
    ax.set_xlim(10, 1e8)

    ax = axes[1]
    n_range = np.linspace(10, 300, 200)
    classical_factor = np.exp(1.9 * n_range ** (1 / 3) * (np.log(n_range)) ** (2 / 3))
    quantum_shor = n_range ** 3
    classical_factor_norm = classical_factor / classical_factor[0]
    quantum_shor_norm = quantum_shor / quantum_shor[0]
    ax.semilogy(n_range, classical_factor_norm, "-", color=CLASSICAL_COLOR, linewidth=2.5,
                label="Clásico: $O(e^{n^{1/3}(\\log n)^{2/3}})$")
    ax.semilogy(n_range, quantum_shor_norm, "-", color=QUANTUM_COLOR, linewidth=2.5,
                label="Shor: $O(n^3)$")
    ax.fill_between(n_range, quantum_shor_norm, classical_factor_norm, alpha=0.12,
                    color=QUANTUM_COLOR, label="Speedup exponencial")
    ax.set_xlabel("Bits del número a factorizar (n)")
    ax.set_ylabel("Operaciones (normalizado, escala log)")
    ax.set_title("Factorización: Shor vs Clásico\nSpeedup exponencial")
    ax.legend(fontsize=9)
    ax.set_ylim(1, 1e15)

    ax = axes[2]
    cities = np.arange(4, 18)
    exact_time = np.array([math.factorial(max(c - 1, 1)) for c in cities], dtype=float)
    nn_time = cities ** 2
    qubo_qubits_needed = cities ** 2
    ax2_twin = ax.twinx()
    ax.semilogy(cities, exact_time / exact_time[0], "o-", color="#D32F2F", linewidth=2,
                label="Exacto: $(n-1)!$")
    ax.semilogy(cities, nn_time / nn_time[0], "s--", color=CLASSICAL_COLOR, linewidth=2,
                label="Heurístico: $O(n^2)$")
    ax2_twin.plot(cities, qubo_qubits_needed, "^:", color=QUANTUM_COLOR, linewidth=2,
                  label="QUBO/QAOA: $n^2$ qubits")
    ax.set_xlabel("Número de ciudades (TSP)")
    ax.set_ylabel("Tiempo relativo (escala log)")
    ax2_twin.set_ylabel("Qubits requeridos", color=QUANTUM_COLOR)
    ax2_twin.tick_params(axis="y", labelcolor=QUANTUM_COLOR)
    for n_hw, hw_name, hw_color in [(11, "IBM Eagle\n127q", "#1976D2"), (20, "IBM Osprey\n433q", "#7B2FBE")]:
        ax2_twin.axhline(n_hw ** 2, linestyle=":", color=hw_color, linewidth=1.2, alpha=0.8)
        ax2_twin.text(17.8, n_hw ** 2 + 3, hw_name, fontsize=8, color=hw_color, ha="right")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2_twin.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=9, loc="upper left")
    ax.set_title("TSP: Escalabilidad Exacto vs QUBO/QAOA")

    plt.tight_layout()
    save_fig("07_complejidad_cuantico_vs_clasico.png")


def fig_07_kernel_advantage_detalle():
    adv_06, q_06, c_06 = quantum_advantage(DF_06, "accuracy", "max")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Quantum Kernel Advantage — Análisis detallado (Liu et al. 2021)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    df_06_s = DF_06.sort_values("accuracy")
    colors = [QUANTUM_COLOR if t == "quantum" else CLASSICAL_COLOR for t in df_06_s["type"]]
    bars = ax.barh(df_06_s["model"], df_06_s["accuracy"], color=colors, edgecolor="white", height=0.6)
    best_cl_acc = DF_06[DF_06["type"] == "classical"]["accuracy"].max()
    ax.axvline(best_cl_acc, color=CLASSICAL_COLOR, linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Mejor clásico: {best_cl_acc:.3f}")
    for bar, val in zip(bars, df_06_s["accuracy"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=QUANTUM_COLOR, label=f"Cuántico (acc={q_06:.3f})"),
        mpatches.Patch(color=CLASSICAL_COLOR, label=f"Clásico  (acc={c_06:.3f})"),
    ], fontsize=10)
    ax.set_title(f"Accuracy en dataset cuántico\nVentaja: {adv_06:+.4f} ({adv_06 / c_06 * 100:+.1f}%)",
                 fontweight="bold")
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0.50, 1.06)

    ax2 = axes[1]
    best_q_row = DF_06[DF_06["type"] == "quantum"].sort_values("accuracy").iloc[-1]
    best_cl_row = DF_06[DF_06["type"] == "classical"].sort_values("accuracy").iloc[-1]
    metrics_labels = ["Accuracy", "F1-Score", "Velocidad\n(1/tiempo norm.)"]
    max_time = DF_06["time_s"].max()
    q_vals = [best_q_row["accuracy"], best_q_row.get("f1", best_q_row["accuracy"]),
              1 - best_q_row["time_s"] / max_time]
    cl_vals = [best_cl_row["accuracy"], best_cl_row.get("f1", best_cl_row["accuracy"]),
               1 - best_cl_row["time_s"] / max_time]
    x = np.arange(len(metrics_labels))
    width = 0.35
    b1 = ax2.bar(x - width / 2, q_vals, width, color=QUANTUM_COLOR,
                 label="Quantum Kernel SVM", edgecolor="white")
    b2 = ax2.bar(x + width / 2, cl_vals, width, color=CLASSICAL_COLOR,
                 label="SVM RBF / XGBoost", edgecolor="white")
    for bar, val in zip(list(b1) + list(b2), q_vals + cl_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_labels)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Valor (normalizado 0-1)")
    ax2.set_title("Comparación multimétrica\nQuantum Kernel SVM vs Mejor Clásico", fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    save_fig("07_kernel_advantage_detalle.png")


def fig_07_paralelismo():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Paralelismo Cuántico — Superposición con n qubits (Hadamard)",
                 fontsize=13, fontweight="bold")
    for ax, n_q in zip(axes, [2, 4, 8]):
        qc = QuantumCircuit(n_q)
        qc.h(range(n_q))
        sv = Statevector(qc)
        amplitudes = np.abs(sv.data) ** 2
        states = [f"|{i:0{n_q}b}⟩" for i in range(2 ** n_q)]
        if 2 ** n_q <= 16:
            ax.bar(range(2 ** n_q), amplitudes, color=QUANTUM_COLOR, edgecolor="white", alpha=0.8)
            ax.set_xticks(range(2 ** n_q))
            ax.set_xticklabels(states, rotation=45, fontsize=7)
        else:
            ax.bar(range(2 ** n_q), amplitudes, color=QUANTUM_COLOR, width=1.0, edgecolor="none", alpha=0.8)
            ax.set_xlabel(f"{2 ** n_q} estados (representación comprimida)")
        ax.set_ylim(0, 1.0 / (2 ** n_q) * 2.5)
        ax.axhline(1.0 / (2 ** n_q), color="orange", linestyle="--", linewidth=1.5,
                   label=f"$1/2^{n_q}$ = {1 / 2 ** n_q:.4f}")
        ax.set_title(f"{n_q} qubits → $2^{{{n_q}}}$ = {2 ** n_q} estados simultáneos\n"
                     f"Clásico necesitaría {2 ** n_q} evaluaciones", fontweight="bold")
        ax.set_ylabel("Probabilidad")
        ax.legend(fontsize=9)
        ax_inset = ax.inset_axes([0.65, 0.55, 0.33, 0.40])
        ax_inset.axis("off")
        ax_inset.text(0.5, 0.5, f"H^⊗{n_q}|0⟩", ha="center", va="center", fontsize=11,
                      color=QUANTUM_COLOR, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor=QUANTUM_COLOR))
    plt.tight_layout()
    save_fig("07_paralelismo_cuantico.png")


def fig_07_grover_benchmark():
    def classical_search(database, target):
        for i, item in enumerate(database):
            if item == target:
                return i
        return -1

    def grover_iterations(n):
        return int(np.pi / 4 * np.sqrt(n))

    search_sizes = [16, 64, 256, 1024, 4096, 16384, 65536]
    benchmark_data = []
    for n in search_sizes:
        db = list(range(n))
        target = n - 1
        t_classical = timeit.timeit(lambda: classical_search(db, target), number=50) / 50 * 1000
        grover_ops = grover_iterations(n)
        benchmark_data.append({
            "N": n, "n_qubits": int(np.ceil(np.log2(n))),
            "classical_ops": n, "grover_ops": grover_ops,
            "speedup": n / grover_ops, "classical_time_ms": round(t_classical, 4),
        })
    bench_df = pd.DataFrame(benchmark_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Benchmark: Grover vs Búsqueda Clásica", fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.loglog(bench_df["N"], bench_df["classical_ops"], "o-", color=CLASSICAL_COLOR,
              linewidth=2.5, label="Clásico: $O(N)$ comparaciones")
    ax.loglog(bench_df["N"], bench_df["grover_ops"], "s-", color=QUANTUM_COLOR,
              linewidth=2.5, label="Grover: $O(\\sqrt{N})$ iteraciones")
    ax.fill_between(bench_df["N"], bench_df["grover_ops"], bench_df["classical_ops"],
                    alpha=0.10, color=QUANTUM_COLOR, label="Región de ventaja")
    for _, row in bench_df.iterrows():
        ax.annotate(f"{row['speedup']:.0f}x", (row["N"], row["grover_ops"]),
                    textcoords="offset points", xytext=(5, -12), fontsize=8, color=QUANTUM_COLOR)
    ax.set_xlabel("Tamaño del espacio de búsqueda (N)")
    ax.set_ylabel("Número de operaciones")
    ax.set_title("Operaciones teóricas (Grover vs Clásico)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogx(bench_df["N"], bench_df["speedup"], "D-", color=ADVANTAGE_COLOR,
                linewidth=2.5, markersize=8, label="Speedup real (N / √N = √N)")
    n_ref = np.array(search_sizes)
    ax.semilogx(n_ref, np.sqrt(n_ref), "--", color="gray", linewidth=1.5,
                label="Teórico: $\\sqrt{N}$")
    for _, row in bench_df.iterrows():
        ax.text(row["N"], row["speedup"] + 1, f"{row['speedup']:.0f}×",
                ha="center", fontsize=9, fontweight="bold", color=ADVANTAGE_COLOR)
    ax.set_xlabel("Tamaño del espacio de búsqueda (N)")
    ax.set_ylabel("Speedup cuántico (veces más rápido)")
    ax.set_title("Speedup cuántico de Grover")
    ax.legend(fontsize=9)
    ax.set_ylim(0, bench_df["speedup"].max() * 1.2)
    plt.tight_layout()
    save_fig("07_grover_benchmark.png")


def fig_07_radar():
    dimensions = [
        "Calidad\nde solución", "Ventaja\nrelativa (%)", "Escalabilidad\nteórica",
        "Robustez\n(consistencia)", "Aplicabilidad\nSKG",
    ]
    profiles = {
        "Clasificación\n(QSVM, dataset cuántico)": {"scores": [9.2, 9.5, 7.0, 9.0, 6.0],
                                                    "color": "#7B2FBE", "fill": "#DBC8F0"},
        "Optimización\n(QAOA, asignación SKG)": {"scores": [7.5, 7.8, 8.5, 6.5, 9.5],
                                                  "color": "#1976D2", "fill": "#BBDEFB"},
        "Portafolio\n(VQE/QAOA)": {"scores": [7.0, 5.3, 8.0, 7.0, 7.5],
                                    "color": "#2E7D32", "fill": "#C8E6C9"},
        "Kernel Advantage\n(Liu 2021, cuántico-nativo)": {"scores": [9.8, 10.0, 9.5, 9.8, 4.5],
                                                           "color": "#F57F17", "fill": "#FFF9C4"},
        "Anomalías\n(OCSVM-ZZ, zz_phase)": {"scores": [8.6, 8.5, 6.5, 8.0, 5.5],
                                             "color": "#C62828", "fill": "#FFCDD2"},
        "Forecasting\n(QK-SVR, zz_lag)": {"scores": [9.0, 9.2, 6.0, 8.5, 5.0],
                                           "color": "#00695C", "fill": "#B2DFDB"},
        "TSP / MDS\n(QAOA, protocolo)": {"scores": [7.0, 3.0, 7.5, 7.0, 8.0],
                                          "color": "#5D4037", "fill": "#D7CCC8"},
    }
    n_dim = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]

    def draw_radar(ax, scores, color, fill, alpha_fill=0.2):
        vals = scores + scores[:1]
        ax.plot(angles, vals, "o-", color=color, linewidth=2.5, markersize=6, zorder=5)
        ax.fill(angles, vals, color=fill, alpha=alpha_fill, zorder=4)

    fig, axes_r = plt.subplots(1, 2, figsize=(18, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Perfil Multidimensional de Ventaja Cuántica\nTFM Computación Cuántica — Juan David Prada | UNIR 2026",
                 fontsize=14, fontweight="bold", y=1.02)
    for ax_r in axes_r:
        for level in [2, 4, 6, 8, 10]:
            ax_r.plot(angles, [level] * (n_dim + 1), "-", color="#ddd", lw=0.7, zorder=1)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(dimensions, fontsize=9.5)
        ax_r.set_ylim(0, 10)
        ax_r.set_yticks([2, 4, 6, 8, 10])
        ax_r.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color="gray")
        ax_r.spines["polar"].set_visible(False)
        ax_r.grid(color="#eee", linestyle="-", linewidth=0.5)

    for name_r, prof in profiles.items():
        draw_radar(axes_r[0], prof["scores"], prof["color"], prof["fill"])
    axes_r[0].set_title("Comparativa de todos los experimentos", fontweight="bold", pad=18)
    axes_r[0].legend([p for p in profiles], loc="upper right", bbox_to_anchor=(1.5, 1.15),
                     fontsize=8.5, framealpha=0.8, title="Experimento", title_fontsize=9)

    for name_r in ["Clasificación\n(QSVM, dataset cuántico)",
                   "Anomalías\n(OCSVM-ZZ, zz_phase)",
                   "Forecasting\n(QK-SVR, zz_lag)",
                   "Optimización\n(QAOA, asignación SKG)"]:
        prof = profiles[name_r]
        draw_radar(axes_r[1], prof["scores"], prof["color"], prof["fill"], alpha_fill=0.25)
    draw_radar(axes_r[1], [5.0] * n_dim, "#999", "#eee", 0.1)
    axes_r[1].set_title("Perfiles de mayor ventaja cuántica", fontweight="bold", pad=18)
    axes_r[1].legend(["Referencia clásica (baseline)"], loc="upper right",
                     bbox_to_anchor=(1.55, 1.15), fontsize=8.5, framealpha=0.8)
    plt.tight_layout()
    save_fig("07_radar_ventaja_cuantica.png")


def fig_07_heatmap():
    data_heatmap = {
        "Dataset\ncuántico-nativo": [
            38.3, -92.8, -30.3, 5.3, 0.0, 8.9, 73.0, 31.2, 86.5
        ],
        "Dataset SKG\n(datos reales)": [
            -4.5, -56.2, -30.3, 5.3, 0.0, -6.7, 73.0, -25.0, -325.8
        ],
        "Dataset de\nreferencia pública": [
            -3.8, 9.3, None, None, 0.0, 6.7, None, -30.0, -154.6
        ],
    }
    index_rows = [
        "CU1: QSVM\n(clasificación)",
        "CU2: QK-SVR\n(regresión)",
        "CU3: QAOA\n(asignación)",
        "CU4: VQE/QAOA\n(portafolio)",
        "CU5: QAOA TSP\n(protocolo justo)",
        "CU6: QSVM\n(umbral Liu)",
        "CU7: QAOA MDS\n(conj. dominante)",
        "CU8: OCSVM-ZZ\n(anomalías)",
        "CU9: QK-SVR\n(lags)",
    ]
    df_hm = pd.DataFrame(data_heatmap, index=index_rows)
    mask = df_hm.isnull()
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle("Ventaja Cuántica (Δ pp, Δ% o P(ópt)) por Caso de Uso y Dataset\n"
                 "Verde = ventaja cuántica | Rojo = clásico superior | Gris = no aplica",
                 fontsize=12, fontweight="bold")
    sns.heatmap(df_hm.astype(float), ax=ax, cmap=plt.cm.RdYlGn, center=0,
                vmin=-80, vmax=80, annot=True, fmt=".1f", linewidths=0.5,
                linecolor="#eee", mask=mask,
                cbar_kws={"label": "Ventaja cuántica (pp o Δ%)  [+ = cuántico mejor]"},
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    for r_i in range(df_hm.shape[0]):
        for c_i in range(df_hm.shape[1]):
            if mask.iloc[r_i, c_i]:
                ax.add_patch(plt.Rectangle((c_i, r_i), 1, 1, fill=True, color="#E0E0E0", zorder=5))
                ax.text(c_i + 0.5, r_i + 0.5, "N/A", ha="center", va="center",
                        fontsize=10, color="#999", zorder=6)
    ax.set_xlabel("Tipo de dataset", fontsize=11)
    ax.set_ylabel("Dominio / Algoritmo cuántico", fontsize=11)
    plt.tight_layout()
    save_fig("07_heatmap_ventaja_por_dominio.png")


# ── Notebook 01 ────────────────────────────────────────────────────────────────

def fig_01_clasificacion_comparacion():
    results_df = DF_01_RESULTS.sort_values("f1_score", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Clasificacion Quantum vs Clasico — Telco Customer Churn", fontsize=14)
    palette = results_df["type"].map({"quantum": "#7B2FBE", "classical": "#2196F3"})
    for ax, metric, label in zip(axes,
                                 ["f1_score", "accuracy", "training_time_s"],
                                 ["F1-Score (weighted)", "Accuracy", "Tiempo de entrenamiento (s)"]):
        bars = ax.barh(results_df["model"], results_df[metric], color=palette.values, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.invert_yaxis()
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    fig.legend(handles=[Patch(facecolor="#7B2FBE", label="Cuantico"),
                        Patch(facecolor="#2196F3", label="Clasico")],
               loc="lower right", bbox_to_anchor=(0.99, 0.01))
    plt.tight_layout()
    save_fig("01_clasificacion_comparacion.png")


def fig_01_qsvm_vs_svm_directo():
    cmp_df = CMP_DF
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Comparativa directa: QSVM (kernel cuántico) vs SVM-RBF (kernel clásico)\n"
                 "Condiciones controladas: mismo SVC | C=1.0 | PCA 2 componentes",
                 fontsize=12, fontweight="bold")
    datasets_lbl = cmp_df["dataset"].tolist()
    x_idx = np.arange(len(datasets_lbl))
    w = 0.35
    bars_q = ax.bar(x_idx - w / 2, cmp_df["acc_qsvm"], w, label="QSVM (kernel cuántico)",
                    color="#7B2FBE", edgecolor="white", alpha=0.9)
    bars_rbf = ax.bar(x_idx + w / 2, cmp_df["acc_svm_rbf"], w, label="SVM-RBF (kernel clásico)",
                      color="#2196F3", edgecolor="white", alpha=0.9)
    for pos, (_, row) in enumerate(cmp_df.iterrows()):
        y_top = max(row["acc_qsvm"], row["acc_svm_rbf"]) + 0.02
        color_delta = "#2E7D32" if row["delta_acc"] > 0 else "#D32F2F"
        ax.annotate(f"Δ={row['delta_acc']:+.3f}", xy=(pos, y_top),
                    ha="center", fontsize=9, fontweight="bold", color=color_delta)
    for bar_set in [bars_q, bars_rbf]:
        for bar in bar_set:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(datasets_lbl, rotation=12, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 0.85)
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Azar (50%)")
    plt.tight_layout()
    save_fig("01_qsvm_vs_svm_directo.png")


# ── Notebook 06 ────────────────────────────────────────────────────────────────

def fig_06_learning_curves():
    n_train_vals = np.array([30, 50, 80, 100, 140, 180])
    acc_qsvm = np.array([0.889, 0.911, 0.856, 0.856, 0.889, 0.911])
    acc_svm_rbf = np.array([0.467, 0.422, 0.489, 0.478, 0.444, 0.556])
    acc_gb = np.array([0.511, 0.489, 0.511, 0.511, 0.500, 0.533])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Curvas de Aprendizaje — QSVM vs Clásicos\n"
                 "Dataset Cuántico-Nativo (Liu 2021, ZZFeatureMap 2 qubits)",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.plot(n_train_vals, acc_qsvm, "o-", color=QUANTUM_COLOR, linewidth=2.5, markersize=8,
            label=f"Quantum Kernel SVM  (media={acc_qsvm.mean():.1%})", zorder=5)
    ax.plot(n_train_vals, acc_svm_rbf, "s--", color=CLASSICAL_COLOR, linewidth=2, markersize=7,
            label=f"SVM-RBF clásico      (media={acc_svm_rbf.mean():.1%})", zorder=4)
    ax.plot(n_train_vals, acc_gb, "^-.", color="#F57C00", linewidth=2, markersize=7,
            label=f"Gradient Boosting    (media={acc_gb.mean():.1%})", zorder=4)
    ax.fill_between(n_train_vals, acc_svm_rbf, acc_qsvm, alpha=0.15,
                    color=QUANTUM_COLOR, label="Región de ventaja cuántica")
    ax.axhline(0.5, color="gray", lw=1, ls=":", alpha=0.6, label="Azar (50%)")
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Tamaño de entrenamiento")
    ax.set_ylim(0.35, 1.05)
    ax.set_xlim(25, 185)
    ax.legend(fontsize=9, loc="upper left")
    gap_mean = acc_qsvm.mean() - acc_svm_rbf.mean()
    mid_y = (acc_qsvm.mean() + acc_svm_rbf.mean()) / 2
    ax.annotate(f"Δ medio = +{gap_mean:.1%}", xy=(100, mid_y), xytext=(70, mid_y + 0.1),
                fontsize=10, color=QUANTUM_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=QUANTUM_COLOR, lw=1.5))

    ax2 = axes[1]
    advantage_pp = (acc_qsvm - acc_svm_rbf) * 100
    bar_colors = [ADVANTAGE_COLOR if v > 0 else "#D32F2F" for v in advantage_pp]
    bars = ax2.bar(n_train_vals.astype(str), advantage_pp, color=bar_colors, edgecolor="white", width=0.6)
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.axhline(advantage_pp.mean(), color=ADVANTAGE_COLOR, lw=1.5, ls="-.",
                alpha=0.7, label=f"Media = +{advantage_pp.mean():.1f}pp")
    for bar, val in zip(bars, advantage_pp):
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2.5
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:+.1f}pp",
                 ha="center", fontsize=9, fontweight="bold",
                 color=ADVANTAGE_COLOR if val >= 0 else "#D32F2F")
    ax2.set_xlabel("Tamaño de entrenamiento")
    ax2.set_ylabel("Ventaja cuántica (pp) = QSVM − SVM-RBF")
    ax2.set_title("Ventaja cuántica por tamaño de muestra\n(positivo = cuántico mejor)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(advantage_pp.min() - 10, advantage_pp.max() + 12)
    plt.tight_layout()
    save_fig("06_learning_curves_quantum_advantage.png")


def fig_06_3datasets_comparativa():
    summary = SUMMARY_06
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    x6 = np.arange(len(summary))
    w6 = 0.35
    # Una sola pareja de colores: cuántico vs clásico (no un color por dataset)
    ax0 = axes[0]
    bq6 = ax0.bar(x6 - w6 / 2, summary["QSVC"] * 100, w6, label="QSVC Cuántico",
                  color=QUANTUM_COLOR, edgecolor="white")
    bc6 = ax0.bar(x6 + w6 / 2, summary["Mejor Clásico"] * 100, w6,
                  label="Mejor Clásico (GB/RF en reales)",
                  color=CLASSICAL_COLOR, edgecolor="white")
    ax0.set_xticks(x6)
    ax0.set_xticklabels(summary["Dataset"], fontsize=9)
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_ylim(40, 118)
    ax0.set_title("Accuracy: QSVC vs Mejor Clásico\npor tipo de dataset", fontweight="bold")
    ax0.axhline(50, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax0.legend(fontsize=8, loc="upper right")
    for b in list(bq6) + list(bc6):
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                 f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    # Veredicto explícito encima de cada grupo
    for i, (_, row) in enumerate(summary.iterrows()):
        win_q = row["Ventaja (pp)"] > 0
        color = QUANTUM_COLOR if win_q else CLASSICAL_COLOR
        top = max(row["QSVC"], row["Mejor Clásico"]) * 100
        ax0.text(i, top + 8.5, row["veredicto"], ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=color)
    ax1 = axes[1]
    gcols6 = [QUANTUM_COLOR if v > 0 else CLASSICAL_COLOR for v in summary["Ventaja (pp)"]]
    bg6 = ax1.bar(x6, summary["Ventaja (pp)"], color=gcols6, edgecolor="white", width=0.5)
    ax1.axhline(0, color="black", lw=1)
    ax1.set_xticks(x6)
    ax1.set_xticklabels(summary["Dataset"], fontsize=9)
    ax1.set_ylabel("Ventaja cuántica (pp)")
    ax1.set_ylim(-12, 18)
    ax1.set_title("Solo Liu > 0: SKG/Telco gana el clásico\n(QSVC − Mejor Clásico operativo)",
                  fontweight="bold")
    for b, (_, row) in zip(bg6, summary.iterrows()):
        yp = b.get_height() + 0.4 if b.get_height() >= 0 else b.get_height() - 1.8
        ax1.text(b.get_x() + b.get_width() / 2, yp, f"{b.get_height():+.1f}pp",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.text(b.get_x() + b.get_width() / 2,
                 14 if row["Ventaja (pp)"] > 0 else -10.5,
                 row["veredicto"], ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color=QUANTUM_COLOR if row["Ventaja (pp)"] > 0 else CLASSICAL_COLOR)
    fig.suptitle("Quantum Kernel vs Mejor Clásico — ventaja SOLO en cuántico-nativo (Liu)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Telco: QSVM 76.0% vs Gradient Boosting 79.8% (−3.8 pp). "
             "El antiguo +6.7 pp vs RBF (n=50) era artefacto, no ventaja operativa.",
             ha="center", fontsize=8, style="italic", color="#444444")
    plt.tight_layout()
    save_fig("06_quantum_kernel_3datasets_comparativa.png")


def fig_08_anomaly_3datasets_comparativa():
    """Barras Q vs C en anomalías (mismo formato que clasificación/regresión)."""
    datasets = [
        "1. Nativo\n(zz_phase)",
        "2. SKG\n(inyectado)",
        "3. Referencia\n(moons)",
    ]
    # Medias multi-repeat Table A (BUSINESS/anomaly_v2_results.md)
    auc_q = np.array([0.864, 0.694, 0.599])
    auc_c = np.array([0.553, 0.993, 0.890])  # IF / LOF / IF
    delta_pp = (auc_q - auc_c) * 100
    veredictos = ["VENTAJA Q", "GANA CLÁSICO", "GANA CLÁSICO"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    x = np.arange(len(datasets))
    w = 0.35
    ax0 = axes[0]
    bq = ax0.bar(x - w / 2, auc_q, w, label="OCSVM-ZZ (cuántico)",
                 color=QUANTUM_COLOR, edgecolor="white")
    bc = ax0.bar(x + w / 2, auc_c, w, label="Mejor clásico (IF/LOF)",
                 color=CLASSICAL_COLOR, edgecolor="white")
    ax0.set_xticks(x)
    ax0.set_xticklabels(datasets, fontsize=9)
    ax0.set_ylabel("ROC-AUC")
    ax0.set_ylim(0.35, 1.18)
    ax0.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax0.set_title("AUC: OCSVM-ZZ vs Mejor Clásico\npor tipo de dataset", fontweight="bold")
    ax0.legend(fontsize=8, loc="upper right")
    for b in list(bq) + list(bc):
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                 f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for i, v in enumerate(veredictos):
        top = max(auc_q[i], auc_c[i])
        color = QUANTUM_COLOR if delta_pp[i] > 0 else CLASSICAL_COLOR
        ax0.text(i, top + 0.08, v, ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=color)

    ax1 = axes[1]
    cols = [QUANTUM_COLOR if v > 0 else CLASSICAL_COLOR for v in delta_pp]
    bars = ax1.bar(x, delta_pp, color=cols, edgecolor="white", width=0.5)
    ax1.axhline(0, color="black", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=9)
    ax1.set_ylabel("Ventaja cuántica (pp de AUC)")
    ax1.set_ylim(-55, 45)
    ax1.set_title("Solo zz_phase > 0: SKG/moons gana el clásico\n(OCSVM-ZZ − Mejor Clásico)",
                  fontweight="bold")
    for b, d, v in zip(bars, delta_pp, veredictos):
        yp = b.get_height() + 1.5 if d >= 0 else b.get_height() - 4
        ax1.text(b.get_x() + b.get_width() / 2, yp, f"{d:+.1f}pp",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.text(b.get_x() + b.get_width() / 2, 35 if d > 0 else -48, v,
                 ha="center", va="center", fontsize=8, fontweight="bold",
                 color=QUANTUM_COLOR if d > 0 else CLASSICAL_COLOR)

    fig.suptitle("Anomalías one-class — ventaja SOLO con novedad alineada al mapa ZZ",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Medias n=160, 5 repeats (Table A). zz_phase: OCSVM-ZZ 0.864 vs IF 0.553. "
             "SKG/moons: IsolationForest / LOF dominan.",
             ha="center", fontsize=8, style="italic", color="#444444")
    plt.tight_layout()
    save_fig("08_anomaly_3datasets_comparativa.png")


def fig_06_escalabilidad():
    scale_df = SCALE_DF_06
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Escalabilidad del Quantum Kernel Advantage", fontsize=13)
    ax = axes[0]
    ax.plot(scale_df["n_samples"], scale_df["acc_classical"], "o-", color="#1976D2",
            label="Mejor clasico (Grad. Boost)", linewidth=2)
    ax.plot(scale_df["n_samples"], scale_df["acc_quantum"], "s-", color="#7B2FBE",
            label="Quantum Kernel SVM", linewidth=2)
    ax.fill_between(scale_df["n_samples"], scale_df["acc_classical"], scale_df["acc_quantum"],
                    alpha=0.15, color="#7B2FBE", label="Ventaja cuantica")
    ax.set_xlabel("Numero de muestras")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Tamano del dataset")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    bars = ax.bar(scale_df["n_samples"].astype(str), scale_df["advantage"],
                  color=["#388E3C" if v > 0 else "#D32F2F" for v in scale_df["advantage"]])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Numero de muestras")
    ax.set_ylabel("Ventaja (Quantum - Clasico)")
    ax.set_title("Diferencia de accuracy")
    for bar, val in zip(bars, scale_df["advantage"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig("06_quantum_kernel_escalabilidad.png")


def fig_06_zzfeaturemap_circuit():
    from qiskit.circuit.library import zz_feature_map

    fm_draw = zz_feature_map(2, reps=2, entanglement="full")
    fig_circ = fm_draw.decompose().draw(output="mpl", style="iqp", fold=40)
    fig_circ.suptitle("Circuito ZZFeatureMap — 2 qubits, 2 repeticiones\n"
                      "El quantum kernel calcula k(x,y) = |⟨φ(x)|φ(y)⟩|²",
                      fontsize=11, fontweight="bold")
    fig_circ.savefig(OUT / "06_zzfeaturemap_circuit.png", dpi=150, bbox_inches="tight")
    plt.close(fig_circ)
    WRITTEN.append(OUT / "06_zzfeaturemap_circuit.png")


def fig_06_boundaries():
    """Fronteras de decisión: requiere QSVC sobre malla (~80s en notebook). Se omite."""
    try:
        import time
        from qiskit.circuit.library import zz_feature_map
        from qiskit.quantum_info import Statevector, SparsePauliOp
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.inspection import DecisionBoundaryDisplay
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.svm import SVC

        np.random.seed(42)
        fm = zz_feature_map(2, reps=2, entanglement="full")
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        n_samples = 50
        X = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
        y = np.array([
            1 if Statevector(fm.assign_parameters(dict(zip(fm.parameters, x))))
            .expectation_value(obs).real >= 0 else 0
            for x in X
        ])
        X = MinMaxScaler((-np.pi, np.pi)).fit_transform(X)
        acc_map = {
            "SVM RBF": 0.5333,
            "Gradient Boosting": 0.4444,
            "Quantum Kernel SVM": 0.6667,
        }
        classifiers = {
            "SVM RBF": SVC(kernel="rbf", C=1.0, random_state=42).fit(X, y),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42).fit(X, y),
        }
        qk = FidelityQuantumKernel(feature_map=fm)
        qsvc = SVC(kernel="precomputed", C=1.0).fit(qk.evaluate(x_vec=X), y)

        x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
        y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
        xx_c, yy_c = np.meshgrid(np.linspace(x_min, x_max, 12), np.linspace(y_min, y_max, 12))
        grid_coarse = np.c_[xx_c.ravel(), yy_c.ravel()]

        t0 = time.time()
        zz_q = qsvc.predict(qk.evaluate(x_vec=grid_coarse, y_vec=X)).reshape(xx_c.shape)
        if time.time() - t0 > 45:
            SKIPPED.append("06_quantum_kernel_boundaries.png (QSVC grid >45s)")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Fronteras de decision: Clasicos vs Quantum Kernel SVM\n"
                     "(Dataset generado por ZZFeatureMap — Liu et al. 2021)", fontsize=13)
        models_to_plot = [
            ("SVM RBF", classifiers["SVM RBF"], "#1976D2", "Clasico"),
            ("Gradient Boosting", classifiers["Gradient Boosting"], "#388E3C", "Clasico"),
            ("Quantum Kernel SVM", qsvc, "#7B2FBE", "CUANTICO"),
        ]
        for ax, (name, clf, color, tag) in zip(axes, models_to_plot):
            if tag == "Clasico":
                DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax, response_method="predict",
                                                       alpha=0.3, cmap="RdYlBu")
            else:
                ax.contourf(xx_c, yy_c, zz_q, alpha=0.3, cmap="RdYlBu", levels=[-0.5, 0.5, 1.5])
                ax.text(0.02, 0.02, "(aprox. 12×12)", transform=ax.transAxes, fontsize=7, color="gray")
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", linewidths=0.5, s=40, zorder=3)
            ax.set_title(f"[{tag}] {name}\nAccuracy: {acc_map.get(name, ''):.3f}",
                         fontsize=11, color=color, fontweight="bold")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
        plt.tight_layout()
        save_fig("06_quantum_kernel_boundaries.png")
    except Exception as exc:
        SKIPPED.append(f"06_quantum_kernel_boundaries.png ({exc})")


# ── Notebook 08 ───────────────────────────────────────────────────────────────

def fig_08_test_cuanticidad():
    """Mediciones de los factores de Q(D) = w1·ΔA + w2·d̂_F + w3·ΔH en 3 datasets."""
    results = RESULTS_08
    names = [r["dataset"] for r in results]
    colors = ["#7B2FBE", "#FF5722", "#1976D2"]
    d_F_ref = max(r["d_F"] for r in results) or 1.0
    d_as = [r["delta_A"] for r in results]
    d_fs = [r["d_F"] for r in results]
    d_fs_hat = [r["d_F"] / d_F_ref for r in results]
    d_hs = [r["delta_H"] for r in results]
    q_scores = [r["Q_score"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        r"Factores de $\mathcal{Q}(\mathcal{D})$ en 3 datasets (NB08) — geometría vs RBF"
        "\n"
        r"$\mathcal{Q}=w_1\Delta A + w_2\hat{d}_F + w_3\Delta H$"
        r"  ·  $\mathcal{Q}$ alto $\neq$ $\Delta_{\mathrm{op}}>0$"
        r"  (SKG: $Q{+}0.35$ pero $\Delta_{\mathrm{op}}{=}{-}6$ pp)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    x_pos = np.arange(len(names))
    w = 0.35

    # ── Fila 1: los tres factores de Q(D) ────────────────────────────────────
    ax = axes[0, 0]
    bars = ax.bar(names, d_as, color=[QUANTUM_COLOR if d > 0 else "#C62828" for d in d_as],
                  alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"$\Delta A = A(K_Q)-A(K_{cl})$")
    ax.set_title(r"Factor 1 — $\Delta A$" "\n" "¿el cuántico ve mejor las clases?")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d in zip(bars, d_as):
        ax.text(b.get_x() + b.get_width() / 2, d + (0.001 if d >= 0 else -0.003),
                f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax = axes[0, 1]
    bars = ax.bar(names, d_fs, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel(r"$d_F=\|K_Q-K_{cl}\|_F/n$")
    ax.set_title(r"Factor 2 — $d_F$ / $\hat{d}_F$" "\n" "¿geometría distinta del clásico?")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d, dh in zip(bars, d_fs, d_fs_hat):
        ax.text(b.get_x() + b.get_width() / 2, d + 0.001,
                f"{d:.3f}\n" + r"$(\hat{d}_F$" + f"={dh:.2f})",
                ha="center", va="bottom", fontsize=8)

    ax = axes[0, 2]
    bars = ax.bar(names, d_hs, color=[QUANTUM_COLOR if d > 0 else "#C62828" for d in d_hs],
                  alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"$\Delta H = H(K_Q)-H(K_{cl})$")
    ax.set_title(r"Factor 3 — $\Delta H$" "\n" "¿diversidad o concentración?")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d in zip(bars, d_hs):
        ax.text(b.get_x() + b.get_width() / 2, d + (0.02 if d >= 0 else -0.04),
                f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                fontsize=9, fontweight="bold")

    # ── Fila 2: mediciones de apoyo + score compuesto ────────────────────────
    ax = axes[1, 0]
    a_qs = [r["A_Q"] for r in results]
    a_cls = [r["A_cl"] for r in results]
    bars1 = ax.bar(x_pos - w / 2, a_qs, w, label=r"$A(K_Q,K_y)$", color=QUANTUM_COLOR, alpha=0.85)
    bars2 = ax.bar(x_pos + w / 2, a_cls, w, label=r"$A(K_{cl},K_y)$", color=CLASSICAL_COLOR, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Kernel Alignment")
    ax.set_title("Alineaciones (base de " r"$\Delta A$" ")")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(a_qs + a_cls) * 1.35 + 0.01)
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 1]
    h_qs = [r["H_Q"] for r in results]
    h_cls = [r["H_cl"] for r in results]
    ax.bar(x_pos - w / 2, h_qs, w, label=r"$H(K_Q)$", color=QUANTUM_COLOR, alpha=0.85)
    ax.bar(x_pos + w / 2, h_cls, w, label=r"$H(K_{cl})$", color=CLASSICAL_COLOR, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Entropía espectral H")
    ax.set_title("Entropías (base de " r"$\Delta H$" ")")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    bars = ax.bar(names, q_scores,
                  color=[ADVANTAGE_COLOR if q > 0 else "#C62828" for q in q_scores],
                  alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"Score $\mathcal{Q}(\mathcal{D})$")
    ax.set_title(r"Score $\mathcal{Q}$ (geometría)" "\n" r"anotar $\Delta_{\mathrm{op}}$ debajo")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, q, r in zip(bars, q_scores, results):
        d_op = OPERATIONAL_08.get(r["dataset"], float("nan"))
        ax.text(b.get_x() + b.get_width() / 2, q + (0.02 if q >= 0 else -0.03),
                f"{q:+.3f}", ha="center", va="bottom" if q >= 0 else "top",
                fontsize=10, fontweight="bold")
        # Δ clasificación (NB06) — otra fuente; no se mezcla en Q
        y_note = min(q_scores) - 0.12
        ax.text(b.get_x() + b.get_width() / 2, y_note,
                f"$\\Delta_{{\\mathrm{{op}}}}={d_op:+.1f}$",
                ha="center", va="top", fontsize=8, color="#424242")
    ax.set_ylim(min(q_scores) - 0.22, max(q_scores) + 0.12)
    ax.text(
        0.5, -0.28,
        r"$\Delta_{\mathrm{acc}}$ = vs mejor clásico (clasificación); no entra en $\mathcal{Q}$",
        transform=ax.transAxes, ha="center", fontsize=8, color="#616161",
    )

    plt.tight_layout()
    save_fig("08_test_cuanticidad_datasets.png")


def fig_08_kernel_heatmaps():
    """NO regenerar visuales sintéticos de kernels.

    Las matrices K_Q/K_cl viven solo en memoria del Notebook 08. Cualquier
    heatmap/contraste regenerado aquí sería construido a mano → no presentable.
    Si existe un PNG huérfano de una regeneración previa, se elimina.
    """
    path = OUT / "08_kernel_heatmaps_comparativa.png"
    if path.exists():
        path.unlink()
        SKIPPED.append(
            "08_kernel_heatmaps_comparativa.png "
            "(eliminado: no hay K exportadas; no conservar reconstrucciones)"
        )
    else:
        SKIPPED.append(
            "08_kernel_heatmaps_comparativa.png "
            "(omitido: re-ejecutar NB08 para matrices reales; no sintetizar)"
        )


def fig_08_barrido_sintetico():
    """Figura HONESTA del barrido NB08 (datos reales del notebook)."""
    df_sweep = DF_SWEEP_08
    alphas = df_sweep["alpha"].values
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        r"Barrido sintético NB08 (datos reales): $\alpha$ = mezcla de etiquetas, $X$ fijo"
        "\n"
        r"Hallazgo robusto: $\rho_S(\alpha,\Delta A)=+0.833$ ($p=0.005$). "
        r"$d_F$/$H$/$g$ constantes por diseño. "
        r"No equivale a $\rho(\mathcal{Q},\mathrm{ventaja})$.",
        fontsize=11, fontweight="bold",
    )
    panels = [
        ("delta_A", r"$\Delta A$",
         r"$\Delta A$ vs $\alpha$" "\n" r"(sí varía · $\rho_S=+0.833$)", "#1565C0"),
        ("A_Q", r"$A(K_Q,K_y)$",
         r"$A_Q$ vs $\alpha$" "\n" "(alineación cuántica)", QUANTUM_COLOR),
        ("A_cl", r"$A(K_{cl},K_y)$",
         r"$A_{cl}$ vs $\alpha$" "\n" "(alineación clásica)", CLASSICAL_COLOR),
        ("d_F", r"$d_F$",
         r"$d_F$ vs $\alpha$" "\n" "(constante: $X$ fijo)", "#6A1B9A"),
        ("H_Q", r"$H(K_Q)$",
         r"$H(K_Q)$ vs $\alpha$" "\n" "(constante: $X$ fijo)", "#00838F"),
        ("advantage_pp", "Ventaja (pp)",
         r"Ventaja QSVM vs $\alpha$" "\n" r"($\rho_S(\alpha,\mathrm{adv})\approx 0.34$, n.s.)",
         "#C62828"),
    ]
    for ax, (col, ylabel, title, color) in zip(axes.flat, panels):
        y_vals = df_sweep[col].values.astype(float)
        ax.plot(alphas, y_vals, "o-", color=color, linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        if np.nanmax(y_vals) > 0 and np.nanmin(y_vals) < 0:
            ax.fill_between(alphas, y_vals, 0, alpha=0.08, color=color)
        ax.set_xlabel(r"$\alpha$ (fracción etiquetas cuánticas)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(alphas)
        ax.set_xticklabels([str(a) for a in alphas], rotation=45)
        if np.nanstd(y_vals) < 1e-12:
            ax.text(0.05, 0.95, "constante\n(protocolo $X$ fijo)",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="#FFECB3", alpha=0.7))
        else:
            rho, pval = spearmanr(alphas, y_vals)
            sign = "↑" if rho > 0 else "↓"
            ax.text(0.05, 0.95, f"$\\rho_S(\\alpha,\\cdot)$={rho:.3f} {sign}\np={pval:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    plt.tight_layout()
    save_fig("08_barrido_sintetico_alpha.png")


def fig_08_correlacion():
    """Correlación HONESTA: indicadores NB08 vs ventaja NB08 (vs RBF), n=3.

    No mezclar Δ operativo (SUMMARY_06) en esta figura: son protocolos distintos.
    La línea gris del barrido usa solo (ΔA, ventaja) — lo que sí varía con α.
    """
    results = RESULTS_08
    df_results = pd.DataFrame([
        {"Dataset": r["dataset"], "Q_score": r["Q_score"], "ΔA": r["delta_A"],
         "d_F": r["d_F"], "Ventaja_RBF(pp)": r["advantage_pp"]}
        for r in results
    ])
    df_sweep = DF_SWEEP_08
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Indicadores NB08 vs ventaja NB08 (QSVM−SVM-RBF) · n=3\n"
        "Línea gris: barrido α (solo ΔA varía; d_F/H constantes). "
        "No es Δ operativo vs GB/RF.",
        fontsize=11, fontweight="bold",
    )
    panel_config = [
        ("Q_score", "Q_score", "Score compuesto Q(D)", "(NB08)", "#2E7D32"),
        ("delta_A", "ΔA", r"$\Delta A = A(K_Q)-A(K_{cl})$", "(Huang 2021)", "#1565C0"),
        ("d_F", "d_F", r"Distancia Frobenius $d_F$", "(Liu 2021)", "#6A1B9A"),
    ]
    for ax, (r_col, df_col, xlabel, ref, color) in zip(axes, panel_config):
        # Barrido: solo tiene sentido vs ΔA (d_F y Q_sweep no están bien definidos)
        if r_col == "delta_A":
            ax.plot(df_sweep["delta_A"], df_sweep["advantage_pp"], "o-",
                    color="lightgray", linewidth=1.5, markersize=4, alpha=0.7,
                    label=r"Barrido $\alpha$ ($\Delta A$)")
        for r, marker in zip(results, ["s", "D", "^"]):
            ax.scatter(r[r_col], r["advantage_pp"], s=120, marker=marker, color=color,
                       zorder=5, label=r["dataset"], edgecolors="white", linewidths=1.5)
            ax.annotate(r["dataset"], (r[r_col], r["advantage_pp"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        # Marcar también Δ_op como referencia hueca (no entra en ρ)
        for r, marker in zip(results, ["s", "D", "^"]):
            ax.scatter(r[r_col], OPERATIONAL_08[r["dataset"]], s=70, marker=marker,
                       facecolors="none", edgecolors="gray", linewidths=1.2, zorder=4,
                       label=None)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel(f"{xlabel}\n{ref}", fontsize=9)
        ax.set_ylabel("Ventaja (pp)")
        rho, pval = spearmanr(df_results[df_col].values, df_results["Ventaja_RBF(pp)"].values)
        ax.set_title(f"$\\rho_S$ vs RBF (n=3)={rho:+.3f} (p={pval:.3f})\n"
                     r"marcadores huecos = $\Delta_{\mathrm{op}}$ (no entran en $\rho$)",
                     fontsize=9)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    save_fig("08_correlacion_indicadores_ventaja.png")


# ── Notebook 09 ────────────────────────────────────────────────────────────────

def fig_09_mapeos_comparativa():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mapeos Cuántico-Compatibles: Efecto sobre Q(D) y Ventaja QSVM\n"
                 "(Framework QCR — Quantum-Compatible Representation)",
                 fontsize=13, fontweight="bold")
    for ax_row, ds_name in enumerate(["SKG Incidentes", "Telco Churn"]):
        baseline = BASELINE_09[ds_name]
        strategies_data = ALL_RESULTS_09[ds_name]
        strategy_names = ["Baseline"] + [r["strategy"] for r in strategies_data]
        q_vals = [baseline["Q_score"]] + [r["Q_score"] for r in strategies_data]
        adv_vals = [baseline["advantage_pp"]] + [r["advantage_pp"] for r in strategies_data]
        x = np.arange(len(strategy_names))
        colors_q = ["#90A4AE"] + ["#2E7D32" if v > baseline["Q_score"] else "#C62828" for v in q_vals[1:]]
        colors_adv = ["#90A4AE"] + ["#1565C0" if v > baseline["advantage_pp"] else "#E65100"
                                    for v in adv_vals[1:]]
        ax = axes[ax_row, 0]
        bars = ax.bar(x, q_vals, color=colors_q, alpha=0.85, edgecolor="white", width=0.6)
        ax.axhline(baseline["Q_score"], color="gray", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Score Q(D)")
        ax.set_title(f"{ds_name} — Score Q(D)\n(verde = mejora sobre baseline)")
        for b, v in zip(bars, q_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.001 if v >= 0 else v - 0.002,
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax = axes[ax_row, 1]
        bars = ax.bar(x, adv_vals, color=colors_adv, alpha=0.85, edgecolor="white", width=0.6)
        ax.axhline(baseline["advantage_pp"], color="gray", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Ventaja QSVM (pp)")
        ax.set_title(f"{ds_name} — Ventaja QSVM\n(azul = mejora sobre baseline)")
        for b, v in zip(bars, adv_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3 if v >= 0 else v - 0.5,
                    f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    plt.tight_layout()
    save_fig("09_mapeos_comparativa.png")


def fig_09_barrido_rotaciones():
    """NO regenerar con curvas gaussianas sintéticas.

    El PNG válido es el exportado por Notebook 09 (θ_opt SKG=154.3°, Telco=92.6°).
    Este script no tiene las series (θ, Q, d_F) serializadas → no inventarlas.
    """
    path = OUT / "09_barrido_rotaciones.png"
    if path.exists():
        SKIPPED.append(
            "09_barrido_rotaciones.png "
            "(conservado del Notebook 09; no regenerar con proxy sintético)"
        )
    else:
        SKIPPED.append(
            "09_barrido_rotaciones.png "
            "(ausente: re-ejecutar Notebook 09; no sintetizar)"
        )


def fig_09_qcr_framework():
    """Diagnóstico 3 datasets (nativo | SKG | Telco) — protocolo NB08.

    Misma medición para los tres: ΔA, d_F, ΔH + Q + Δ_RBF / Δ_op.
    El contraste con el nativo es el punto: Q>0 no basta (SKG) y ΔH
    negativo tumba Telco; solo el nativo tiene Δ_op > 0.
    Componentes post-QCR no se inventan (NB09 no logueó ΔA/d_F/ΔH post).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.suptitle(
        r"Diagnóstico $\mathcal{Q}(\mathcal{D})$: nativo · SKG · Telco (NB08, mismo protocolo)"
        "\n"
        r"Componentes $\Delta A$, $d_F$, $\Delta H$ · "
        r"$\Delta_{\mathrm{acc}}$ = vs mejor clásico (clasificación) · "
        r"Q>0 vs RBF $\neq$ ventaja operativa",
        fontsize=12, fontweight="bold",
    )

    components = [r"$\Delta A$", r"$d_F$", r"$\Delta H$"]
    # Lecturas honestas por dataset (cifras = RESULTS_08 + OPERATIONAL_08)
    meta = {
        "Cuántico-nativo": {
            "subtitle": "control — alineado al mapa",
            "aviso": "Único con $\\Delta_{\\mathrm{op}}>0$\n"
                     "Régimen favorable al cuántico\n"
                     "(generador = ZZFeatureMap)",
            "box_fc": "#E8F5E9", "box_ec": "#2E7D32",
        },
        "SKG Incidentes": {
            "subtitle": "NO cuántico-nativo",
            "aviso": "Q>0 vs RBF, pero GB/RF ganan\n"
                     "QML v2: classical\n"
                     "(Q>0 ≠ compatibilidad operativa)",
            "box_fc": "#FFEBEE", "box_ec": "#C62828",
        },
        "Telco Churn": {
            "subtitle": r"cuello: $\Delta H$ (concentración)",
            "aviso": r"$\Delta H$ muy negativo tumba $\mathcal{Q}$"
                     "\nQML v2: classical"
                     "\n(lifts vs RBF no rescatan)",
            "box_fc": "#FFF8E1", "box_ec": "#F9A825",
        },
    }

    for ax, r in zip(axes, RESULTS_08):
        name = r["dataset"]
        m = meta[name]
        vals = [r["delta_A"], r["d_F"], r["delta_H"]]
        colors = ["#C62828" if v < 0 else "#1565C0" for v in vals]
        bars = ax.bar(components, vals, color=colors, alpha=0.9, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel("Valor del componente")
        ax.set_title(f"{name}\n{m['subtitle']}", fontsize=11)
        ymin = min(vals) - 0.28
        ymax = max(0.55, max(vals) + 0.40)
        ax.set_ylim(ymin, ymax)
        for b, v in zip(bars, vals):
            if v < -0.2:
                ax.text(
                    b.get_x() + b.get_width() / 2, v / 2, f"{v:+.3f}",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white",
                )
            else:
                ax.text(
                    b.get_x() + b.get_width() / 2, v + 0.02, f"{v:+.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )
        ax.text(
            0.5, 0.98, m["aviso"],
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor=m["box_fc"], edgecolor=m["box_ec"], alpha=0.95),
        )
        d_op = OPERATIONAL_08.get(name, float("nan"))
        footer = (
            f"Q={r['Q_score']:+.3f}\n"
            f"Δ_RBF={r['advantage_pp']:+.1f}pp  (NB08)\n"
            f"Δ_acc={d_op:+.1f}pp  (clasificación)"
        )
        ax.text(
            0.98, 0.02, footer,
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
        )

    plt.tight_layout()
    save_fig("09_qcr_framework_diagnostico.png")


def _blue_bars(ax, labels, vals, ylab, title, fmt="{:+.3f}", highlight0=True,
               ylim=None):
    """Barras azules compartidas por los perfiles por caso de uso."""
    colors = ["#0D47A1", "#1976D2", "#90CAF9"]
    bars = ax.bar(labels, vals, color=colors[: len(labels)], edgecolor="white",
                  alpha=0.95, width=0.65)
    ax.axhline(0, color="#37474F", lw=0.9, ls="--")
    if highlight0 and bars:
        bars[0].set_linewidth(2.2)
        bars[0].set_edgecolor("#01579B")
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=10.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
        ymin, ymax = ylim
    else:
        ymax = max(vals) if vals else 0.0
        ymin = min(vals) if vals else 0.0
    pad = max(0.06 * (ymax - ymin + 1e-9), 0.01)
    for b, v in zip(bars, vals):
        inside = v < ymin + 0.35 * (ymax - ymin) and v < -0.05 * max(abs(ymin), abs(ymax), 1e-9)
        if inside and abs(v) > 0.05:
            ax.text(b.get_x() + b.get_width() / 2, v / 2, fmt.format(v),
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        else:
            ax.text(b.get_x() + b.get_width() / 2,
                    v + pad if v >= 0 else v - pad,
                    fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)


def _load_estructura_por_caso() -> pd.DataFrame:
    path = OUT_METRICS / "estructura_por_caso_uso.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. Ejecutar: python3 POCs/scripts/medir_estructura_por_caso.py"
        )
    return pd.read_csv(path)


# Escalas compartidas para comparar las MISMAS métricas entre familias
_ESTRUCTURA_YLIMS = {
    "delta_A": (-0.40, 0.35),
    "d_F": (0.0, 0.015),
    "delta_H": (0.0, 2.5),
}


def _fig_estructura_familia(family: str, title_es: str, out_name: str,
                            alias: str | None = None) -> None:
    """Una figura = un caso de uso = SOLO (ΔA, d_F, ΔH). Sin accuracy/R²/AUC."""
    df = _load_estructura_por_caso()
    sub = df[df["family"] == family].copy()
    if sub.empty:
        SKIPPED.append(f"{out_name}: sin filas para family={family}")
        return
    labels = sub["dataset"].tolist()
    dA = sub["delta_A"].tolist()
    dF = sub["d_F"].tolist()
    dH = sub["delta_H"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    fig.suptitle(
        f"{title_es} — métricas de estructura cuántica "
        r"$(\Delta A,\ d_F,\ \Delta H)$"
        "\n"
        "Mismo protocolo en las tres familias: ZZ 2q, n=60, 3 repeats, "
        r"baseline = RBF Frobenius-best.  "
        "No se mezclan accuracy / $R^2$ / AUC.",
        fontsize=11.5, fontweight="bold",
    )
    _blue_bars(axes[0], labels, dA, r"$\Delta A$",
               r"$\Delta A$ (Huang / Cortes)" "\n" r"$A(K_Q)-A(K_{\mathrm{RBF}})$",
               fmt="{:+.3f}", ylim=_ESTRUCTURA_YLIMS["delta_A"])
    _blue_bars(axes[1], labels, dF, r"$d_F$",
               r"$d_F$ (Liu)" "\n" r"$\|K_Q-K_{\mathrm{RBF}}\|_F/n$",
               fmt="{:.4f}", ylim=_ESTRUCTURA_YLIMS["d_F"])
    _blue_bars(axes[2], labels, dH, r"$\Delta H$",
               r"$\Delta H$ (Thanasilp)" "\n" r"$H(K_Q)-H(K_{\mathrm{RBF}})$",
               fmt="{:+.3f}", ylim=_ESTRUCTURA_YLIMS["delta_H"])
    plt.tight_layout()
    path = OUT / out_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    WRITTEN.append(path)
    if alias:
        alias_path = OUT / alias
        plt.savefig(alias_path, dpi=150, bbox_inches="tight")
        WRITTEN.append(alias_path)
    plt.close()


def fig_08_perfil_clasificacion():
    """Clasificación: solo ΔA, d_F, ΔH (mismas métricas que regresión/anomalías)."""
    _fig_estructura_familia(
        "clasificacion", "Clasificación", "08_perfil_clasificacion.png",
        alias="08_q_vector_nativo_vs_reales.png",
    )


def fig_08_perfil_regresion():
    """Regresión: solo ΔA, d_F, ΔH (mismas métricas que clasificación/anomalías)."""
    _fig_estructura_familia(
        "regresion", "Regresión", "08_perfil_regresion.png",
    )


def fig_08_perfil_anomalias():
    """Anomalías: solo ΔA, d_F, ΔH (mismas métricas que clasificación/regresión)."""
    _fig_estructura_familia(
        "anomalias", "Anomalías", "08_perfil_anomalias.png",
    )


def fig_08_estructura_tres_casos():
    """UN gráfico grande con 3 paneles: uno por métrica (ΔA, d_F, ΔH).

    En cada panel: eje X = caso de uso (clasificación / regresión / anomalías);
    dentro de cada caso, una barra por rol de dataset (Nativo / SKG / Referencia).
    Misma métrica y misma escala para poder comparar los casos de uso.
    """
    df = _load_estructura_por_caso()
    cases = [
        ("clasificacion", "Clasificación", ["Nativo", "SKG", "Telco"]),
        ("regresion", "Regresión", ["Nativo", "SKG", "Friedman"]),
        ("anomalias", "Anomalías", ["zz_phase", "SKG", "moons"]),
    ]
    roles = ["Nativo", "SKG", "Referencia"]
    role_colors = ["#0D47A1", "#1976D2", "#90CAF9"]
    metrics = [
        ("delta_A", r"$\Delta A$", r"$\Delta A$ (Huang / Cortes)" "\n"
         r"$A(K_Q)-A(K_{\mathrm{RBF}})$", "{:+.2f}"),
        ("d_F", r"$d_F$", r"$d_F$ (Liu)" "\n"
         r"$\|K_Q-K_{\mathrm{RBF}}\|_F/n$", "{:.4f}"),
        ("delta_H", r"$\Delta H$", r"$\Delta H$ (Thanasilp)" "\n"
         r"$H(K_Q)-H(K_{\mathrm{RBF}})$", "{:+.2f}"),
    ]

    data: dict[tuple[str, int], dict[str, float]] = {}
    ds_names: dict[tuple[str, int], str] = {}
    for fam, _, names in cases:
        for k, ds in enumerate(names):
            row = df[(df["family"] == fam) & (df["dataset"] == ds)]
            if row.empty:
                SKIPPED.append(f"estructura_tres_casos: falta {fam}/{ds}")
                return
            r = row.iloc[0]
            data[(fam, k)] = {c: float(r[c]) for c, *_ in metrics}
            ds_names[(fam, k)] = ds

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    fig.suptitle(
        r"Estructura cuántica $(\Delta A,\ d_F,\ \Delta H)$: una métrica por panel, "
        "comparando los casos de uso\n"
        "Mismo protocolo (ZZ 2q, n=60, 3 repeats, RBF Frobenius-best); "
        "sin accuracy / $R^2$ / AUC",
        fontsize=12, fontweight="bold",
    )
    width = 0.26
    x = np.arange(len(cases))
    for ax, (col, ylab, title, fmt) in zip(axes, metrics):
        allv = [data[(fam, k)][col] for fam, _, _ in cases for k in range(3)]
        lo, hi = min(allv + [0.0]), max(allv + [0.0])
        span = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.18 * span if lo < 0 else 0.0, hi + 0.22 * span)
        for k, (role, color) in enumerate(zip(roles, role_colors)):
            vals = [data[(fam, k)][col] for fam, _, _ in cases]
            xs = x + (k - 1) * width
            bars = ax.bar(xs, vals, width, color=color, edgecolor="white",
                          label=role if col == "delta_A" else None)
            ybot = ax.get_ylim()[0]
            for b, v, (fam, _, _) in zip(bars, vals, cases):
                rot = 90 if col == "d_F" else 0
                ax.text(b.get_x() + b.get_width() / 2,
                        v + (0.02 * span if v >= 0 else -0.02 * span),
                        fmt.format(v), ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7.5, fontweight="bold", rotation=rot)
                covered = v >= 0 and ybot >= 0 and k < 2
                ax.text(b.get_x() + b.get_width() / 2, ybot + 0.01 * span,
                        ds_names[(fam, k)], ha="center", va="bottom",
                        fontsize=7, rotation=90,
                        color="white" if covered else "#37474F")
        ax.axhline(0, color="#37474F", lw=0.9, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in cases], fontsize=10)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=11)
    handles = [mpatches.Patch(color=c, label=r) for r, c in zip(roles, role_colors)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               title="Rol del dataset dentro de cada caso de uso", frameon=False)
    plt.tight_layout(rect=(0, 0.09, 1, 0.93))
    save_fig("08_estructura_comparativa_casos_uso.png")


def fig_08_q_vector_vs_nativo():
    """Genera perfiles por familia (detalle) + compuesto 3 paneles (comparativa)."""
    fig_08_perfil_clasificacion()
    fig_08_perfil_regresion()
    fig_08_perfil_anomalias()
    fig_08_estructura_tres_casos()


def main():
    print("Generando figuras restantes del TFM...")
    fig_07_complejidad()
    fig_07_kernel_advantage_detalle()
    fig_07_paralelismo()
    fig_07_grover_benchmark()
    fig_07_radar()
    fig_07_heatmap()
    fig_01_clasificacion_comparacion()
    fig_01_qsvm_vs_svm_directo()
    fig_06_learning_curves()
    fig_06_3datasets_comparativa()
    fig_08_anomaly_3datasets_comparativa()
    fig_06_escalabilidad()
    fig_06_zzfeaturemap_circuit()
    fig_06_boundaries()
    fig_08_test_cuanticidad()
    fig_08_kernel_heatmaps()
    fig_08_barrido_sintetico()
    fig_08_correlacion()
    fig_08_q_vector_vs_nativo()
    fig_09_mapeos_comparativa()
    fig_09_barrido_rotaciones()
    fig_09_qcr_framework()

    print("\n── Archivos escritos ──")
    for path in sorted(WRITTEN):
        size = path.stat().st_size
        print(f"  {path.name:45s}  {size:>10,} bytes")
    if SKIPPED:
        print("\n── Omitidos ──")
        for msg in SKIPPED:
            print(f"  SKIPPED: {msg}")
    print(f"\nTotal: {len(WRITTEN)} figuras generadas en {OUT}")


if __name__ == "__main__":
    main()
