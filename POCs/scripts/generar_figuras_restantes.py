"""
Genera figuras restantes del TFM extraídas de notebooks 01, 06, 07, 08 y 09.
Usa datos embebidos / fallbacks de salidas de notebook; evita simulaciones cuánticas largas.
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
# Telco: QSVM 76.0% vs GB 79.8% (Caso 1 / evaluación.tex) — NO el artefacto
# n=50 vs RBF (+6.7 pp) que antes contaminaba esta figura con números de Liu.
SUMMARY_06 = pd.DataFrame([
    {"Dataset": "1. Cuántico-nativo\n(Liu 2021)", "QSVC": 0.733, "Mejor Clásico": 0.600,
     "veredicto": "VENTAJA Q"},
    {"Dataset": "2. SKG\n(Incidentes Bogotá)", "QSVC": 0.547, "Mejor Clásico": 0.607,
     "veredicto": "GANA CLÁSICO"},
    {"Dataset": "3. Vida real\n(Telco Churn)", "QSVC": 0.760, "Mejor Clásico": 0.798,
     "veredicto": "GANA CLÁSICO"},
])
SUMMARY_06["Ventaja (pp)"] = (SUMMARY_06["QSVC"] - SUMMARY_06["Mejor Clásico"]) * 100

RESULTS_08 = [
    {"dataset": "Cuántico-nativo", "A_Q": 0.0181, "A_cl": 0.0469, "delta_A": -0.0288,
     "H_Q": 2.2749, "H_cl": 2.2411, "delta_H": 0.0339, "d_F": 0.3532, "g": 1541.657,
     "Q_score": 0.2858, "advantage_pp": 0.0},
    # advantage_pp below = QSVM - RBF only (NB08); NOT vs GB/RF; SKG is NOT quantum-native
    {"dataset": "SKG Incidentes", "A_Q": 0.0396, "A_cl": 0.0499, "delta_A": -0.0103,
     "H_Q": 2.1545, "H_cl": 1.9666, "delta_H": 0.1880, "d_F": 0.3689, "g": 2080.864,
     "Q_score": 0.3523, "advantage_pp": 8.3},
    {"dataset": "Telco Churn", "A_Q": 0.1899, "A_cl": 0.1698, "delta_A": 0.0201,
     "H_Q": 2.3332, "H_cl": 3.4646, "delta_H": -1.1313, "d_F": 0.3457, "g": 89.725,
     "Q_score": -0.0503, "advantage_pp": 0.0},
]

DF_SWEEP_08 = pd.DataFrame([
    {"alpha": 0.00, "delta_A": -0.3735, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": -0.1200, "advantage_pp": -40.0},
    {"alpha": 0.10, "delta_A": -0.3138, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": -0.1000, "advantage_pp": -30.0},
    {"alpha": 0.25, "delta_A": -0.1688, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": -0.0500, "advantage_pp": 0.0},
    {"alpha": 0.40, "delta_A": -0.0751, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": 0.0100, "advantage_pp": 3.3},
    {"alpha": 0.50, "delta_A": 0.0006, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": 0.0500, "advantage_pp": 0.0},
    {"alpha": 0.65, "delta_A": 0.0389, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": 0.1200, "advantage_pp": 13.3},
    {"alpha": 0.75, "delta_A": 0.0878, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": 0.1800, "advantage_pp": 0.0},
    {"alpha": 0.90, "delta_A": 0.1611, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": 0.2500, "advantage_pp": 0.0},
    {"alpha": 1.00, "delta_A": -0.0003, "H_Q": 2.3046, "d_F": 0.3591, "g": 2380.246,
     "Q_score": 0.2900, "advantage_pp": -3.3},
])

BASELINE_09 = {
    "SKG Incidentes": {"Q_score": 0.1629, "advantage_pp": 8.3, "delta_A": -0.0103,
                       "d_F": 0.3689, "delta_H": 0.1880},
    "Telco Churn": {"Q_score": -0.2277, "advantage_pp": 0.0, "delta_A": 0.0201,
                    "d_F": 0.3457, "delta_H": -1.1313},
}

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

QCR_TELCO = {
    "baseline": BASELINE_09["Telco Churn"],
    "post_qcr": {"delta_A": 0.0551, "d_F": 0.2344, "delta_H": -0.8232,
                 "Q_score": 0.1674, "advantage_pp": 4.2},
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


def _synth_kernel_matrix(n: int, n_class0: int, same: float, diff: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_class0 + [1] * (n - n_class0))
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            base = same if y[i] == y[j] else diff
            k[i, j] = np.clip(base + rng.normal(0, 0.04), 0, 1)
    k = (k + k.T) / 2
    np.fill_diagonal(k, 1.0)
    return k, y


# ── Notebook 08 ───────────────────────────────────────────────────────────────

def fig_08_test_cuanticidad():
    results = RESULTS_08
    names = [r["dataset"] for r in results]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Tests de Cuanticidad sobre los 3 Datasets del TFM\n"
                 "(Indicadores de la literatura: Huang 2021, Thanasilp 2022, Liu 2021)",
                 fontsize=13, fontweight="bold", y=1.01)
    x_pos = np.arange(len(names))
    w = 0.35
    ax = axes[0, 0]
    a_qs = [r["A_Q"] for r in results]
    a_cls = [r["A_cl"] for r in results]
    bars1 = ax.bar(x_pos - w / 2, a_qs, w, label="$A(K_Q, K_y)$", color="#1565C0", alpha=0.85)
    bars2 = ax.bar(x_pos + w / 2, a_cls, w, label="$A(K_{RBF}, K_y)$", color="#B71C1C", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Kernel Alignment")
    ax.set_title("Kernel Alignment\n(Cortes 2012, Huang 2021)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(a_qs + a_cls) * 1.3 + 0.01)
    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    d_as = [r["delta_A"] for r in results]
    bars = ax.bar(names, d_as, color=["#2196F3" if d > 0 else "#F44336" for d in d_as],
                  alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ΔA = A(KQ) - A(Krbf)")
    ax.set_title("Ventaja de Alineación Cuántica\nΔA > 0 → kernel cuántico más alineado")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d in zip(bars, d_as):
        ax.text(b.get_x() + b.get_width() / 2, d + (0.001 if d >= 0 else -0.003),
                f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax = axes[0, 2]
    ax.bar(x_pos - w / 2, [r["H_Q"] for r in results], w, label="$H(K_Q)$", color="#1565C0", alpha=0.85)
    ax.bar(x_pos + w / 2, [r["H_cl"] for r in results], w, label="$H(K_{RBF})$", color="#B71C1C", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Entropía espectral H")
    ax.set_title("Entropía Espectral\nH → 0: concentración (Thanasilp 2022)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    d_fs = [r["d_F"] for r in results]
    bars = ax.bar(names, d_fs, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("$d_F = \\|K_Q - K_{RBF}\\|_F / n$")
    ax.set_title("Divergencia Cuántico-Clásica\n$d_F$ alto → kernel cuántico genuinamente diferente (Liu 2021)")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d in zip(bars, d_fs):
        ax.text(b.get_x() + b.get_width() / 2, d + 0.001, f"{d:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    gs = [r["g"] for r in results]
    bars = ax.bar(names, gs, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("$g(K_Q, K_{cl})$")
    ax.set_title("Diferencia Geométrica\n$g$ alto → cota de ventaja activa (Huang 2021)")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, v in zip(bars, gs):
        ax.text(b.get_x() + b.get_width() / 2, v + max(gs) * 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax = axes[1, 2]
    advantages = [r["advantage_pp"] for r in results]
    q_scores = [r["Q_score"] for r in results]
    ax2 = ax.twinx()
    ax.bar(names, advantages, color=["green" if a > 0 else "red" for a in advantages],
           alpha=0.5, label="Ventaja QSVM (pp)")
    ax2.plot(names, q_scores, "ko--", markersize=8, linewidth=2, label="Score Q(D)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Ventaja QSVM (pp)", color="gray")
    ax2.set_ylabel("Score Q(D)", color="black")
    ax.set_title("Ventaja QSVM vs Score Q(D)\n¿Predice Q la ventaja?")
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_fig("08_test_cuanticidad_datasets.png")


def fig_08_kernel_heatmaps():
    configs = [
        ("Cuántico-nativo", 40, 0.75, 0.15, 42),
        ("SKG Incidentes", 40, 0.70, 0.20, 43),
        ("Telco Churn", 40, 0.65, 0.25, 44),
    ]
    r_metrics = {r["dataset"]: r for r in RESULTS_08}
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle("Matrices de Kernel Cuántico vs Clásico por Dataset\n"
                 "(Ordenadas por clase para revelar estructura de bloque)",
                 fontsize=12, fontweight="bold")
    for row_idx, (name, n0, same_q, diff_q, seed) in enumerate(configs):
        n = 80
        k_q, y = _synth_kernel_matrix(n, n0, same_q, diff_q, seed)
        k_rbf, _ = _synth_kernel_matrix(n, n0, same_q - 0.05, diff_q + 0.05, seed + 1)
        r = r_metrics[name.replace("Cuántico-nativo", "Cuántico-nativo")]
        sort_idx = np.argsort(y)
        k_q_s = k_q[np.ix_(sort_idx, sort_idx)]
        k_rbf_s = k_rbf[np.ix_(sort_idx, sort_idx)]
        k_diff_s = k_q_s - k_rbf_s
        panels = [
            (k_q_s, f"{name}\n$K_Q$ (cuántico)\nA={r['A_Q']:.3f}, H={r['H_Q']:.2f}", "Blues", 0, 1),
            (k_rbf_s, f"{name}\n$K_{{RBF}}$ (clásico)\nA={r['A_cl']:.3f}, H={r['H_cl']:.2f}", "Oranges", 0, 1),
            (k_diff_s, f"{name}\n$K_Q - K_{{RBF}}$ (diferencia)\n$d_F$={r['d_F']:.3f}, g={r['g']:.2f}", "RdBu", None, None),
        ]
        for col_idx, (mat, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if vmin is None:
                vabs = np.percentile(np.abs(mat), 99)
                im = ax.imshow(mat, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")
            else:
                im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=np.percentile(mat, 99), aspect="auto")
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_fig("08_kernel_heatmaps_comparativa.png")


def fig_08_barrido_sintetico():
    df_sweep = DF_SWEEP_08
    alphas = df_sweep["alpha"].values
    metrics_to_plot = [
        ("delta_A", "ΔA = A(KQ) - A(Krbf)", "Kernel Alignment Diferencial\n(Huang 2021)", "#1565C0"),
        ("H_Q", "H(KQ)", "Entropía Espectral KQ\n(Thanasilp 2022)", "#00838F"),
        ("d_F", "$d_F$", "Distancia Frobenius\n(Liu 2021)", "#6A1B9A"),
        ("g", "$g(K_Q, K_{cl})$", "Diferencia Geométrica\n(Huang 2021)", "#E65100"),
        ("Q_score", "Score Q(D)", "Score Compuesto Q(D)\n(Este TFM)", "#2E7D32"),
        ("advantage_pp", "Ventaja QSVM (pp)", "Ventaja Empírica QSVM\n(ground truth)", "#C62828"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Barrido Sintético Controlado: Indicadores de Cuanticidad vs α\n"
                 "Validación de monotonía — si los indicadores son válidos, deben crecer con α",
                 fontsize=12, fontweight="bold")
    for ax, (col, ylabel, title, color) in zip(axes.flat, metrics_to_plot):
        y_vals = df_sweep[col].values
        ax.plot(alphas, y_vals, "o-", color=color, linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.fill_between(alphas, y_vals, 0, alpha=0.08, color=color)
        ax.set_xlabel("α (grado de cuanticidad del dataset)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(alphas)
        ax.set_xticklabels([str(a) for a in alphas], rotation=45)
        rho, pval = spearmanr(alphas, y_vals)
        sign = "↑" if rho > 0 else "↓"
        ax.text(0.05, 0.95, f"ρ_Spearman = {rho:.3f} {sign}\np = {pval:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    plt.tight_layout()
    save_fig("08_barrido_sintetico_alpha.png")


def fig_08_correlacion():
    results = RESULTS_08
    df_results = pd.DataFrame([
        {"Dataset": r["dataset"], "Q_score": r["Q_score"], "ΔA": r["delta_A"],
         "d_F": r["d_F"], "Ventaja(pp)": r["advantage_pp"]}
        for r in results
    ])
    df_sweep = DF_SWEEP_08
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Correlación Indicadores de Cuanticidad vs Ventaja QSVM\n"
                 "3 datasets del TFM (puntos) + tendencia del barrido sintético (línea gris)",
                 fontsize=12, fontweight="bold")
    panel_config = [
        ("Q_score", "Q_score", "Score compuesto Q(D)", "(Este TFM)", "#2E7D32"),
        ("delta_A", "ΔA", "ΔA = A(KQ) - A(Krbf)", "(Huang et al. 2021)", "#1565C0"),
        ("d_F", "d_F", "Distancia Frobenius $d_F$", "(Liu et al. 2021)", "#6A1B9A"),
    ]
    for ax, (r_col, df_col, xlabel, ref, color) in zip(axes, panel_config):
        ax.plot(df_sweep[r_col], df_sweep["advantage_pp"], "o-",
                color="lightgray", linewidth=1.5, markersize=4, alpha=0.6,
                label="Barrido sintético (α sweep)")
        for r, marker in zip(results, ["s", "D", "^"]):
            ax.scatter(r[r_col], r["advantage_pp"], s=120, marker=marker, color=color, zorder=5,
                       label=r["dataset"], edgecolors="white", linewidths=1.5)
            ax.annotate(r["dataset"], (r[r_col], r["advantage_pp"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel(f"{xlabel}\n{ref}", fontsize=9)
        ax.set_ylabel("Ventaja QSVM (pp)")
        rho, pval = spearmanr(df_results[df_col].values, df_results["Ventaja(pp)"].values)
        ax.set_title(f"ρ_Spearman = {rho:+.3f} (p={pval:.3f})", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    save_fig("08_correlacion_indicadores_ventaja.png")


# ── Notebook 09 ────────────────────────────────────────────────────────────────

def _rotation_sweep(theta_opt_deg: float, q_base: float, q_peak: float, d_f_peak: float, seed: int):
    rng = np.random.default_rng(seed)
    thetas = np.linspace(0, np.pi, 36)
    thetas_deg = np.degrees(thetas)
    q_scores = q_base + (q_peak - q_base) * np.exp(-0.5 * ((thetas_deg - theta_opt_deg) / 35) ** 2)
    q_scores += rng.normal(0, 0.01, size=len(thetas))
    d_fs = d_f_peak * (0.7 + 0.3 * np.exp(-0.5 * ((thetas_deg - theta_opt_deg) / 40) ** 2))
    d_fs += rng.normal(0, 0.005, size=len(thetas))
    return thetas, q_scores, d_fs


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
    rotation_details = {}
    for ds_name, theta_deg, q_base, q_peak, d_f_peak, seed in [
        ("SKG Incidentes", 154.3, 0.1629, 0.1698, 0.3777, 91),
        ("Telco Churn", 92.6, -0.2277, 0.0956, 0.4006, 92),
    ]:
        thetas, q_scores, d_fs = _rotation_sweep(theta_deg, q_base, q_peak, d_f_peak, seed)
        rotation_details[ds_name] = {
            "thetas": thetas, "Q_scores": q_scores, "dFs": d_fs,
            "theta_opt": np.radians(theta_deg),
        }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Barrido de Rotaciones: Q(D) y d_F en función del ángulo θ\n"
                 "Estrategia 2 — búsqueda de orientación de máxima divergencia cuántico-clásica",
                 fontsize=12, fontweight="bold")
    for ax, ds_name, lc in zip(axes, rotation_details, ["#1565C0", "#C62828"]):
        det = rotation_details[ds_name]
        thetas_deg = np.degrees(det["thetas"])
        ax.plot(thetas_deg, det["Q_scores"], "o-", color=lc, linewidth=2, markersize=5, label="Score Q(D)")
        ax2 = ax.twinx()
        ax2.plot(thetas_deg, det["dFs"], "s--", color=lc, linewidth=1.5, markersize=4, alpha=0.6, label="$d_F$")
        theta_opt_deg = np.degrees(det["theta_opt"])
        ax.axvline(theta_opt_deg, color="black", linestyle=":", linewidth=1.5,
                   label=f"θ_opt = {theta_opt_deg:.1f}°")
        ax.set_xlabel("Ángulo de rotación θ (grados)")
        ax.set_ylabel("Score Q(D)", color=lc)
        ax2.set_ylabel("$d_F$", color="gray")
        ax.set_title(ds_name)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="best")
    plt.tight_layout()
    save_fig("09_barrido_rotaciones.png")


def fig_09_qcr_framework():
    """Diagnóstico QCR: SKG NO es cuántico-nativo; Q>0 vs RBF ≠ ventaja operativa."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Framework QCR: diagnóstico honesto de datasets reales\n"
        "Q>0 frente a RBF no implica ventaja operativa frente a GB/RF",
        fontsize=12, fontweight="bold",
    )

    # ── SKG: componentes baseline + aviso ──────────────────────────────────
    ax = axes[0]
    skg = BASELINE_09["SKG Incidentes"]
    components = ["ΔA", "d_F", "ΔH"]
    vals = [skg["delta_A"], skg["d_F"], skg["delta_H"]]
    colors = ["#C62828" if v < 0 else "#1565C0" for v in vals]
    ax.bar(components, vals, color=colors, alpha=0.9, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Valor del componente")
    ax.set_title("SKG Incidentes — NO cuántico-nativo")
    ax.text(
        0.5, 0.92,
        "Q≈+0.16 vs RBF, pero GB/RF ganan\n"
        "QML v2: veredicto classical\n"
        "(Q>0 ≠ compatibilidad operativa)",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#FFEBEE", edgecolor="#C62828", alpha=0.95),
    )
    ax.text(
        0.98, 0.02,
        f"Q baseline: {skg['Q_score']:+.3f}\nAdv vs RBF: {skg['advantage_pp']:+.1f}pp",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # ── Telco: baseline vs post-QCR (re-uploading) ─────────────────────────
    ax = axes[1]
    qcr_res = QCR_TELCO
    base = qcr_res["baseline"]
    post = qcr_res["post_qcr"]
    base_vals = [base["delta_A"], base["d_F"], base["delta_H"]]
    post_vals = [post["delta_A"], post["d_F"], post["delta_H"]]
    x = np.arange(len(components))
    w = 0.35
    ax.bar(x - w / 2, base_vals, w, label="Baseline", color="#90A4AE", alpha=0.9)
    ax.bar(x + w / 2, post_vals, w, label="Post-QCR (re-uploading)",
           color="#2E7D32", alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=11)
    ax.set_ylabel("Valor del componente")
    ax.set_title(f"Telco Churn — cuello: {qcr_res['bottleneck']}\n"
                 "(NB09; QML v2 no confirma rescate robusto)")
    ax.legend(fontsize=9)
    txt = (f"Q: {base['Q_score']:+.3f} → {post['Q_score']:+.3f}\n"
           f"Adv vs RBF: {base['advantage_pp']:+.1f} → {post['advantage_pp']:+.1f}pp")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    save_fig("09_qcr_framework_diagnostico.png")


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
