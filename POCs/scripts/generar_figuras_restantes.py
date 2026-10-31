"""
Genera figuras restantes del TFM extraídas de notebooks 01, 06, 07, 08, 09 y 11.

Criterio de integridad (no negociable):
  - Alterar resultados o inventar datos es inadmisible.
  - Preferir CSV en POCs/data/output/metrics/ (salida real de notebooks).
  - Fallbacks embebidos = transcripción del log del notebook, no “embellecidos”.
  - Si un artefacto falta (p.ej. matrices K, series θ), omitir — no sintetizar.
"""

from __future__ import annotations

import math
import sys
import timeit
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr

POCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POCS))


warnings.filterwarnings("ignore")

OUT = POCS / "data" / "output" / "figures"
OUT_METRICS = POCS / "data" / "output" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)
OUT_METRICS.mkdir(parents=True, exist_ok=True)

QUANTUM_COLOR = "#7B2FBE"
CLASSICAL_COLOR = "#1976D2"
ADVANTAGE_COLOR = "#2E7D32"

plt.rcParams.update(
    {
        "figure.figsize": (14, 6),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

WRITTEN: list[Path] = []
SKIPPED: list[str] = []


def save_fig(name: str, paleta_unir: bool = False) -> Path:
    """Guarda la figura. ``paleta_unir=True`` marca el PNG como ya coloreado con la
    paleta UNIR; el resto se recolorea al final de ``main`` (recolorear_figuras_unir)."""
    path = OUT / name
    meta = {"paleta": "unir"} if paleta_unir else None
    plt.savefig(path, dpi=150, bbox_inches="tight", metadata=meta)
    plt.close()
    WRITTEN.append(path)
    return path


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


# ── Datos: salidas ejecutadas de los notebooks ───────────────────────────────
# Ninguna cifra se escribe aquí a mano; si falta un CSV, la figura no se genera.


def leer_metrica(nombre: str) -> pd.DataFrame:
    ruta = OUT_METRICS / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}: ejecute el notebook que lo produce.")
    tabla = pd.read_csv(ruta)
    for col in tabla.select_dtypes(include="object"):
        tabla[col] = tabla[col].str.replace(r"\s*\(Liu 2021\)", "", regex=True)
    return tabla


DF_01_RESULTS = leer_metrica("01_clasificacion_resultados.csv")
CMP_DF = leer_metrica("01_qsvm_vs_svm_directo.csv")
CMP_DF = CMP_DF[~CMP_DF["dataset"].str.startswith("Cuántico-nativo")].reset_index(drop=True)

# Notebook 08: la ventaja ``advantage_pp_rbf`` es vs. SVM-RBF; ``advantage_pp_op`` (Δ_op)
# es la ventaja operativa del Notebook 06 (protocolos distintos, no se mezclan).
_df08 = leer_metrica("08_test_cuanticidad_resultados.csv").rename(
    columns={
        "Dataset": "dataset",
        "A(KQ,Ky)": "A_Q",
        "A(Krbf,Ky)": "A_cl",
        "ΔA": "delta_A",
        "H(KQ)": "H_Q",
        "H(Krbf)": "H_cl",
        "ΔH": "delta_H",
        "Δ_RBF(pp)": "advantage_pp_rbf",
        "Δ_op(pp)": "advantage_pp_op",
    }
)
RESULTS_08 = []
OPERATIONAL_08 = {}
for _, row in _df08[~_df08["dataset"].str.startswith("Cuántico-nativo")].iterrows():
    RESULTS_08.append(
        {
            "dataset": row["dataset"],
            "A_Q": float(row["A_Q"]),
            "A_cl": float(row["A_cl"]),
            "delta_A": float(row["delta_A"]),
            "H_Q": float(row["H_Q"]),
            "H_cl": float(row["H_cl"]),
            "delta_H": float(row["delta_H"]),
            "d_F": float(row["d_F"]),
            "g": float(row["g"]),
            "Q_score": float(row["Q_score"]),
            "advantage_pp": float(row["advantage_pp_rbf"]),
        }
    )
    OPERATIONAL_08[row["dataset"]] = float(row["advantage_pp_op"])

# Barrido α del Notebook 08. X fijo ⇒ d_F/H/g constantes.
# No afirmar ρ(Q, ventaja); lo que se sostiene es ρ_S(α, ΔA).
DF_SWEEP_08 = leer_metrica("08_barrido_sintetico_alpha.csv")

# Notebook 09: línea base y estrategias de mapeo por dataset.
_df09 = leer_metrica("09_mapeos_resultados.csv")
_indicadores_08 = {r["dataset"]: r for r in RESULTS_08}
BASELINE_09: dict[str, dict] = {}
ALL_RESULTS_09: dict[str, list] = {}
for _ds in ("SKG Incidentes", "Telco Churn"):
    _base = _df09[(_df09["dataset"] == _ds) & (_df09["strategy"] == "Baseline")].iloc[0]
    BASELINE_09[_ds] = {
        "Q_score": float(_base["Q_score"]),
        "advantage_pp": float(_base["advantage_pp"]),
        "delta_A": _indicadores_08[_ds]["delta_A"],
        "d_F": _indicadores_08[_ds]["d_F"],
        "delta_H": _indicadores_08[_ds]["delta_H"],
    }
    ALL_RESULTS_09[_ds] = [
        {"strategy": r["strategy"], "Q_score": float(r["Q_score"]), "advantage_pp": float(r["advantage_pp"])}
        for _, r in _df09[(_df09["dataset"] == _ds) & (_df09["strategy"] != "Baseline")].iterrows()
    ]

# Post-QCR Telco: el log del Notebook 09 solo imprime Q y la ventaja; ΔA, d_F y ΔH
# posteriores no se midieron y no se reconstruyen.
_qcr = next(r for r in ALL_RESULTS_09["Telco Churn"] if r["strategy"] == "Re-uploading reps=3")
QCR_TELCO = {
    "baseline": BASELINE_09["Telco Churn"],
    "post_qcr": {"Q_score": _qcr["Q_score"], "advantage_pp": _qcr["advantage_pp"]},
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
    quantum_shor = n_range**3
    classical_factor_norm = classical_factor / classical_factor[0]
    quantum_shor_norm = quantum_shor / quantum_shor[0]
    ax.semilogy(
        n_range,
        classical_factor_norm,
        "-",
        color=CLASSICAL_COLOR,
        linewidth=2.5,
        label="Clásico: $O(e^{n^{1/3}(\\log n)^{2/3}})$",
    )
    ax.semilogy(n_range, quantum_shor_norm, "-", color=QUANTUM_COLOR, linewidth=2.5, label="Shor: $O(n^3)$")
    ax.fill_between(
        n_range, quantum_shor_norm, classical_factor_norm, alpha=0.12, color=QUANTUM_COLOR, label="Speedup exponencial"
    )
    ax.set_xlabel("Bits del número a factorizar (n)")
    ax.set_ylabel("Operaciones (normalizado, escala log)")
    ax.set_title("Factorización: Shor vs Clásico\nSpeedup exponencial")
    ax.legend(fontsize=9)
    ax.set_ylim(1, 1e15)

    ax = axes[2]
    cities = np.arange(4, 18)
    exact_time = np.array([math.factorial(max(c - 1, 1)) for c in cities], dtype=float)
    nn_time = cities**2
    qubo_qubits_needed = cities**2
    ax2_twin = ax.twinx()
    ax.semilogy(cities, exact_time / exact_time[0], "o-", color="#D32F2F", linewidth=2, label="Exacto: $(n-1)!$")
    ax.semilogy(cities, nn_time / nn_time[0], "s--", color=CLASSICAL_COLOR, linewidth=2, label="Heurístico: $O(n^2)$")
    ax2_twin.plot(cities, qubo_qubits_needed, "^:", color=QUANTUM_COLOR, linewidth=2, label="QUBO/QAOA: $n^2$ qubits")
    ax.set_xlabel("Número de ciudades (TSP)")
    ax.set_ylabel("Tiempo relativo (escala log)")
    ax2_twin.set_ylabel("Qubits requeridos", color=QUANTUM_COLOR)
    ax2_twin.tick_params(axis="y", labelcolor=QUANTUM_COLOR)
    for n_hw, hw_name, hw_color in [(11, "IBM Eagle\n127q", "#1976D2"), (20, "IBM Osprey\n433q", "#7B2FBE")]:
        ax2_twin.axhline(n_hw**2, linestyle=":", color=hw_color, linewidth=1.2, alpha=0.8)
        ax2_twin.text(17.8, n_hw**2 + 3, hw_name, fontsize=8, color=hw_color, ha="right")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2_twin.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=9, loc="upper left")
    ax.set_title("TSP: Escalabilidad Exacto vs QUBO/QAOA")

    plt.tight_layout()
    save_fig("07_complejidad_cuantico_vs_clasico.png")


def fig_07_paralelismo():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        "Superposición con n qubits (Hadamard): el estado tiene $2^n$ amplitudes", fontsize=13, fontweight="bold"
    )
    for ax, n_q in zip(axes, [2, 4, 8]):
        qc = QuantumCircuit(n_q)
        qc.h(range(n_q))
        sv = Statevector(qc)
        amplitudes = np.abs(sv.data) ** 2
        states = [f"|{i:0{n_q}b}⟩" for i in range(2**n_q)]
        if 2**n_q <= 16:
            ax.bar(range(2**n_q), amplitudes, color=QUANTUM_COLOR, edgecolor="white", alpha=0.8)
            ax.set_xticks(range(2**n_q))
            ax.set_xticklabels(states, rotation=45, fontsize=7)
        else:
            ax.bar(range(2**n_q), amplitudes, color=QUANTUM_COLOR, width=1.0, edgecolor="none", alpha=0.8)
            ax.set_xlabel(f"{2**n_q} estados (representación comprimida)")
        ax.set_ylim(0, 1.0 / (2**n_q) * 2.5)
        ax.axhline(
            1.0 / (2**n_q), color="orange", linestyle="--", linewidth=1.5, label=f"$1/2^{n_q}$ = {1 / 2**n_q:.4f}"
        )
        ax.set_title(
            f"{n_q} qubits → $2^{{{n_q}}}$ = {2**n_q} amplitudes\nuna medición devuelve un solo resultado",
            fontweight="bold",
        )
        ax.set_ylabel("Probabilidad")
        ax.legend(fontsize=9)
        ax_inset = ax.inset_axes([0.65, 0.55, 0.33, 0.40])
        ax_inset.axis("off")
        ax_inset.text(
            0.5,
            0.5,
            f"H^⊗{n_q}|0⟩",
            ha="center",
            va="center",
            fontsize=11,
            color=QUANTUM_COLOR,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F3E5F5", "edgecolor": QUANTUM_COLOR},
        )
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
        t_classical = timeit.timeit(lambda db=db, target=target: classical_search(db, target), number=50) / 50 * 1000
        grover_ops = grover_iterations(n)
        benchmark_data.append(
            {
                "N": n,
                "n_qubits": int(np.ceil(np.log2(n))),
                "classical_ops": n,
                "grover_ops": grover_ops,
                "speedup": n / grover_ops,
                "classical_time_ms": round(t_classical, 4),
            }
        )
    bench_df = pd.DataFrame(benchmark_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Conteo teórico de operaciones: Grover vs búsqueda clásica (sin ejecución en hardware)",
        fontsize=13,
        fontweight="bold",
    )
    ax = axes[0]
    ax.loglog(
        bench_df["N"],
        bench_df["classical_ops"],
        "o-",
        color=CLASSICAL_COLOR,
        linewidth=2.5,
        label="Clásico: $O(N)$ comparaciones",
    )
    ax.loglog(
        bench_df["N"],
        bench_df["grover_ops"],
        "s-",
        color=QUANTUM_COLOR,
        linewidth=2.5,
        label="Grover: $O(\\sqrt{N})$ iteraciones",
    )
    ax.fill_between(
        bench_df["N"],
        bench_df["grover_ops"],
        bench_df["classical_ops"],
        alpha=0.10,
        color=QUANTUM_COLOR,
        label="Diferencia teórica",
    )
    for _, row in bench_df.iterrows():
        ax.annotate(
            f"{row['speedup']:.0f}x",
            (row["N"], row["grover_ops"]),
            textcoords="offset points",
            xytext=(5, -12),
            fontsize=8,
            color=QUANTUM_COLOR,
        )
    ax.set_xlabel("Tamaño del espacio de búsqueda (N)")
    ax.set_ylabel("Número de operaciones")
    ax.set_title("Operaciones teóricas (Grover vs Clásico)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogx(
        bench_df["N"],
        bench_df["speedup"],
        "D-",
        color=ADVANTAGE_COLOR,
        linewidth=2.5,
        markersize=8,
        label="Speedup teórico (N / iteraciones de Grover)",
    )
    n_ref = np.array(search_sizes)
    ax.semilogx(n_ref, np.sqrt(n_ref), "--", color="gray", linewidth=1.5, label="Teórico: $\\sqrt{N}$")
    for _, row in bench_df.iterrows():
        ax.text(
            row["N"],
            row["speedup"] + 1,
            f"{row['speedup']:.0f}×",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=ADVANTAGE_COLOR,
        )
    ax.set_xlabel("Tamaño del espacio de búsqueda (N)")
    ax.set_ylabel("Speedup teórico en número de operaciones")
    ax.set_title("Speedup teórico de Grover")
    ax.legend(fontsize=9)
    ax.set_ylim(0, bench_df["speedup"].max() * 1.2)
    plt.tight_layout()
    save_fig("07_grover_benchmark.png")


# ── Notebook 01 ────────────────────────────────────────────────────────────────


def fig_01_clasificacion_comparacion():
    results_df = DF_01_RESULTS.sort_values("f1_score", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Clasificacion Quantum vs Clasico — Telco Customer Churn", fontsize=14)
    palette = results_df["type"].map({"quantum": "#7B2FBE", "classical": "#2196F3"})
    for ax, metric, label in zip(
        axes,
        ["f1_score", "accuracy", "training_time_s"],
        ["F1-Score (weighted)", "Accuracy", "Tiempo de entrenamiento (s)"],
    ):
        bars = ax.barh(results_df["model"], results_df[metric], color=palette.values, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.invert_yaxis()
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    fig.legend(
        handles=[Patch(facecolor="#7B2FBE", label="Cuantico"), Patch(facecolor="#2196F3", label="Clasico")],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
    )
    plt.tight_layout()
    save_fig("01_clasificacion_comparacion.png")


def fig_01_qsvm_vs_svm_directo():
    cmp_df = CMP_DF
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "Comparativa directa: QSVM (kernel cuántico) vs SVM-RBF (kernel clásico)\n"
        "Condiciones controladas: mismo SVC | C=1.0 | PCA 2 componentes",
        fontsize=12,
        fontweight="bold",
    )
    datasets_lbl = cmp_df["dataset"].tolist()
    x_idx = np.arange(len(datasets_lbl))
    w = 0.35
    bars_q = ax.bar(
        x_idx - w / 2,
        cmp_df["acc_qsvm"],
        w,
        label="QSVM (kernel cuántico)",
        color="#7B2FBE",
        edgecolor="white",
        alpha=0.9,
    )
    bars_rbf = ax.bar(
        x_idx + w / 2,
        cmp_df["acc_svm_rbf"],
        w,
        label="SVM-RBF (kernel clásico)",
        color="#2196F3",
        edgecolor="white",
        alpha=0.9,
    )
    for pos, (_, row) in enumerate(cmp_df.iterrows()):
        y_top = max(row["acc_qsvm"], row["acc_svm_rbf"]) + 0.02
        color_delta = "#2E7D32" if row["delta_acc"] > 0 else "#D32F2F"
        ax.annotate(
            f"Δ={row['delta_acc']:+.3f}", xy=(pos, y_top), ha="center", fontsize=9, fontweight="bold", color=color_delta
        )
    for bar_set in [bars_q, bars_rbf]:
        for bar in bar_set:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(datasets_lbl, rotation=12, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 0.85)
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Azar (50%)")
    plt.tight_layout()
    save_fig("01_qsvm_vs_svm_directo.png")


# ── Notebook 06 ────────────────────────────────────────────────────────────────


def fig_06_zzfeaturemap_circuit():
    from qiskit.circuit.library import zz_feature_map

    fm_draw = zz_feature_map(2, reps=2, entanglement="full")
    fig_circ = fm_draw.decompose().draw(output="mpl", style="iqp", fold=40)
    fig_circ.suptitle(
        "Circuito ZZFeatureMap — 2 qubits, 2 repeticiones\nEl quantum kernel calcula k(x,y) = |⟨φ(x)|φ(y)⟩|²",
        fontsize=11,
        fontweight="bold",
    )
    fig_circ.savefig(OUT / "06_zzfeaturemap_circuit.png", dpi=150, bbox_inches="tight")
    plt.close(fig_circ)
    WRITTEN.append(OUT / "06_zzfeaturemap_circuit.png")


# ── Notebook 08 ───────────────────────────────────────────────────────────────


def fig_08_test_cuanticidad():
    """Mediciones de los factores de Q(D) = w1·ΔA + w2·d̂_F + w3·ΔH en los datasets reales (SKG, Telco)."""
    results = RESULTS_08
    skg = next(r for r in results if r["dataset"] == "SKG Incidentes")
    names = [r["dataset"] for r in results]
    colors = ["#FF5722", "#1976D2"]
    d_F_ref = max(r["d_F"] for r in results) or 1.0
    d_as = [r["delta_A"] for r in results]
    d_fs = [r["d_F"] for r in results]
    d_fs_hat = [r["d_F"] / d_F_ref for r in results]
    d_hs = [r["delta_H"] for r in results]
    q_scores = [r["Q_score"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        r"Factores de $\mathcal{Q}(\mathcal{D})$ en SKG y Telco (NB08) — geometría vs RBF"
        "\n"
        r"$\mathcal{Q}=w_1\Delta A + w_2\hat{d}_F + w_3\Delta H$"
        r"  ·  $\mathcal{Q}$ alto $\neq$ $\Delta_{\mathrm{op}}>0$"
        + (
            f"  (SKG: $Q{{{skg['Q_score']:+.2f}}}$ pero "
            f"$\\Delta_{{\\mathrm{{op}}}}{{=}}{{{OPERATIONAL_08['SKG Incidentes']:+.1f}}}$ pp)"
        ),
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    x_pos = np.arange(len(names))
    w = 0.35

    # ── Fila 1: los tres factores de Q(D) ────────────────────────────────────
    ax = axes[0, 0]
    bars = ax.bar(
        names, d_as, color=[QUANTUM_COLOR if d > 0 else "#C62828" for d in d_as], alpha=0.85, edgecolor="white"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"$\Delta A = A(K_Q)-A(K_{cl})$")
    ax.set_title(r"Factor 1 — $\Delta A$" "\n" "¿el cuántico ve mejor las clases?")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d in zip(bars, d_as):
        ax.text(
            b.get_x() + b.get_width() / 2,
            d + (0.001 if d >= 0 else -0.003),
            f"{d:+.3f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    ax = axes[0, 1]
    bars = ax.bar(names, d_fs, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel(r"$d_F=\|K_Q-K_{cl}\|_F/n$")
    ax.set_title(r"Factor 2 — $d_F$ / $\hat{d}_F$" "\n" "¿geometría distinta del clásico?")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d, dh in zip(bars, d_fs, d_fs_hat):
        ax.text(
            b.get_x() + b.get_width() / 2,
            d + 0.001,
            f"{d:.3f}\n" + r"$(\hat{d}_F$" + f"={dh:.2f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[0, 2]
    bars = ax.bar(
        names, d_hs, color=[QUANTUM_COLOR if d > 0 else "#C62828" for d in d_hs], alpha=0.85, edgecolor="white"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"$\Delta H = H(K_Q)-H(K_{cl})$")
    ax.set_title(r"Factor 3 — $\Delta H$" "\n" "¿diversidad o concentración?")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, d in zip(bars, d_hs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            d + (0.02 if d >= 0 else -0.04),
            f"{d:+.3f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    for ax in axes[0]:
        ax.margins(y=0.3)

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
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.002,
            f"{b.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

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
    bars = ax.bar(
        names,
        q_scores,
        color=[ADVANTAGE_COLOR if q > 0 else "#C62828" for q in q_scores],
        alpha=0.85,
        edgecolor="white",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(r"Score $\mathcal{Q}(\mathcal{D})$")
    ax.set_title(r"Score $\mathcal{Q}$ (geometría)" "\n" r"anotar $\Delta_{\mathrm{op}}$ debajo")
    ax.set_xticklabels(names, rotation=12, ha="right")
    for b, q, r in zip(bars, q_scores, results):
        d_op = OPERATIONAL_08.get(r["dataset"], float("nan"))
        ax.text(
            b.get_x() + b.get_width() / 2,
            q + (0.02 if q >= 0 else -0.03),
            f"{q:+.3f}",
            ha="center",
            va="bottom" if q >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
        # Δ clasificación (NB06) — otra fuente; no se mezcla en Q
        y_note = min(q_scores) - 0.12
        ax.text(
            b.get_x() + b.get_width() / 2,
            y_note,
            f"$\\Delta_{{\\mathrm{{op}}}}={d_op:+.1f}$",
            ha="center",
            va="top",
            fontsize=8,
            color="#424242",
        )
    ax.set_ylim(min(q_scores) - 0.22, max(q_scores) + 0.12)
    ax.text(
        0.5,
        -0.28,
        r"$\Delta_{\mathrm{acc}}$ = vs mejor clásico (clasificación); no entra en $\mathcal{Q}$",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#616161",
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
            "08_kernel_heatmaps_comparativa.png (eliminado: no hay K exportadas; no conservar reconstrucciones)"
        )
    else:
        SKIPPED.append(
            "08_kernel_heatmaps_comparativa.png (omitido: re-ejecutar NB08 para matrices reales; no sintetizar)"
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
        fontsize=11,
        fontweight="bold",
    )
    panels = [
        ("delta_A", r"$\Delta A$", r"$\Delta A$ vs $\alpha$" "\n" r"(sí varía · $\rho_S=+0.833$)", "#1565C0"),
        ("A_Q", r"$A(K_Q,K_y)$", r"$A_Q$ vs $\alpha$" "\n" "(alineación cuántica)", QUANTUM_COLOR),
        ("A_cl", r"$A(K_{cl},K_y)$", r"$A_{cl}$ vs $\alpha$" "\n" "(alineación clásica)", CLASSICAL_COLOR),
        ("d_F", r"$d_F$", r"$d_F$ vs $\alpha$" "\n" "(constante: $X$ fijo)", "#6A1B9A"),
        ("H_Q", r"$H(K_Q)$", r"$H(K_Q)$ vs $\alpha$" "\n" "(constante: $X$ fijo)", "#00838F"),
        (
            "advantage_pp",
            "Ventaja (pp)",
            r"Ventaja QSVM vs $\alpha$" "\n" r"($\rho_S(\alpha,\mathrm{adv})\approx 0.34$, n.s.)",
            "#C62828",
        ),
    ]
    for ax, (col, ylabel, title, color) in zip(axes.flat, panels):
        y_vals = df_sweep[col].values.astype(float)
        ax.plot(
            alphas,
            y_vals,
            "o-",
            color=color,
            linewidth=2.5,
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2,
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        if np.nanmax(y_vals) > 0 and np.nanmin(y_vals) < 0:
            ax.fill_between(alphas, y_vals, 0, alpha=0.08, color=color)
        ax.set_xlabel(r"$\alpha$ (fracción etiquetas cuánticas)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(alphas)
        ax.set_xticklabels([str(a) for a in alphas], rotation=45)
        if np.nanstd(y_vals) < 1e-12:
            ax.text(
                0.05,
                0.95,
                "constante\n(protocolo $X$ fijo)",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox={"boxstyle": "round", "facecolor": "#FFECB3", "alpha": 0.7},
            )
        else:
            rho, pval = spearmanr(alphas, y_vals)
            sign = "↑" if rho > 0 else "↓"
            ax.text(
                0.05,
                0.95,
                f"$\\rho_S(\\alpha,\\cdot)$={rho:.3f} {sign}\np={pval:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.4},
            )
    plt.tight_layout()
    save_fig("08_barrido_sintetico_alpha.png")


# ── Notebook 09 ────────────────────────────────────────────────────────────────


def fig_09_mapeos_comparativa():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Mapeos Cuántico-Compatibles: Efecto sobre Q(D) y Ventaja QSVM\n"
        "(Framework QCR — Quantum-Compatible Representation)",
        fontsize=13,
        fontweight="bold",
    )
    for ax_row, ds_name in enumerate(["SKG Incidentes", "Telco Churn"]):
        baseline = BASELINE_09[ds_name]
        strategies_data = ALL_RESULTS_09[ds_name]
        strategy_names = ["Baseline"] + [r["strategy"] for r in strategies_data]
        q_vals = [baseline["Q_score"]] + [r["Q_score"] for r in strategies_data]
        adv_vals = [baseline["advantage_pp"]] + [r["advantage_pp"] for r in strategies_data]
        x = np.arange(len(strategy_names))
        colors_q = ["#C0C0C8"] + ["#0E4A75" if v > baseline["Q_score"] else "#858591" for v in q_vals[1:]]
        colors_adv = ["#C0C0C8"] + ["#219BC3" if v > baseline["advantage_pp"] else "#858591" for v in adv_vals[1:]]
        ax = axes[ax_row, 0]
        bars = ax.bar(x, q_vals, color=colors_q, alpha=0.85, edgecolor="white", width=0.6)
        ax.axhline(baseline["Q_score"], color="gray", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Score Q(D)")
        ax.set_title(f"{ds_name} — Score Q(D)\n(azul oscuro = mejora, gris = empeora)")
        for b, v in zip(bars, q_vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.001 if v >= 0 else v - 0.002,
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )
        ax = axes[ax_row, 1]
        bars = ax.bar(x, adv_vals, color=colors_adv, alpha=0.85, edgecolor="white", width=0.6)
        ax.axhline(baseline["advantage_pp"], color="gray", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Ventaja QSVM (pp)")
        ax.set_title(f"{ds_name} — Ventaja QSVM\n(azul = mejora, gris = empeora)")
        for b, v in zip(bars, adv_vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.3 if v >= 0 else v - 0.5,
                f"{v:+.1f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )
    plt.tight_layout()
    save_fig("09_mapeos_comparativa.png", paleta_unir=True)


def fig_09_barrido_rotaciones():
    """NO regenerar con curvas gaussianas sintéticas.

    El PNG válido es el exportado por Notebook 09 (θ_opt SKG=154.3°, Telco=92.6°).
    Este script no tiene las series (θ, Q, d_F) serializadas → no inventarlas.
    """
    path = OUT / "09_barrido_rotaciones.png"
    if path.exists():
        SKIPPED.append("09_barrido_rotaciones.png (conservado del Notebook 09; no regenerar con proxy sintético)")
    else:
        SKIPPED.append("09_barrido_rotaciones.png (ausente: re-ejecutar Notebook 09; no sintetizar)")


def fig_09_qcr_framework():
    """Diagnóstico de los datasets reales (SKG | Telco) — protocolo NB08.

    Misma medición para ambos: ΔA, d_F, ΔH + Q + Δ_RBF / Δ_op.
    Q>0 no basta (SKG) y ΔH negativo tumba Telco.
    Componentes post-QCR no se inventan (NB09 no logueó ΔA/d_F/ΔH post).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.8))
    fig.suptitle(
        r"Diagnóstico $\mathcal{Q}(\mathcal{D})$: SKG · Telco (NB08, mismo protocolo)"
        "\n"
        r"Componentes $\Delta A$, $d_F$, $\Delta H$ · "
        r"$\Delta_{\mathrm{acc}}$ = vs mejor clásico (clasificación) · "
        r"Q>0 vs RBF $\neq$ ventaja operativa",
        fontsize=12,
        fontweight="bold",
    )

    components = [r"$\Delta A$", r"$d_F$", r"$\Delta H$"]
    # Lecturas honestas por dataset (cifras = RESULTS_08 + OPERATIONAL_08)
    meta = {
        "SKG Incidentes": {
            "subtitle": "$\\mathcal{Q}>0$ sin ventaja operativa",
            "aviso": "Q>0 vs RBF, pero GB/RF ganan\nQML v2: classical\n(Q>0 ≠ compatibilidad operativa)",
            "box_fc": "#FFEBEE",
            "box_ec": "#C62828",
        },
        "Telco Churn": {
            "subtitle": r"cuello: $\Delta H$ (concentración)",
            "aviso": r"$\Delta H$ muy negativo tumba $\mathcal{Q}$"
            "\nQML v2: classical"
            "\n(lifts vs RBF no rescatan)",
            "box_fc": "#FFF8E1",
            "box_ec": "#F9A825",
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
                    b.get_x() + b.get_width() / 2,
                    v / 2,
                    f"{v:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
            else:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.02,
                    f"{v:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        ax.text(
            0.5,
            0.98,
            m["aviso"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": m["box_fc"], "edgecolor": m["box_ec"], "alpha": 0.95},
        )
        d_op = OPERATIONAL_08.get(name, float("nan"))
        footer = (
            f"Q={r['Q_score']:+.3f}\nΔ_RBF={r['advantage_pp']:+.1f}pp  (NB08)\nΔ_acc={d_op:+.1f}pp  (clasificación)"
        )
        ax.text(
            0.98,
            0.02,
            footer,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
        )

    plt.tight_layout()
    save_fig("09_qcr_framework_diagnostico.png")


def _blue_bars(ax, labels, vals, ylab, title, fmt="{:+.3f}", highlight0=True, ylim=None):
    """Barras azules compartidas por los perfiles por caso de uso."""
    colors = ["#0D47A1", "#1976D2", "#90CAF9"]
    bars = ax.bar(labels, vals, color=colors[: len(labels)], edgecolor="white", alpha=0.95, width=0.65)
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
    pad = 0.04 * (ymax - ymin + 1e-9)
    for b, v in zip(bars, vals):
        inside = v < ymin + 0.35 * (ymax - ymin) and v < -0.05 * max(abs(ymin), abs(ymax), 1e-9)
        if inside and abs(v) > 0.05:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v / 2,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + pad if v >= 0 else v - pad,
                fmt.format(v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )
    ax.tick_params(axis="x", labelsize=9)


_DATASETS_SINTETICOS_A_MEDIDA = {"Nativo", "zz_phase", "zz_lag"}


def _load_estructura_por_caso() -> pd.DataFrame:
    path = OUT_METRICS / "estructura_por_caso_uso.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecutar: python3 POCs/scripts/medir_estructura_por_caso.py")
    return pd.read_csv(path)


# Escalas compartidas para comparar las MISMAS métricas entre familias
_ESTRUCTURA_YLIMS = {
    "delta_A": (-0.40, 0.35),
    "d_F": (0.0, 0.015),
    "delta_H": (0.0, 2.5),
}


def _fig_estructura_familia(family: str, title_es: str, out_name: str) -> None:
    """Una figura = un caso de uso = SOLO (ΔA, d_F, ΔH). Sin accuracy/R²/AUC."""
    df = _load_estructura_por_caso()
    sub = df[(df["family"] == family) & ~df["dataset"].isin(_DATASETS_SINTETICOS_A_MEDIDA)].copy()
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
        "Mismo protocolo en las cuatro familias: ZZ 2q, n=60, 3 repeats, "
        r"baseline = RBF Frobenius-best.  "
        "No se mezclan accuracy / $R^2$ / AUC.",
        fontsize=11.5,
        fontweight="bold",
    )
    _blue_bars(
        axes[0],
        labels,
        dA,
        r"$\Delta A$",
        r"$\Delta A$ (Huang / Cortes)" "\n" r"$A(K_Q)-A(K_{\mathrm{RBF}})$",
        fmt="{:+.3f}",
        ylim=_ESTRUCTURA_YLIMS["delta_A"],
    )
    _blue_bars(
        axes[1],
        labels,
        dF,
        r"$d_F$",
        r"$d_F$" "\n" r"$\|K_Q-K_{\mathrm{RBF}}\|_F/n$",
        fmt="{:.4f}",
        ylim=_ESTRUCTURA_YLIMS["d_F"],
    )
    _blue_bars(
        axes[2],
        labels,
        dH,
        r"$\Delta H$",
        r"$\Delta H$ (Thanasilp)" "\n" r"$H(K_Q)-H(K_{\mathrm{RBF}})$",
        fmt="{:+.3f}",
        ylim=_ESTRUCTURA_YLIMS["delta_H"],
    )
    plt.tight_layout()
    path = OUT / out_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    WRITTEN.append(path)
    plt.close()


def fig_08_perfil_clasificacion():
    """Clasificación: solo ΔA, d_F, ΔH (mismas métricas que regresión/anomalías)."""
    _fig_estructura_familia(
        "clasificacion",
        "Clasificación",
        "08_perfil_clasificacion.png",
    )


def fig_08_perfil_regresion():
    """Regresión: solo ΔA, d_F, ΔH (mismas métricas que clasificación/anomalías)."""
    _fig_estructura_familia(
        "regresion",
        "Regresión",
        "08_perfil_regresion.png",
    )


def fig_08_perfil_anomalias():
    """Anomalías: solo ΔA, d_F, ΔH (mismas métricas que clasificación/regresión)."""
    _fig_estructura_familia(
        "anomalias",
        "Anomalías",
        "08_perfil_anomalias.png",
    )


def fig_08_perfil_optimizacion():
    """Optimización (paisaje QUBO): mismas métricas ΔA, d_F, ΔH."""
    _fig_estructura_familia(
        "optimizacion",
        "Optimización (paisaje QUBO)",
        "08_perfil_optimizacion.png",
    )


def fig_08_estructura_tres_casos():
    """UN gráfico grande: un panel por indicador (ΔA, ΔH, d_F, g, Q).

    En cada panel: eje X = caso de uso (clasificación / regresión / anomalías /
    optimización);
    dentro de cada caso, una barra por dataset real o de referencia.
    Misma métrica y misma escala para poder comparar los casos de uso.
    """
    df = _load_estructura_por_caso()
    df = df[~df["dataset"].isin(_DATASETS_SINTETICOS_A_MEDIDA)].reset_index(drop=True)
    # Compuesto del TFM con los MISMOS componentes medidos y g como cuarto término
    # (pesos 0.4/0.2/0.2/0.2; d̂_F y ĝ normalizados por el máximo). Solo informativo.
    from qml_v2.estructura_casos import add_q_composite

    df = add_q_composite(df)
    cases = [
        ("clasificacion", "Clasificación", ["SKG", "Telco"]),
        ("regresion", "Regresión", ["SKG", "Friedman"]),
        ("anomalias", "Anomalías", ["SKG", "moons"]),
        ("optimizacion", "Optimización\n(paisaje QUBO)", ["Asignación", "MaxCut", "TSP"]),
    ]
    roles = ["SKG (en optimización: asignación)", "Referencia (en optimización: MaxCut)", "TSP (solo optimización)"]
    role_colors = ["#0D47A1", "#1976D2", "#90CAF9"]
    metrics = [
        (
            "delta_A",
            r"$\Delta A$",
            r"Alineación $\Delta A$ (Cortes / Huang)"
            "\n"
            r"$A(K_Q)-A(K_{\mathrm{RBF}})$",
            "{:+.2f}",
        ),
        (
            "delta_H",
            r"$\Delta H$",
            r"Entropía espectral $\Delta H$ (Thanasilp)"
            "\n"
            r"$H(K_Q)-H(K_{\mathrm{RBF}})$",
            "{:+.2f}",
        ),
        (
            "d_F",
            r"$d_F$",
            r"Distancia Frobenius $d_F$"
            "\n"
            r"$\|K_Q-K_{\mathrm{RBF}}\|_F/n$",
            "{:.4f}",
        ),
        (
            "g",
            r"$g$",
            r"Diferencia geométrica $g$ (Huang)"
            "\n"
            r"$\sqrt{\|K_Q^{1/2}(K_{\mathrm{RBF}}+\lambda I)^{-1}K_Q^{1/2}\|_2}$,  $\lambda=10^{-3}$",
            "{:.0f}",
        ),
        (
            "Q_comp",
            r"$\mathcal{Q}$",
            r"Compuesto $\mathcal{Q}$ (este TFM)"
            "\n"
            r"$0.4\,\Delta A+0.2\,\hat d_F+0.2\,\Delta H+0.2\,\hat g$",
            "{:+.2f}",
        ),
    ]

    n_barras = {fam: len(names) for fam, _, names in cases}
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

    fig, axes_grid = plt.subplots(2, 3, figsize=(17.0, 10.4))
    axes = axes_grid.ravel()
    fig.suptitle(
        "Indicadores de estructura cuántica: un panel por indicador, "
        "comparando los casos de uso\n"
        "Mismo protocolo (ZZ 2q, n=60, 3 repeats, RBF Frobenius-best); "
        "sin accuracy / $R^2$ / AUC",
        fontsize=12.5,
        fontweight="bold",
    )
    width = 0.26
    x = np.arange(len(cases))
    for ax, (col, ylab, title, fmt) in zip(axes, metrics):
        allv = [v[col] for v in data.values()]
        lo, hi = min(allv + [0.0]), max(allv + [0.0])
        span = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.30 * span if lo < 0 else 0.0, hi + 0.22 * span)
        for k, color in enumerate(role_colors):
            presentes = [(i, fam) for i, (fam, _, _) in enumerate(cases) if k < n_barras[fam]]
            vals = [data[(fam, k)][col] for _, fam in presentes]
            xs = [x[i] + (k - (n_barras[fam] - 1) / 2) * width for i, fam in presentes]
            bars = ax.bar(xs, vals, width, color=color, edgecolor="white")
            ybot = ax.get_ylim()[0]
            for b, v, (_, fam) in zip(bars, vals, presentes):
                rot = 90 if col in ("d_F", "g") else 0
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + (0.02 * span if v >= 0 else -0.02 * span),
                    fmt.format(v),
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5,
                    fontweight="bold",
                    rotation=rot,
                )
                covered = v >= 0 and ybot >= 0 and k < 2
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    ybot + 0.01 * span,
                    ds_names[(fam, k)],
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="white" if covered else "#37474F",
                )
        ax.axhline(0, color="#37474F", lw=0.9, ls="--")
        if col == "delta_A" and "delta_A_null" in df.columns:
            nm = float(df["delta_A_null"].mean())
            ns = float(df["delta_A_null_std"].mean())
            ax.axhspan(
                nm - 2 * ns, nm + 2 * ns, color="#BBDEFB", alpha=0.45, zorder=0, label="ruido: $y$ permutada (±2σ)"
            )
            ax.legend(loc="upper right", fontsize=7.5, frameon=False)
        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in cases], fontsize=10)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=11)
    ax_note = axes[len(metrics)]
    ax_note.axis("off")
    handles = [mpatches.Patch(color=c, label=r) for r, c in zip(roles, role_colors)]
    ax_note.legend(
        handles=handles,
        loc="upper left",
        fontsize=10.5,
        frameon=False,
        title="Posición de la barra dentro de cada caso de uso",
        title_fontsize=10.5,
    )
    ax_note.text(
        0.0,
        0.50,
        "Cómo leerlo\n"
        r"Solo $\Delta A$ usa las etiquetas; $\Delta H$, $d_F$ y $g$"
        "\n"
        r"solo ven $X$. Ningún dataset supera el ruido de $\Delta A$"
        "\n"
        "(banda azul): el resto queda dentro del ruido o por debajo (RBF alinea igual o mejor).\n"
        r"$\Delta H$, $d_F$, $g$ casi planos; excepción: anomalías-SKG"
        "\n"
        "(outliers 8σ comprimen el MinMax; artefacto de escala).\n"
        r"En optimización $\Delta H,d_F,g$ solo dependen del nº de bits."
        "\n"
        r"Optimización = paisaje QUBO (bitstrings→energía), no QAOA."
        "\n"
        r"$\mathcal{Q}$: informativo, no es regla de decisión.",
        transform=ax_note.transAxes,
        fontsize=10,
        va="top",
        color="#263238",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig("08_estructura_comparativa_casos_uso.png")


def fig_08_estructura_por_caso():
    """Genera perfiles por familia (detalle) + compuesto de cinco paneles (comparativa)."""
    fig_08_perfil_clasificacion()
    fig_08_perfil_regresion()
    fig_08_perfil_anomalias()
    fig_08_perfil_optimizacion()
    fig_08_estructura_tres_casos()


def main():
    print("Generando figuras restantes del TFM...")
    fig_07_complejidad()
    fig_07_paralelismo()
    fig_07_grover_benchmark()
    fig_01_clasificacion_comparacion()
    fig_01_qsvm_vs_svm_directo()
    fig_06_zzfeaturemap_circuit()
    fig_08_test_cuanticidad()
    fig_08_kernel_heatmaps()
    fig_08_barrido_sintetico()
    fig_08_estructura_por_caso()
    fig_09_mapeos_comparativa()
    fig_09_barrido_rotaciones()
    fig_09_qcr_framework()

    from recolorear_figuras_unir import procesar as _recolorear

    for path in WRITTEN:
        _recolorear(path)

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
