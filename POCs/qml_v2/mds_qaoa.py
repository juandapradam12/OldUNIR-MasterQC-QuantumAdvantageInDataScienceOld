"""Caso 7 — Conjunto dominante mínimo (MDS) de 6 nodos con QAOA (QML v2).

Experimento REAL y reproducible (sin datos simulados a mano): construye el QUBO
del Cap. 5 (``C(x) = A·Σ x_v + B·Σ_v (1 − Σ_{u∈N[v]} x_u)^2``, A=1, B=2),
lo pasa a Hamiltoniano de Ising, ejecuta QAOA (``QAOAAnsatz`` de Qiskit,
profundidad p ∈ {1, 2, 3}) con COBYLA y reinicios aleatorios, simulando el
vector de estados exacto, y guarda en ``POCs/data/output/metrics``:

* ``10_mds_qaoa_reinicios.csv``      energía final y P(óptimo) por (p, reinicio)
* ``10_mds_qaoa_trayectorias.csv``   energía en cada evaluación de COBYLA
* ``10_mds_qaoa_probabilidades.csv`` distribución del mejor reinicio para p=3
                                      (probabilidad exacta y conteos de 6000 shots)
* ``10_mds_qaoa_escalabilidad.csv``  Greedy / ILP / QAOA en grafos Erdős–Rényi (n = 6..15)
* ``10_mds_qaoa_resumen.json``       parámetros y resumen (referencia exacta, semillas)

Observación metodológica: la penalización cuadrática (1 − Σ)² castiga también
cubrir un vértice más de una vez, así que este QUBO codifica un *dominante
perfecto*; coincide con el MDS usual solo si existe un dominante perfecto
mínimo (lo que se comprueba por fuerza bruta en esta instancia).

Las cadenas de bits se guardan tal cual; léelas con ``dtype=str`` para no perder ceros a la izquierda.

Uso:  python -m qml_v2.mds_qaoa
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "data" / "output" / "metrics"

NODES = ["Mario", "Sarah", "Raúl", "Ana", "Enrique", "Saúl"]
# Instancia usada en el Cap. 5 (6 aristas). Ver también POCs/scripts/generar_figuras_categoria_C.py
EDGES = [("Mario", "Sarah"), ("Mario", "Raúl"), ("Mario", "Ana"),
         ("Enrique", "Sarah"), ("Enrique", "Raúl"), ("Saúl", "Ana")]
A, B = 1.0, 2.0
SHOTS = 6000
N_RESTARTS = 20
MAXITER = 300
SEED = 42


def closed_neighbourhoods(nodes: list[str], edges: list[tuple[str, str]]) -> dict[int, list[int]]:
    idx = {n: i for i, n in enumerate(nodes)}
    adj: dict[int, set[int]] = {i: {i} for i in range(len(nodes))}
    for u, v in edges:
        adj[idx[u]].add(idx[v])
        adj[idx[v]].add(idx[u])
    return {i: sorted(s) for i, s in adj.items()}


def qubo_cost(bits: tuple[int, ...] | np.ndarray, closed: dict[int, list[int]]) -> float:
    cost = A * float(sum(bits))
    for v, cl in closed.items():
        cost += B * (1.0 - sum(bits[u] for u in cl)) ** 2
    return cost


def build_qp(n: int, closed: dict[int, list[int]]) -> QuadraticProgram:
    """QUBO como QuadraticProgram: expandimos (1 − Σ x_u)^2 con x² = x."""
    qp = QuadraticProgram("mds")
    for i in range(n):
        qp.binary_var(f"x{i}")
    const = B * n
    lin = {f"x{i}": A for i in range(n)}
    quad: dict[tuple[str, str], float] = {}
    for _, cl in closed.items():
        for u in cl:
            lin[f"x{u}"] = lin.get(f"x{u}", 0.0) + B * (-2.0 + 1.0)  # -2 x_u + x_u^2 = -x_u
        for a, b in itertools.combinations(cl, 2):
            key = (f"x{a}", f"x{b}")
            quad[key] = quad.get(key, 0.0) + 2.0 * B
    qp.minimize(constant=const, linear=lin, quadratic=quad)
    return qp


def energies_diag(n: int, closed: dict[int, list[int]]) -> np.ndarray:
    """E[k] = coste QUBO del bitstring k (qubit i = bit i de k, convención little-endian de Qiskit)."""
    e = np.empty(2 ** n)
    for k in range(2 ** n):
        bits = tuple((k >> i) & 1 for i in range(n))
        e[k] = qubo_cost(bits, closed)
    return e


def qaoa_probs(theta: np.ndarray, diag_ising: np.ndarray, n: int, p: int) -> np.ndarray:
    """Probabilidades exactas de QAOA (vector de estados), idénticas a ``QAOAAnsatz``.

    Orden de parámetros de Qiskit: theta = [beta_1..beta_p, gamma_1..gamma_p]; cada capa
    aplica exp(-i gamma H_C) y luego el mezclador exp(-i beta sum X). Se verifica contra
    ``QAOAAnsatz`` + ``Statevector`` en ``verificar_simulador``.
    """
    betas, gammas = theta[:p], theta[p:]
    psi = np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=complex)
    for beta, gamma in zip(betas, gammas):
        psi = psi * np.exp(-1j * gamma * diag_ising)
        t = psi.reshape([2] * n)
        c, s_ = np.cos(beta), -1j * np.sin(beta)
        for ax in range(n):
            a = np.take(t, 0, axis=ax)
            b = np.take(t, 1, axis=ax)
            t = np.stack([c * a + s_ * b, s_ * a + c * b], axis=ax)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2


def verificar_simulador(ising, diag_ising: np.ndarray, n: int, ps=(1, 2, 3), n_check: int = 5, seed: int = 0) -> float:
    """Máxima diferencia de probabilidades entre el simulador rápido y QAOAAnsatz de Qiskit."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for p in ps:
        ansatz = QAOAAnsatz(cost_operator=ising, reps=p)
        for _ in range(n_check):
            th = rng.uniform(0, 2 * np.pi, 2 * p)
            ref = Statevector(ansatz.assign_parameters(th)).probabilities()
            worst = max(worst, float(np.max(np.abs(ref - qaoa_probs(th, diag_ising, n, p)))))
    return worst


def bitstring(k: int, n: int) -> str:
    return format(k, f"0{n}b")  # qubit n-1 a la izquierda (Qiskit)


def run(seed: int = SEED, n_restarts: int = N_RESTARTS, maxiter: int = MAXITER,
        ps: tuple[int, ...] = (1, 2, 3), nodes=NODES, edges=EDGES, verbose: bool = True):
    n = len(nodes)
    closed = closed_neighbourhoods(nodes, edges)
    qp = build_qp(n, closed)
    ising, offset = qp.to_ising() if hasattr(qp, "to_ising") else _to_ising(qp)
    diag = energies_diag(n, closed)

    # Comprobación: el Ising + offset reproduce el QUBO en todos los bitstrings.
    for k in range(2 ** n):
        sv = Statevector.from_int(k, 2 ** n)
        e_ising = float(sv.expectation_value(ising).real) + offset
        assert abs(e_ising - diag[k]) < 1e-9, (k, e_ising, diag[k])

    diag_ising = diag - offset
    dif_sim = verificar_simulador(ising, diag_ising, n, ps=ps)
    assert dif_sim < 1e-9, dif_sim

    opt_k = int(np.argmin(diag))
    e_opt = float(diag[opt_k])
    n_opt = int(np.sum(np.isclose(diag, e_opt)))
    brute = sorted(range(2 ** n), key=lambda k: diag[k])[:5]
    info = {"nodes": nodes, "edges": edges, "A": A, "B": B, "n_qubits": n,
            "optimo_bitstring": bitstring(opt_k, n),
            "optimo_conjunto": [nodes[i] for i in range(n) if (opt_k >> i) & 1],
            "optimo_coste": e_opt, "n_optimos": n_opt,
            "top5_fuerza_bruta": [(bitstring(k, n), float(diag[k])) for k in brute],
            "shots": SHOTS, "n_restarts": n_restarts, "maxiter": maxiter, "seed": seed,
            "simulacion": "vector de estados exacto (numpy), verificado contra QAOAAnsatz + qiskit.quantum_info.Statevector",
            "ising_offset": offset,
            "max_dif_vs_QAOAAnsatz": dif_sim}

    rows, traj_rows = [], []
    best_by_p: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
    for p in ps:
        ansatz = QAOAAnsatz(cost_operator=ising, reps=p)
        for r in range(n_restarts):
            rng = np.random.default_rng(seed + 1000 * p + r)
            x0 = rng.uniform(0.0, 2.0 * np.pi, size=ansatz.num_parameters)
            history: list[float] = []

            def f(theta: np.ndarray) -> float:
                probs = qaoa_probs(theta, diag_ising, n, p)
                e = float(np.dot(probs, diag))
                history.append(e)
                return e

            t0 = time.perf_counter()
            res = COBYLA(maxiter=maxiter).minimize(f, x0)
            dt = time.perf_counter() - t0
            theta = np.asarray(res.x)
            probs = qaoa_probs(theta, diag_ising, n, p)
            e_fin = float(np.dot(probs, diag))
            rows.append(dict(p=p, restart=r, n_params=ansatz.num_parameters, energy_final=e_fin,
                             p_opt=float(probs[opt_k]),
                             p_opt_visto_en_shots=float(1.0 - (1.0 - probs[opt_k]) ** SHOTS),
                             n_evals=len(history), time_s=dt))
            for it, e in enumerate(history):
                traj_rows.append(dict(p=p, restart=r, iteration=it, energy=e))
            if p not in best_by_p or e_fin < best_by_p[p][0]:
                best_by_p[p] = (e_fin, theta, probs)
            if verbose:
                print(f"p={p} r={r:2d}  <C>={e_fin:7.4f}  P(opt)={probs[opt_k]:.3f}  evals={len(history)}  {dt:5.1f}s")

    # distribución del mejor reinicio de p=3 (probabilidades exactas + 6000 shots con semilla)
    p_ref = max(ps)
    e_b, th_b, probs_b = best_by_p[p_ref]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(SHOTS, probs_b / probs_b.sum())
    dist = pd.DataFrame({
        "bitstring": [bitstring(k, n) for k in range(2 ** n)],
        "conjunto": ["{" + ", ".join(nodes[i] for i in range(n) if (k >> i) & 1) + "}" for k in range(2 ** n)],
        "coste_qubo": diag, "prob_exacta": probs_b, "conteos_6000_shots": counts,
        "frecuencia_shots": counts / SHOTS})
    dist = dist.sort_values("prob_exacta", ascending=False).reset_index(drop=True)
    return pd.DataFrame(rows), pd.DataFrame(traj_rows), dist, info


def escalabilidad(ns=(6, 8, 10, 12, 15), density: float = 0.45, seed: int = SEED, p: int = 3,
                  n_restarts: int = N_RESTARTS, maxiter: int = MAXITER) -> pd.DataFrame:
    """Tiempos y calidad de Greedy, ILP (scipy.optimize.milp / HiGHS), fuerza bruta y QAOA en grafos Erdős–Rényi."""
    import networkx as nx
    from scipy.optimize import Bounds, LinearConstraint, milp

    rows = []
    for n in ns:
        g = nx.gnp_random_graph(n, density, seed=seed + n)
        nodes = [f"v{i}" for i in range(n)]
        edges = [(nodes[u], nodes[v]) for u, v in g.edges()]
        closed = closed_neighbourhoods(nodes, edges)

        t0 = time.perf_counter()
        covered, chosen = set(), []
        while len(covered) < n:
            v = max(range(n), key=lambda x: len(set(closed[x]) - covered))
            chosen.append(v)
            covered |= set(closed[v])
        t_greedy = time.perf_counter() - t0

        t0 = time.perf_counter()
        A_cov = np.zeros((n, n))
        for v in range(n):
            A_cov[v, closed[v]] = 1.0
        res_ilp = milp(c=np.ones(n), constraints=LinearConstraint(A_cov, lb=np.ones(n), ub=np.inf),
                       integrality=np.ones(n), bounds=Bounds(0, 1))
        ilp_size = int(round(res_ilp.fun))
        t_ilp = time.perf_counter() - t0

        diag = energies_diag(n, closed)
        qp = build_qp(n, closed)
        ising, offset = qp.to_ising()
        diag_ising = diag - offset
        e_opt = float(diag.min())
        opt_k = int(np.argmin(diag))
        t0 = time.perf_counter()
        best_cost_sampled, best_pk = np.inf, 0.0
        rng = np.random.default_rng(seed + n)
        for r in range(n_restarts):
            x0 = np.random.default_rng(seed + 1000 * p + r).uniform(0, 2 * np.pi, 2 * p)
            res = COBYLA(maxiter=maxiter).minimize(
                lambda th: float(np.dot(qaoa_probs(th, diag_ising, n, p), diag)), x0)
            pr = qaoa_probs(np.asarray(res.x), diag_ising, n, p)
            cnt = rng.multinomial(SHOTS, pr / pr.sum())
            seen = np.flatnonzero(cnt)
            best_cost_sampled = min(best_cost_sampled, float(diag[seen].min()))
            best_pk = max(best_pk, float(pr[opt_k]))
        t_q = time.perf_counter() - t0
        rows.append(dict(n=n, n_aristas=len(edges), ilp_tamano=ilp_size, greedy_tamano=len(chosen),
                         qubo_optimo=e_opt, qaoa_mejor_coste_muestreado=best_cost_sampled,
                         qaoa_mejor_p_opt=best_pk, t_greedy_s=t_greedy, t_ilp_s=t_ilp,
                         t_qaoa_20reinicios_s=t_q))
        print(rows[-1])
    return pd.DataFrame(rows)


def _to_ising(qp):  # pragma: no cover - compatibilidad con versiones antiguas
    from qiskit_optimization.converters import QuadraticProgramToQubo
    return QuadraticProgramToQubo().convert(qp).to_ising()


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    reinicios, trayectorias, dist, info = run()
    esc = escalabilidad()
    esc.to_csv(METRICS / "10_mds_qaoa_escalabilidad.csv", index=False)
    reinicios.to_csv(METRICS / "10_mds_qaoa_reinicios.csv", index=False)
    trayectorias.to_csv(METRICS / "10_mds_qaoa_trayectorias.csv", index=False)
    dist.to_csv(METRICS / "10_mds_qaoa_probabilidades.csv", index=False)
    resumen = (reinicios.groupby("p")
               .agg(energia_media=("energy_final", "mean"), energia_std=("energy_final", "std"),
                    energia_min=("energy_final", "min"), p_opt_media=("p_opt", "mean"),
                    p_opt_max=("p_opt", "max"),
                    p_visto_shots_media=("p_opt_visto_en_shots", "mean"), evals_media=("n_evals", "mean"),
                    tiempo_medio_s=("time_s", "mean")).round(6))
    info["resumen_por_p"] = json.loads(resumen.to_json(orient="index"))
    (METRICS / "10_mds_qaoa_resumen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(resumen)
    print(dist.head(8))


if __name__ == "__main__":
    main()
