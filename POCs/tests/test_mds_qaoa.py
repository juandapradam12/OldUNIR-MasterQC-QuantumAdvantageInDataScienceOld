"""Caso 7 (MDS con QAOA): consistencia del QUBO y de las métricas de escalabilidad."""

import numpy as np

from qml_v2.mds_qaoa import closed_neighbourhoods, energies_diag, escalabilidad


def test_qubo_optimo_de_una_estrella_es_su_centro():
    nodos = ["c", "a", "b", "d"]
    aristas = [("c", "a"), ("c", "b"), ("c", "d")]
    cerrado = closed_neighbourhoods(nodos, aristas)
    diag = energies_diag(4, cerrado)
    assert int(np.argmin(diag)) == 1
    assert diag.min() == 1.0


def test_escalabilidad_incluye_referencia_nula_y_factibilidad():
    esc = escalabilidad(ns=(6,), n_restarts=1, maxiter=20)
    fila = esc.iloc[0]
    assert fila["p_uniforme_opt"] == fila["n_optimos_qubo"] / 2 ** 6
    assert 0.0 <= fila["p_hallar_uniforme_mismo_presupuesto"] <= 1.0
    assert fila["qaoa_tamano_mejor_muestra"] >= 0
    assert fila["qaoa_mejor_muestra_es_dominante"] in (True, False)
    assert fila["ilp_tamano"] <= fila["greedy_tamano"]
