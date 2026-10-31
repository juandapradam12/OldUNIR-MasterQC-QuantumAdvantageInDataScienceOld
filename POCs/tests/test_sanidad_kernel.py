"""Prueba de sanidad de la implementación (no es un resultado de la tesis).

Si las etiquetas se generan con el observable del propio ``ZZFeatureMap``, el QSVM debe
aprenderlas con holgura. Eso solo comprueba que el kernel cuántico y el SVM funcionan; que lo
logre no demuestra ventaja alguna, porque la tarea está definida con el mismo mapa que el modelo.
Por eso estos datos no se usan en ningún experimento de la tesis.
"""

import numpy as np
import pytest

pytest.importorskip("qiskit_machine_learning")

from qml_v2.classification import QuantumKernelSVC, make_quantum_native  # noqa: E402


def test_qsvm_aprende_etiquetas_de_su_propio_mapa():
    X, y = make_quantum_native(n_samples=80, random_state=0)
    modelo = QuantumKernelSVC(n_qubits=2, random_state=0)
    modelo.fit(X[:50], y[:50])
    precision = float(np.mean(modelo.predict(X[50:]) == y[50:]))
    assert precision > 0.8
