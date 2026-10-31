"""Kernel cuántico de fidelidad y modelos sklearn sobre matrices de Gram precalculadas."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC


def zz_map(n_qubits: int, reps: int, entanglement: str):
    """ZZ feature map as a plain QuantumCircuit."""
    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)


def feature_map_expectation(fmap, pauli: str, X: np.ndarray) -> np.ndarray:
    """<pauli> on |phi(x)> for every row of ``X`` (exact statevector)."""
    obs = SparsePauliOp.from_list([(pauli, 1.0)])
    return np.array(
        [Statevector(fmap.assign_parameters(dict(zip(fmap.parameters, x)))).expectation_value(obs).real for x in X]
    )


def fidelity_kernel(fmap, random_state: int) -> FidelityQuantumKernel:
    algorithm_globals.random_seed = random_state
    return FidelityQuantumKernel(feature_map=fmap)


def zz_fidelity_kernel(X: np.ndarray, n_qubits: int, reps: int, entanglement: str, seed: int) -> np.ndarray:
    return np.asarray(fidelity_kernel(zz_map(n_qubits, reps, entanglement), seed).evaluate(X), dtype=float)


class QuantumKernelModel:
    """Fidelity quantum kernel + sklearn estimator with ``kernel="precomputed"``.

    The circuit is the model of similarity; the decision is classical. Train points
    are stored so ``predict`` does not need a second ``X_train`` argument.
    """

    def __init__(self, feature_map, estimator, random_state: int = 42):
        self.kernel = fidelity_kernel(feature_map, random_state)
        self.estimator = estimator
        self.X_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> QuantumKernelModel:
        self.X_train_ = np.asarray(X, dtype=float)
        K = self.kernel.evaluate(x_vec=self.X_train_)
        if y is None:
            self.estimator.fit(K)
        else:
            self.estimator.fit(K, y)
        return self

    def _gram_vs_train(self, X: np.ndarray, caller: str) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError(f"Call fit() before {caller}().")
        return self.kernel.evaluate(x_vec=np.asarray(X, dtype=float), y_vec=self.X_train_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self._gram_vs_train(X, "predict"))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.decision_function(self._gram_vs_train(X, "decision_function"))


class FidelityKernelSVC(QuantumKernelModel):
    """Quantum-kernel SVC; the feature map is injectable."""

    def __init__(
        self,
        feature_map,
        C: float = 1.0,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ):
        svc = SVC(kernel="precomputed", C=C, class_weight=class_weight, random_state=random_state)
        super().__init__(feature_map, svc, random_state)
