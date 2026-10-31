"""Second feature map experiment: advantage follows alignment, not ZZ.

Cross table (synthetic only):

* Generators: labels = sign(<obs>) under ZZFeatureMap vs PauliFeatureMap(Y/YY).
* Kernels: fidelity QSVM with the same two maps + classical SVC-RBF.

Hypothesis: the matching (generator, kernel) pair wins Table A; the
mismatched quantum kernel does not inherit the advantage. ZZ is one
validated map in the TFM corpus — not the exclusive source of advantage.

Run::

    cd POCs
    PYTHONPATH=. python3 -m qml_v2.feature_map_alignment --quick
    PYTHONPATH=. python3 -m qml_v2.feature_map_alignment --n-total 120 --n-repeats 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
POCS_DIR = Path(__file__).resolve().parents[1]
MapKind = Literal["zz", "pauli_y"]


@dataclass
class AlignConfig:
    n_qubits: int = 2
    reps: int = 2
    entanglement: str = "full"
    svm_C: float = 1.0
    n_total: int = 120
    test_size: float = 0.3
    n_repeats: int = 5
    random_state: int = 42
    class_weight: str | None = "balanced"


def build_feature_map(kind: MapKind, n_qubits: int, reps: int, entanglement: str):
    """Return a plain QuantumCircuit feature map."""
    if kind == "zz":
        try:
            from qiskit.circuit.library import zz_feature_map as _zz

            return _zz(n_qubits, reps=reps, entanglement=entanglement)
        except ImportError:
            from qiskit.circuit.library import ZZFeatureMap

            return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    if kind == "pauli_y":
        # Distinct geometry from ZZ: single-Y + YY interactions.
        try:
            from qiskit.circuit.library import pauli_feature_map as _pauli

            return _pauli(
                n_qubits,
                reps=reps,
                entanglement=entanglement,
                paulis=["Y", "YY"],
            )
        except ImportError:
            from qiskit.circuit.library import PauliFeatureMap

            return PauliFeatureMap(
                feature_dimension=n_qubits,
                reps=reps,
                entanglement=entanglement,
                paulis=["Y", "YY"],
            )
    raise ValueError(f"Unknown feature map kind: {kind}")


def _obs_pauli(kind: MapKind, n_qubits: int) -> str:
    if kind == "zz":
        return "Z" * n_qubits if n_qubits != 2 else "ZZ"
    if kind == "pauli_y":
        return "Y" * n_qubits if n_qubits != 2 else "YY"
    raise ValueError(kind)


def make_map_native(
    kind: MapKind,
    n_samples: int,
    n_qubits: int = 2,
    reps: int = 2,
    entanglement: str = "full",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """X in [-pi, pi]^{n_q}; y = sign(<obs>) on |phi_kind(x)|."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    rng = np.random.RandomState(random_state)
    X = rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits))
    fmap = build_feature_map(kind, n_qubits, reps, entanglement)
    obs = SparsePauliOp.from_list([(_obs_pauli(kind, n_qubits), 1.0)])
    y = np.empty(n_samples, dtype=int)
    for i, x in enumerate(X):
        bound = fmap.assign_parameters(dict(zip(fmap.parameters, x)))
        ev = Statevector(bound).expectation_value(obs).real
        y[i] = 1 if ev >= 0 else 0
    return X, y


class FidelityKernelSVC:
    """Fidelity quantum kernel + SVC; feature map is injectable."""

    def __init__(
        self,
        feature_map,
        C: float = 1.0,
        random_state: int = 42,
        class_weight: str | None = "balanced",
    ):
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        algorithm_globals.random_seed = random_state
        self.kernel = FidelityQuantumKernel(feature_map=feature_map)
        self.C = C
        self.random_state = random_state
        self.class_weight = class_weight
        self.X_train_: np.ndarray | None = None
        self.svc_: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FidelityKernelSVC:
        self.X_train_ = np.asarray(X, dtype=float)
        K = self.kernel.evaluate(x_vec=self.X_train_)
        self.svc_ = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.svc_.fit(K, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.svc_ is None:
            raise RuntimeError("Call fit() before predict().")
        K = self.kernel.evaluate(x_vec=np.asarray(X, dtype=float), y_vec=self.X_train_)
        return self.svc_.predict(K)


def _one_repeat(
    X: np.ndarray,
    y: np.ndarray,
    cfg: AlignConfig,
    generator: MapKind,
    repeat: int,
) -> list[dict]:
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.test_size,
        random_state=cfg.random_state + repeat,
    )
    tr, te = next(sss.split(X, y))
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    rows: list[dict] = []

    models: dict[str, object] = {
        "QSVM-ZZ": FidelityKernelSVC(
            build_feature_map("zz", cfg.n_qubits, cfg.reps, cfg.entanglement),
            C=cfg.svm_C,
            random_state=cfg.random_state + repeat,
            class_weight=cfg.class_weight,
        ),
        "QSVM-PauliY": FidelityKernelSVC(
            build_feature_map("pauli_y", cfg.n_qubits, cfg.reps, cfg.entanglement),
            C=cfg.svm_C,
            random_state=cfg.random_state + repeat,
            class_weight=cfg.class_weight,
        ),
        "SVM-RBF": SVC(
            kernel="rbf",
            C=cfg.svm_C,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state + repeat,
        ),
    }
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append(
            {
                "generator": generator,
                "model": name,
                "repeat": repeat,
                "accuracy": float(accuracy_score(yte, pred)),
                "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
                "n_train": int(len(ytr)),
                "n_test": int(len(yte)),
                "aligned": (
                    (generator == "zz" and name == "QSVM-ZZ") or (generator == "pauli_y" and name == "QSVM-PauliY")
                ),
            }
        )
    return rows


def run_alignment(cfg: AlignConfig | None = None) -> pd.DataFrame:
    cfg = cfg or AlignConfig()
    all_rows: list[dict] = []
    for gen in ("zz", "pauli_y"):
        X, y = make_map_native(
            gen,  # type: ignore[arg-type]
            n_samples=cfg.n_total,
            n_qubits=cfg.n_qubits,
            reps=cfg.reps,
            entanglement=cfg.entanglement,
            random_state=cfg.random_state,
        )
        # Need both classes for stratified splits.
        if len(np.unique(y)) < 2:
            raise RuntimeError(f"Generator {gen} produced a single class; increase n_total.")
        for r in range(cfg.n_repeats):
            all_rows.extend(_one_repeat(X, y, cfg, gen, r))  # type: ignore[arg-type]
    return pd.DataFrame(all_rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["generator", "model"], as_index=False)
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"),
            n_repeats=("repeat", "nunique"),
            aligned=("aligned", "first"),
        )
        .sort_values(["generator", "acc_mean"], ascending=[True, False])
    )
    return g


def write_business_results(df: pd.DataFrame, cfg: AlignConfig) -> Path:
    out_md = REPO_ROOT / "BUSINESS" / "feature_map_alignment_v2_results.md"
    out_csv = POCS_DIR / "data" / "output" / "metrics" / "feature_map_alignment_v2.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = summarize(df)

    lines = [
        "# Feature-map alignment results (ZZ vs Pauli-Y)",
        "",
        "Generated by `python -m qml_v2.feature_map_alignment`.",
        "Advantage should follow the **matching** (generator, kernel) pair.",
        "",
        f"Config: `{json.dumps(asdict(cfg))}`",
        "",
        "## Summary",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Headline",
        "",
    ]
    for gen in ("zz", "pauli_y"):
        sub = summary[summary["generator"] == gen]
        best = sub.iloc[0]
        match = sub[sub["aligned"] == True]  # noqa: E712
        match_acc = float(match.iloc[0]["acc_mean"]) if len(match) else float("nan")
        rbf = sub[sub["model"] == "SVM-RBF"]
        rbf_acc = float(rbf.iloc[0]["acc_mean"]) if len(rbf) else float("nan")
        lines.append(
            f"- **generator={gen}**: best={best['model']} "
            f"acc={best['acc_mean']:.3f}; matched kernel={match_acc:.3f}; "
            f"RBF={rbf_acc:.3f}"
        )
    lines.append("")
    lines.append(
        "Reading: if matched >> mismatched quantum and >> RBF, alignment (not ZZ exclusivity) explains the lift."
    )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-total", type=int, default=120)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--n-qubits", type=int, default=2)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="n_total=40, n_repeats=2")
    args = p.parse_args(argv)
    cfg = AlignConfig(
        n_total=40 if args.quick else args.n_total,
        n_repeats=2 if args.quick else args.n_repeats,
        n_qubits=args.n_qubits,
        reps=args.reps,
        random_state=args.random_state,
    )
    df = run_alignment(cfg)
    path = write_business_results(df, cfg)
    print(summarize(df).to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
