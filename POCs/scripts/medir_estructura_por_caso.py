#!/usr/bin/env python3
"""Mide ΔA, ΔH, d_F, g por caso de uso (clasificación, regresión, anomalías, optimización).

Toda la lógica vive en qml_v2.estructura_casos (la usa también el notebook 11).
Salida: POCs/data/output/metrics/estructura_por_caso_uso.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

POCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POCS))

from qml_v2.estructura_casos import CFG, measure_all  # noqa: E402

OUT = POCS / "data" / "output" / "metrics"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"n_sample={CFG.n_sample}, n_repeats={CFG.n_repeats}, qubits={CFG.n_qubits}")
    df = measure_all()
    if df.empty:
        raise SystemExit("No se pudo medir ningún caso.")
    path = OUT / "estructura_por_caso_uso.csv"
    df.to_csv(path, index=False)
    print(f"\nGuardado: {path}")
    print(
        df[["family", "dataset", "delta_A", "delta_A_null", "delta_H", "d_F", "g", "g_legacy"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
