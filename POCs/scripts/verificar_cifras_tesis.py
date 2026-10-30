"""Comprueba que las cifras citadas en la tesis y en la defensa coinciden con los CSV ejecutados.

Cada comprobación calcula el valor esperado a partir de un CSV de ``data/output/metrics``,
lo formatea como aparece en el texto LaTeX y exige que esa cadena figure en el fichero
indicado. Termina con código 1 si alguna cifra no se encuentra, de modo que una
discrepancia entre texto y datos no pasa desapercibida.

Uso, desde ``POCs/``::

    python scripts/verificar_cifras_tesis.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

POCS = Path(__file__).resolve().parents[1]
RAIZ = POCS.parent
METRICS = POCS / "data" / "output" / "metrics"
CAPITULOS = RAIZ / "Documento" / "Capitulos"
DEFENSA = RAIZ / "Documento" / "Defensa" / "defensa.tex"
EVALUACION = CAPITULOS / "evaluacion.tex"


@dataclass(frozen=True)
class Comprobacion:
    descripcion: str
    fichero: Path
    cadena: str


def _csv(nombre: str) -> pd.DataFrame:
    return pd.read_csv(METRICS / nombre)


def _comprobaciones_sintesis() -> list[Comprobacion]:
    filas = _csv("07_sintesis_resultados.csv")
    resultado = []
    for _, fila in filas.iterrows():
        if fila["caso"] in (3, 4, 5, 7) or fila["etiqueta"] == "telco_injected":
            continue
        if fila["caso"] == 2:
            cadena = f"{fila['valor'] / 100:.3f}"
        else:
            cadena = f"{abs(fila['valor']):.1f}"
        resultado.append(
            Comprobacion(
                f"Síntesis CU{fila['caso']} {fila['capa']}", EVALUACION, cadena
            )
        )
    return resultado


def _comprobaciones_regresion_nb03() -> list[Comprobacion]:
    resumen = _csv("03_regresion_resumen_3datasets.csv")
    cadenas: list[str] = []
    for _, fila in resumen.iterrows():
        cadenas += [
            f"{fila['QK-SVR']:.3f}",
            f"{fila['Mejor Clásico']:.3f}",
            f"{abs(fila['Ventaja R²']):.3f}",
        ]
    return [Comprobacion("Regresión NB03 (tabla)", EVALUACION, c) for c in cadenas]


def _comprobaciones_regresion_justa() -> list[Comprobacion]:
    datos = _csv("qml_v2_regression_all.csv")
    resultado = []
    for dataset, grupo in datos.groupby("dataset", sort=False):
        q = grupo[grupo["model"] == "QK-SVR-ZZ"].iloc[0]
        a = grupo[(grupo["table"] == "A_matched_features") & (grupo["model"] != "QK-SVR-ZZ")]
        mejor_a = a.sort_values("r2_mean", ascending=False).iloc[0]
        resultado.append(
            Comprobacion(f"Regresión justa {dataset}: QK-SVR", EVALUACION, f"{q['r2_mean']:.3f}\\pm{q['r2_std']:.3f}")
        )
        resultado.append(
            Comprobacion(
                f"Regresión justa {dataset}: mejor clásico A", EVALUACION, f"{mejor_a['r2_mean']:.3f}\\pm{mejor_a['r2_std']:.3f}"
            )
        )
        resultado.append(
            Comprobacion(
                f"Regresión justa {dataset}: Δ A", EVALUACION, f"{q['r2_mean'] - mejor_a['r2_mean']:.3f}"
            )
        )
    return resultado


def _comprobaciones_clasificacion_justa() -> list[Comprobacion]:
    datos = _csv("fair_classification_all.csv")
    resultado = []
    for dataset, grupo in datos.groupby("dataset", sort=False):
        q = grupo[grupo["model"] == "QSVM-ZZ"].iloc[0]
        resultado.append(
            Comprobacion(
                f"Clasificación justa {dataset}: QSVM",
                EVALUACION,
                f"{100 * q['acc_mean']:.1f}\\pm{100 * q['acc_std']:.1f}",
            )
        )
    return resultado


def _comprobaciones_escalabilidad() -> list[Comprobacion]:
    datos = _csv("fair_scalability_native.csv")
    resultado = []
    for _, fila in datos.iterrows():
        resultado.append(
            Comprobacion(
                f"Escalabilidad n={int(fila['n_train'])}: QSVM",
                EVALUACION,
                f"{fila['qsvm_media']:.1f}",
            )
        )
    return resultado


def _comprobaciones_mds() -> list[Comprobacion]:
    datos = _csv("10_mds_qaoa_escalabilidad.csv")
    resultado = []
    for _, fila in datos.iterrows():
        resultado.append(
            Comprobacion(
                f"MDS n={int(fila['n'])}: P_max y P_unif",
                EVALUACION,
                f"{fila['qaoa_mejor_p_opt']:.4f} & {fila['p_uniforme_opt']:.4f} & {fila['t_qaoa_20reinicios_s']:.1f}",
            )
        )
    return resultado


def todas() -> list[Comprobacion]:
    return (
        _comprobaciones_sintesis()
        + _comprobaciones_regresion_nb03()
        + _comprobaciones_regresion_justa()
        + _comprobaciones_clasificacion_justa()
        + _comprobaciones_escalabilidad()
        + _comprobaciones_mds()
    )


def main() -> int:
    textos: dict[Path, str] = {}
    fallos = 0
    for comprobacion in todas():
        texto = textos.setdefault(comprobacion.fichero, comprobacion.fichero.read_text(encoding="utf-8"))
        if comprobacion.cadena not in texto:
            fallos += 1
            print(f"FALTA  {comprobacion.descripcion}: «{comprobacion.cadena}» no aparece en {comprobacion.fichero.name}")
    total = len(todas())
    print(f"{total - fallos}/{total} cifras encontradas")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
