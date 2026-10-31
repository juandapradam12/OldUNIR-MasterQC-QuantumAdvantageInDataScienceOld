"""Comprueba que cada cifra de los nueve casos de uso coincide con la salida ejecutada.

Cada *hecho* declara (i) los valores esperados, que se calculan a partir de un CSV/JSON de
``data/output/metrics`` o de la salida impresa de un notebook, (ii) en qué ficheros y en qué
sección deben figurar y (iii) un ancla (palabra o fila de tabla) que debe aparecer junto a ellos.
Así, un número correcto pero colocado en el lugar equivocado también se detecta.

Modos:

* sin argumentos: comprueba todos los hechos; código de salida 1 si falta alguno.
* ``--auditoria``: recorre las secciones de resultados de la tesis y lista los números que no
  figuran en ningún CSV del caso ni entre los valores derivados de los hechos y que tampoco están
  en ``EXENCIONES`` (cada exención lleva su justificación).

Uso, desde ``POCs/``::

    python scripts/verificar_cifras_tesis.py [--auditoria]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

POCS = Path(__file__).resolve().parents[1]
RAIZ = POCS.parent
METRICS = POCS / "data" / "output" / "metrics"
NOTEBOOKS = POCS / "notebooks"
CAPITULOS = RAIZ / "Documento" / "Capitulos"

FICHEROS = {
    "eval": CAPITULOS / "evaluacion.tex",
    "desa": CAPITULOS / "desarrollo.tex",
    "conc": CAPITULOS / "conclusiones.tex",
    "resumen": CAPITULOS / "resumen.tex",
    "abstract": CAPITULOS / "abstract.tex",
    "anexos": CAPITULOS / "anexos.tex",
    "defensa": RAIZ / "Documento" / "Defensa" / "defensa.tex",
    "readme": RAIZ / "README.md",
    "readme_pocs": POCS / "README.md",
    "intro": CAPITULOS / "introduccion.tex",
    "contexto": CAPITULOS / "contexto.tex",
    "objetivos": CAPITULOS / "objetivos.tex",
    "investigacion": CAPITULOS / "investigacion_original.tex",
    "futuro": CAPITULOS / "futuro.tex",
}
CON_SECCIONES_POR_CASO = {"eval", "desa"}


@dataclass(frozen=True)
class Destino:
    fichero: str
    caso: int | None = None


@dataclass(frozen=True)
class Hecho:
    caso: int
    descripcion: str
    valores: tuple[str, ...]
    destinos: tuple[Destino, ...]
    ancla: str = ""


def en(*claves: str, caso: int | None = None) -> tuple[Destino, ...]:
    return tuple(Destino(c, caso if c in CON_SECCIONES_POR_CASO else None) for c in claves)


# ----------------------------------------------------------------------------- lectura de datos
def csv(nombre: str) -> pd.DataFrame:
    return pd.read_csv(METRICS / nombre)


def salida_notebook(numero: str) -> str:
    ruta = next(NOTEBOOKS.glob(f"{numero}_*.ipynb"))
    celdas = json.loads(ruta.read_text(encoding="utf-8"))["cells"]
    trozos = []
    for celda in celdas:
        for salida in celda.get("outputs", []):
            trozos.append("".join(salida.get("text", [])))
    return "\n".join(trozos)


def de_notebook(numero: str, patron: str) -> str:
    """Primer grupo capturado por ``patron`` en la salida impresa del notebook."""
    coincidencia = re.search(patron, salida_notebook(numero))
    if coincidencia is None:
        raise LookupError(f"NB{numero}: patrón sin coincidencias: {patron}")
    return coincidencia.group(1)


def f(x: float, d: int = 1, escala: float = 1.0) -> str:
    cuanto = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(round(abs(float(x)) * escala, 9)))).quantize(cuanto, rounding=ROUND_HALF_UP))


def pm(media: float, desv: float, d: int = 1, escala: float = 1.0) -> str:
    signo = "-" if media < 0 else ""
    return f"{signo}{f(media, d, escala)}±{f(desv, d, escala)}"


# ----------------------------------------------------------------------------- normalización LaTeX
def normalizar(texto: str) -> str:
    texto = re.sub(r"(?m)^\s*%.*$", "", texto)
    texto = re.sub(r"(?<![\d.])(\d{2,3})\{,\}(\d{3})(?!\d)", r"\1\2", texto)
    for origen, destino in (
        ("\\%", "%"),
        ("{,}", "."),
        ("\\,", ""),
        ("\\!", ""),
        ("\\pm", "±"),
        ("$", ""),
        ("~", " "),
        ("\\#", "#"),
        ("−", "-"),
        ("{+}", "+"),
        ("{-}", "-"),
        ("{=}", "="),
        ("\\_", "_"),
        ("---", "—"),
    ):
        texto = texto.replace(origen, destino)
    texto = re.sub(r"(?<=\d)\\ ", "", texto)
    return re.sub(r"[ \t]+", " ", texto)


def bloque_de_caso(texto: str, caso: int) -> str:
    inicio = re.search(rf"\\section\{{Caso de uso {caso}\b", texto)
    if inicio is None:
        return ""
    fin = re.search(r"\\(section|chapter)\{", texto[inicio.end() :])
    return texto[inicio.start() : inicio.end() + fin.start()] if fin else texto[inicio.start() :]


_CACHE: dict[tuple[str, int | None], list[str]] = {}


def fragmentos(destino: Destino) -> list[str]:
    """Párrafos y líneas del bloque, ya normalizados."""
    clave = (destino.fichero, destino.caso)
    if clave not in _CACHE:
        texto = FICHEROS[destino.fichero].read_text(encoding="utf-8")
        if destino.caso is not None:
            texto = bloque_de_caso(texto, destino.caso)
        texto = normalizar(texto)
        parrafos = [p for p in re.split(r"\n\s*\n", texto) if p.strip()]
        _CACHE[clave] = parrafos + [linea for linea in texto.split("\n") if linea.strip()]
    return _CACHE[clave]


def contiene_valor(fragmento: str, valor: str) -> bool:
    patron = rf"(?<![\d.]){re.escape(valor)}(?!\d|\.\d)"
    return re.search(patron, fragmento) is not None


def comprobar(hecho: Hecho, destino: Destino) -> bool:
    for fragmento in fragmentos(destino):
        if hecho.ancla and not re.search(hecho.ancla, fragmento):
            continue
        if all(contiene_valor(fragmento, v) for v in hecho.valores):
            return True
    return False


# ----------------------------------------------------------------------------- hechos por caso
def caso1() -> list[Hecho]:
    h: list[Hecho] = []
    resumen = csv("01_clasificacion_resumen_3datasets.csv")
    skg, telco = (resumen.iloc[i] for i in (1, 2))
    h.append(
        Hecho(
            1,
            "NB01 SKG",
            (f(skg["QSVM"], 1, 100), f(skg["Mejor Clásico"], 1, 100), f(skg["Ventaja (pp)"])),
            en("eval", caso=1),
            "Sobre el dataset real de incidentes",
        )
    )
    h.append(
        Hecho(
            1,
            "NB01 Telco",
            (f(telco["QSVM"], 1, 100), f(telco["Mejor Clásico"], 1, 100), f(telco["Ventaja (pp)"])),
            en("eval", caso=1),
            "Sobre el dataset de referencia",
        )
    )
    h.append(
        Hecho(
            1,
            "NB01 SKG/Telco en conclusiones",
            (f(skg["Ventaja (pp)"]), f(telco["Ventaja (pp)"])),
            en("conc"),
            "Notebook",
        )
    )
    pca_skg = float(de_notebook("01", r"PCA varianza explicada: ([\d.]+)%"))
    pca_telco = float(de_notebook("01", r"Varianza explicada por PCA: ([\d.]+)%"))
    h.append(Hecho(1, "PCA SKG (NB01)", (f"{pca_skg:.1f}", f"{100 - pca_skg:.1f}"), en("eval", caso=1), "PCA"))
    h.append(
        Hecho(
            1,
            "Telco: variables, particiones y PCA (NB01)",
            (
                de_notebook("01", r"Features tras engineering: (\d+)"),
                de_notebook("01", r"Train : \((\d+), 19\)"),
                de_notebook("01", r"Test  : \((\d+), 19\)"),
                f"{pca_telco:.1f}",
            ),
            en("eval", caso=1),
            "asimétrica",
        )
    )

    fair = csv("fair_classification_all.csv")
    for dataset in ("skg", "telco"):
        g = fair[fair["dataset"] == dataset]
        q = g[g["model"] == "QSVM-ZZ"].iloc[0]
        a = g[(g["table"] == "A_matched_features") & (g["model"] != "QSVM-ZZ")].nlargest(1, "acc_mean").iloc[0]
        valores = [
            pm(q.acc_mean, q.acc_std, 1, 100),
            pm(a.acc_mean, a.acc_std, 1, 100),
            f((q.acc_mean - a.acc_mean) * 100),
        ]
        b = g[g["table"] == "B_matched_n_full_features"]
        if not b.empty:
            mb = b.nlargest(1, "acc_mean").iloc[0]
            valores += [pm(mb.acc_mean, mb.acc_std, 1, 100), f((q.acc_mean - mb.acc_mean) * 100)]
        ancla = {"skg": "SKG Incidentes &", "telco": "Telco Churn &"}[dataset]
        h.append(Hecho(1, f"Protocolo justo clasificación: {dataset}", tuple(valores), en("eval", caso=1), ancla))
    skg_a = fair[(fair["dataset"] == "skg") & (fair["model"] == "QSVM-ZZ")].iloc[0]
    telco_a = fair[(fair["dataset"] == "telco") & (fair["model"] == "QSVM-ZZ")].iloc[0]
    h.append(
        Hecho(
            1,
            "Protocolo justo SKG/Telco en conclusiones",
            (f(skg_a.delta_acc_vs_best_classical, 1, 100), f(telco_a.delta_acc_vs_best_classical, 1, 100)),
            en("conc"),
            "protocolo justo",
        )
    )
    h.append(
        Hecho(
            1,
            "Protocolo justo SKG (fair_classification_skg)",
            (pm(skg_a.acc_mean, skg_a.acc_std, 1, 100),),
            en("eval", caso=1),
            "SKG",
        )
    )
    skg_cl = csv("01_clasificacion_skg_incidentes.csv").set_index("Modelo")["Accuracy"]
    total = int(de_notebook("01", r"Dataset SKG — total filas: ([\d,]+)").replace(",", ""))
    prueba_cl = -(-total // 5)
    particion = (de_notebook("01", r"Train: (\d+) \| Test"), de_notebook("01", r"Test: (\d+)\n"))
    h.append(
        Hecho(
            1,
            "SKG (NB01): clásicos",
            (f(skg_cl["Gradient Boosting"], 1, 100), f(skg_cl["Logistic Reg."], 1, 100)),
            en("desa", caso=1),
            "quedaron entre ambos",
        )
    )
    h.append(Hecho(1, "SKG (NB01): submuestra del QSVM", particion, en("desa", caso=1), "Submuestreo"))
    h.append(
        Hecho(
            1,
            "SKG (NB01): partición de los clásicos",
            (f"{total - prueba_cl}", f"{prueba_cl}"),
            en("desa", caso=1),
            "reserva",
        )
    )
    directo = csv("01_qsvm_vs_svm_directo.csv")
    for _, r in directo.iloc[1:].iterrows():
        h.append(
            Hecho(
                1,
                f"Comparativa directa: {r.dataset}",
                (f(r.acc_qsvm, 3), f(r.acc_svm_rbf, 3), f(r.delta_acc, 3)),
                en("desa", caso=1),
                r.dataset.split(" (")[0].split(" Customer")[0].split(" movilidad")[0],
            )
        )
    map9 = csv("09_mapeos_resultados.csv")
    base = map9[map9["strategy"] == "Baseline"].set_index("dataset")["Q_score"]
    h.append(Hecho(1, "Q baseline de NB09 (SKG)", (f(base["SKG Incidentes"], 3),), en("desa", caso=1), "Dataset SKG"))
    h.append(Hecho(1, "Q baseline de NB09 (Telco)", (f(base["Telco Churn"], 3),), en("desa", caso=1), "Telco Churn:"))
    h.append(
        Hecho(
            1,
            "Contraste directo: SKG",
            (f(directo.delta_acc.iloc[1], 1, 100), f(directo.pca_var_exp.iloc[1], 1, 100)),
            en("desa", caso=1),
            "Dataset SKG",
        )
    )
    h.append(
        Hecho(
            1,
            "Contraste directo: Telco",
            (f(directo.delta_acc.iloc[2], 1, 100), str(int(directo.n_train.iloc[2]))),
            en("desa", caso=1),
            "Telco Churn:",
        )
    )
    tiempos = csv("01_clasificacion_resultados.csv").set_index("model")["training_time_s"]
    clasicos = tiempos.drop(["Quantum Kernel SVM", "VQC"])
    nb6 = csv("06_quantum_kernel_advantage.csv").set_index("model")["time_s"]
    clasicos6 = nb6.drop("Quantum Kernel SVM")
    h.append(
        Hecho(
            1,
            "Tiempos de simulación del kernel (conclusiones)",
            (
                f(tiempos["Quantum Kernel SVM"], 1),
                f(clasicos.min(), 1),
                f(clasicos.max(), 1),
                f(nb6["Quantum Kernel SVM"], 1),
                f(clasicos6.min(), 3),
                f(clasicos6.max(), 2),
            ),
            en("conc"),
            "O\\(n\\^2\\)|O\\(n.2\\)",
        )
    )
    total_inc = int(de_notebook("01", r"SKG — total filas: ([\d,]+)").replace(",", "").replace(".", ""))
    celdas01 = [
        c
        for c in json.loads(next(NOTEBOOKS.glob("01_*.ipynb")).read_text(encoding="utf-8"))["cells"]
        if "SKG_QK_TRAIN" in "".join(c["source"])
    ]
    codigo01 = "".join("".join(c["source"]) for c in celdas01)
    qk_train = int(re.search(r"SKG_QK_TRAIN\s*=\s*(\d+)", codigo01).group(1))
    qk_test = int(re.search(r"SKG_QK_TEST\s*=\s*(\d+)", codigo01).group(1))
    entrenamiento_clasico = total_inc - math.ceil(0.2 * total_inc)
    h.append(
        Hecho(
            1,
            "NB01 SKG: instancias del escalado y la PCA",
            (str(2 * (qk_train + qk_test)),),
            en("desa", caso=1),
            "Estandarización",
        )
    )
    h.append(
        Hecho(
            1,
            "NB01 SKG: entrenamiento de los clásicos",
            (f"{entrenamiento_clasico:,}".replace(",", "."),),
            en("desa", caso=1),
            "submuestreo a",
        )
    )
    return h


def caso2() -> list[Hecho]:
    h: list[Hecho] = []
    res = csv("03_regresion_resumen_3datasets.csv")
    skg, fri = (res.iloc[i] for i in (1, 2))
    for nombre, fila, ancla in (
        ("SKG", skg, "SKG Corredores"),
        ("Friedman", fri, "Friedman"),
    ):
        h.append(
            Hecho(
                2,
                f"NB03 {nombre} (tabla)",
                (f(fila["QK-SVR"], 3), f(fila["Mejor Clásico"], 3), f(fila["Ventaja R²"], 3)),
                en("eval", caso=2),
                ancla,
            )
        )
    skg_bc = csv("03_regresion_skg_bc.csv").set_index("Modelo")
    gb = skg_bc.loc["Gradient Boosting"]
    h.append(
        Hecho(
            2,
            "NB03 SKG: RMSE y R² de QK-SVR y GB",
            (f(skg["QK-SVR"], 3), f(gb.RMSE, 2), f(gb.R2, 3), f(skg["Ventaja R²"], 3)),
            en("eval", caso=2),
            "RMSE",
        )
    )
    h.append(Hecho(2, "NB03 SKG en conclusiones", (f(gb.R2, 3), f(skg["QK-SVR"], 3)), en("conc"), "corredores"))
    rmse_q = csv("03_regresion_resultados.csv").set_index("model")
    friedman = rmse_q["r2"]
    h.append(
        Hecho(
            2,
            "NB03 Friedman: XGBoost, regresión lineal y RF",
            (f(friedman["XGBoost"], 3), f(friedman["Reg. Lineal"], 3), f(friedman["Random Forest"], 3)),
            en("eval", caso=2),
            "Friedman",
        )
    )
    h.append(
        Hecho(
            2, "NB03 Friedman en conclusiones", (f(fri["QK-SVR"], 3), f(fri["Mejor Clásico"], 3)), en("conc"), "XGBoost"
        )
    )
    pca_skg = float(de_notebook("03", r"PCA varianza explicada: ([\d.]+)%"))
    h.append(Hecho(2, "PCA SKG (NB03)", (f"{pca_skg:.1f}", f"{100 - pca_skg:.1f}"), en("eval", caso=2), "PCA"))
    total = int(de_notebook("03", r"total filas: ([\d,]+)").replace(",", ""))
    prueba = math.ceil(0.2 * total)
    entrenamiento = total - prueba
    h.append(
        Hecho(
            2,
            "SKG: aprovechamiento de datos de entrenamiento",
            (f"{100 * 150 / entrenamiento:.2f}", str(entrenamiento), str(total)),
            en("eval", caso=2),
            "Desperdicio de datos",
        )
    )
    h.append(
        Hecho(
            2,
            "SKG: tamaño de entrenamiento clásico (desarrollo)",
            (str(entrenamiento),),
            en("desa", caso=2),
            "Submuestreo",
        )
    )
    h.append(
        Hecho(
            2,
            "SKG: aprovechamiento (desarrollo)",
            (f"{100 * 150 / entrenamiento:.2f}", str(entrenamiento)),
            en("desa", caso=2),
            "aprovechamiento|supone usar",
        )
    )
    h.append(
        Hecho(
            2,
            "SKG: particiones asimétricas",
            (str(entrenamiento), str(prueba)),
            en("eval", caso=2),
            "asimetría es mucho mayor",
        )
    )
    h.append(
        Hecho(
            2,
            "Friedman: particiones asimétricas",
            ("300", "210", "90", "150", "60"),
            en("eval", caso=2),
            "asimetría es distinta",
        )
    )
    h.append(
        Hecho(
            2,
            "Mallado de hiperparámetros del QK-SVR en Friedman",
            ("0.5", "2", "50", "100", "10"),
            en("eval", caso=2),
            "malla de 9",
        )
    )
    retenida = [
        float(de_notebook("01", r"PCA varianza explicada: ([\d.]+)%")),
        float(de_notebook("01", r"Varianza explicada por PCA: ([\d.]+)%")),
        pca_skg,
        *(100 * csv("fair_classification_all.csv").groupby("dataset")["pca_var"].first().dropna()),
        *(
            100
            * csv("qml_v2_regression_all.csv").groupby("dataset")["pca_var"].first().dropna().loc[["skg", "friedman"]]
        ),
    ]
    descartada = (f"{100 - max(retenida):.0f}", f"{100 - min(retenida):.0f}")
    h.append(
        Hecho(2, "Rango de varianza descartada por la PCA (resto de cuadernos)", descartada, en("conc"), "varianza")
    )
    nb08 = [
        float(x) for x in re.findall(r"PCA varianza explicada: ([\d.]+)%\s+\|\s+muestra: n=80", salida_notebook("08"))
    ]
    descartada_nb08 = tuple(f"{100 - x:.0f}" for x in sorted(nb08)[:2])
    h.append(
        Hecho(
            2,
            "Varianza descartada en NB08 (SKG y Telco)",
            (min(descartada_nb08), max(descartada_nb08)),
            en("conc"),
            "(varianza|PCA)",
        )
    )

    reg = csv("qml_v2_regression_all.csv")
    filas = {
        "skg": ("SKG Corredores &", "A_matched_features"),
        "friedman": ("Friedman #1 \\(2 comp.\\) &", "A_matched_features"),
        "friedman_full": ("Friedman #1 \\(5 qubits\\) &", "A_matched_features"),
    }
    for dataset, (ancla, _) in filas.items():
        g = reg[reg["dataset"] == dataset]
        q = g[g["model"] == "QK-SVR-ZZ"].iloc[0]
        a = g[(g["table"] == "A_matched_features") & (g["model"] != "QK-SVR-ZZ")].nlargest(1, "r2_mean").iloc[0]
        valores = [pm(q.r2_mean, q.r2_std, 3), pm(a.r2_mean, a.r2_std, 3), f(q.r2_mean - a.r2_mean, 3)]
        b = g[g["table"] == "B_matched_n_full_features"]
        if not b.empty:
            mb = b.nlargest(1, "r2_mean").iloc[0]
            valores += [pm(mb.r2_mean, mb.r2_std, 3), f(q.r2_mean - mb.r2_mean, 3)]
        h.append(Hecho(2, f"Protocolo justo regresión: {dataset}", tuple(valores), en("eval", caso=2), ancla))
    skg_q = reg[(reg["dataset"] == "skg") & (reg["model"] == "QK-SVR-ZZ")].iloc[0]
    fri_q = reg[(reg["dataset"] == "friedman") & (reg["model"] == "QK-SVR-ZZ")].iloc[0]
    h.append(
        Hecho(
            2,
            "Δ justo SKG y Friedman en conclusiones",
            (f(skg_q.r2_mean - 0.031554, 3), f(fri_q.delta_r2_vs_best_classical, 3)),
            en("conc"),
            "mismas variables",
        )
    )
    return h


def caso3() -> list[Hecho]:
    h: list[Hecho] = []
    resumen = csv("qml_v2_optimization_summary.csv").set_index("model")
    bruto = csv("qml_v2_optimization_raw.csv")
    voraz = bruto[bruto["model"] == "Greedy"]
    qaoa = bruto[bruto["model"] == "QAOA-p1"]
    optimo = bruto[bruto["model"] == "Hungarian"].set_index("instance_id")["cost"]
    voraz_optimo = int((np.isclose(voraz.set_index("instance_id")["cost"], optimo)).sum())
    qaoa_optimo = int((np.isclose(qaoa.set_index("instance_id")["cost"], optimo)).sum())
    n = int(resumen.loc["QAOA-p1", "n"])
    h.append(
        Hecho(
            3,
            "Tabla de asignación",
            (
                f(resumen.loc["Hungarian", "cost_mean"], 3),
                f(resumen.loc["Greedy", "cost_mean"], 3),
                f(resumen.loc["Greedy", "gap_mean"], 1, 100),
                f"{qaoa_optimo}/{n}",
                f"{voraz_optimo}/{n}",
                f(voraz["gap_vs_pulp"].max(), 1, 100),
            ),
            en("eval", caso=3),
            "Greedy",
        )
    )
    h.append(
        Hecho(
            3,
            "Tiempos de QAOA (s) y exactos",
            (f"{round(qaoa['time_s'].min(), -1):.0f}", f"{round(qaoa['time_s'].max(), -1):.0f}"),
            en("eval", caso=3),
            "QAOA",
        )
    )
    h.append(Hecho(3, "Qubits de la codificación n^2 con n=6", (str(6**2),), en("desa", caso=3), "n=6 ya requiere"))
    h.append(Hecho(3, "Tiempo de los exactos", ("0.02",), en("eval", caso=3), "Hungarian"))
    assert resumen.loc[["Hungarian", "PuLP-CBC"], "time_mean_s"].max() < 0.02

    semillas = csv("02_optimizacion_qaoa_semillas.csv")
    comp = csv("02_optimizacion_ventaja_cuantica.csv")
    voraz_nb = float(comp.loc[comp["metodo"].str.startswith("Greedy"), "coste"].iloc[0])
    exceso = 100 * (semillas["coste"].mean() - voraz_nb) / voraz_nb
    p1 = semillas[semillas["p"] == 1]
    p2 = semillas[semillas["p"] == 2]
    fuente = (POCS / "qml_v2" / "optimization.py").read_text(encoding="utf-8")
    cfg = {
        k: re.search(rf"{k}: \w+ = ([\d.]+)", fuente).group(1)
        for k in ("n_instances", "cobyla_maxiter", "qaoa_restarts", "random_state", "cost_low", "cost_high")
    }
    semillas_txt = f"{cfg['random_state']}--{int(cfg['random_state']) + int(cfg['n_instances']) - 1}"
    h.append(Hecho(3, "Semillas de las instancias", (semillas_txt,), en("eval", "desa", caso=3), "semillas"))
    h.append(
        Hecho(
            3,
            "Configuración de QAOA",
            (cfg["cobyla_maxiter"], cfg["qaoa_restarts"]),
            en("eval", "desa", caso=3),
            "COBYLA",
        )
    )
    h.append(
        Hecho(
            3,
            "Desarrollo: brecha y tiempos",
            (f(voraz["gap_vs_pulp"].max(), 1, 100), f(resumen.loc["Greedy", "gap_mean"], 1, 100), "0.02"),
            en("desa", caso=3),
            "Greedy lo falla",
        )
    )
    h.append(
        Hecho(
            3,
            "NB02: ejecuciones óptimas y exceso frente a Greedy",
            (
                f"{int(semillas['es_optimo'].sum())} de {len(semillas)}",
                f"{int(p1['es_optimo'].sum())} de {len(p1)}",
                f"{exceso:.1f}",
            ),
            en("eval", caso=3),
            "Notebook 02",
        )
    )
    assert int(p2["es_optimo"].sum()) == int(p1["es_optimo"].sum())
    h.append(
        Hecho(3, "NB02: choques", (de_notebook("02", r"hora=8 h -> (\d+) incidentes"),), en("eval", caso=3), "choques")
    )
    h.append(
        Hecho(
            3,
            "Caso 3: QAOA recupera las cinco instancias",
            ("cinco",),
            en("conc", "resumen"),
            "asignación|Caso~3|Caso 3",
        )
    )
    return h


def caso6() -> list[Hecho]:
    res = csv("06_qkernel_resumen_3datasets.csv")
    skg, telco = (res.iloc[i] for i in (1, 2))
    adv = csv("06_quantum_kernel_advantage.csv")
    h = [
        Hecho(
            6,
            "NB06 SKG",
            (f(skg["QSVC"], 1, 100), f(skg["Mejor Clásico"], 1, 100), f(skg["Ventaja (pp)"])),
            en("eval", "desa", caso=6),
            "SKG",
        ),
        Hecho(
            6,
            "NB06 Telco",
            (f(telco["QSVC"], 1, 100), f(telco["Mejor Clásico"], 1, 100), f(telco["Ventaja (pp)"])),
            en("eval", "desa", caso=6),
            "Telco",
        ),
        Hecho(6, "Partición única de SKG", ("150",), en("eval", caso=6), "puntos de test"),
    ]
    assert len(adv) >= 2
    assert (
        de_notebook("06", r"Train: (\d+) \| Test: \d+") == "105"
        and de_notebook("06", r"Train: \d+ \| Test: (\d+)") == "45"
    )
    n_skg_tr = de_notebook("06", r"Train: (\d+) \| Test: 150")
    t_skg = de_notebook("06", r"QSVC accuracy: [\d.]+%\s+\[([\d.]+)s\]")
    h.append(
        Hecho(
            6,
            "NB06 SKG: entrenamiento y tiempo del QSVC",
            (n_skg_tr, f(float(t_skg), 0)),
            en("desa", caso=6),
            "coste cuadrático",
        )
    )
    return h


def caso7() -> list[Hecho]:
    j = json.loads((METRICS / "10_mds_qaoa_resumen.json").read_text(encoding="utf-8"))
    por_p = j["resumen_por_p"]
    esc = csv("10_mds_qaoa_escalabilidad.csv")
    reinicios = csv("10_mds_qaoa_reinicios.csv")
    n_dom = int(esc["qaoa_mejor_muestra_es_dominante"].sum())
    n_ig = int(
        ((esc["qaoa_mejor_coste_muestreado"] == esc["qubo_optimo"]) & esc["qaoa_mejor_muestra_es_dominante"]).sum()
    )
    assert n_dom == n_ig
    p1, p3 = por_p["1"], por_p["3"]
    probabilidades = csv("10_mds_qaoa_probabilidades.csv")
    frecuencia = float(probabilidades.sort_values("prob_exacta", ascending=False).iloc[0]["frecuencia_shots"])
    r3 = reinicios[reinicios["p"] == 3]
    visto99 = int((r3["p_opt_visto_en_shots"] >= 0.99).sum())
    assert max(esc["t_greedy_s"].max(), esc["t_ilp_s"].max()) < 0.025
    base = 1 / 2 ** j["n_qubits"]
    factor1, factor3 = p1["p_opt_media"] / base, p3["p_opt_media"] / base
    factor_max = p3["p_opt_max"] / base
    h = [
        Hecho(
            7,
            "Red de 6 nodos: probabilidades de medir el óptimo",
            (
                f(p3["p_opt_media"], 1, 100),
                f(p3["p_opt_max"], 1, 100),
                f(p1["p_opt_media"], 1, 100),
                f(base, 1, 100),
                f"{factor1:.1f}",
                f"{factor3:.1f}",
                f"{factor_max:.0f}",
            ),
            en("eval", caso=7),
            "concentra",
        ),
        Hecho(
            7,
            "Probabilidad media y mejor reinicio en resumen/abstract/defensa",
            (f(p3["p_opt_media"], 1, 100), f(p3["p_opt_max"], 1, 100)),
            en("resumen", "abstract"),
            "p=3|\\$p=3",
        ),
        Hecho(
            7,
            "Mejor energía esperada por profundidad",
            (f(por_p["3"]["energia_min"], 2), f(por_p["1"]["energia_min"], 1)),
            en("eval", caso=7),
            "Convergencia",
        ),
        Hecho(
            7,
            "Instancias ER con MDS óptimo",
            (f"{n_dom} de {len(esc)}", f"{len(esc)} de {len(esc)}", f(100 * (n_dom / len(esc) - 1), 0)),
            en("eval", caso=7),
            "Medido como porcentaje",
        ),
        Hecho(7, "Instancias ER en resumen/abstract", (f"{n_dom} de {len(esc)}",), en("resumen"), "Erd"),
        Hecho(7, "Instancias ER en conclusiones", (f"{n_dom}",), en("conc"), "Caso~7|MDS"),
        Hecho(
            7,
            "Tiempos de simulación del QAOA",
            (f(esc["t_qaoa_20reinicios_s"].iloc[0]), f(esc["t_qaoa_20reinicios_s"].iloc[-1])),
            en("eval", caso=7),
            "crece",
        ),
        Hecho(
            7,
            "Probabilidad del óptimo del QUBO en el mejor reinicio (n=6 y n=15)",
            (
                f(esc["qaoa_mejor_p_opt"].iloc[0], 0, 100),
                f(esc["qaoa_mejor_p_opt"].iloc[-1], 1, 100),
                f"{esc['enriquecimiento_vs_uniforme'].min():.0f}",
                f"{esc['enriquecimiento_vs_uniforme'].max():.0f}",
            ),
            en("eval", caso=7),
            "cae",
        ),
        Hecho(
            7,
            "Reinicios cercanos al óptimo (60 reinicios)",
            (f"{len(reinicios)}",),
            en("eval", caso=7),
            "reinicios solo uno",
        ),
        Hecho(
            7,
            "Probabilidades por profundidad (desarrollo)",
            (
                f(p1["p_opt_media"], 1, 100),
                f(p1["p_opt_max"], 1, 100),
                f(por_p["2"]["p_opt_media"], 1, 100),
                f(por_p["2"]["p_opt_max"], 1, 100),
                f(p3["p_opt_media"], 1, 100),
                f(p3["energia_min"], 2),
                f(p3["p_opt_max"], 1, 100),
            ),
            en("desa", caso=7),
            "Resultados medidos",
        ),
        Hecho(
            7,
            "Frecuencia en shots del mejor reinicio",
            (f(frecuencia, 1, 100), f(p3["p_opt_max"], 1, 100)),
            en("desa", caso=7),
            "shots",
        ),
        Hecho(
            7,
            "Reinicios con el óptimo visto con probabilidad > 99 %",
            (f"{visto99} de {len(reinicios[reinicios['p'] == 3])}",),
            en("desa", caso=7),
            "reinicios",
        ),
        Hecho(
            7,
            "Tiempos Greedy/ILP y QAOA (desarrollo)",
            (f(esc["t_qaoa_20reinicios_s"].iloc[0]), f(esc["t_qaoa_20reinicios_s"].iloc[-1]), "25"),
            en("desa", caso=7),
            "Greedy e ILP",
        ),
        Hecho(7, "Máximo tiempo de Greedy/ILP por debajo de 25 ms", ("25",), en("desa", caso=7), "25 ms"),
    ]
    for _, r in esc.iterrows():
        valores = (
            f"{int(r.n_aristas)}",
            f"{int(r.ilp_tamano)}",
            f"{int(r.greedy_tamano)}",
            f"{int(r.qubo_optimo)}",
            f"{int(r.qaoa_tamano_mejor_muestra)} ({'sí' if r.qaoa_mejor_muestra_es_dominante else 'no'}",
            f"{r.qaoa_mejor_p_opt:.4f}",
            f"{r.p_uniforme_opt:.4f}",
            f"{r.t_qaoa_20reinicios_s:.1f}",
        )
        h.append(Hecho(7, f"Tabla de escalabilidad MDS n={int(r.n)}", valores, en("eval", caso=7), rf"^{int(r.n)}\s+&"))
    return h


def caso8() -> list[Hecho]:
    h: list[Hecho] = []
    sint = csv("07_sintesis_resultados.csv")
    s8 = sint[sint["caso"] == 8].set_index("etiqueta")["valor"]
    datos = {}
    for nombre in ("zz_phase", "moons", "blobs", "telco_injected", "skg_injected"):
        d = csv(f"fair_anomaly_{nombre}.csv")
        datos[nombre] = d

    def auc(nombre: str, modelo: str, tabla: str = "A") -> float:
        d = datos[nombre]
        d = d[d["table"].str.startswith(tabla)]
        return float(d[d["model"].str.startswith(modelo)].iloc[0]["auc_mean"])

    h.append(
        Hecho(
            8,
            "moons",
            (f(auc("moons", "IsolationForest"), 3), f(auc("moons", "OCSVM-ZZ"), 3)),
            en("eval", caso=8),
            "moons",
        )
    )
    h.append(Hecho(8, "blobs", (f(auc("blobs", "OCSVM-ZZ"), 3),), en("eval", caso=8), "blobs"))
    h.append(Hecho(8, "telco_injected", (f(auc("telco_injected", "IsolationForest"), 3),), en("eval", caso=8), "telco"))
    h.append(Hecho(8, "skg_injected", (f(auc("skg_injected", "LOF"), 3),), en("eval", caso=8), "skg"))
    h.append(
        Hecho(
            8,
            "Conclusiones: Caso 8",
            tuple(f(s8[k], 1) for k in ("telco_injected", "skg_injected", "moons", "blobs")),
            en("conc"),
            "Caso 8",
        )
    )
    h.append(
        Hecho(
            8,
            "Síntesis del caso 8",
            tuple(f(s8[k], 1) for k in ("skg_injected", "moons", "blobs")),
            en("eval"),
            "OCSVM-ZZ",
        )
    )
    cfg8 = json.loads((METRICS / "qml_v2_anomaly_config.json").read_text(encoding="utf-8"))
    h.append(
        Hecho(
            8,
            "Configuración",
            (str(cfg8["n_total"]), str(cfg8["n_repeats"]), f"{100 * cfg8['contamination']:.0f}"),
            en("eval", caso=8),
            "n=160",
        )
    )
    h.append(
        Hecho(
            8,
            "Configuración (desarrollo)",
            (str(cfg8["n_total"]), str(cfg8["n_repeats"]), f"{100 * cfg8['contamination']:.0f}", str(cfg8["nu"])),
            en("desa", caso=8),
            "Configuración reportada",
        )
    )
    return h


def caso9() -> list[Hecho]:
    h: list[Hecho] = []
    sine = csv("fair_timeseries_sine.csv")
    skg = csv("fair_timeseries_skg_speed.csv")

    def fila(d: pd.DataFrame, modelo: str, tabla: str = "A") -> pd.Series:
        d = d[d["table"].str.startswith(tabla)]
        return d[d["model"] == modelo].iloc[0]

    sine_cl = (
        sine[(sine["table"].str.startswith("A")) & (sine["model"] != "QK-SVR-ZZ")].nsmallest(1, "mae_mean").iloc[0]
    )
    h.append(
        Hecho(9, "sine", (f(sine_cl.mae_mean, 3), f(fila(sine, "QK-SVR-ZZ").mae_mean, 3)), en("eval", caso=9), "sine")
    )
    skg_cl = skg[(skg["table"].str.startswith("A")) & (skg["model"] != "QK-SVR-ZZ")].nsmallest(1, "mae_mean").iloc[0]
    h.append(
        Hecho(9, "skg_speed", (f(skg_cl.mae_mean, 2), f(fila(skg, "QK-SVR-ZZ").mae_mean, 2)), en("eval", caso=9), "skg")
    )
    cfg9 = json.loads((METRICS / "qml_v2_timeseries_config.json").read_text(encoding="utf-8"))
    h.append(
        Hecho(
            9, "Configuración", (str(cfg9["n_total"]), str(cfg9["n_repeats"])), en("desa", caso=9), "ventana contigua"
        )
    )
    puntos_sine = max(cfg9["n_total"] * 3, 400)
    fuente_ts = (POCS / "qml_v2" / "timeseries.py").read_text(encoding="utf-8")
    ruido = float(re.search(r"def make_sine_series\([^)]*noise: float = ([\d.]+)", fuente_ts).group(1))
    skg_n = pd.read_csv(RAIZ / "Datos" / "bctraveltimes_history.csv", usecols=["tid"])["tid"].value_counts()
    h.append(
        Hecho(9, "Partición temporal", (str(round((1 - cfg9["test_size"]) * 100)),), en("desa", caso=9), "División")
    )
    h.append(Hecho(9, "Serie sine", (str(puntos_sine), f(ruido, 2)), en("desa", caso=9), "sine"))
    h.append(Hecho(9, "Corredores SKG", (str(skg_n.max()), str(len(skg_n))), en("desa", caso=9), "skg_speed"))
    h.append(
        Hecho(
            9,
            "README: Caso 9",
            (f(skg_cl.mae_mean, 2), f(fila(skg, "QK-SVR-ZZ").mae_mean, 2)),
            en("readme"),
            "Forecasting",
        )
    )
    h.append(Hecho(9, "Repeticiones", (str(cfg9["n_repeats"]),), en("eval", caso=9), "repeticiones"))
    h.append(
        Hecho(
            9,
            "Tabla del desarrollo (MAE)",
            (
                f(sine_cl.mae_mean, 3),
                f(fila(sine, "QK-SVR-ZZ").mae_mean, 3),
                f(skg_cl.mae_mean, 2),
                f(fila(skg, "QK-SVR-ZZ").mae_mean, 2),
            ),
            en("desa", caso=9),
            "sine",
        )
    )
    refs = csv("fair_timeseries_references.csv").set_index(["dataset", "reference"])["mae_mean"]
    sine_b = sine[sine["table"].str.startswith("B")].nsmallest(1, "mae_mean").iloc[0]
    skg_b = skg[skg["table"].str.startswith("B")].nsmallest(1, "mae_mean").iloc[0]
    skg_q = fila(skg, "QK-SVR-ZZ")
    h.append(
        Hecho(
            9,
            "sine: media del entrenamiento y Tabla B",
            (f(refs["sine", "train_mean"], 3), f(sine_b.mae_mean, 3)),
            en("eval", caso=9),
            "sine",
        )
    )
    h.append(
        Hecho(
            9,
            "skg_speed: dispersión, referencias y Tabla B",
            (
                pm(skg_cl.mae_mean, skg_cl.mae_std, 2),
                pm(skg_q.mae_mean, skg_q.mae_std, 2),
                f(refs["skg_speed", "train_mean"], 2),
                f(refs["skg_speed", "persistence"], 2),
                f(skg_b.mae_mean, 2),
            ),
            en("eval", caso=9),
            "skg_speed",
        )
    )
    h.append(
        Hecho(
            9, "skg_speed: ventanas por corredor", (str(int(skg_q.n_train + skg_q.n_test)),), en("desa", caso=9), "138"
        )
    )
    sint = csv("07_sintesis_resultados.csv")
    s9 = sint[sint["caso"] == 9].set_index("etiqueta")["valor"]
    h.append(
        Hecho(
            9,
            "Síntesis del caso 9",
            tuple(f(s9[k], 1) for k in ("skg_speed", "sine")),
            en("eval"),
            "QK-SVR lags",
        )
    )
    return h


def sintesis_general() -> list[Hecho]:
    sint = csv("07_sintesis_resultados.csv")
    h = []
    fila = sint[(sint["caso"] == 1)].set_index("capa")["valor"]
    h.append(Hecho(1, "Síntesis CU1", tuple(f(fila[c], 1) for c in ("skg", "referencia")), en("eval"), "QSVM"))
    fila = sint[(sint["caso"] == 2)].set_index("capa")["valor"]
    h.append(Hecho(2, "Síntesis CU2", tuple(f(fila[c] / 100, 3) for c in ("skg", "referencia")), en("eval"), "QK-SVR"))
    fila = sint[(sint["caso"] == 6)].set_index("capa")["valor"]
    h.append(Hecho(6, "Síntesis CU6", tuple(f(fila[c], 1) for c in ("skg", "referencia")), en("eval"), "umbral"))
    return h


def investigacion_nb08() -> list[Hecho]:
    pca = re.findall(r"PCA varianza explicada: ([\d.]+)%", salida_notebook("08"))
    return [
        Hecho(0, "NB08: varianza de la PCA en SKG y Telco", (pca[1], pca[2]), en("investigacion"), "configuración base")
    ]


def entorno() -> list[Hecho]:
    """Versiones del entorno B frente a lo instalado, al lock y a los metadatos de los cuadernos."""
    from importlib import metadata

    paquetes = {
        "Qiskit": "qiskit",
        "qiskit-aer": "qiskit-aer",
        "qiskit-algorithms": "qiskit-algorithms",
        "qiskit-machine-learning": "qiskit-machine-learning",
        "qiskit-optimization": "qiskit-optimization",
        "qiskit-finance": "qiskit-finance",
        "scikit-learn": "scikit-learn",
        "XGBoost": "xgboost",
        "SciPy": "scipy",
        "NumPy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "NetworkX": "networkx",
        "yfinance": "yfinance",
    }
    h = [
        Hecho(0, f"Versión instalada de {nombre}", (metadata.version(dist),), en("anexos"), re.escape(nombre))
        for nombre, dist in paquetes.items()
    ]
    h.append(Hecho(0, "Versión de PuLP", (metadata.version("pulp"),), en("anexos"), "PuLP"))
    for nombre, dist in (
        ("Qiskit", "qiskit"),
        ("scikit-learn", "scikit-learn"),
        ("XGBoost", "xgboost"),
        ("SciPy", "scipy"),
        ("pandas", "pandas"),
        ("NumPy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("NetworkX", "networkx"),
    ):
        h.append(
            Hecho(0, f"README: versión de {nombre}", (metadata.version(dist),), en("readme_pocs"), re.escape(nombre))
        )
    versiones = {
        json.loads(r.read_text(encoding="utf-8"))["metadata"].get("language_info", {}).get("version")
        for r in sorted(NOTEBOOKS.glob("*.ipynb"))
        if not r.name.startswith("01b")
    }
    assert versiones == {platform.python_version()}, versiones
    h.append(
        Hecho(
            0, "Python de los cuadernos", (platform.python_version(),), en("anexos"), "metadatos de todos los cuadernos"
        )
    )
    h.append(Hecho(0, "Python en README", (platform.python_version(),), en("readme_pocs"), "reejecución"))
    bloqueadas = {
        linea.split("==")[0].lower(): linea.split("==")[1]
        for linea in (POCS / "requirements-lock.txt").read_text(encoding="utf-8").split()
        if "==" in linea
    }
    for dist in [*paquetes.values(), "pulp"]:
        assert bloqueadas[dist] == metadata.version(dist), dist
    return h


def datos_de_entrada() -> list[Hecho]:
    """Tamaños de los conjuntos de datos; solo se comprueban si la carpeta ``Datos`` está disponible."""
    carpeta = RAIZ / "Datos"
    if not carpeta.exists():
        return []
    miles = lambda n: f"{int(n):,}".replace(",", ".")  # noqa: E731
    incidentes = pd.read_csv(carpeta / "incidents_movility.csv", low_memory=False, usecols=["gravity"])
    simples = int((incidentes["gravity"] == 25.0).sum())
    lesionados = int((incidentes["gravity"] == 26.0).sum())
    tiempos = pd.read_csv(carpeta / "bctraveltimes_history.csv", low_memory=False, usecols=["speed"])
    agentes = pd.read_csv(carpeta / "agents.csv", low_memory=False, usecols=["dev_id"])
    telco = pd.read_csv(POCS / "data" / "input" / "telco_customer_churn.csv", usecols=[0])
    return [
        Hecho(
            1,
            "SKG incidentes: registros y clases",
            (miles(simples + lesionados), str(simples), str(lesionados)),
            en("desa", caso=1),
            "Se utilizaron datos reales",
        ),
        Hecho(1, "SKG incidentes en el anexo", (miles(simples + lesionados),), en("anexos"), "Incidentes NUSE"),
        Hecho(2, "SKG tiempos de viaje", (miles(len(tiempos)),), en("desa", caso=2), "históricos"),
        Hecho(2, "SKG tiempos de viaje en el anexo", (miles(len(tiempos)),), en("anexos"), "Tiempos de viaje"),
        Hecho(
            3,
            "SKG agentes en el anexo",
            (miles(len(agentes)), str(agentes["dev_id"].nunique())),
            en("anexos"),
            "Agentes en campo",
        ),
        Hecho(
            3,
            "SKG agentes: lecturas, dispositivos y localidades",
            (
                miles(len(agentes)).replace(".", ""),
                str(agentes["dev_id"].nunique()),
                str(
                    pd.read_csv(carpeta / "agents.csv", low_memory=False, usecols=["localidad_id"])[
                        "localidad_id"
                    ].nunique()
                ),
            ),
            en("futuro"),
            "Validación cruzada sobre",
        ),
        Hecho(1, "Telco en desarrollo y anexo", (miles(len(telco)),), en("desa", "anexos", caso=1), "Telco"),
    ]


def readme() -> list[Hecho]:
    sint = csv("07_sintesis_resultados.csv")
    c2 = sint[sint["caso"] == 2].set_index("capa")["valor"]
    c1 = sint[sint["caso"] == 1].set_index("capa")["valor"]
    j = json.loads((METRICS / "10_mds_qaoa_resumen.json").read_text(encoding="utf-8"))
    esc = csv("10_mds_qaoa_escalabilidad.csv")
    n_dom = int(esc["qaoa_mejor_muestra_es_dominante"].sum())
    return [
        Hecho(2, "README: regresión", (f(c2["skg"], 1), f(c2["referencia"], 1)), en("readme"), "Quantum-kernel SVR"),
        Hecho(
            1, "README: clasificación", (f(c1["skg"], 1), f(c1["referencia"], 1)), en("readme"), "Quantum kernel SVM"
        ),
        Hecho(
            7,
            "README: MDS",
            (f(j["resumen_por_p"]["3"]["p_opt_media"], 1, 100), f"{n_dom} of {len(esc)}", f"{len(esc)} of {len(esc)}"),
            en("readme"),
            "dominating set",
        ),
    ]


def todos() -> list[Hecho]:
    hechos: list[Hecho] = []
    for generador in (
        caso1,
        caso2,
        caso3,
        caso4,
        caso5,
        caso6,
        caso7,
        caso8,
        caso9,
        sintesis_general,
        readme,
        investigacion_nb08,
        entorno,
        datos_de_entrada,
    ):
        hechos += generador()
    return hechos


def caso4() -> list[Hecho]:
    resumen = csv("qml_v2_portfolio_summary.csv").set_index(["configuracion", "metodo"])
    cruda = csv("qml_v2_portfolio_raw.csv")
    ref = json.loads((METRICS / "qml_v2_portfolio_reference.json").read_text(encoding="utf-8"))
    optimo = ref["optimo_exacto"]
    assert np.isclose(ref["coste_voraz"], optimo) and ref["activos_optimo"] == ["EC", "CIB", "MSFT"]
    assert (resumen["semillas"] == 10).all()
    alta_vqe, alta_qaoa = resumen.loc[("penalizacion_alta", "VQE")], resumen.loc[("penalizacion_alta", "QAOA")]
    baja_vqe, baja_qaoa = resumen.loc[("penalizacion_baja", "VQE")], resumen.loc[("penalizacion_baja", "QAOA")]
    assert baja_vqe["factibles"] == 0 and baja_qaoa["factibles"] == 0
    peor = cruda["coste"].max()
    sesiones = len(pd.read_csv(POCS / "data" / "output" / "input" / "portfolio_prices.csv"))
    aciertos = lambda fila: f"{int(fila['optimo_encontrado'])}/{int(fila['semillas'])}"  # noqa: E731
    en_texto = lambda fila: f"{int(fila['optimo_encontrado'])} de {int(fila['semillas'])}"  # noqa: E731
    azar = f(100 / math.comb(6, 3), 0)
    t_vqe = de_notebook("04", r"VQE\s+10\.0\s+\w+\s+True\s+[\d.]+\s+[\w, ]+?\s+300\s+([\d.]+)")
    t_qaoa = de_notebook("04", r"QAOA\s+10\.0\s+\w+\s+True\s+[\d.]+\s+[\w, ]+?\s+300\s+([\d.]+)")
    h = [
        Hecho(4, "Óptimo exacto y sesiones", (f(optimo, 4), str(sesiones)), en("eval", caso=4), "selecciona 3 de 6"),
        Hecho(
            4,
            "Fila VQE penalización 10",
            (
                aciertos(alta_vqe),
                f"{int(alta_vqe['factibles'])}/10",
                f(alta_vqe["brecha_media_pct"]),
                f(alta_vqe["brecha_mediana_pct"]),
                f(alta_vqe["brecha_max_pct"]),
                f(alta_vqe["coste_medio"], 4),
            ),
            en("eval", caso=4),
            r"VQE \(TwoLocal, 5 reinicios\)",
        ),
        Hecho(
            4,
            "Fila QAOA penalización 10",
            (
                aciertos(alta_qaoa),
                f"{int(alta_qaoa['factibles'])}/10",
                f(alta_qaoa["brecha_media_pct"]),
                f(alta_qaoa["brecha_max_pct"]),
                f(alta_qaoa["coste_medio"], 4),
            ),
            en("eval", caso=4),
            r"QAOA \(\$?p=3\$?, 5 reinicios\)",
        ),
        Hecho(4, "Filas con penalización 0.01", ("0/10",), en("eval", caso=4), r"VQE \(1 inicio\)"),
        Hecho(4, "Filas con penalización 0.01 (QAOA)", ("0/10",), en("eval", caso=4), r"QAOA \(\$?p=3\$?, 1 inicio\)"),
        Hecho(
            4,
            "Peor coste y óptimo (pie de tabla)",
            (f(peor, 4), f(optimo, 4)),
            en("eval", caso=4),
            "brechas porcentuales grandes",
        ),
        Hecho(
            4,
            "Aciertos del texto y azar",
            (en_texto(alta_vqe), en_texto(alta_qaoa), azar, "20 ejecuciones"),
            en("eval", caso=4),
            "La penalización determina",
        ),
        Hecho(
            4,
            "Mediana y máximo QAOA en texto",
            (f(alta_qaoa["brecha_max_pct"]),),
            en("eval", caso=4),
            "mediana de brecha",
        ),
        Hecho(4, "Diferencia con el clásico", ("60", "40"), en("eval", caso=4), "frecuencia de acierto"),
        Hecho(4, "Tiempos del Notebook 04 (300 evaluaciones)", (t_qaoa, t_vqe), en("eval", caso=4), "300 evaluaciones"),
        Hecho(4, "Planteamiento en desarrollo", ("0.5", "252", "20"), en("desa", caso=4), "Se aplicó VQE"),
        Hecho(
            4,
            "Protocolo en desarrollo",
            ("10", "5", "300", "200", "4096", "0.01"),
            en("desa", caso=4),
            "usa para ambos métodos",
        ),
        Hecho(
            4,
            "Síntesis CU4",
            ("40", f"{int(alta_qaoa['optimo_encontrado'])}/10", f"{int(alta_vqe['optimo_encontrado'])}/10"),
            en("eval"),
            r"^4\s+&",
        ),
        Hecho(4, "Resumen", (en_texto(alta_qaoa), en_texto(alta_vqe)), en("resumen"), "portafolio"),
        Hecho(
            4,
            "Abstract",
            (en_texto(alta_qaoa).replace(" de ", " of "), en_texto(alta_vqe).replace(" de ", " of ")),
            en("abstract"),
            "portfolio",
        ),
        Hecho(
            4,
            "Conclusiones",
            (f"{int(alta_qaoa['optimo_encontrado'])}", f"{int(alta_vqe['optimo_encontrado'])}", "0.01"),
            en("conc"),
            "portafolio",
        ),
        Hecho(4, "README", (aciertos(alta_qaoa), aciertos(alta_vqe)), en("readme"), "portfolio"),
    ]
    return h


def caso5() -> list[Hecho]:
    col = csv("qml_v2_tsp_colombia_summary.csv").set_index("model")
    nb = csv("05_tsp_resultados.csv").set_index("solver")
    eu = csv("qml_v2_tsp_optimization_summary.csv").set_index("model")
    exacto = col.loc["NN+2-opt", "cost_mean"]
    for solver in ("BruteForce (exacto)", "NN+2-opt", "QAOA-p1", "SA-QUBO"):
        assert np.isclose(nb.loc[solver, "longitud_km"], exacto), solver
    assert np.isclose(nb.loc["Nearest Neighbor", "longitud_km"], col.loc["NearestNeighbor", "cost_mean"])
    nn_km, nn_gap = f(col.loc["NearestNeighbor", "cost_mean"]), f(col.loc["NearestNeighbor", "gap_mean"], 1, 100)
    assert (col["feasible_rate"] == 1).all() and int(col.loc["QAOA-p1", "n_instances"]) == 1
    h = [
        Hecho(5, "Longitud óptima y NN (evaluación)", (f(exacto), nn_km, nn_gap), en("eval", caso=5), "Colombia"),
        Hecho(5, "Tabla de Colombia", (f(exacto), nn_km, nn_gap), en("desa", caso=5), r"NearestNeighbor|BruteForce"),
        Hecho(
            5,
            "Gap de SA-QUBO en réplicas Euclidean",
            (f(eu.loc["SA-QUBO", "gap_mean"], 2),),
            en("desa", "eval", caso=5),
            "Euclidean",
        ),
        Hecho(5, "QAOA y NN+2-opt con gap 0 en Euclidean", ("0",), en("eval", caso=5), "Euclidean"),
        Hecho(
            5,
            "NB05: 12 capitales (NN y NN+2-opt) y tours del exacto",
            (
                de_notebook("05", r"NN: ([\d.]+) km"),
                de_notebook("05", r"NN\+2-opt: ([\d.]+) km"),
                str(math.factorial(11)),
            ),
            en("desa", caso=5),
            "Sobre las 12 capitales",
        ),
        Hecho(
            5,
            "Memoria de la evolución densa de 16 qubits (GiB)",
            ("16", str(2**32 * 16 // 2**30)),
            en("desa", caso=5),
            "64",
        ),
        Hecho(5, "Resumen / abstract", (f(exacto),), en("resumen", "abstract"), "TSP"),
        Hecho(5, "Conclusiones", (f(exacto),), en("conc"), "Caso 5"),
    ]
    return h


# ----------------------------------------------------------------------------- auditoría de cobertura
PREFIJOS_CASO = {
    1: ("fair_classification", "01_", "06_"),
    2: ("03_", "qml_v2_regression", "regresion_"),
    3: ("02_", "qml_v2_optimization"),
    4: ("qml_v2_portfolio", "04_"),
    5: ("qml_v2_tsp", "05_"),
    6: ("06_",),
    7: ("10_mds",),
    8: ("fair_anomaly", "qml_v2_anomaly", "07_sintesis"),
    9: ("fair_timeseries", "qml_v2_timeseries", "07_sintesis"),
    0: ("",),
}

# Números de las secciones de resultados que no proceden de un CSV y por qué no se comprueban.
EXENCIONES: tuple[tuple[str, str | None, str], ...] = (
    (r"^(19|20)\d\d$", None, "años"),
    (r"^95$", r"IC|CI\b|intervalo|confianza|confidence", "nivel de confianza de un intervalo, no un resultado"),
    (
        r"^(70\.5|30\.3|1024)$",
        r"versión inicial|Tampoco se mantiene|versión previa|versiones anteriores",
        "cifras de versiones anteriores citadas únicamente para retirarlas (no reproducibles)",
    ),
    (r"^30$", r"70/30", "partición 70/30 (el caso 6 comprueba Train 105 / Test 45 en la salida del Notebook 06)"),
)

# Líneas que no contienen resultados: dibujos TikZ, definición de colores y referencias bibliográficas.
LINEAS_EXENTAS: tuple[tuple[str, str], ...] = (
    (r"\\(node|draw|path|fill|definecolor|foreach|coordinate)\b|rectangle|^\s*\(.*\)\s*--", "coordenadas TikZ"),
    (r"\(\d{4}\)\.\s+\\emph\{|arXiv:\d{4}\.\d{4,5}", "referencias bibliográficas del cierre de la defensa"),
)


def _valores_csv(prefijos: tuple[str, ...]) -> np.ndarray:
    valores: set[float] = set()

    def anadir(v: object) -> None:
        try:
            x = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if np.isfinite(x):
            for k in (x, 100 * x):
                valores.add(round(abs(k), 7))

    def recorrer(o: object) -> None:
        if isinstance(o, dict):
            for v in o.values():
                recorrer(v)
        elif isinstance(o, list):
            for v in o:
                recorrer(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            anadir(o)

    for ruta in METRICS.iterdir():
        if not ruta.name.startswith(prefijos):
            continue
        if ruta.suffix == ".csv":
            try:
                tabla = pd.read_csv(ruta)
            except Exception:
                continue
            for col in tabla.columns:
                if pd.api.types.is_numeric_dtype(tabla[col]):
                    for v in tabla[col].dropna().unique():
                        anadir(v)
        elif ruta.suffix == ".json":
            recorrer(json.loads(ruta.read_text(encoding="utf-8")))
    return np.array(sorted(valores))


def _en_pool(pool: np.ndarray, token: str) -> bool:
    decimales = len(token.split(".")[1]) if "." in token else 0
    x, tol = abs(float(token)), 0.5 * 10 ** (-decimales) + 1e-9
    i = np.searchsorted(pool, x - tol)
    return bool(i < len(pool) and pool[i] <= x + tol)


def _numeros(linea: str) -> list[str]:
    linea = re.sub(
        r"\\(cite[a-z]*|ref|label|eqref|nolinkurl|url|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", "", linea
    )
    linea = re.sub(r"\\texttt\{[^}]*\}", "", normalizar(linea))
    linea = re.sub(
        r"(Caso|Cap[ií]tulo|Tabla|Figura|Secci[oó]n|Ec\.|Notebook|NB|Anexo|Cuadro|CU|Fig\.|Bloque|Sec\.|§)\s*\d+(\.\d+)?",
        "",
        linea,
    )
    linea = re.sub(r"(?<![\w.])(n|p|N|k|d|reps|B|q|A|T|L)\s*[=≈]\s*\d+", "", linea)
    return re.findall(r"(?<![\w.\\])(\d+(?:\.\d+)?)(?![\w])", linea)


def auditar() -> int:
    cubiertos = {v for h in todos() for val in h.valores for v in re.findall(r"\d+(?:\.\d+)?", val)}
    pools: dict[int, np.ndarray] = {}
    sin_cubrir = 0
    for clave in (
        "eval",
        "desa",
        "conc",
        "resumen",
        "abstract",
        "defensa",
        "readme",
        "readme_pocs",
        "anexos",
        "intro",
        "contexto",
        "objetivos",
        "investigacion",
        "futuro",
    ):
        caso = 0
        for numero, linea in enumerate(FICHEROS[clave].read_text(encoding="utf-8").split("\n"), 1):
            m = re.search(r"\\section\{Caso de uso (\d)", linea)
            if m:
                caso = int(m.group(1))
            elif re.search(r"\\(section|chapter)\{", linea):
                caso = 0
            if linea.strip().startswith("%") or any(re.search(p, linea) for p, _ in LINEAS_EXENTAS):
                continue
            if clave not in CON_SECCIONES_POR_CASO:
                caso = 0
            if caso not in pools:
                pools[caso] = _valores_csv(PREFIJOS_CASO[caso])
            pool = pools[caso]
            malos = []
            for t in _numeros(linea):
                if t in cubiertos or any(
                    re.search(p, t) and (ambito is None or re.search(ambito, linea)) for p, ambito, _ in EXENCIONES
                ):
                    continue
                if t.isdigit() and int(t) <= 10:
                    continue
                if not _en_pool(pool, t):
                    malos.append(t)
            if malos:
                sin_cubrir += len(malos)
                print(f"{clave}:{numero} CU{caso} {malos} | {linea.strip()[:150]}")
    print(f"{sin_cubrir} números sin cubrir")
    return 1 if sin_cubrir else 0


# ----------------------------------------------------------------------------- ejecución
def verificar() -> int:
    hechos = todos()
    fallos = 0
    comprobaciones = 0
    for hecho in hechos:
        for destino in hecho.destinos:
            comprobaciones += 1
            if not comprobar(hecho, destino):
                fallos += 1
                donde = destino.fichero + (f" §Caso {destino.caso}" if destino.caso else "")
                print(
                    f"FALTA  CU{hecho.caso} {hecho.descripcion}: {list(hecho.valores)} "
                    f"(ancla «{hecho.ancla}») en {donde}"
                )
    print(f"{comprobaciones - fallos}/{comprobaciones} comprobaciones superadas en {len(hechos)} hechos")
    return 1 if fallos else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auditoria", action="store_true")
    args = parser.parse_args()
    return auditar() if args.auditoria else verificar()


if __name__ == "__main__":
    sys.exit(main())
