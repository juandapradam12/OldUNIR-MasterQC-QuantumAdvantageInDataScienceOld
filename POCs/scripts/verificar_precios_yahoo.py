"""Compara los precios guardados del Notebook 04 con una descarga nueva de Yahoo Finance.

Yahoo reajusta los cierres históricos cuando paga dividendos, por lo que los niveles de precio pueden
cambiar entre descargas; lo que importa para el problema son los log-retornos. El script exige que los
log-retornos diarios de la descarga nueva coincidan con los guardados (tolerancia absoluta 1e-5).

Uso, desde ``POCs/``::

    python scripts/verificar_precios_yahoo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

POCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POCS))
from qml_v2.portfolio import PRICES, TICKERS  # noqa: E402


def main() -> int:
    import yfinance as yf

    warnings.filterwarnings("ignore")
    guardado = pd.read_csv(PRICES, index_col=0, parse_dates=True)[TICKERS]
    bruto = yf.download(TICKERS, start="2022-01-01", end="2024-01-01", auto_adjust=True, progress=False)
    nuevo = bruto["Close"].dropna()[TICKERS]
    print(f"Guardado: {len(guardado)} sesiones ({guardado.index[0].date()} a {guardado.index[-1].date()})")
    print(f"Descarga: {len(nuevo)} sesiones ({nuevo.index[0].date()} a {nuevo.index[-1].date()})")
    if not guardado.index.equals(nuevo.index):
        print("FALLO: las fechas no coinciden")
        return 1
    factor = (nuevo / guardado).iloc[[0, -1]].round(6)
    print("Cociente descarga/guardado (primera y última sesión):")
    print(factor.to_string())
    r_guardado = np.log(guardado / guardado.shift(1)).dropna()
    r_nuevo = np.log(nuevo / nuevo.shift(1)).dropna()
    diferencia = float((r_guardado - r_nuevo).abs().max().max())
    print(f"Máxima diferencia absoluta en log-retornos: {diferencia:.2e}")
    return 0 if diferencia < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
