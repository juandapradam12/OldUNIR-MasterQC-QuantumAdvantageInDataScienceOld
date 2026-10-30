"""Recolorea las figuras del TFM con la paleta de la plantilla de UNIR.

Solo cambia el color de los píxeles cromáticos (según su tono); no toca
geometría, texto, ejes ni datos. Los píxeles ya grises/negros/blancos se dejan
tal cual. Es idempotente: marca el PNG con un metadato ``paleta=unir`` y lo
omite si ya está procesado.

Paleta (azules y grises de la plantilla):
    azul UNIR   #219BC3   azul oscuro #0E4A75
    gris medio  #858591   gris carbón #4A4A57   gris claro #C0C0C8

Correspondencia por tono de la figura original:
    azules        -> azul UNIR
    violetas/rosa -> azul oscuro
    rojos         -> gris medio
    verdes        -> gris carbón
    naranja/amar. -> gris claro
    turquesas     -> azul oscuro

La saturación y el valor de cada píxel se conservan como tinte y sombra (los
tonos claros u oscuros de un mismo color siguen distinguiéndose y el suavizado
de bordes se mantiene).

Uso:
    python POCs/scripts/recolorear_figuras_unir.py [ruta ...]
Sin argumentos procesa las figuras de POCs/data/output/figures y
Documento/Imagenes/zz_phase_*.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

RAIZ = Path(__file__).resolve().parents[2]
MARCA = "paleta"
VALOR = "unir"

AZUL = (0x21, 0x9B, 0xC3)
NAVY = (0x0E, 0x4A, 0x75)
GRIS = (0x85, 0x85, 0x91)
CARBON = (0x4A, 0x4A, 0x57)
CLARO = (0xC0, 0xC0, 0xC8)

# (tono_min, tono_max, destino) con tono en grados [0, 360)
FAMILIAS = [
    (0, 14, GRIS),
    (14, 70, CLARO),
    (70, 170, CARBON),
    (170, 195, NAVY),
    (195, 255, AZUL),
    (255, 345, NAVY),
    (345, 360, GRIS),
]

UMBRAL_SATURACION = 0.03
SAT_PLENA = 0.60
VAL_PLENO = 0.75


def _rgb_a_hsv(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0.0)
    h = np.zeros_like(mx)
    dd = np.maximum(d, 1e-9)
    h = np.where(mx == r, ((g - b) / dd) % 6, h)
    h = np.where(mx == g, (b - r) / dd + 2, h)
    h = np.where(mx == b, (r - g) / dd + 4, h)
    h = np.where(d == 0, 0.0, h) * 60.0
    return h % 360.0, s, mx


def recolorear(img: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float64) / 255.0
    rgb, alfa = arr[..., :3], arr[..., 3:]
    h, s, v = _rgb_a_hsv(rgb)
    cromatico = (s > UMBRAL_SATURACION) & (v > 0.05)

    destino = np.zeros_like(rgb)
    for lo, hi, col in FAMILIAS:
        m = (h >= lo) & (h < hi)
        destino[m] = np.array(col, dtype=np.float64) / 255.0

    # Saturación -> cuánto color destino (frente a blanco); valor -> cuánto oscurecer.
    tinte = np.clip(s / SAT_PLENA, 0.0, 1.0)[..., None]
    oscuridad = np.clip(v / VAL_PLENO, 0.0, 1.0)[..., None]
    nuevo = (1.0 - tinte * (1.0 - destino)) * oscuridad
    peso = np.clip((s - UMBRAL_SATURACION) / 0.03, 0.0, 1.0)[..., None]
    mezcla = rgb * (1.0 - peso) + nuevo * peso
    salida = np.where(cromatico[..., None], mezcla, rgb)
    out = np.concatenate([np.clip(salida, 0, 1), alfa], axis=-1)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA")


def procesar(ruta: Path) -> bool:
    img = Image.open(ruta)
    if img.info.get(MARCA) == VALOR:
        return False
    nuevo = recolorear(img)
    if img.mode != "RGBA" and "A" not in img.getbands():
        nuevo = nuevo.convert("RGB")
    meta = PngImagePlugin.PngInfo()
    for k, val in img.info.items():
        if isinstance(val, str) and k != MARCA:
            meta.add_text(k, val)
    meta.add_text(MARCA, VALOR)
    nuevo.save(ruta, pnginfo=meta, optimize=True)
    return True


def main(argv: list[str]) -> None:
    if argv:
        rutas = [Path(a) for a in argv]
    else:
        rutas = sorted((RAIZ / "POCs/data/output/figures").glob("*.png"))
        rutas += sorted((RAIZ / "Documento/Imagenes").glob("zz_phase_*.png"))
    for r in rutas:
        print(("recoloreada " if procesar(r) else "omitida     ") + str(r.resolve().relative_to(RAIZ) if r.resolve().is_relative_to(RAIZ) else r))


if __name__ == "__main__":
    main(sys.argv[1:])
