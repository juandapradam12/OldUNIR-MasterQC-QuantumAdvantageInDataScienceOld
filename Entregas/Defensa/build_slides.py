#!/usr/bin/env python3
"""Build TFM defense deck: ≤20 slides, image-heavy, minimal text."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "POCs" / "data" / "output" / "figures"
IMG = ROOT / "Documento" / "Imagenes"
OUT = Path(__file__).resolve().parent / "Defensa_TFM_QuantumAdvantage.pptx"

# Widescreen 16:9
W, H = Inches(13.333), Inches(7.5)

# Visual direction: ink + paper (academic), teal accent — not purple/glow
INK = RGBColor(0x1A, 0x23, 0x2E)
PAPER = RGBColor(0xF7, 0xF5, 0xF0)
MUTED = RGBColor(0x5C, 0x6B, 0x73)
ACCENT = RGBColor(0x0E, 0x7C, 0x7B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xD4, 0xCF, 0xC4)


def _set_run(run, size=18, bold=False, color=INK, font="Georgia"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def add_bg(slide, color=PAPER):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    # send to back
    spTree = slide.shapes._spTree
    sp = shape._element
    spTree.remove(sp)
    spTree.insert(2, sp)


def add_accent_bar(slide, y=0):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, y, W, Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()


def add_footer(slide, n, total=18):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12), Inches(0.3))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = f"TFM · UNIR · Computación Cuántica  ·  {n}/{total}"
    p.alignment = PP_ALIGN.RIGHT
    _set_run(p.runs[0], size=10, color=MUTED, font="Calibri")


def textbox(slide, left, top, width, height, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, font="Georgia"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    _set_run(p.runs[0], size=size, bold=bold, color=color, font=font)
    return box


def bullets(slide, left, top, width, height, items, size=16, color=INK):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        label = item if not item else ("•  " + item if item.strip() else "")
        p.text = label
        if p.runs:
            _set_run(p.runs[0], size=size, color=color, font="Calibri")
    return box


def picture(slide, path: Path, left, top, width=None, height=None):
    if not path.exists():
        textbox(slide, left, top, width or Inches(6), height or Inches(3), f"[missing] {path.name}", size=12, color=MUTED)
        return None
    kwargs = {"image_file": str(path), "left": left, "top": top}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    return slide.shapes.add_picture(**kwargs)


def section_title(slide, title, subtitle=None):
    add_bg(slide)
    add_accent_bar(slide)
    textbox(slide, Inches(0.7), Inches(0.35), Inches(12), Inches(0.55), title, size=28, bold=True, color=INK)
    if subtitle:
        textbox(slide, Inches(0.7), Inches(0.95), Inches(12), Inches(0.4), subtitle, size=14, color=MUTED, font="Calibri")


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]
    total = 18
    n = 0

    def new():
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        return s

    # ── 1 Portada ──────────────────────────────────────────────
    s = new()
    add_bg(s, INK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.9), W, Inches(1.6))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    logo = IMG / "logo_unir.png"
    if logo.exists():
        picture(s, logo, Inches(0.7), Inches(0.5), height=Inches(0.7))
    textbox(s, Inches(0.7), Inches(2.0), Inches(12), Inches(1.2),
            "Análisis de la ventaja cuántica\nen casos aplicados de ciencia de datos",
            size=32, bold=True, color=WHITE)
    textbox(s, Inches(0.7), Inches(4.0), Inches(12), Inches(0.8),
            "Estudio empírico · Kernels cuánticos · Algoritmos híbridos · Score de cuanticidad · Framework QCR",
            size=14, color=RGBColor(0xB8, 0xC4, 0xCC), font="Calibri")
    textbox(s, Inches(0.7), Inches(6.15), Inches(12), Inches(1.0),
            "Juan David Prada Malagón  ·  Dir. Ricardo Rafael Domínguez Reyes\n"
            "Máster en Computación Cuántica · UNIR · Defensa TFM",
            size=14, color=WHITE, font="Calibri")

    # ── 2 Motivación ───────────────────────────────────────────
    s = new()
    section_title(s, "Motivación", "Brecha entre promesas teóricas y evidencia reproducible (era NISQ)")
    bullets(s, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.5), [
        "Metodologías cuánticas maduras (QAOA, VQE, kernels)",
        "Hardware NISQ: ruido, qubits limitados",
        "Pocos estudios comparativos justos sobre problemas de industria",
        "Necesidad: condiciones medibles de ventaja — no hype",
    ], size=18)
    picture(s, FIGS / "07_complejidad_cuantico_vs_clasico.png", Inches(6.5), Inches(1.5), width=Inches(6.3))
    add_footer(s, n, total)

    # ── 3 Formulación de la tesis ──────────────────────────────
    s = new()
    section_title(s, "Formulación de la tesis", "Pregunta central")
    # quote card
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(1.8), Inches(12), Inches(2.4))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = RULE
    textbox(s, Inches(1.0), Inches(2.1), Inches(11.4), Inches(2.0),
            "¿En qué condiciones, y para qué tipos de problemas de ciencia de datos,\n"
            "los algoritmos cuánticos NISQ ofrecen una ventaja real y medible\n"
            "frente a los algoritmos clásicos?",
            size=22, bold=True, color=INK, align=PP_ALIGN.CENTER)
    textbox(s, Inches(0.7), Inches(4.6), Inches(12), Inches(1.5),
            "Hipótesis de trabajo: la ventaja es real pero condicionada —\n"
            "alineación feature map ↔ generador de datos, o problemas QUBO con diseño híbrido.",
            size=16, color=MUTED, align=PP_ALIGN.CENTER, font="Calibri")
    add_footer(s, n, total)

    # ── 4 Objetivos ────────────────────────────────────────────
    s = new()
    section_title(s, "Objetivos", "General y específicos (síntesis visual)")
    items = [
        ("01", "Comparar cuántico vs clásico\nen 9 casos de uso"),
        ("02", "Caracterizar el patrón\nhíbrido QAOA / VQE"),
        ("03", "Proponer score de\ncuanticidad Q(D)"),
        ("04", "Plantear framework QCR\n(espacio de representación)"),
    ]
    for i, (num, label) in enumerate(items):
        left = Inches(0.6 + i * 3.15)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(2.0), Inches(2.95), Inches(3.2))
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = RULE
        textbox(s, left + Inches(0.15), Inches(2.25), Inches(2.65), Inches(0.6), num, size=28, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
        textbox(s, left + Inches(0.15), Inches(3.1), Inches(2.65), Inches(1.8), label, size=15, color=INK, align=PP_ALIGN.CENTER, font="Calibri")
    add_footer(s, n, total)

    # ── 5 Metodología ──────────────────────────────────────────
    s = new()
    section_title(s, "Metodología", "Nueve casos · tres capas de datos · protocolos justos")
    picture(s, FIGS / "07_dashboard_ventaja_cuantica.png", Inches(0.4), Inches(1.35), width=Inches(12.5))
    add_footer(s, n, total)

    # ── 6 Diseño experimental ──────────────────────────────────
    s = new()
    section_title(s, "Diseño experimental", "Familias de problema × capas de datos")
    bullets(s, Inches(0.7), Inches(1.5), Inches(5.8), Inches(5), [
        "Clasificación — QSVM (kernel ZZ)",
        "Regresión — QK-SVR",
        "Optimización — QAOA / VQE (híbrido + COBYLA)",
        "Anomalías one-class — OCSVM-ZZ",
        "Forecasting por lags — QK-SVR",
        "",
        "Datos: sintéticos cuántico-nativos · SKG · públicos",
        "Stack: Qiskit 2.x · simulación statevector · baselines scikit-learn",
    ], size=16)
    picture(s, FIGS / "07_heatmap_ventaja_por_dominio.png", Inches(6.4), Inches(1.45), width=Inches(6.4))
    add_footer(s, n, total)

    # ── 7 Algoritmos: kernels ──────────────────────────────────
    s = new()
    section_title(s, "Algoritmos — kernels cuánticos", "ZZFeatureMap → fidelidad → SVM / SVR / OCSVM")
    picture(s, FIGS / "06_zzfeaturemap_circuit.png", Inches(0.5), Inches(1.4), width=Inches(6.2))
    bullets(s, Inches(7.0), Inches(1.6), Inches(5.5), Inches(4.5), [
        "Codificación: ZZFeatureMap (2 qubits, reps=2)",
        "Kernel: κ(xᵢ,xⱼ)=|⟨φ(xᵢ)|φ(xⱼ)⟩|²",
        "Modelos: QSVM · QK-SVR · OCSVM-ZZ",
        "Control: SVM/SVR-RBF, mismos C / PCA",
        "Principio: ventaja si mapa ↔ generador",
    ], size=16)
    add_footer(s, n, total)

    # ── 8 Algoritmos: híbridos ─────────────────────────────────
    s = new()
    section_title(s, "Algoritmos — híbridos QAOA / VQE", "Módulo cuántico + optimizador clásico (COBYLA)")
    picture(s, FIGS / "10_qaoa_convergencia.png", Inches(0.4), Inches(1.4), width=Inches(6.3))
    picture(s, FIGS / "10_mds_grafo_solucion.png", Inches(6.9), Inches(1.4), width=Inches(5.9))
    textbox(s, Inches(0.7), Inches(6.35), Inches(12), Inches(0.45),
            "QUBO → Hamiltoniano → ansatz variacional · Casos 3, 4, 5 (TSP) y 7 (conjunto dominante)",
            size=13, color=MUTED, font="Calibri", align=PP_ALIGN.CENTER)
    add_footer(s, n, total)

    # ── 9 Problemas de investigación ───────────────────────────
    s = new()
    section_title(s, "Problemas de investigación planteados", "Contribución original del TFM")
    cards = [
        ("P1 — Tests de cuanticidad",
         "¿Existe un indicador calculable a priori\nque prediga si un algoritmo cuántico\nmostrará ventaja?"),
        ("P2 — Espacio de representación",
         "Si el dataset no tiene estructura cuántica,\n¿puede transformarse a un espacio donde\nsí la tenga, sin perder señal?"),
    ]
    for i, (title, body) in enumerate(cards):
        left = Inches(0.6 + i * 6.3)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.7), Inches(6.0), Inches(4.0))
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = RULE
        top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.7), Inches(6.0), Inches(0.7))
        top.fill.solid()
        top.fill.fore_color.rgb = ACCENT
        top.line.fill.background()
        textbox(s, left + Inches(0.2), Inches(1.82), Inches(5.6), Inches(0.5), title, size=16, bold=True, color=WHITE, font="Calibri")
        textbox(s, left + Inches(0.35), Inches(2.8), Inches(5.3), Inches(2.5), body, size=18, color=INK, align=PP_ALIGN.CENTER)
    add_footer(s, n, total)

    # ── 10 Formalización Q(D) y QCR ────────────────────────────
    s = new()
    section_title(s, "Formalización", "Score Q(D) · Framework QCR (planteado formalmente)")
    picture(s, FIGS / "09_qcr_framework_diagnostico.png", Inches(0.4), Inches(1.35), width=Inches(7.2))
    bullets(s, Inches(7.8), Inches(1.6), Inches(4.8), Inches(4.5), [
        "Q(D): alineación, entropía espectral, d_F",
        "Diagnóstico antes de entrenar",
        "QCR: buscar ψ que maximice Q(T(D))",
        "Conjetura abierta — evidencia preliminar",
        "No se afirma demostración fuerte",
    ], size=15)
    add_footer(s, n, total)

    # ── 11 Resultados: alineación ──────────────────────────────
    s = new()
    section_title(s, "Resultados — principio de alineación", "El feature map alineado gana; ZZ es el mapa medido aquí")
    picture(s, FIGS / "06_quantum_kernel_boundaries.png", Inches(0.3), Inches(1.3), width=Inches(6.4))
    picture(s, FIGS / "06_learning_curves_quantum_advantage.png", Inches(6.8), Inches(1.3), width=Inches(6.1))
    add_footer(s, n, total)

    # ── 12 Resultados: clasificación ───────────────────────────
    s = new()
    section_title(s, "Resultados — clasificación", "+38.3 pp en datos alineados · clásico en datos reales sin alineación")
    picture(s, FIGS / "01_escalabilidad_ventaja_clasificacion.png", Inches(0.3), Inches(1.3), width=Inches(6.4))
    picture(s, FIGS / "01_qsvm_vs_svm_directo.png", Inches(6.8), Inches(1.5), width=Inches(6.1))
    add_footer(s, n, total)

    # ── 13 Resultados: optimización híbrida ────────────────────
    s = new()
    section_title(s, "Resultados — optimización híbrida", "QAOA −30.3% coste vs Greedy · TSP empate óptimo (modelado)")
    picture(s, FIGS / "10_qaoa_histograma_medicion.png", Inches(0.4), Inches(1.35), width=Inches(6.2))
    picture(s, FIGS / "07_kernel_advantage_detalle.png", Inches(6.8), Inches(1.35), width=Inches(6.0))
    add_footer(s, n, total)

    # ── 14 Resultados: anomalías ───────────────────────────────
    s = new()
    section_title(s, "Resultados — anomalías one-class", "Ventaja solo si la novedad es de fase (zz_phase)")
    picture(s, IMG / "zz_phase_concept_diagram.png", Inches(0.3), Inches(1.35), width=Inches(6.3))
    picture(s, IMG / "zz_phase_scores_if_vs_zz.png", Inches(6.7), Inches(1.5), width=Inches(6.2))
    add_footer(s, n, total)

    # ── 15 Resultados: Q(D) ────────────────────────────────────
    s = new()
    section_title(s, "Resultados — score de cuanticidad Q(D)", "ρ_S = 0.833 (p = 0.005, n = 9) en barrido sintético")
    picture(s, FIGS / "08_correlacion_indicadores_ventaja.png", Inches(0.3), Inches(1.3), width=Inches(6.4))
    picture(s, FIGS / "08_barrido_sintetico_alpha.png", Inches(6.8), Inches(1.3), width=Inches(6.1))
    add_footer(s, n, total)

    # ── 16 Resultados: QCR / mapeos ────────────────────────────
    s = new()
    section_title(s, "Resultados — mapeos y QCR", "Evidencia preliminar · validación fuerte = post-TFM")
    picture(s, FIGS / "09_mapeos_comparativa.png", Inches(0.3), Inches(1.3), width=Inches(6.4))
    picture(s, FIGS / "09_barrido_rotaciones.png", Inches(6.8), Inches(1.3), width=Inches(6.1))
    add_footer(s, n, total)

    # ── 17 Síntesis ────────────────────────────────────────────
    s = new()
    section_title(s, "Síntesis", "Ventaja real pero condicionada · adopción selectiva e híbrida")
    picture(s, FIGS / "07_radar_ventaja_cuantica.png", Inches(1.5), Inches(1.25), width=Inches(10.2))
    add_footer(s, n, total)

    # ── 18 Cierre ──────────────────────────────────────────────
    s = new()
    add_bg(s, INK)
    add_accent_bar(s, Inches(0))
    textbox(s, Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.2),
            "La ventaja cuántica es real pero condicionada.\n"
            "Diagnosticar · remapear si procede · activar de forma selectiva.",
            size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    textbox(s, Inches(0.8), Inches(4.4), Inches(11.7), Inches(1.2),
            "Aportes: mapa empírico (9 casos) · Q(D) · QCR formalmente planteado · arquitectura híbrida\n\n¿Preguntas?",
            size=16, color=RGBColor(0xB8, 0xC4, 0xCC), align=PP_ALIGN.CENTER, font="Calibri")
    textbox(s, Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.5),
            "Gracias",
            size=20, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    assert n <= 20, n
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Wrote {OUT} ({n} slides)")


if __name__ == "__main__":
    build()
