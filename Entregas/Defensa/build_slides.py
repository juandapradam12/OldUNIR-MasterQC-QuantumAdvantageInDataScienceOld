#!/usr/bin/env python3
"""TFM defense deck v2 — tighter visual craft, image-led, minimal copy."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from pptx.oxml import parse_xml

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "POCs" / "data" / "output" / "figures"
IMG = ROOT / "Documento" / "Imagenes"
OUT = Path(__file__).resolve().parent / "Defensa_TFM_QuantumAdvantage.pptx"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.55)

# Palette: charcoal + warm paper + single teal accent
INK = RGBColor(0x14, 0x18, 0x1C)
PAPER = RGBColor(0xFB, 0xFA, 0xF7)
PANEL = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0x6B, 0x73, 0x7A)
LINE = RGBColor(0xE6, 0xE1, 0xD8)
ACCENT = RGBColor(0x0B, 0x6E, 0x6A)
ACCENT_SOFT = RGBColor(0xD8, 0xEF, 0xED)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x0E, 0x12, 0x16)


def _font(run, size, bold=False, color=INK, name="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name


def _fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _send_back(slide, shape):
    spTree = slide.shapes._spTree
    sp = shape._element
    spTree.remove(sp)
    spTree.insert(2, sp)


def bg(slide, color=PAPER):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    _fill(sh, color)
    _send_back(slide, sh)


def rect(slide, left, top, width, height, color, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    return sh


def txt(slide, left, top, width, height, text, size=18, bold=False, color=INK,
        align=PP_ALIGN.LEFT, name="Calibri", anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    try:
        tf._txBody.bodyPr.set(qn("a:anchor"), "t" if anchor == MSO_ANCHOR.TOP else "ctr")
    except Exception:
        pass
    # multi-line support
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(4)
        if p.runs:
            _font(p.runs[0], size=size, bold=bold, color=color, name=name)
    return box


def footer(slide, n, total=18):
    rect(slide, 0, Inches(7.15), W, Inches(0.35), PAPER)
    txt(slide, MARGIN, Inches(7.18), Inches(8), Inches(0.28),
        "UNIR · Máster en Computación Cuántica · TFM",
        size=10, color=MUTED)
    txt(slide, Inches(10.5), Inches(7.18), Inches(2.3), Inches(0.28),
        f"{n} / {total}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def header(slide, kicker: str, title: str):
    """Compact header band with accent rule."""
    txt(slide, MARGIN, Inches(0.28), Inches(12), Inches(0.28),
        kicker.upper(), size=11, bold=True, color=ACCENT, name="Calibri")
    txt(slide, MARGIN, Inches(0.52), Inches(12.2), Inches(0.55),
        title, size=26, bold=True, color=INK, name="Calibri")
    rect(slide, MARGIN, Inches(1.15), Inches(1.4), Inches(0.05), ACCENT)


def fit_picture(slide, path: Path, left, top, box_w, box_h):
    """Place image centered inside a box, preserving aspect ratio."""
    if not path.exists():
        rect(slide, left, top, box_w, box_h, PANEL, line=LINE)
        txt(slide, left, top + box_h / 2, box_w, Inches(0.4),
            f"Missing: {path.name}", size=12, color=MUTED, align=PP_ALIGN.CENTER)
        return None
    # Add subtle panel behind
    rect(slide, left, top, box_w, box_h, PANEL, line=LINE)
    pic = slide.shapes.add_picture(str(path), left, top)
    # scale to fit
    avail_w, avail_h = int(box_w), int(box_h)
    pw, ph = pic.width, pic.height
    scale = min(avail_w / pw, avail_h / ph)
    new_w = int(pw * scale)
    new_h = int(ph * scale)
    pic.width = new_w
    pic.height = new_h
    pic.left = int(left + (avail_w - new_w) / 2)
    pic.top = int(top + (avail_h - new_h) / 2)
    return pic


def metric_chip(slide, left, top, width, height, value, label):
    rect(slide, left, top, width, height, PANEL, line=LINE)
    accent = rect(slide, left, top, Inches(0.08), height, ACCENT)
    txt(slide, left + Inches(0.25), top + Inches(0.25), width - Inches(0.4), Inches(0.55),
        value, size=22, bold=True, color=ACCENT)
    txt(slide, left + Inches(0.25), top + Inches(0.85), width - Inches(0.4), Inches(0.7),
        label, size=12, color=MUTED)


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]
    total = 18
    n = 0

    def slide():
        nonlocal n
        n += 1
        return prs.slides.add_slide(blank)

    # ═══════════════════════════════════════════════════════════
    # 1 · Cover
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s, DARK)
    # left accent panel
    rect(s, 0, 0, Inches(0.22), H, ACCENT)
    if (IMG / "logo_unir.png").exists():
        s.shapes.add_picture(str(IMG / "logo_unir.png"), Inches(0.7), Inches(0.55), height=Inches(0.55))
    txt(s, Inches(0.7), Inches(1.8), Inches(11.5), Inches(0.4),
        "TRABAJO DE FIN DE MÁSTER  ·  DEFENSA", size=12, bold=True, color=ACCENT)
    txt(s, Inches(0.7), Inches(2.3), Inches(11.8), Inches(1.8),
        "Ventaja cuántica en ciencia de datos:\ncondiciones empíricas, métodos híbridos\ny criterios de activación selectiva",
        size=30, bold=True, color=WHITE, name="Calibri")
    txt(s, Inches(0.7), Inches(5.0), Inches(11.5), Inches(0.8),
        "Juan David Prada Malagón\nDirector: Ricardo Rafael Domínguez Reyes  ·  UNIR  ·  Máster en Computación Cuántica",
        size=14, color=RGBColor(0xA8, 0xB0, 0xB8))
    # bottom keyline
    rect(s, Inches(0.7), Inches(6.6), Inches(3.2), Inches(0.06), ACCENT)
    txt(s, Inches(0.7), Inches(6.8), Inches(11), Inches(0.35),
        "9 casos de uso  ·  kernels ZZ  ·  QAOA/VQE  ·  score Q(D)  ·  framework QCR",
        size=12, color=RGBColor(0x8A, 0x93, 0x9B))

    # ═══════════════════════════════════════════════════════════
    # 2 · Motivation
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Contexto", "La brecha NISQ")
    # three columns
    cols = [
        ("Promesa", "Aceleración teórica en\naprendizaje y optimización"),
        ("Realidad NISQ", "Ruido, pocos qubits,\nsimulación costosa"),
        ("Hueco", "Falta evidencia justa\nen problemas de industria"),
    ]
    for i, (t, b) in enumerate(cols):
        left = MARGIN + i * Inches(4.15)
        rect(s, left, Inches(1.55), Inches(3.95), Inches(2.4), PANEL, line=LINE)
        rect(s, left, Inches(1.55), Inches(3.95), Inches(0.08), ACCENT)
        txt(s, left + Inches(0.3), Inches(1.9), Inches(3.35), Inches(0.45), t, size=16, bold=True, color=ACCENT)
        txt(s, left + Inches(0.3), Inches(2.5), Inches(3.35), Inches(1.2), b, size=15, color=INK)
    fit_picture(s, FIGS / "07_complejidad_cuantico_vs_clasico.png",
                MARGIN, Inches(4.2), Inches(12.2), Inches(2.7))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 3 · Thesis question
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Formulación de la tesis", "Pregunta de investigación")
    rect(s, MARGIN, Inches(1.7), Inches(12.2), Inches(3.2), PANEL, line=LINE)
    rect(s, MARGIN, Inches(1.7), Inches(0.12), Inches(3.2), ACCENT)
    txt(s, Inches(1.1), Inches(2.2), Inches(11), Inches(2.2),
        "¿En qué condiciones, y para qué tipos de problemas\nde ciencia de datos, los algoritmos cuánticos NISQ\nofrecen una ventaja real y medible frente a los clásicos?",
        size=24, bold=True, color=INK, align=PP_ALIGN.LEFT)
    txt(s, MARGIN, Inches(5.3), Inches(12.2), Inches(1.2),
        "Hipótesis operativa\n"
        "La ventaja es real pero condicionada: alineación feature map ↔ generador de datos,\n"
        "o problemas QUBO con arquitectura híbrida cuántico–clásica.",
        size=15, color=MUTED)
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 4 · Objectives as outcomes
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Objetivos", "Qué entrega este TFM")
    rows = [
        ("01", "Mapa empírico", "Comparativa justa en 9 casos de uso industriales"),
        ("02", "Patrón híbrido", "QAOA / VQE + optimizador clásico, criterios de activación"),
        ("03", "Score Q(D)", "Diagnóstico a priori de compatibilidad con kernels cuánticos"),
        ("04", "Framework QCR", "Espacio de representación — planteado formalmente"),
    ]
    for i, (num, title, desc) in enumerate(rows):
        top = Inches(1.5) + i * Inches(1.25)
        rect(s, MARGIN, top, Inches(12.2), Inches(1.1), PANEL, line=LINE)
        txt(s, Inches(0.85), top + Inches(0.28), Inches(1.0), Inches(0.55), num, size=22, bold=True, color=ACCENT)
        txt(s, Inches(2.1), top + Inches(0.22), Inches(4), Inches(0.4), title, size=18, bold=True, color=INK)
        txt(s, Inches(2.1), top + Inches(0.58), Inches(10), Inches(0.4), desc, size=14, color=MUTED)
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 5 · Methodology overview (dashboard)
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Metodología", "Nueve casos · tres capas de datos · protocolos justos")
    fit_picture(s, FIGS / "07_dashboard_ventaja_cuantica.png",
                MARGIN, Inches(1.4), Inches(12.2), Inches(5.5))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 6 · Experimental design
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Diseño experimental", "Familias de problema × evidencia visual")
    fit_picture(s, FIGS / "07_heatmap_ventaja_por_dominio.png",
                MARGIN, Inches(1.4), Inches(7.4), Inches(5.4))
    # side legend cards
    families = [
        ("Kernels", "QSVM · QK-SVR · OCSVM-ZZ"),
        ("Híbridos", "QAOA · VQE + COBYLA"),
        ("Datos", "Sintético nativo · SKG · público"),
        ("Stack", "Qiskit 2.x · statevector"),
    ]
    for i, (t, d) in enumerate(families):
        top = Inches(1.5) + i * Inches(1.25)
        rect(s, Inches(8.2), top, Inches(4.55), Inches(1.1), PANEL, line=LINE)
        txt(s, Inches(8.45), top + Inches(0.22), Inches(4.1), Inches(0.35), t, size=15, bold=True, color=ACCENT)
        txt(s, Inches(8.45), top + Inches(0.55), Inches(4.1), Inches(0.4), d, size=13, color=INK)
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 7 · Algorithms — kernels
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Algoritmos", "Kernels cuánticos (ZZFeatureMap)")
    fit_picture(s, FIGS / "06_zzfeaturemap_circuit.png",
                MARGIN, Inches(1.4), Inches(7.6), Inches(5.4))
    steps = [
        ("1", "Codificar", "ZZFeatureMap · 2 qubits · reps=2"),
        ("2", "Kernel", "κ = |⟨φ(xᵢ)|φ(xⱼ)⟩|²"),
        ("3", "Modelo", "QSVM / QK-SVR / OCSVM"),
        ("4", "Control", "RBF · mismo C · mismo PCA"),
    ]
    for i, (num, t, d) in enumerate(steps):
        top = Inches(1.5) + i * Inches(1.25)
        rect(s, Inches(8.4), top, Inches(4.35), Inches(1.1), PANEL, line=LINE)
        txt(s, Inches(8.6), top + Inches(0.28), Inches(0.5), Inches(0.45), num, size=20, bold=True, color=ACCENT)
        txt(s, Inches(9.2), top + Inches(0.2), Inches(3.3), Inches(0.35), t, size=15, bold=True, color=INK)
        txt(s, Inches(9.2), top + Inches(0.55), Inches(3.3), Inches(0.4), d, size=12, color=MUTED)
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 8 · Algorithms — hybrid
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Algoritmos", "Híbridos QAOA / VQE")
    fit_picture(s, FIGS / "10_qaoa_convergencia.png",
                MARGIN, Inches(1.4), Inches(6.0), Inches(5.4))
    fit_picture(s, FIGS / "10_mds_grafo_solucion.png",
                Inches(6.7), Inches(1.4), Inches(6.0), Inches(5.4))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 9 · Research problems
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Contribución original", "Dos problemas de investigación")
    # P1
    rect(s, MARGIN, Inches(1.55), Inches(6.0), Inches(5.2), PANEL, line=LINE)
    rect(s, MARGIN, Inches(1.55), Inches(6.0), Inches(0.7), ACCENT)
    txt(s, Inches(0.85), Inches(1.7), Inches(5.4), Inches(0.45), "P1  ·  Tests de cuanticidad", size=16, bold=True, color=WHITE)
    txt(s, Inches(0.85), Inches(2.7), Inches(5.4), Inches(2.5),
        "¿Existe un indicador\ncalculable a priori que\nprediga si un algoritmo\ncuántico mostrará ventaja?",
        size=20, bold=True, color=INK)
    txt(s, Inches(0.85), Inches(5.5), Inches(5.4), Inches(0.8),
        "Respuesta del TFM: score Q(D)\n(alineación · entropía · d_F)",
        size=14, color=MUTED)
    # P2
    rect(s, Inches(6.8), Inches(1.55), Inches(6.0), Inches(5.2), PANEL, line=LINE)
    rect(s, Inches(6.8), Inches(1.55), Inches(6.0), Inches(0.7), DARK)
    txt(s, Inches(7.1), Inches(1.7), Inches(5.4), Inches(0.45), "P2  ·  Espacio de representación", size=16, bold=True, color=WHITE)
    txt(s, Inches(7.1), Inches(2.7), Inches(5.4), Inches(2.5),
        "Si el dataset no tiene\nestructura cuántica, ¿puede\ntransformarse a un espacio\ndonde sí la tenga?",
        size=20, bold=True, color=INK)
    txt(s, Inches(7.1), Inches(5.5), Inches(5.4), Inches(0.8),
        "Respuesta del TFM: framework QCR\n(planteado formalmente · conjetura abierta)",
        size=14, color=MUTED)
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 10 · Formalization figure
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Formalización", "Diagnóstico Q(D) → decisión de activación / remapeo")
    fit_picture(s, FIGS / "09_qcr_framework_diagnostico.png",
                MARGIN, Inches(1.4), Inches(12.2), Inches(5.5))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 11 · Results — alignment
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Resultados", "Principio de alineación feature map ↔ generador")
    fit_picture(s, FIGS / "06_quantum_kernel_boundaries.png",
                MARGIN, Inches(1.4), Inches(6.0), Inches(5.4))
    fit_picture(s, FIGS / "06_learning_curves_quantum_advantage.png",
                Inches(6.7), Inches(1.4), Inches(6.0), Inches(5.4))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 12 · Results — classification + metrics
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Resultados · Clasificación", "QSVM vs clásicos")
    fit_picture(s, FIGS / "01_escalabilidad_ventaja_clasificacion.png",
                MARGIN, Inches(1.4), Inches(8.0), Inches(5.4))
    metric_chip(s, Inches(8.4), Inches(1.5), Inches(4.35), Inches(1.55),
                "+38.3 pp", "Ventaja QSVM en datos\nalineados al mapa")
    metric_chip(s, Inches(8.4), Inches(3.25), Inches(4.35), Inches(1.55),
                "+40.9 pp", "Media en escalabilidad\nn = 30–180")
    metric_chip(s, Inches(8.4), Inches(5.0), Inches(4.35), Inches(1.55),
                "Clásico", "Gana en Telco / SKG\nsin alineación")
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 13 · Results — hybrid optimization
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Resultados · Optimización híbrida", "QAOA / VQE")
    fit_picture(s, FIGS / "10_qaoa_histograma_medicion.png",
                MARGIN, Inches(1.4), Inches(8.0), Inches(5.4))
    metric_chip(s, Inches(8.4), Inches(1.5), Inches(4.35), Inches(1.55),
                "−30.3%", "Coste QAOA vs Greedy\nasignación de agentes")
    metric_chip(s, Inches(8.4), Inches(3.25), Inches(4.35), Inches(1.55),
                "Empate", "TSP n=4: QAOA = NN+2-opt\n(valor de modelado)")
    metric_chip(s, Inches(8.4), Inches(5.0), Inches(4.35), Inches(1.55),
                "~73%", "P(óptimo) MDS\npatrón híbrido")
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 14 · Results — anomalies
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Resultados · Anomalías one-class", "Ventaja solo si la novedad es de fase")
    fit_picture(s, IMG / "zz_phase_concept_diagram.png",
                MARGIN, Inches(1.4), Inches(6.0), Inches(5.4))
    fit_picture(s, IMG / "zz_phase_scores_if_vs_zz.png",
                Inches(6.7), Inches(1.4), Inches(6.0), Inches(5.4))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 15 · Results — Q(D)
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Resultados · Score Q(D)", "Correlación con ventaja de kernel")
    fit_picture(s, FIGS / "08_correlacion_indicadores_ventaja.png",
                MARGIN, Inches(1.4), Inches(6.0), Inches(5.4))
    fit_picture(s, FIGS / "08_barrido_sintetico_alpha.png",
                Inches(6.7), Inches(1.4), Inches(6.0), Inches(5.4))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 16 · Results — QCR mappings
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Resultados · QCR / mapeos", "Evidencia preliminar — no demostración fuerte")
    fit_picture(s, FIGS / "09_mapeos_comparativa.png",
                MARGIN, Inches(1.4), Inches(6.0), Inches(5.4))
    fit_picture(s, FIGS / "09_barrido_rotaciones.png",
                Inches(6.7), Inches(1.4), Inches(6.0), Inches(5.4))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 17 · Synthesis
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s)
    header(s, "Síntesis", "Ventaja real pero condicionada")
    fit_picture(s, FIGS / "07_radar_ventaja_cuantica.png",
                Inches(1.8), Inches(1.35), Inches(9.7), Inches(5.5))
    footer(s, n, total)

    # ═══════════════════════════════════════════════════════════
    # 18 · Close
    # ═══════════════════════════════════════════════════════════
    s = slide()
    bg(s, DARK)
    rect(s, 0, 0, Inches(0.22), H, ACCENT)
    txt(s, Inches(0.9), Inches(1.8), Inches(11.5), Inches(0.4),
        "CONCLUSIÓN", size=12, bold=True, color=ACCENT)
    txt(s, Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.8),
        "Diagnosticar compatibilidad.\nRemapear solo si procede.\nActivar cuántica de forma selectiva\ndentro de un stack híbrido.",
        size=28, bold=True, color=WHITE)
    txt(s, Inches(0.9), Inches(5.3), Inches(11.5), Inches(0.9),
        "Aportes: mapa empírico (9 casos)  ·  Q(D)  ·  QCR formalmente planteado  ·  arquitectura híbrida",
        size=14, color=RGBColor(0xA8, 0xB0, 0xB8))
    rect(s, Inches(0.9), Inches(6.4), Inches(2.2), Inches(0.06), ACCENT)
    txt(s, Inches(0.9), Inches(6.6), Inches(11.5), Inches(0.4),
        "Gracias  ·  ¿Preguntas?", size=18, bold=True, color=WHITE)

    assert n <= 20
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Wrote {OUT} ({n} slides, {OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
