# TFM - Computación Cuántica Aplicada

**Trabajo de Fin de Máster**  
**Universidad Internacional de La Rioja (UNIR)**  
**Máster en Inteligencia Artificial**

**Autor:** Juan David Prada Montoya  
**Director:** Ricardo Rafael Domínguez Reyes  
**Empresa:** SKG Tecnología  

---

## 📋 Descripción del Proyecto

Este TFM investiga la aplicabilidad práctica de algoritmos de computación cuántica en problemas empresariales reales, comparándolos con soluciones clásicas de Machine Learning en tres áreas:

1. **Clasificación Supervisada**: Quantum Kernels y VQC vs SVM, Random Forest
2. **Optimización Combinatoria**: QAOA vs heurísticas clásicas  
3. **Regresión**: Quantum Neural Networks vs modelos clásicos

---

## 🗂️ Estructura del Repositorio

```
TFM_ComputacionCuantica/
├── Documento/              # 📄 Documento LaTeX del TFM
│   ├── Capitulos/         # Capítulos del trabajo
│   ├── Estilos/           # Plantilla UNIR
│   └── main.tex           # Documento principal
│
├── POCs/                   # 💻 Pruebas de Concepto (código)
│   ├── src/               # Código fuente modular
│   ├── scripts/           # Scripts ejecutables CLI
│   ├── notebooks/         # Análisis interactivos
│   ├── data/              # Datasets y resultados
│   ├── utils/             # Utilidades compartidas
│   └── README.md          # 📖 Documentación técnica detallada
│
├── Entregas/              # 📦 Entregas parciales del TFM
│   ├── Entrega1/         # Primera entrega (contexto + objetivos)
│   ├── Entrega2/         # Segunda entrega (desarrollo)
│   └── Entrega3/         # Tercera entrega (evaluación + conclusiones)
│
└── Revisiones/            # ✍️ Feedback del tutor
│   ├── Estilos/
│   │   └── unir.sty              # Hoja de estilos UNIR
│   └── Imagenes/                 # Figuras e imágenes del documento
├── .vscode/
│   ├── settings.json             # Configuración LaTeX Workshop
│   └── tasks.json                # Tarea de compilación (Cmd+Shift+B)
├── programacion.pdf              # Calendario y entregas del TFM
├── Solicitud TFM.docx            # Propuesta original del TFM
└── .gitignore
```

---

## 🛠️ Compilación

El proyecto usa **LaTeX** (BasicTeX 2026) con `pdflatex`.

### Desde VS Code
Pulsa `Cmd+Shift+B` — compila dos veces y abre el PDF en Preview automáticamente.

### Desde terminal
```bash
cd Documento
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # segunda pasada para referencias
open -a Preview main.pdf
```

> **Requisito:** tener BasicTeX instalado (`brew install --cask basictex`) con el PATH `/usr/local/texlive/2026basic/bin/universal-darwin` en `~/.zshrc`.

---

## 📅 Estado de entregas

| Entrega | Contenido | Estado |
|---------|-----------|--------|
| Entrega 1 (Sem. 6) | Resumen, Abstract, Cap. 1–4 | ✅ Completa |
| Entrega 2 (Sem. 11) | Cap. 5–6 + borrador intermedio | 🔄 En progreso |
| Entrega 3 (Sem. 15) | Cap. 7–8 + artículo de investigación | ⏳ Pendiente |

---

## 📚 Tecnologías y frameworks (previstos)

- [Qiskit](https://qiskit.org/) / [PennyLane](https://pennylane.ai/) — implementación de algoritmos cuánticos
- [scikit-learn](https://scikit-learn.org/) — algoritmos clásicos de referencia
- Python 3.x — entorno de desarrollo y experimentación
