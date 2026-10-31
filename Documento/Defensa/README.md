# Defensa TFM (Beamer)

Presentación en LaTeX/Beamer alineada a la línea cronológica del documento.

## Dónde están las slides

```
Documento/Defensa/defensa.pdf   ← PDF listo para proyectar
Documento/Defensa/defensa.tex   ← fuente Beamer
```

Figuras: `POCs/data/output/figures/` y `Documento/Imagenes/`.

## Importante: rama correcta

Los cambios recientes de la defensa **no están en `main`** hasta mergear el PR.
Hay que usar:

```bash
git fetch origin
git checkout cursor/defensa-mas-resultados-c00b
git pull origin cursor/defensa-mas-resultados-c00b
```

Luego abrir `Documento/Defensa/defensa.pdf`.

PR: https://github.com/juandapradam12/UNIR-MasterQC-QuantumAdvantageInDataScience/pull/12

## Compilar

```bash
cd Documento/Defensa
tectonic defensa.tex
# o: pdflatex defensa.tex && pdflatex defensa.tex
```

## Secciones

1. Título
2. Motivación (+ diagrama Cap. 1–8)
3. Introducción (pregunta de investigación + alcance)
4. Contexto: era NISQ
5. Planteamiento y metodología (cuatro fases, modelos comparados, datasets utilizados)
6. Resultados Q vs C (mapa de errores, síntesis por caso de uso y conjunto de datos, criterio de activación)
7. Conjeturas: camino empírico → $\mathcal{Q}$ → QCR
8. Conclusiones, limitaciones y trabajo futuro (una diapositiva cada una) + referencias
