# Defensa TFM (Beamer)

Presentación en LaTeX/Beamer alineada a la línea cronológica del documento.

## Compilar

```bash
cd Documento/Defensa
tectonic -X compile defensa.tex
# o: pdflatex defensa.tex && pdflatex defensa.tex
```

Salida: `defensa.pdf`

## Secciones

1. Título / agenda  
2. Motivación (+ diagrama Cap. 1–8)  
3. Introducción (problema + alcance 4 familias × 3 datasets)  
4. Contexto NISQ y estado de la técnica  
5. Metodología (4 fases, formalismos kernel / QAOA)  
6. Resultados de modelos  
7. Conjeturas $\mathcal{Q}(\mathcal{D})$ y QCR  
8. Análisis empírico de las conjeturas  
9. Conclusiones y limitaciones  
10. Trabajo futuro + referencias  

Figuras: `POCs/data/output/figures/` y `Documento/Imagenes/`.
