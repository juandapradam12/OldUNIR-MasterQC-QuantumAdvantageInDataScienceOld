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

1. Título / agenda  
2. Motivación (+ diagrama Cap. 1–8)  
3. Introducción (problema + alcance)  
4. Contexto NISQ y estado de la técnica  
5. Metodología (fases + diseño 4×3; **sin** slides de formalismo sueltas)  
6. Resultados de modelos (**muchos plots**; se pueden recortar después)  
7. Conjeturas: camino empírico → $\mathcal{Q}$ → QCR  
8. Evidencia de las conjeturas  
9. Conclusiones (3) + limitaciones (2)  
10. Trabajo futuro (2) + referencias  
