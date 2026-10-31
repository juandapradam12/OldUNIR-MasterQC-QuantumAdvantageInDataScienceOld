# Autoevaluación frente a la rúbrica (`rubrica.pdf`)

Estado según lo que se puede comprobar en el repositorio. No es una nota: es una lista de evidencias y de lo que falta.
Pesos: criterios específicos 60 %, generales 20 %, exposición 20 %. La exposición exige una valoración mínima o el TFM queda suspenso.

| Criterio (peso) | Evidencia comprobada | Brecha / acción | Estado |
|---|---|---|---|
| Temática y originalidad (10 %) | Marco cuántico aplicado a ciencia de datos de telecom/movilidad; aportación propia: perfil `Q(D)` y framework QCR | Citar explícitamente en `introduccion.tex` las asignaturas del máster con las que se relaciona (la rúbrica pide "más de dos") | Pendiente (necesita el listado de asignaturas del alumno) |
| Estructura, estilo y bibliografía (10 %) | 136 páginas (>50). Las 37 citas del texto resuelven en la lista. Lista ahora en **orden alfabético** (APA) y sin las 2 entradas no citadas (`carlsson2009`, `edelsbrunner2010`) | Revisión ortográfica completa del texto (no automatizada aquí); 32 avisos `Overfull hbox` en LaTeX; decidir APA frente a IEEE y mantenerlo | Parcial |
| Alcance (5 %) | Nueve casos de uso, cuatro familias en el análisis de estructura, tres datasets por familia | — | Cubierto |
| Marco teórico y referencias (10 %) | Capítulo de estado de la técnica; Huang, Liu, Thanasilp, Cortes en la sección de indicadores; figura de Huang (CC BY 4.0) con atribución | Las referencias cubren texto e imágenes: la figura de Huang está citada en el pie | Cubierto |
| Desarrollo de la contribución (30 %) | Notebook 11 reproducible, protocolo único, control nulo, `g` de Huang, paisaje QUBO, correcciones documentadas | Declarar en el texto que la medición de optimización es del paisaje, no de QAOA (hecho); mantener la lista de limitaciones | Fuerte, con límites declarados |
| Objetivos, planteamiento, desarrollo y conclusiones (15 %) | Se añadió el objetivo específico 8 (perfil `Q(D)` + QCR) y la tabla de trazabilidad objetivos → conclusiones (`conclusiones.tex`) | Revisar que la numeración de la tabla coincide con la lista si se reordenan conclusiones | Mejorado |
| Presentación (10 %) | `defensa.pdf` de 30 diapositivas (resultados resumidos en un mapa de errores y una síntesis; conclusiones, limitaciones y trabajo futuro condensados en una diapositiva cada una) | **Ajuste de tiempo**: la rúbrica penaliza no ajustarse al tiempo y premia la síntesis. Falta saber el tiempo asignado para fijar una ruta principal y mover el resto a anexo | Pendiente (necesita el tiempo asignado) |
| Dominio del contenido (10 %) | Sección de limitaciones y "criterio de integridad" en las slides | Preparar respuestas a las preguntas esperables (ver abajo); no leer | Depende del alumno |

## Preguntas que el tribunal puede hacer (y dónde está la respuesta)
1. ¿Por qué solo ΔA discrimina? Solo ΔA usa `y`; ΔH, d_F y g solo ven `X` (notebook 11, §7).
2. ¿Qué dice `g`? Con la definición de Huang (norma espectral, λ=1e-3) vale ≈76–85 en casi todos los casos: no discrimina (§7.1).
3. ¿Qué mediste en optimización? El paisaje del QUBO, no QAOA; con 2 qubits y 9–10 variables, PCA(2) retiene ≈1/3 de la varianza.
4. ¿Por qué Q no decide? Lo dominan indicadores planos y el escalar ordenó mal frente a Δ_op (contraejemplo SKG).
5. ¿Qué corregiste? Submuestreo de anomalías-SKG (ΔA=0 espurio) y `g` (≈3000 por mal condicionamiento).
