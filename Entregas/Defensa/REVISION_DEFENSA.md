# Revisión rápida del documento (pre-defensa)

## Coherencia general — OK

| Eje | Estado |
|---|---|
| Pregunta central | Clara y repetida (intro → evaluación → conclusiones) |
| Mensaje de alineación feature map ↔ generador | Unificado; ZZ como mapa validado, no exclusivo |
| 9 casos | Consistentes en resumen, síntesis, semáforo, dashboard |
| Q(D) | Empírico; ρ_S = 0.833 en barrido sintético |
| QCR | Correctamente acotado como *planteado formalmente* |
| Híbridos | Casos 3–5 y 7; prototipo QAOA–COBYLA |

## Algoritmos — puntos a verbalizar

1. **Kernels:** `ZZFeatureMap` (2q, reps=2) → fidelidad → QSVM / QK-SVR / OCSVM-ZZ; control RBF con mismo `SVC`/`C`/PCA.
2. **Híbridos:** problema → QUBO → Hamiltoniano → QAOA/VQE + COBYLA; clásico pre/post-procesa.
3. **TSP:** protocolo justo; empate óptimo = valor de *modelado*, no speedup.
4. **Anomalías:** one-class (no fraude); ventaja solo en `zz_phase`.
5. **Pauli-Y:** tabla cruzada confirma que la ventaja sigue al emparejamiento mapa–generador.

## Limitaciones que debes decir tú (antes de que te las pregunten)

- Simulación statevector (no ruido NISQ real).
- PCA a 2 componentes (~68% varianza descartada).
- QCR sin demostración fuerte.
- Correlación Q(D) en datos reales con n pequeño (orientativa); el barrido sintético es el ancla estadística.

## Detalles menores (no bloquean defensa)

- README del repo habla a veces de “MSc in AI”; el `main.tex` dice **Máster en Computación Cuántica** — usar este en slides.
- Caption histórico del dashboard ya actualizado a 9 casos en el paquete mergeado.
- Em-dashes (—) pueden avisar en LaTeX; cosmético.
