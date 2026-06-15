# Computación Cuántica aplicada a Machine Learning
## Solicitud de Datasets - Resumen del Proyecto

**Estudiante:** Juan David Prada Malagón  
**Posición:** Científico de Datos Sr.  
**Institución:** UNIR - Máster en Investigación en Ingeniería de Software  
**Empresa:** SKG Tecnología  
**Fecha:** Mayo 2026

---

## Sobre SKG Tecnología

SKG Tecnología es una empresa colombiana especializada en soluciones de Smart Cities. Trabajamos con entidades públicas y privadas en áreas de movilidad urbana, seguridad, gestión de recursos naturales y servicios de ciudad inteligente. Entre nuestros clientes se encuentran la Secretaría de Movilidad de Bogotá, la Policía Nacional, la CAR y PESCAR, entre otros.

Nuestros proyectos generan datos operacionales ricos y complejos que representan una oportunidad única para investigar la aplicabilidad de algoritmos de Machine Learning cuántico en problemas de negocio reales.

---

## Objetivo del Proyecto

Este TFM investiga si los algoritmos cuánticos ofrecen ventajas reales frente a los algoritmos clásicos de Machine Learning en problemas representativos del negocio de SKG.

Se compararán modelos cuánticos y clásicos en términos de rendimiento predictivo, tiempo de entrenamiento y escalabilidad, utilizando datos reales de los proyectos de la empresa.

---

## Casos de Uso

### Caso 1: Regresión — Predicción de Velocidades en Bogotá

**Contexto:** SKG gestiona datos de movilidad urbana para la Secretaría de Movilidad de Bogotá. Uno de los retos centrales es predecir la velocidad promedio en segmentos viales en función de variables históricas, temporales y contextuales.

**Objetivo:** Predecir la velocidad (km/h) en un segmento vial dado un conjunto de condiciones.

**Modelos a comparar:**
- Cuánticos: Variational Quantum Regressor (VQR), Quantum Kernel Regression
- Clásicos: SVR, Random Forest Regressor, XGBoost, Regresión Lineal

**Dataset requerido:**
- Tamaño: 5.000-50.000 registros históricos
- Variables: segmento vial, hora, día de semana, mes, clima, eventos especiales, velocidad promedio histórica
- Target: velocidad promedio en km/h (variable numérica continua)

---

### Caso 2: Optimización — Asignación de Agentes a Focos de Seguridad

**Contexto:** SKG desarrolla soluciones para la Policía Nacional orientadas a optimizar el despliegue de recursos en campo. El problema consiste en asignar el número disponible de agentes a los focos de seguridad más críticos de la ciudad, maximizando la cobertura y el impacto.

**Objetivo:** Encontrar la asignación óptima de N agentes a M focos de seguridad sujeto a restricciones de capacidad y prioridad.

**Modelos a comparar:**
- Cuánticos: QAOA (Quantum Approximate Optimization Algorithm)
- Clásicos: Algoritmos Genéticos, Simulated Annealing, Programación Lineal

**Dataset requerido:**
- Número de agentes disponibles por turno
- Listado de focos de seguridad con nivel de criticidad
- Restricciones operativas (zonas, distancias, tiempos de respuesta)
- Historial de asignaciones anteriores (si existe)
- Tamaño orientativo: 20-100 agentes, 30-200 focos

---

### Caso 3: Clasificación — Por Definir con el Equipo

**Contexto:** SKG tiene proyectos en múltiples dominios que pueden proveer un dataset adecuado para un problema de clasificación. Se propone trabajar con alguna de las siguientes dos opciones, según la disponibilidad de datos en el equipo:

**Opciones candidatas:**
- Clasificación de incidentes de tráfico (tipo de evento o nivel de severidad) — datos de Movilidad de Bogotá
- Clasificación de alertas o eventos en sistemas de monitoreo ambiental (CAR, PESCAR)

**Modelos a comparar:**
- Cuánticos: Quantum Kernel SVM, Variational Quantum Classifier (VQC)
- Clásicos: SVM, Random Forest, XGBoost, Gradient Boosting

**Dataset requerido (general):**
- Tamaño: 1.000-10.000 registros
- Variables: 5-20 features relevantes al problema
- Target: variable categórica (binaria o multiclase)

---

## Requisitos Generales de Datasets

- Datos reales de SKG (no sintéticos)
- Anonimizados o pseudonimizados si contienen información sensible o restringida
- Formato: CSV, Excel o exportación de base de datos
- Calidad aceptable o documentación clara sobre valores faltantes y anomalías conocidas
- Documentación mínima por dataset:
  - Diccionario de datos (descripción de cada columna)
  - Contexto de negocio (qué representa el dataset, cómo se recolectó)
  - Definición del objetivo (qué queremos predecir u optimizar)

---

## Privacidad y Cumplimiento

Compromiso total con la privacidad de los datos:
- Anonimización completa de datos sensibles antes de su uso
- Cumplimiento estricto con la normativa de protección de datos aplicable
- Uso exclusivo para fines de investigación académica, sin uso comercial
- Posibilidad de eliminar los datos al finalizar el TFM si así se requiere
- Disponibilidad para firmar acuerdos de confidencialidad y coordinar con el área legal

---

## Timeline Propuesto

| Semana | Actividad |
|--------|-----------|
| 1-2 | Identificación de datasets candidatos y evaluación de disponibilidad |
| 3 | Anonimización si es necesaria y preparación de documentación |
| 4 | Entrega de datasets y validación inicial |
| 5-8 | Experimentación y análisis comparativo |
| 9 | Presentación de resultados preliminares al equipo |

El timeline es adaptable según la complejidad y la disponibilidad del equipo.

---

## Beneficios para SKG

**Para el equipo de datos:**
- Exploración de técnicas cuánticas aplicadas a problemas propios de la empresa
- Análisis profundo de datasets operacionales con una perspectiva investigativa

**Para SKG Tecnología:**
- Evaluación sin inversión adicional de si la computación cuántica puede aportar valor en los proyectos actuales
- Generación de conocimiento aplicado sobre movilidad, seguridad y optimización de recursos
- Posible publicación académica con reconocimiento a la empresa

---

## Resumen Ejecutivo (Una Página)

**Qué se investiga:** Si los algoritmos cuánticos de Machine Learning superan a los clásicos en problemas reales de SKG.

**Por qué es relevante para SKG:** Los tres casos de uso están directamente alineados con proyectos activos de la empresa:
- Movilidad de Bogotá: predicción de velocidades para mejorar la planificación del tráfico
- Policía Nacional: optimización del despliegue de agentes de campo en focos de seguridad
- Movilidad de Bogotá o CAR/PESCAR: clasificación de incidentes de tráfico o alertas ambientales

**Qué se necesita de SKG:**
- Tres datasets reales (uno por caso de uso)
- Documentación básica de cada dataset
- Una reunión de 30 minutos para alinear y priorizar

**Compromiso de confidencialidad:** Uso exclusivo para investigación académica. Datos anonimizados y posibilidad de firma de NDA.

**Duración del proyecto:** 9 semanas desde la entrega de datos hasta la presentación de resultados.

---

## Próximos Pasos

1. Reunión de presentación (30 min) para discutir los casos de uso y la viabilidad de los datasets
2. Identificación y priorización de fuentes de datos disponibles en SKG
3. Preparación y entrega de datasets con documentación
4. Inicio de experimentos y análisis comparativo
5. Presentación de resultados al equipo

---

## Contacto

**Juan David Prada Malagón**  
Científico de Datos Sr. — SKG Tecnología  
jprada@skgtecnologia.com
