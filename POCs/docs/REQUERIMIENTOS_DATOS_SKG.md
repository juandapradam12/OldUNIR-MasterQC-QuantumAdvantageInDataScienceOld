# Requerimientos de Datos — SKG Tecnología

## 📋 Resumen

Para completar las pruebas de concepto del TFM, necesitamos datasets reales de SKG Tecnología que permitan comparar algoritmos cuánticos vs clásicos en tres familias de problemas:

1. **Clasificación Supervisada**
2. **Optimización Combinatoria**
3. **Regresión**

---

## 1. Clasificación Supervisada

### **Objetivo:**
Predecir una variable categórica (binaria o multiclase) basándose en features históricos.

### **Casos de uso sugeridos:**
- ✅ **Clasificación de tickets de soporte** (urgente/no urgente, tipo de problema)
- ✅ **Predicción de éxito de proyectos** (completado a tiempo / retrasado / cancelado)
- ✅ **Segmentación de clientes por propensión a compra** (alto/medio/bajo potencial)
- ✅ **Clasificación de riesgo de proyectos** (bajo/medio/alto riesgo)

### **Requisitos del dataset:**
- **Tamaño:** Mínimo 500 filas, ideal 1,000-5,000
- **Features:** 5-20 columnas predictoras (pueden ser numéricas o categóricas)
- **Target:** Variable categórica (2-5 clases)
- **Formato:** CSV con headers
- **Privacidad:** Datos **anonimizados** (sin nombres, emails, datos sensibles)

### **Ejemplo de estructura:**

```csv
ticket_id,priority,categoria,num_mensajes,tiempo_resolucion_prev,cliente_tipo,asignado_equipo,target_urgencia
1,media,tecnico,5,24,enterprise,backend,0
2,alta,funcional,12,48,pyme,frontend,1
3,baja,consulta,2,8,individual,soporte,0
...
```

---

## 2. Optimización Combinatoria

### **Objetivo:**
Encontrar la asignación/selección óptima que maximice/minimice una función objetivo con restricciones.

### **Casos de uso sugeridos:**
- ✅ **Asignación de desarrolladores a proyectos** (minimizar coste, maximizar skill match)
- ✅ **Planificación de recursos cloud** (minimizar coste, respetar SLA)
- ✅ **Scheduling de despliegues** (minimizar conflictos, maximizar disponibilidad)
- ✅ **Selección de tecnologías para proyecto** (maximizar ROI, respetar presupuesto)

### **Requisitos del dataset:**
- **Tamaño:** 10-30 elementos a optimizar (proyectos, recursos, tareas)
- **Formato:** 
  - **Matriz de costes/beneficios** (CSV)
  - **Restricciones** (archivo de texto o JSON)
- **Ejemplo:** Matriz de habilidades (desarrolladores × tecnologías) con coste de asignación

### **Ejemplo de estructura:**

```csv
# Matriz de skills: desarrollador × proyecto (puntuación 0-10)
desarrollador,proyecto_A,proyecto_B,proyecto_C,disponibilidad_horas
dev_1,8,5,3,120
dev_2,4,9,7,80
dev_3,6,6,8,160
...
```

**Restricciones (JSON):**
```json
{
  "max_devs_por_proyecto": 3,
  "horas_totales_disponibles": 360,
  "skills_minimos_requeridos": {"proyecto_A": 7, "proyecto_B": 6}
}
```

---

## 3. Regresión

### **Objetivo:**
Predecir una variable numérica continua basándose en features históricos.

### **Casos de uso sugeridos:**
- ✅ **Predicción de horas de trabajo de proyectos** (horas estimadas vs reales)
- ✅ **Estimación de ingresos por proyecto/cliente** (revenue mensual)
- ✅ **Predicción de uso de recursos cloud** (CPU%, memoria, storage)
- ✅ **Estimación de tiempo de resolución de tickets** (minutos/horas)

### **Requisitos del dataset:**
- **Tamaño:** Mínimo 500 filas, ideal 1,000-5,000
- **Features:** 5-15 columnas predictoras
- **Target:** Variable numérica continua (no discreta)
- **Formato:** CSV con headers
- **Privacidad:** Datos **anonimizados**

### **Ejemplo de estructura:**

```csv
proyecto_id,tipo,complejidad,num_devs,tecnologias,cliente_tipo,horas_estimadas,target_horas_reales
1,web,media,3,5,enterprise,240,280
2,mobile,alta,5,8,pyme,480,520
3,api,baja,2,3,individual,80,75
...
```

---

## 🔒 Consideraciones de Privacidad

Para **todos los datasets**:
- ✅ **Anonimizar** IDs de clientes/proyectos (usar `cliente_001`, `proyecto_A`)
- ✅ **No incluir** nombres, emails, datos personales
- ✅ **Agregar ruido** si es necesario (pequeñas variaciones aleatorias en valores numéricos)
- ✅ **Generalizar categorías** (ej: "Barcelona" → "Cataluña")
- ✅ Los datos serán usados **exclusivamente** para fines académicos del TFM
- ✅ No se publicarán en repositorio público (estarán en `.gitignore`)

---

## 📤 Formato de Entrega

**Ubicación:** `POCs/data/raw/`

**Archivos esperados:**
```
POCs/data/raw/
├── skg_clasificacion.csv         # Dataset de clasificación
├── skg_optimizacion.csv          # Matriz de optimización
├── skg_optimizacion_restricciones.json  # Restricciones
└── skg_regresion.csv             # Dataset de regresión
```

**Metadata (opcional):**
Archivo `skg_metadata.txt` con:
- Descripción del problema de negocio
- Definición de cada columna
- Rango de valores esperados
- Cualquier consideración especial

---

## ⏰ Timeline

**Fecha esperada de entrega:** **[Definir con supervisor]**

**Uso en el TFM:**
- Los datasets se usarán en Cap. 5 (Desarrollo) y Cap. 6 (Evaluación)
- Los resultados se incluirán en el documento final del TFM
- Se mencionará a SKG Tecnología en los agradecimientos (si se autoriza)

---

## 📞 Contacto

**Estudiante:** Juan David Prada Malagón  
**Director TFM:** Ricardo Rafael Domínguez Reyes  
**Universidad:** UNIR — Máster en Computación Cuántica

¿Preguntas sobre los datos? Contactar con el estudiante.
