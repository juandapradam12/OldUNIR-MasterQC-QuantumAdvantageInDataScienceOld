# Pruebas de Concepto (POCs) — TFM Computación Cuántica

**Current code (2026):** QML v2 experiments live in [`qml_v2/`](qml_v2/). The old README `src/models/` tree was never implemented; original TFM experiments stay inlined in `notebooks/01_*.ipynb`–`10_*.ipynb`.

```bash
cd POCs
pip install -r requirements.txt
python -m qml_v2.classification --quick          # smoke
python -m qml_v2.anomaly --quick                 # one-class AD (not fraud)
python -m qml_v2.classification --n-total 100 --n-repeats 3
pytest tests/ -q
```

Notebooks are optional; prefer the `.py` modules (Cursor Agents shows `.ipynb` as JSON).

---

Este directorio contiene las pruebas de concepto para la comparación empírica entre algoritmos cuánticos y clásicos.

## Layout actual

```
POCs/
├── qml_v2/
│   ├── classification.py
│   ├── regression.py
│   ├── optimization.py          # same-model PuLP + greedy + QAOA (assignment)
│   ├── maxcutm_optimization.py  # MaxCut bench
│   ├── tsp_optimization.py      # fair closed-tour TSP (not VRP)
│   ├── quantumness.py           # Q_rob / ZZ-structure score
│   ├── qcr.py                   # Quantum-Compatible Representation pre-maps
│   ├── clustering.py            # fair Spectral-ZZ vs classical clustering
│   ├── anomaly.py               # one-class OCSVM-ZZ vs IF/RBF/LOF (not fraud)
│   ├── timeseries.py            # fair lag one-step forecast (QK-SVR)
│   ├── __init__.py
│   └── __main__.py
├── tests/
├── notebooks/             # TFM record (01–10) + thin 01b driver
├── data/
├── scripts/               # Thesis figure generators
├── config.yaml
└── requirements.txt
```

## Arquitectura prevista en el README original (no implementada)

El README original describía `src/models/`, `pipelines/`, `cli/`. That tree does not exist. New work goes under `qml_v2/`.

El proyecto sigue una arquitectura limpia y modular:

```
POCs/
├── main.py                      # 🎯 Orquestador principal
├── config.yaml                  # ⚙️ Configuración global
├── requirements.txt             # 📦 Dependencias
│
├── src/                         # Código fuente
│   ├── config/                  # Configuración
│   │   ├── datasets.py         # Config centralizada de datasets
│   │   └── models.py           # Config centralizada de modelos
│   ├── services/                # Servicios de datos
│   │   ├── loaders.py          # Cargadores de datasets
│   │   └── preprocessors.py    # Preprocesadores
│   └── models/                  # Modelos
│       ├── quantum/            # Modelos cuánticos (QK, VQC, VQR, QAOA)
│       └── classical/          # Modelos clásicos
│           ├── svm.py          # SVM clasificador y regresor
│           ├── random_forest.py # Random Forest
│           ├── gradient_boosting.py # Gradient Boosting
│           ├── xgboost_model.py # XGBoost
│           └── linear_regression.py # Regresión lineal
│
├── pipelines/                   # 🔄 Lógica de orquestación
│   ├── classification.py       # Pipeline de clasificación
│   ├── regression.py           # Pipeline de regresión
│   └── optimization.py         # Pipeline de optimización
│
├── cli/                         # 💻 Command Line Interfaces
│   ├── clasificacion.py        # CLI para clasificación
│   ├── regresion.py            # CLI para regresión
│   └── optimizacion.py         # CLI para optimización
│
├── utils/                       # 🛠️ Utilidades compartidas
│   ├── helpers.py              # Funciones auxiliares
│   ├── evaluation.py           # Métricas y evaluación
│   └── download_datasets.py    # Descarga de datasets públicos
│
├── notebooks/                   # Notebooks de análisis
│   ├── 01_clasificacion_quantum_vs_clasico.ipynb
│   ├── 02_optimizacion_QAOA_vs_clasico.ipynb
│   └── 03_regresion_VQC_vs_clasico.ipynb
│
├── data/                        # 📊 Datos
│   ├── input/                  # Datasets de entrada
│   └── output/                 # 📈 Salidas de modelos
│       ├── figures/            # Gráficos generados
│       └── metrics/            # Métricas en CSV/JSON
│
├── docs/                        # Documentación
│   └── REQUERIMIENTOS_DATOS_SKG.md
│
└── tests/                       # Tests unitarios
    └── __init__.py
```

### 🎯 Separación de Responsabilidades

**`pipelines/`** - Lógica de negocio
- Orquestan el flujo completo de un experimento
- Clases reutilizables (`ClassificationPipeline`, `RegressionPipeline`, etc.)
- Independientes de la línea de comandos
- Pueden ser importadas desde notebooks o scripts

**`cli/`** - Interfaces de usuario
- Parsean argumentos con `argparse`
- Punto de entrada ejecutable
- Llaman a los pipelines con los parámetros adecuados
- Gestionan errores y output para el usuario

**`main.py`** - Orquestador global
- Ejecuta múltiples CLIs en secuencia
- Ideal para batch processing
- Resumen de resultados

## 🚀 Uso Rápido

### Opción 1: main.py (Orquestador Principal - RECOMENDADO)

El archivo `main.py` ejecuta experimentos llamando a los CLIs:

```bash
# Ejecutar TODOS los experimentos
python main.py --all

# Ejecutar experimento específico
python main.py --task classification --dataset telco_churn --model quantum_kernel_svm

# Ejecutar todos los modelos de clasificación con un dataset
python main.py --task classification --dataset telco_churn --model all

# Ejecutar con todos los datasets de clasificación
python main.py --task classification --dataset all --model all

# Prueba rápida con muestreo
python main.py --task classification --dataset fraud_detection --model all --sample-size 1000

# Ver ayuda
python main.py --help
```

### Opción 2: CLIs Directos

Ejecuta los CLIs como módulos de Python:

```bash
# Clasificación
python -m cli.clasificacion --dataset telco_churn --model quantum_kernel_svm

# Ver opciones
python -m cli.clasificacion --help

# Regresión (TODO)
python -m cli.regresion --dataset house_prices --model vqr

# Optimización (TODO)
python -m cli.optimizacion --dataset portfolio --model qaoa
```

### Opción 3: Usar Pipelines Programáticamente

Desde Python, notebooks o scripts:

```python
from pipelines.classification import ClassificationPipeline

# Configurar modelos
quantum_models = [{
    'name': 'Quantum Kernel SVM',
    'type': 'quantum_kernel_svm',
    'params': {'feature_dimension': 4, 'reps': 2}
}]

classical_models = [{
    'name': 'SVM Clásico',
    'type': 'svm',
    'params': {'kernel': 'rbf', 'C': 1.0}
}]

# Ejecutar pipeline
pipeline = ClassificationPipeline('telco_churn', sample_size=500)
results = pipeline.run(quantum_models, classical_models)
```

### Opción 4: Notebooks Interactivos

Para análisis y visualización:

```bash
jupyter notebook notebooks/01_clasificacion_quantum_vs_clasico.ipynb
```

## 🚀 Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar herramientas de descarga de datos (opcional)
pip install opendatasets kaggle
```

## 📥 Descarga de Datasets

### **Datasets Públicos (automático):**

```bash
# Ejecutar script de descarga
python utils/download_datasets.py
```

Esto descargará:
- ✅ **Telco Customer Churn** (Kaggle)
- ✅ **Credit Card Fraud Detection** (Kaggle - subset)
- ✅ **House Prices** (Kaggle)
- ✅ **California Housing** (sklearn - incluido)

**Requisito:** Configurar Kaggle API (ver `utils/download_datasets.py`)

### **Datasets de SKG Tecnología:**

Coordinar con supervisor de prácticas. Ver `docs/REQUERIMIENTOS_DATOS_SKG.md` para especificaciones detalladas.

Ubicación esperada: `data/raw/skg_*.csv`

## 📊 Casos de Uso

### 1. Clasificación Supervisada
**Notebook:** `notebooks/01_clasificacion_quantum_vs_clasico.ipynb`

Comparación de:
- **Cuántico:** Quantum Kernel Methods, Variational Quantum Classifier (VQC)
- **Clásico:** SVM (kernels RBF, polinomial), Random Forest, Gradient Boosting

**Métricas:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, tiempo de entrenamiento

---

### 2. Optimización Combinatoria
**Notebook:** `notebooks/02_optimizacion_QAOA_vs_clasico.ipynb`

Comparación de:
- **Cuántico:** QAOA (diferentes profundidades p=1,2,3)
- **Clásico:** Algoritmos genéticos, Simulated Annealing, Gurobi (óptimo)

**Métricas:** Valor de función objetivo, razón de aproximación, tiempo de convergencia

---

### 3. Regresión
**Notebook:** `notebooks/03_regresion_VQC_vs_clasico.ipynb`

Comparación de:
- **Cuántico:** VQC para regresión
- **Clásico:** Linear Regression, Ridge, Random Forest Regressor, XGBoost

**Métricas:** MSE, RMSE, MAE, R², tiempo de entrenamiento

---

## 🛠️ Tecnologías

- **Quantum:** Qiskit 2.6.1 (`qiskit-aer` 0.18.2, `qiskit-algorithms` 0.4.0, `qiskit-machine-learning` 0.9.1, `qiskit-optimization` 0.7.0, `qiskit-finance` 0.4.1)
- **Clásico:** scikit-learn 1.9.1, XGBoost 3.4.1, SciPy 1.18.1
- **Análisis:** pandas 3.1.0, NumPy 2.5.4, matplotlib 3.11.2, seaborn 0.13.2, NetworkX 3.7
- **Optimización:** PuLP 2.9.0 (CBC) y `scipy.optimize.milp` (HiGHS)
- **Entorno de reejecución:** Python 3.12.3 en Linux; versiones exactas en `requirements-lock.txt` (el entorno del autor: Python 3.14, Qiskit 2.3.0; véase el Anexo A de la tesis)

## 📝 Notas

- Los datasets empresariales están en `data/raw/` y **NO** se suben al repo por privacidad
- Los resultados se guardan automáticamente en `results/`
- Cada notebook es autónomo y ejecutable independientemente
