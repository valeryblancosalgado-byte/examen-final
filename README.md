# Sistema de Diagnóstico de Enfermedad Renal Crónica - Nexus Health Institute

## 1. Instrucciones de Ejecución

1.  **Instalar Python versión:** 3.12.10
2.  **Instalar dependencias:** `py -m pip install -r requirements.txt`
3.  **Análisis Exploratorio:** Ejecutar el notebook `EDA.ipynb`.
4.  **Entrenamiento y MLOps:** Ejecutar `py train.py`. Esto genera el modelo final (`pipeline_final.joblib`) y registra los 3 runs en MLflow.
5.  **API de Predicción:** El sistema está listo para ser probado con el `TestClient` o ejecutarse con `py -m uvicorn api:app --reload`
    en local en la url `http://127.0.0.1:8000/docs`.

## 2. Análisis de Insights (Fase 2)

* **Hemoglobina y Creatinina:** Se observó una separación clara: los pacientes enfermos (CKD) muestran consistentemente **niveles bajos de Hemoglobina** (anemia) y **niveles muy altos de Creatinina Sérica**, confirmando su fuerte valor predictivo.
* [cite_start]**Correlaciones:** Se identificó una alta multicolinealidad entre pares clínicos (ej., Hemoglobina y Packed Cell Volume), lo cual justificó la posterior eliminación de una de las dos variables en la Fase 4[cite: 48].
* **Comorbilidades:** La presencia de **Hipertensión** y **Diabetes Mellitus** se relacionó directamente con la clase CKD, actuando como factores de riesgo clave.

## 3. Justificación de Diseño (Fases 1 y 3)

* **Estrategia de Imputación:** Se utilizó la **Mediana** para rellenar los valores nulos en las características numéricas. Esto fue elegido debido a la presencia de **valores atípicos (outliers)** extremos (vistos en Creatinina y Glucosa), ya que la mediana es más **robusta** que la media.
* **Ingeniería de Características:**
    * **Ratio Urea/Creatinina:** Se creó para capturar la relación renal entre estos dos metabolitos, proporcionando una característica única que aumenta la capacidad discriminativa del modelo.
    * **Presión Arterial Categórica:** Se utilizó para simplificar la interpretación de la presión arterial en rangos clínicos (Normal, Elevada, Hipertensión), mejorando la capacidad de decisión del clasificador RandomForest.
* **Selección de Características (Fase 4):** Se eligió el método **Wrapper (Regresión Logística L1)**. [cite_start]Este método penaliza y elimina características con poco impacto, resultando en un **Modelo Optimizado** que mantiene el alto rendimiento con un subconjunto mínimo de variables, cumpliendo directamente con el reto del examen[cite: 11, 70].

## 4. Tabla Comparativa de Modelos (Resultados MLflow)

| Modelo | Accuracy | F1-Score | AUC-ROC | N.º de Features | Observaciones |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 99.16% | 0.9933 | 1.000 | 25+ | Alto rendimiento, pero computacionalmente pesado. |
| **Optimized** | 99.16% | 0.9933 | 0.999 | 13-15 | **Mejor Balance.** Mantiene rendimiento con menos variables. |
| **Tuned** | 99.16% | 0.9933 | 0.998 | 13-15 | Confirmó la robustez, con ganancias mínimas en el tuning. |

## 5. Conclusión Final

El sistema de diagnóstico alcanza un rendimiento excepcional, con una **Accuracy** y un **F1-Score** superiores al 99%. [cite_start]La elección del **Modelo Optimizado** con características seleccionadas garantiza que la solución sea no solo precisa, sino también **escalable y eficiente** para la integración clínica, cumpliendo con la misión de construir un sistema robusto[cite: 7].
