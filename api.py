import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- CONFIGURACIÓN INICIAL ---
app = FastAPI(
    title="API de Diagnóstico Renal - Nexus Health",
    description="Sistema de predicción de Enfermedad Renal Crónica (CKD) basado en biomarcadores.",
    version="1.0"
)

# Cargar el modelo entrenado (asegúrate de que el archivo esté en la misma carpeta)
try:
    model = joblib.load('pipeline_final.joblib')
    print("✅ Modelo 'pipeline_final.joblib' cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: No se encontró 'pipeline_final.joblib'. Ejecuta primero train.py.")

# --- DEFINICIÓN DEL SCHEMA DE ENTRADA ---
# Usamos 'Field(alias=...)' para permitir que el JSON de entrada tenga espacios,
# tal como están en el dataset original (Requisito 91).
class PatientData(BaseModel):
    Age: float
    # Usamos alias para permitir claves con espacios en el JSON
    Blood_Pressure: float = Field(..., alias="Blood Pressure")
    Specific_Gravity: float = Field(..., alias="Specific Gravity")
    Albumin: float
    Sugar: float
    Red_Blood_Cells: str = Field(..., alias="Red Blood Cells")
    Pus_Cell: str = Field(..., alias="Pus Cell")
    Pus_Cell_clumps: str = Field(..., alias="Pus Cell clumps")
    Bacteria: str
    Blood_Glucose_Random: float = Field(..., alias="Blood Glucose Random")
    Blood_Urea: float = Field(..., alias="Blood Urea")
    Serum_Creatinine: float = Field(..., alias="Serum Creatinine")
    Sodium: float
    Potassium: float
    Hemoglobin: float
    Packed_Cell_Volume: float = Field(..., alias="Packed Cell Volume")
    White_Blood_Cell_Count: float = Field(..., alias="White Blood Cell Count")
    Red_Blood_Cell_Count: float = Field(..., alias="Red Blood Cell Count")
    Hypertension: str
    Diabetes_Mellitus: str = Field(..., alias="Diabetes Mellitus")
    Coronary_Artery_Disease: str = Field(..., alias="Coronary Artery Disease")
    Appetite: str
    Pedal_Edema: str = Field(..., alias="Pedal Edema")
    Anemia: str

    class Config:
        populate_by_name = True

# --- ENDPOINT DE PREDICCIÓN ---
@app.post("/predict")
def predict(data: PatientData):
    try:
        # 1. Convertir datos de entrada a DataFrame
        # by_alias=True permite usar los nombres con espacios ("Blood Pressure")
        input_data = data.model_dump(by_alias=True)
        df = pd.DataFrame([input_data])

        # 2. Mapeo de Nombres (Originales -> Nombres cortos del entrenamiento)
        # Tu modelo fue entrenado con 'bp', 'hemo', etc., no con los nombres largos.
        rename_map = {
            'Age': 'age', 'Blood Pressure': 'bp', 'Specific Gravity': 'sg', 'Albumin': 'al',
            'Sugar': 'su', 'Red Blood Cells': 'rbc', 'Pus Cell': 'pc', 'Pus Cell clumps': 'pcc',
            'Bacteria': 'ba', 'Blood Glucose Random': 'bgr', 'Blood Urea': 'bu',
            'Serum Creatinine': 'sc', 'Sodium': 'sod', 'Potassium': 'pot', 'Hemoglobin': 'hemo',
            'Packed Cell Volume': 'pcv', 'White Blood Cell Count': 'wc',
            'Red Blood Cell Count': 'rc', 'Hypertension': 'htn', 'Diabetes Mellitus': 'dm',
            'Coronary Artery Disease': 'cad', 'Appetite': 'appet', 'Pedal Edema': 'pe',
            'Anemia': 'ane'
        }
        df.rename(columns=rename_map, inplace=True)

        # 3. Ingeniería de Características "On-the-fly"
        # CRÍTICO: El pipeline guardado espera recibir 'hemo_pcv_ratio' y 'risk_score',
        # pero estas se calculan, no se ingresan. Debemos recrearlas aquí.
        
        # A. Feature: Ratio Hemoglobina / PCV
        df['hemo_pcv_ratio'] = df['hemo'] / (df['pcv'] + 1e-5)

        # B. Feature: Risk Score
        # Primero necesitamos convertir texto (yes/no) a números (1/0) para sumar
        risk_map = {'yes': 1, 'no': 0, 'si': 1, 'ckd': 1, 'notckd': 0}
        
        # Aplicamos mapeo de seguridad
        htn_val = df['htn'].map(risk_map).fillna(0) if df['htn'].dtype == 'object' else df['htn']
        dm_val = df['dm'].map(risk_map).fillna(0) if df['dm'].dtype == 'object' else df['dm']
        
        df['risk_score'] = htn_val + dm_val

        # 4. Predicción
        # El pipeline se encarga de imputar nulos, escalar y hacer one-hot encoding
        prediction = model.predict(df)[0]
        prob_array = model.predict_proba(df)[0]
        # Obtenemos la probabilidad de la clase positiva (1 = CKD)
        probability = prob_array[1]

        # 5. Respuesta JSON
        return {
            "prediction_class": int(prediction),
            "diagnosis": "Chronic Kidney Disease (CKD)" if prediction == 1 else "Healthy (No CKD)",
            "probability_of_ckd": float(round(probability, 4))
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno en el servidor: {str(e)}")

# Endpoint de chequeo
@app.get("/")
def index():
    return {"message": "API Nexus Health funcionando. Usa /docs para probar el endpoint /predict."}