"""
apirest.py
----------
API REST mínima para exponer el modelo de predicción mediante FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import os
import train

app = FastAPI(title="NYC Taxi Trip Duration API", version="0.2")

# -------------------------
# Ubicación del modelo
# -------------------------
MODEL_PATHS = [
    "./data/model_lgbm.pkl"
]

def find_model_path() -> Optional[str]:
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None


def load_model(model_path: str):
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo: {e}")


# -------------------------
# Modelo Pydantic "data"
# -------------------------
class Record(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int
    pickup_datetime: str   # puede ser str; pandas lo convierte


# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "API de predicción activa"}


@app.post("/predict")
def predict(records: List[Record]):
    if not records:
        raise HTTPException(status_code=400, detail="Se requiere al menos un registro para predecir.")

    # Convertir a lista de dicts
    list_dicts = [r.model_dump() for r in records]

    df = pd.DataFrame(list_dicts)

    # Transformaciones
    try:
        df = train.add_distance_feature(df)
        df = train.add_time_features(df)
        weather_df = train.fetch_weather_data()
        df = train.merge_weather(df, weather_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar features: {e}")

    # Modelo
    model_path = find_model_path()
    if model_path is None:
        raise HTTPException(status_code=500, detail="Modelo no encontrado. Entrene el modelo primero (POST /train).")

    try:
        model = load_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Features esperadas
    try:
        X = df[train.FEATURES]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Faltan columnas necesarias: {e}")

    # Predicción
    preds = model.predict(X)

    return {"predictions": [{"prediction": float(p)} for p in preds]}


@app.post("/train")
def retrain(sync: bool = True):
    try:
        train.main()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante entrenamiento: {e}")

    mp = find_model_path()
    return {"status": "trained", "model_path": mp}
