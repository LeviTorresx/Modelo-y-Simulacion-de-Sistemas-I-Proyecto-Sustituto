"""
predict.py
-----------
Este script carga el modelo entrenado y genera predicciones sobre un conjunto de datos de prueba.
Guarda las predicciones en un archivo submission.csv.
"""

import os
import pandas as pd
import joblib
from train import (
    add_distance_feature, add_time_features,
    fetch_weather_data, merge_weather, load_data, FEATURES
)

# =========================================================
# CONFIGURACIÓN
# =========================================================
DATA_PATH = "./data/test.zip"      # Datos de entrada para predicción
MODEL_PATH = "./data/model_lgbm.pkl"  # Modelo entrenado
OUTPUT_PATH = "./data/submission.csv"  # Resultado de predicciones


# =========================================================
# FUNCIONES DE PREDICCIÓN
# =========================================================
def load_model(model_path: str):
    """
    Carga el modelo entrenado desde un archivo .pkl.

    Parámetros:
        model_path (str): Ruta al archivo del modelo.

    Retorna:
        modelo cargado (LightGBM)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    print(f"Modelo cargado desde: {model_path}")
    return joblib.load(model_path)


def make_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera predicciones usando las mismas variables que en el entrenamiento.

    Parámetros:
        model: modelo entrenado
        df: conjunto de datos con las variables predictoras

    Retorna:
        DataFrame con las columnas ['id', 'trip_duration']
    """
    X_pred = df[FEATURES]
    df["trip_duration"] = model.predict(X_pred)
    return df[["id", "trip_duration"]]


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main():
    """Ejecuta el proceso de predicción completo."""
    print("Iniciando predicción...\n")

    model = load_model(MODEL_PATH)
    test_df = load_data(DATA_PATH)
    print(f"Datos cargados: {test_df.shape}")

    test_df = add_distance_feature(test_df)
    test_df = add_time_features(test_df)

    weather_df = fetch_weather_data()
    test_merged = merge_weather(test_df, weather_df)

    submission = make_predictions(model, test_merged)

    print(f"Predicciones generadas: {submission.shape}")
    print(f"Primeras filas:\n{submission.head()}")

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Predicciones guardadas en: {OUTPUT_PATH}")


# =========================================================
# PUNTO DE ENTRADA
# =========================================================
if __name__ == "__main__":
    main()
