"""
train.py
---------
Este script entrena un modelo de predicción de duración de viajes en taxi usando LightGBM.
El flujo es:
1. Cargar los datos de entrenamiento.
2. Crear características derivadas (distancia, tiempo, clima).
3. Entrenar el modelo.
4. Guardar el modelo entrenado en formato .pkl.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
from lightgbm import LGBMRegressor
import joblib

# =========================================================
# CONFIGURACIÓN GLOBAL
# =========================================================
DATA_PATH = "./data/train.zip"  # Ruta del dataset de entrenamiento

# Coordenadas de Nueva York (ubicación de referencia del dataset)
NYC_COORDS = Point(40.7128, -74.0060)

# Rango de fechas para los datos meteorológicos
START_DATE = datetime(2015, 12, 31)
END_DATE = datetime(2016, 7, 31)


# =========================================================
# CARGA DE DATOS
# =========================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset de entrenamiento desde un archivo .csv, .gz o .zip.

    Parámetros:
        path (str): Ruta al archivo.

    Retorna:
        pd.DataFrame: Datos cargados.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    # Detecta tipo de compresión según extensión
    if path.endswith(".gz"):
        compression = "gzip"
    elif path.endswith(".zip"):
        compression = "zip"
    else:
        compression = None

    print(f"Cargando datos desde: {path} (compresión={compression})")
    df = pd.read_csv(path, compression=compression)

    print(f"Datos cargados: {df.shape}")
    return df


# =========================================================
# CÁLCULO DE DISTANCIA
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos geográficos (en km) usando la fórmula Haversine.
    """
    R = 6371.0  # Radio de la Tierra (km)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega una nueva columna 'distance_km' calculando la distancia entre punto de recogida y destino.
    """
    df["distance_km"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )
    return df


# =========================================================
# CARACTERÍSTICAS TEMPORALES
# =========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae información temporal de la columna pickup_datetime.
    Crea columnas: día, mes, hora, semana, día de la semana, etc.
    """
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_week"] = df["pickup_datetime"].dt.isocalendar().week
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_datetime_hour_trunc"] = df["pickup_datetime"].dt.floor("h")
    return df


# =========================================================
# DATOS METEOROLÓGICOS
# =========================================================
def fetch_weather_data() -> pd.DataFrame:
    """
    Descarga datos meteorológicos horarios de Nueva York usando la API de Meteostat.
    Retorna un DataFrame con variables climáticas (temperatura, precipitación, etc.)
    """
    print("Descargando datos meteorológicos...")
    data_hourly = Hourly(NYC_COORDS, START_DATE, END_DATE).fetch()
    data_hourly = data_hourly.reset_index().rename(columns={"time": "time_ny"})
    print(f"Datos climáticos descargados: {data_hourly.shape}")
    return data_hourly


def merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une los datos del viaje con los datos climáticos según la hora del viaje.
    """
    merged = df.merge(weather_df, how="left", left_on="pickup_datetime_hour_trunc", right_on="time_ny")
    print(f"Datos combinados: {merged.shape}")
    return merged


# =========================================================
# SELECCIÓN DE VARIABLES Y ENTRENAMIENTO
# =========================================================
FEATURES = [
    "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "distance_km",
    "pickup_day", "pickup_hour", "pickup_dayofweek", "temp", "prcp"
]


def train_split(df: pd.DataFrame):
    """
    Divide el dataset en variables predictoras (X) y variable objetivo (y).
    """
    X = df[FEATURES]
    y = df["trip_duration"]
    return X, y


def train_model(X_train, y_train) -> LGBMRegressor:
    """
    Entrena un modelo LightGBM para predecir la duración de los viajes.
    """
    print("Entrenando modelo LightGBM...")
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")
    return model


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================
def main():
    """Ejecuta el flujo completo de entrenamiento."""
    print("Iniciando entrenamiento...\n")

    df = load_data(DATA_PATH)
    df = add_distance_feature(df)
    df = add_time_features(df)

    weather_df = fetch_weather_data()
    df = merge_weather(df, weather_df)

    X_train, y_train = train_split(df)
    model = train_model(X_train, y_train)

    joblib.dump(model, "./data/model_lgbm.pkl")
    print("Modelo guardado en ./data/model_lgbm.pkl")


# =========================================================
# PUNTO DE ENTRADA
# =========================================================
if __name__ == "__main__":
    main()
