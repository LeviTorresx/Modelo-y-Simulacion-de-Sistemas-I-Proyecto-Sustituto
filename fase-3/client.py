"""
client.py
---------
Ejemplos de consumo programático del API REST desplegado en fase-3.
Asume que el API está corriendo en http://localhost:8000
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def call_train():
    """Llamada a /train para reentrenar el modelo."""
    url = f"{BASE_URL}/train"
    print("\n=== Ejecutando entrenamiento vía API ===")
    resp = requests.post(url, json={})
    print("Status:", resp.status_code)

    try:
        print("JSON:", json.dumps(resp.json(), indent=2))
    except Exception:
        print("Error interpretando respuesta:", resp.text)

    # Pausa opcional para asegurarse de que el archivo se guarde
    time.sleep(2)


def call_predict():
    """Llamada a /predict después de entrenar."""
    url = f"{BASE_URL}/predict"
    payload = [
        {
            "id": "test_1",
            "pickup_latitude": 40.7128,
            "pickup_longitude": -74.0060,
            "dropoff_latitude": 40.7306,
            "dropoff_longitude": -73.9352,
            "passenger_count": 1,
            "pickup_datetime": "2016-01-01 08:00:00"
        }
    ]

    print("\n=== Solicitando predicción ===")
    resp = requests.post(url, json=payload)
    print("Status:", resp.status_code)

    try:
        print("JSON:", json.dumps(resp.json(), indent=2))
    except Exception:
        print("Error interpretando respuesta:", resp.text)


if __name__ == "__main__":
    # Primero entrenamiento
    call_train()

    # Luego predicción
    call_predict()
