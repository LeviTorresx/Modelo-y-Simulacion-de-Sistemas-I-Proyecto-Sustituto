#!/bin/sh
# entrypoint.sh
# --------------------------------------------------------
# Este script orquesta la ejecución dentro del contenedor.
# 1. Ejecuta el entrenamiento (train.py)
# 2. Ejecuta la predicción (predict.py)
# Si ocurre algún error, el script se detiene inmediatamente.
# --------------------------------------------------------

# Terminar si ocurre un error
set -e

echo "Iniciando entrenamiento..."
python train.py    # Ejecuta el script de entrenamiento

echo "Ejecutando predicción..."
python predict.py  # Ejecuta el script de predicción

echo "Proceso completado con éxito."
