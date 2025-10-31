#!/bin/sh
# entrypoint.sh

# Salir si hay algún error
set -e

echo "Iniciando entrenamiento-"
python train.py

echo "Ejecutando predicción-"
python predict.py

echo "Proceso finalizado."
