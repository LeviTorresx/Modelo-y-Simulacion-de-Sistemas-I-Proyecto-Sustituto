# Proyecto Sustituto - Modelo y Simulación de Sistemas I

## Descripción
El objetivo del proyecto es completar la formación anterior llevando un modelo predictivo a un estado listo para que sea integrado en sistema de producción.
Este proyecto corresponde a la competencia de Kaggle:  [NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/)

El objetivo es predecir la duración de viajes en taxi en la ciudad de Nueva York utilizando técnicas de machine learning.  

El trabajo se divide en tres fases:  

1. **Fase 1**: Notebook con el flujo completo de entrenamiento y predicción del modelo.  
2. **Fase 2**: Separación del código en scripts (`train.py`, `predict.py`) y construcción de un contenedor con Docker.  
3. **Fase 3**: Exposición del modelo como un servicio REST con `apirest.py` y `client.py`, más un Dockerfile extendido.  

En este README se documentan los pasos para ejecutar las tres fases.

---

## Fase 1

Para la predicción del modelo utilizamos la entrega del usuario Moiz Essaji [Moiz Essaji NYC Taxi Trip Duration submit](https://www.kaggle.com/code/moizessaji/nyc-taxi-trip)

Dentro del directorio `fase-1` encontraremos:  

- Un **notebook Jupyter/Colab** que incluye:  
  - Descarga y carga de datos desde Kaggle.  
  - Preprocesamiento y creación de variables (distancia, tiempo, clima).  
  - Entrenamiento del modelo de machine learning.  
  - Generación de predicciones sobre el conjunto de prueba.  
  - Creación del archivo `submission.csv` listo para subir a Kaggle.  

---

## Requisitos

- Python 3.10+  
- Entorno recomendado: **Google Colab** (no requiere instalación local).  
- Librerías principales:  
  - `numpy`, `pandas`  
  - `scikit-learn`  
  - `lightgbm`  
  - `meteostat`  
  - `kagglehub`  

---

## Ejecución del Notebook

1. Abrir el archivo `fase-1/notebook.ipynb` en **Google Colab**.  
2. Conectar a KaggleHub para descargar el dataset (NYC Taxi Trip Duration).  
3. Ejecutar las celdas en orden sin omitir ninguna.  
4. Al finalizar, se generará un archivo llamado `submission.csv` con las predicciones.  

---

## Resultados esperados

- Un dataset enriquecido con características adicionales (distancia, clima, tiempo).  
- Un modelo entrenado con LightGBM.  
- Archivo `submission.csv` con las predicciones finales.  

---

## Próximas fases

- **Fase 2**: migrar el flujo de entrenamiento y predicción a scripts (`train.py`, `predict.py`) y crear un contenedor Docker para su ejecución.  
- **Fase 3**: exponer el modelo como un API REST (`apirest.py`) y añadir un cliente (`client.py`) para consultar el servicio en contenedor.  

---
