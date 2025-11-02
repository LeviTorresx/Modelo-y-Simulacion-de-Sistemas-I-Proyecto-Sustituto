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

## Fase 2

En esta fase se busca **migrar el flujo del modelo** desarrollado en la Fase 1 hacia un entorno más estructurado y modular, mediante la **separación del código en scripts independientes** y su **empaquetamiento dentro de un contenedor Docker**.  

El propósito es preparar el modelo para su integración en un sistema productivo o para su ejecución automatizada en entornos controlados.

Dentro del directorio `fase-2` encontraremos:

- **`train.py`** → Entrena el modelo de machine learning utilizando el conjunto de datos de Kaggle, genera el archivo `model.pkl` y guarda el modelo entrenado.  
- **`predict.py`** → Carga el modelo entrenado desde `model.pkl` y realiza predicciones sobre un nuevo conjunto de datos (por ejemplo, `test.csv`).  
- **`requirements.txt`** → Contiene las librerías necesarias para ejecutar ambos scripts.  
- **`Dockerfile`** → Define la configuración del contenedor que incluye todo el entorno de ejecución.
- **`docker-compose.yml`** → Automatiza la construcción y ejecución del contenedor enviando los resultados a la carpeta `data/`.

Dentro de `fase-2` encontramos un directorio adicional que se llama **`data`** que incluye:

- **`test.zip`** → Es dataset con formato .csv comprimido que vamos a pasar por el modelo para generar la predicción.
- **`train.zip`** → Es el conjunto de entrenamiento en formato .csv comprimido que incluye tanto las variables de entrada como la etiqueta objetivo (trip_duration) y se usa para entrenar el modelo.
- **`model.pkl`** → Archivo binario con el modelo ya entrenado (formato `pickle`), se genera después de ejecutar el script `train.py`.
- **`submission.csv`** → Es la predicción realizada con el modelo entrenado, se genera después de ejecutar el script `predict.py`.

---

## Requisitos

Antes de ejecutar el entorno Docker debemos tener algunas consideraciones:

- Tener instalado Docker en la máquina donde se va a ejecutar el contenedor. Se recomienda Docker Desktop para visualizar de una forma más visual la construcción y despliegue del contenedor.
- Si estamos trabajando en un entorno Windows, debemos asegurarnos de que Docker engine esté activo para poder ejecutar.
- Si estamos trabajando en un entorno Windows, para que podamos ejecutar los archivos .sh que en nuestro caso es entrypoint.sh debemos usar LF en vez de CRLF que es como viene por defecto. Esto es fácilmente modificable desde un compilador como Visual Studio Code.

---

## Paso a paso

Nos ubicamos sobre la raíz repositorio donde tenemos nuestro proyecto y entramos al directorio fase-2 y ejecutamos el siguiente comando:

docker compose up --build

Esto lo que hará será construir la imagen Docker a partir del Dockerfile, instalando las dependencias necesarias (pandas, scikit-learn, lightgbm, etc.) y levantar el contenedor modelo-taxi definido en docker-compose.yml, montando el volumen ./data:/app/data para que los datos y resultados persistan fuera del contenedor.

---

## Ejecución del contenedor

Cuando el contenedor se inicia hará lo siguiente:

1. Cuando el contenedor se inicia, el script train.py:

- Carga train.zip desde data/ y el script se encarga de descomprimirlo.
- Preprocesa los datos.
- Entrena el modelo, en nuestro caso LightGBM
- Guarda el modelo en disco, en nuestro caso en la carpeta data con el nombre 'model_lgbm.pkl'

2. Después del entrenamiento, el contenedor ejecuta predict.py:

- Carga test.zip desde data/ y el script se encarga de descomprimirlo.
- Usa el modelo entrenado (model_lgbm.pkl) para generar predicciones.
- Guarda en data/ un archivo submission.csv que será nuestra predicción.

3. Cuando la ejecución finaliza, nuestro contenedor automáticamente se apaga.

4. En caso de que queramos cambiar nuestros datos de entrada test y train, debemos tener en cuenta que nuestro script está leyendo los archivos comprimidos en .zip así que modificamos estos archivos en data/ y ejecutamos nuevamente nuestro contenedor con

docker compose up --build

---

## Resultados esperados

Nuestros resultados esperados estarán almacenados en el directorio data de nuestro repositorio raíz.

- El modelo entrenado con LightGBM en formato pickle 'model_lgbm.pkl'.
- Archivo `submission.csv` con las predicciones finales.  

---

## Próximas fases

- **Fase 3**: exponer el modelo como un API REST (`apirest.py`) y añadir un cliente (`client.py`) para consultar el servicio en contenedor.  

---
