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
- **`dependencies.txt`** → Contiene las librerías necesarias para ejecutar ambos scripts.  
- **`Dockerfile`** → Define la configuración de la imagen que incluye todo el entorno de ejecución.
- **`docker-compose.yml`** → Automatiza la construcción y ejecución del contenedor enviando los resultados a la carpeta `data/`.

Dentro de `fase-2` encontramos un directorio adicional que se llama **`data`** que incluye:

- **`test.zip`** → Es dataset con formato .csv comprimido que vamos a pasar por el modelo para generar la predicción.
- **`train.zip`** → Es el conjunto de entrenamiento en formato .csv comprimido que incluye tanto las variables de entrada como la etiqueta objetivo (trip_duration) y se usa para entrenar el modelo.


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

- **`model_lgbm.pkl`** → Archivo binario con el modelo ya entrenado (formato `pickle`), se genera después de ejecutar el script `train.py`.
- **`submission.csv`** → Es la predicción realizada con el modelo entrenado, se genera después de ejecutar el script `predict.py`.
  
---

## Fase 3

En esta fase lo que hacemos es exponer el modelo como un API REST (`apirest.py`) y añadir un cliente (`client.py`) para consultar el servicio en contenedor.  

En esta fase se completa el ciclo de despliegue del modelo, pasando de un flujo ejecutado exclusivamente dentro del contenedor (Fase 2) a un servicio REST profesional que expone:

- Un endpoint para entrenar el modelo

- Un endpoint para realizar predicciones

- Un script cliente para consumir el API desde cualquier máquina

Además, el Dockerfile se actualiza para que el contenedor ejecute exclusivamente la API mediante Uvicorn + FastAPI.

Dentro del directorio `fase-3` encontraremos los mismos archivos que en fase-2 con la adición de dos scripts adicionales y una diferencia clave en el Dockerfile y en el docker-compose.yml:

- **`apirest.py`** → Este script expone un servicio con dos endpoints principales:
  - @POST /train: ejecuta train.main() y guarda el modelo en `data/model_lgbm.pkl`
  - @POST /predict: recibe una lista de registros JSON y la API realiza transformación, enriquecimiento, validación de columnas, carga el modelo entrenado y hace la predicción con este. El endpoint está en capacidad de predecir N registros en un solo llamado.
- **`client.py`** → Este script simula un cliente externo que consume la API, lo que hace es ejecutar primero la API /train, espera que termine el entrenamiento y luego envía un registro a /predict finalizando con el resultado formateado.  
- **`Dockerfile`** → Además de la configuración de la fase 2, añadimos las dependencias joblib, FastAPI y uvicorn y posteriormente expone el puerto 8000.
- **`docker-compose.yml`** → Se añade la línea restart: unless-stopped que hará que el contenedor no se apague automáticament, debido a que necesitamos que se quede escuchando las peticiones hasta que decidamos apagarlo.

A diferencia de fase-2 no necesitaremos el script `entrypoint.sh` ya que debemos ejecutar `client.py` desde una terminal diferente de donde ejecutamos el contenedor.

## Requisitos

Antes de ejecutar el entorno Docker debemos tener algunas consideraciones:

- Tener instalado Docker en la máquina donde se va a ejecutar el contenedor. Se recomienda Docker Desktop para visualizar de una forma más visual la construcción y despliegue del contenedor.
- Si estamos trabajando en un entorno Windows, debemos asegurarnos de que Docker engine esté activo para poder ejecutar.
- Como debemos ejecutar el script de python `client.py` desde nuestra máquina, debemos instalar la librería requests para ejecutar este script, además de tener claramente Python 3.10 o superior. En caso de no tener instalado requests, se ejecuta el siguiente comando en terminal:

pip install requests

## Paso a paso

Nos ubicamos sobre la raíz repositorio donde tenemos nuestro proyecto y entramos al directorio fase-3 y ejecutamos el siguiente comando:

docker compose up --build

Esto lo que hará será construir la imagen Docker a partir del Dockerfile, instalando las dependencias necesarias (pandas, scikit-learn, lightgbm, etc.) y levantar el contenedor modelo-taxi definido en docker-compose.yml, montando el volumen ./data:/app/data para que los datos y resultados persistan fuera del contenedor y levantando el servidor web configurado en el puerto 8000.

En bien levantemos nuestro contenedor, lo que haremos será abrir una nueva terminal sin cerrar en la que está corriendo el contenedor. Esta nueva terminal debemos ubicarnos sobre la raíz del repositorio, entrar al directorio fase-3 donde está nuestro script `client.py` y ejecutamos el siguiente comando:

python client.py

Lo que hará este script es consumir primero el endpoint /train, si nos dirigimos a la terminal donde está corriendo el contenedor podremos observar como se hace la petición y el proceso de entrenamiento que ya conocemos de la fase-2. Después de entrenar el modelo y tenerlo en el directorio /data, va a consumir el endpoint /predict enviando el cuerpo de una petición los datos de un viaje en formato JSON y nuestra API lo que hará será realizar transformaciones, cálculos y aplicar el modelo entrenado. Finalmente recibiremos una respuesta HTML estructurada con un JSON con los resultados, un código HTTP y mensaje de éxito o error.

## Ejecución del contenedor

1. Se levanta el servidor API REST (apirest.py)

Cuando el contenedor inicia:

- Se ejecuta el script apirest.py.

- Este script levanta un servidor web Flask/FastAPI escuchando en el puerto 8000.

- El servidor expone endpoints HTTP (como /predict) que permiten procesar solicitudes externas.

- Durante el arranque, el script:

  - Carga el modelo entrenado (model_lgbm.pkl) desde la carpeta data/.

  - Prepara la lógica para recibir datos, procesarlos y generar predicciones bajo demanda.

En otras palabras: el contenedor inicia y queda “vivo” ejecutando la API, a diferencia de la Fase-2 donde el contenedor corría y se apagaba.

2. Ejecución del cliente (client.py) — Opcional y externo al contenedor

En esta fase, el cliente no se ejecuta dentro del contenedor, sino desde nuestra máquina local o desde otro servicio.

El archivo client.py:

- Consume el endpoint /train de la API.

- Espera a que el contenedor termine de ejecutar el entrenamiento (visble desde la terminal donde corre el contenedor).

- Envía datos al endpoint /predict de la API.

- Recibe como respuesta las predicciones generadas por el modelo.

- Se comunica vía HTTP con el contenedor levantado.

3. Modificación de los datos de entrada

En el caso de que queramos cambiar los datos usados para la predicción, modificamos en el script client.py el payload en la función call_predict(), recordando que la API está capacitada para recibir una lista de registros en una sola llamada.

4. Comportamiento del contenedor

El contenedor no se apaga automáticamente, se quedará corriendo ya que la API se queda escuchando en espera de solicitudes, para detenerlo usamos Control+C en la terminal donde se ejecuta el contenedor y este se detendrá de forma segura.


## Resultados esperados

Nuestros resultados esperados serán visibles en la terminal donde ejecutamos el script `client.py` la cual nos mostrará una respuesta HTTP con el código HTTP de la petición y el cuerpo de un JSON.

Ejemplo de los resultados esperados para código 200 (OK):
=== Ejecutando entrenamiento vía API ===
Status: 200
JSON: {
  "status": "trained",
  "model_path": "./data/model_lgbm.pkl"
}

=== Solicitando predicción ===
Status: 200
JSON: {
  "predictions": [
    {
      "prediction": 1679.078195277709
    }
  ]
}
  
---
