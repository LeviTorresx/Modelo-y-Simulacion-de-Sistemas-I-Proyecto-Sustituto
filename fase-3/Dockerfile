# ============================================
# Imagen base ligera de Python 3.12
# ============================================
FROM python:3.12-slim

# ============================================
# Establecer el directorio de trabajo dentro del contenedor
# ============================================
WORKDIR /app

# ============================================
# Actualizar índices de paquete e instalar dependencias del sistema
# - libgomp1 es necesario para LightGBM
# - curl/ca-certificates pueden ser útiles para descargas seguras desde Python
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Copiar el archivo de dependencias al contenedor
# ============================================
COPY dependencies.txt .

# ============================================
# Instalar las dependencias de Python definidas en dependencies.txt
# - --no-cache-dir evita cache en pip para reducir tamaño de imagen final
# ============================================
RUN pip install --no-cache-dir -r dependencies.txt

# ============================================
# Copiar el código fuente de la aplicación al contenedor
# - Incluye train.py, predict.py, apirest.py, client.py, entrypoint.sh, etc.
# ============================================
COPY . .

# ============================================
# Exponer el puerto donde correrá el API (FastAPI/Uvicorn)
# ============================================
EXPOSE 8000

# ============================================
# Comando por defecto: lanzar Uvicorn para servir FastAPI
# - Ejecuta apirest:app en host 0.0.0.0 puerto 8000
# - Se puede sobrescribir con docker compose o docker run command
# ============================================
CMD ["uvicorn", "apirest:app", "--host", "0.0.0.0", "--port", "8000"]
