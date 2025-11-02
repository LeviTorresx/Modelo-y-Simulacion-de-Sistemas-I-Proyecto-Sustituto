# ============================================
# Imagen base ligera de Python 3.12
# ============================================
FROM python:3.12-slim

# ============================================
# Definir el directorio de trabajo dentro del contenedor
# ============================================
WORKDIR /app

# ============================================
# Instalación de dependencias del sistema requeridas por LightGBM
# - libgomp1 es necesario para la ejecución paralela de LightGBM
# - Se limpia el caché de apt para reducir el tamaño final de la imagen
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Copiar archivo de dependencias Python al contenedor
# ============================================
COPY dependencies.txt .

# ============================================
# Instalar las librerías de Python necesarias
# ============================================
RUN pip install --no-cache-dir -r dependencies.txt

# ============================================
# Copiar el resto del código fuente y los datos
# al directorio de trabajo del contenedor
# ============================================
COPY . .

# ============================================
# Dar permisos de ejecución al script de entrada
# ============================================
RUN chmod +x entrypoint.sh

# ============================================
# Comando por defecto: ejecutar el script de entrada
# que entrena el modelo y genera las predicciones
# ============================================
ENTRYPOINT ["./entrypoint.sh"]