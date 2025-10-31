# ============================================
# lightweight Python base image
# ============================================
FROM python:3.12-slim

# ============================================
# work directory
# ============================================
WORKDIR /app

# ============================================
# system dependencies (for LightGBM)
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# copy and install Python dependencies
# ============================================
COPY dependencies.txt .

RUN pip install --no-cache-dir -r dependencies.txt

# ============================================
# copy source code and data
# ============================================
COPY . .

# ============================================
# default command (train + predict)
# ============================================

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]