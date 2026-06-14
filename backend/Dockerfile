# ===== Backend: FastAPI + trading engine =====
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source
COPY backend/ ./backend/

WORKDIR /app/backend

# Persist the SQLite database outside the layer
ENV XAU_DB_PATH=/data/xau_bot.db
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
