# ============================================================================
# Image unique pour une VM (Oracle Cloud Always Free, Ubuntu, ARM ou x86).
# Reproduit le build Railway (nixpacks.toml) : Python 3.12 + build Vite,
# puis uvicorn sert l'API ET frontend/dist (main.py, _FRONTEND_DIST).
# Build depuis la racine du dépôt :
#   docker build -f deploy/oracle/Dockerfile .
# ============================================================================

# --- 1) Frontend --------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2) Backend ---------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Mêmes dépendances que Railway (nixpacks installe backend/requirements.txt).
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --upgrade pip && pip install -r backend/requirements.txt

COPY backend/ backend/
COPY --from=frontend /app/frontend/dist frontend/dist

# /data existe => database.py et agent_manager.py y écrivent d'eux-mêmes,
# comme sur le volume Railway. Monté depuis l'hôte par docker-compose.
RUN mkdir -p /data
ENV XAU_DB_PATH=/data/xau_bot.db
VOLUME ["/data"]

EXPOSE 8000

# "/" est public (sert index.html) : même chemin que le healthcheck Railway.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=8)" || exit 1

# Même commande que railway.json, port fixe (pas de $PORT hors Railway).
CMD ["uvicorn", "--app-dir", "backend", "main:app", "--host", "0.0.0.0", "--port", "8000"]
