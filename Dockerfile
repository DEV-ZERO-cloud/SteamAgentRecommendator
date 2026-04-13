# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Instala dependencias pesadas en una imagen separada para reducir el tamaño final
FROM python:3.11-slim AS builder

# Evitar prompts interactivos durante apt
ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema necesarias para compilar pyswip y faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# SWI-Prolog (repositorio oficial)
RUN apt-get update && apt-get install -y --no-install-recommends \
    swi-prolog \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo para el build
WORKDIR /install

# Copiar requirements antes del código para aprovechar caché de Docker
COPY requirements.txt .

# Instalar dependencias Python en un prefix separado
RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Solo SWI-Prolog runtime (sin build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    swi-prolog-nox \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias Python desde el stage builder
COPY --from=builder /install/deps /usr/local

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Modelo de embeddings (se puede sobreescribir con --env)
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    # SWI-Prolog necesita saber dónde están sus librerías
    SWI_HOME_DIR=/usr/lib/swi-prolog

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copiar el código fuente
COPY --chown=appuser:appuser . .

# Crear directorios que necesita el engine en runtime
RUN mkdir -p /app/src/embeddings /app/src/data && \
    chown -R appuser:appuser /app/src/embeddings /app/src/data

USER appuser

# Puerto por defecto (ajusta según tu API/servidor)
EXPOSE 8000

# Punto de entrada — ajusta según cómo arranca tu aplicación
# Ejemplos:
#   FastAPI  : uvicorn src.api.main:app --host 0.0.0.0 --port 8000
#   Flask    : python -m flask --app src.api.main run --host 0.0.0.0 --port 8000
#   Script   : python main.py
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]