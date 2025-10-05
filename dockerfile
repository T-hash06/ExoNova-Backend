# Imagen base ligera de Python
FROM python:3.12-slim

# Configurar entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear carpeta de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para FastAPI + IA ligera
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar UV (gestor de paquetes)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copiar archivos de dependencias primero para aprovechar caché
COPY pyproject.toml uv.lock ./

# Instalar dependencias de Python usando UV
RUN uv sync --frozen --no-dev

# Copiar el código fuente y modelos
COPY src/ ./src/
COPY models/ ./models/

# Exponer el puerto (FastAPI por defecto usa 8000)
EXPOSE 8000

# Variable de entorno para Azure Container Apps
ENV PORT=8000

# Comando para ejecutar FastAPI con Uvicorn (producción)
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
