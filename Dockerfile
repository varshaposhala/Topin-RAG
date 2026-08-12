# Multi-stage build keeps the runtime image smaller (important on Render).
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Slim deps only — do NOT install sentence-transformers/torch (OOMs on Render).
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .
COPY --from=frontend /frontend/dist ./frontend/dist

# Bake ONNX MiniLM into the image so first search does not download under memory pressure.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN mkdir -p /app/.fastembed_cache \
    && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/.fastembed_cache')"

ENV EMBEDDINGS_BACKEND=fast
ENV SKIP_LLM_INTRO=1
ENV SKIP_RERANK=1
ENV TOPIN_API_MODE=1
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache

EXPOSE 8000

# Render injects $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 75"]
