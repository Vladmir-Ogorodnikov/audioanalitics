# syntax=docker/dockerfile:1
FROM node:22-alpine AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/tsconfig.json frontend/vite.config.ts frontend/index.html ./
COPY frontend/src ./src
RUN npm install && npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    AUDIOANALYTIC_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY --from=frontend-builder /build/src/audioanalytic/web/static ./src/audioanalytic/web/static

RUN pip install --no-cache-dir -e ".[postgres]"

EXPOSE 8000
CMD ["audioanalytic", "serve", "--host", "0.0.0.0", "--port", "8000"]
