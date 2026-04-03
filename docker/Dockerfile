# --- Stage 1: Build frontend ---
FROM node:20-slim AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit
COPY ui/ ./
RUN npm run build

# --- Stage 2: Production backend ---
FROM python:3.12-slim AS production
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml ./
COPY src/ src/
COPY README.md ./
RUN pip install --no-cache-dir .

# Config + prompts (platform reads these at runtime)
COPY config/ config/
COPY prompts/ prompts/
COPY plugins/ plugins/

# Built frontend
COPY --from=ui-build /ui/dist/ ui/dist/

# Data directory (mounted as volume in production)
RUN mkdir -p data

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/health'); assert r.status_code == 200"

CMD ["python", "-m", "uvicorn", "unstruck.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
