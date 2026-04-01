FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (production only, no dev deps)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Do NOT copy .env — secrets must be injected at runtime via env vars

EXPOSE 8000

CMD ["python", "-m", "mas.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
