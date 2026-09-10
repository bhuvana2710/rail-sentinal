# Dockerfile for Person 3's ML API (Isolation Forest + LSTM + SHAP)
# Build once -> runs forever. No Colab, no manual cell execution.

FROM python:3.11-slim

WORKDIR /app

# System deps some ML wheels need to build cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better Docker layer caching — only re-runs
# this slow step if requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app.py, train.py, predict.py, etc.)
COPY . .

# Train the models AT BUILD TIME so the image is self-contained.
# This replaces "manually running train.py in a notebook cell" —
# it now happens once, automatically, whenever you rebuild the image.
RUN python train.py

# Document the port (informational; actual publishing happens in
# docker-compose.yml or `docker run -p`)
EXPOSE 8000

# Basic container-level health check so `docker ps` shows (healthy)/(unhealthy)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import requests,sys; sys.exit(0 if requests.get('http://localhost:8000/health',timeout=3).ok else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
