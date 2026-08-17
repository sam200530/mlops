# Serving image for the fraud detection API.
#
# Multi-stage: wheels are built in a stage that carries a compiler, and the
# runtime stage installs only prebuilt wheels. That keeps gcc and the build
# headers out of the shipped image.
#
# Only requirements-api.txt is installed — not the full training stack. Serving a
# prediction does not need matplotlib, seaborn, jupyter or optuna, and leaving
# them out cuts both image size and vulnerability surface.

# ---------- build stage ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential for any source-only dependency; libgomp1 headers for LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements-api.txt .
RUN pip wheel --wheel-dir /wheels -r requirements-api.txt

# ---------- runtime stage ----------
FROM python:3.12-slim AS runtime

# PYTHONUNBUFFERED so container logs appear immediately rather than on flush.
# FRAUD_PROJECT_ROOT points the path resolver at the image layout.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUD_PROJECT_ROOT=/app \
    PYTHONPATH=/app

# libgomp1 is a runtime requirement of LightGBM's OpenMP threading.
# curl is used by the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements-api.txt .
RUN pip install --no-index --find-links=/wheels -r requirements-api.txt \
    && rm -rf /wheels

# Application code only. Data and notebooks are excluded via .dockerignore.
COPY src/ ./src/
COPY api/ ./api/
COPY database/ ./database/
COPY configs/ ./configs/

# The model artifact is mounted at runtime rather than baked in: a 20-40 MB
# binary in the image would force an image rebuild for every retrain, and would
# make the image's contents depend on training output.
RUN mkdir -p /app/models /app/data/processed

# Run as an unprivileged user.
RUN useradd --create-home --shell /bin/bash --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Hits /health, which reports degraded (200) rather than failing when only Redis
# or Postgres are down — the service can still score transactions.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
