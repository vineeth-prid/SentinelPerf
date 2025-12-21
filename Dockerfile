# SentinelPerf AI - Autonomous Performance Engineering Agent
# Multi-stage build for minimal image size

FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# Copy source code
COPY sentinelperf/ /build/sentinelperf/
COPY setup.py .
COPY pyproject.toml .

# -------------------------------------------------------------------
# Final image
# -------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="SentinelPerf"
LABEL description="Autonomous Performance Engineering Agent"
LABEL version="0.1.0"

WORKDIR /app

# Install k6
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -sS https://dl.k6.io/key.gpg | gpg --dearmor -o /usr/share/keyrings/k6-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" > /etc/apt/sources.list.d/k6.list \
    && apt-get update && apt-get install -y --no-install-recommends k6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages/

# Copy application
COPY --from=builder /build/sentinelperf /app/sentinelperf/
COPY sentinelperf.yaml.example /app/

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default working directory for user configs
WORKDIR /work

# Entrypoint
ENTRYPOINT ["python", "-m", "sentinelperf.cli"]
CMD ["--help"]
