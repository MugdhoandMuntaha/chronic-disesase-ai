# Multi-Service Production Dockerfile for Chronic Disease AI System
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing bytecode
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_ALLOW_FILE_STORE=true \
    MLFLOW_DISABLE_AGENT_HINT=1

# Install essential system dependencies (build tools, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code, data, models, and reports
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY tests/ ./tests/
COPY README.md .

# Expose ports for FastAPI (8000), Streamlit (8501), and MLflow (5000)
EXPOSE 8000 8501 5000

# Default command starts FastAPI microservice
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
