# =====================================
# Base Image
# =====================================
FROM python:3.10-slim

# =====================================
# Environment Variables
# =====================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =====================================
# Working Directory
# =====================================
WORKDIR /app

# =====================================
# System Dependencies (minimal & safe)
# =====================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =====================================
# Install Python Dependencies
# (Cached layer for faster rebuilds)
# =====================================
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================
# Copy Application Code
# =====================================
COPY app/ ./app/

# =====================================
# Copy Static UI Files
# =====================================
COPY static/ ./static/

# =====================================
# Copy Trained Model Artifacts
# IMPORTANT: This is what makes retraining effective
# =====================================
COPY model/ ./model/

# =====================================
# Expose FastAPI Port
# =====================================
EXPOSE 8000

# =====================================
# Run Application
# =====================================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
