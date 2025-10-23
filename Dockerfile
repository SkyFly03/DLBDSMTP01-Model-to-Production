# Dockerfile
# ------------------------------------------------------------
# Minimal container for the Flask API and model runtime
# Purpose:
#   - Install deps and run `python -m app.api` on port 5000
# Behavior:
#   - Trains model on first start if models/turbine_iforest.pkl is missing
# Notes:
#   - Use with `docker build -t turbine-api .`
#   - Run with `docker run -p 5000:5000 turbine-api`
# ------------------------------------------------------------

FROM python:3.10-slim

WORKDIR /app

# System packages for scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code; models/ may be empty initially
COPY app ./app
COPY models ./models

ENV PYTHONPATH=/app
ENV MODEL_PATH=models/turbine_iforest.pkl

EXPOSE 5000
CMD ["python", "-m", "app.api"]
