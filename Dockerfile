FROM python:3.11-slim

WORKDIR /app

# System deps (shap can need build tools depending on wheels; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code + model artifacts
COPY api /app/api
COPY src /app/src
COPY models /app/models

# (Optional) if you want the dataset in container for demos
# COPY data /app/data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
