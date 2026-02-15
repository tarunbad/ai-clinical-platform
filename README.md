# AI Clinical Intelligence Platform

FastAPI service that predicts 30-day readmission risk for diabetic patient encounters, logs predictions to PostgreSQL, and exposes monitoring metrics via Prometheus + Grafana.

## Stack
- FastAPI (REST API)
- XGBoost + scikit-learn pipeline
- PostgreSQL (prediction logs)
- Prometheus (metrics scrape)
- Grafana (dashboard)

## Quickstart (Docker)
```bash
docker compose up --build
