# 🏥 AI Clinical Readmission Risk Platform

A production-oriented machine learning system that predicts 30-day hospital readmission risk using structured clinical data.

This project goes beyond model training. It simulates how real-world ML systems are deployed, monitored, logged, and explained in healthcare environments.

---

## 🚨 Problem Statement

Hospital readmissions within 30 days are:

- Extremely costly
- A major hospital performance metric
- Often preventable with early risk detection

However, most ML projects stop at model accuracy.

In real-world healthcare systems, we also need:

- Explainability
- Logging & audit trails
- Monitoring & observability
- Reliable API serving
- Containerized deployment

This project addresses the full ML lifecycle.

---

## 🏗 System Architecture
```
Client
  ↓
FastAPI (Prediction API)
  ↓
XGBoost Pipeline (Preprocessing + Model)
  ↓
PostgreSQL (Prediction Logging)
  ↓
Prometheus (Metrics Scraping)
  ↓
Grafana (Monitoring Dashboard)
```

Fully containerized using Docker Compose.

---

## 🧠 Model Details

- Algorithm: XGBoost (Binary Classification)
- Dataset: UCI Diabetes 130-US hospitals dataset
- Class imbalance handled using `scale_pos_weight`
- Preprocessing via `ColumnTransformer`
- Explainability via SHAP

### 📈 Performance

- Validation AUC: **0.6697**
- Test AUC: **0.6826**

---

## 🚀 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/predict` | Returns risk probability + classification |
| `/explain` | Returns SHAP top risk drivers |
| `/logs` | Returns recent predictions |
| `/logs/{id}` | Returns full payload + explanations |
| `/stats` | Aggregated system statistics |
| `/metrics` | Prometheus metrics endpoint |

Swagger Docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Observability

The system exposes Prometheus metrics:

- `total_predictions`
- `high_risk_predictions`
- `avg_risk_probability`

These are visualized in Grafana dashboards:

- Prediction volume
- High-risk rate
- Model probability trends
- System health

Prometheus: [http://localhost:9090](http://localhost:9090)

Grafana: [http://localhost:3000](http://localhost:3000)

---

## 🐳 Deployment (Local)

Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ai-clinical-platform.git
cd ai-clinical-platform
```

Run the full stack:
```bash
docker compose up --build
```

Services:

- API → [http://localhost:8000](http://localhost:8000)
- Prometheus → [http://localhost:9090](http://localhost:9090)
- Grafana → [http://localhost:3000](http://localhost:3000)

---

## 🗄 Database

All predictions are logged in PostgreSQL:

- Request payload
- Risk probability
- Classification
- Timestamp
- Explainability output (if available)

This simulates real-world audit and compliance requirements in healthcare AI systems.

---

## 🔎 Explainability

Each prediction can optionally return:

- Top risk-increasing features
- Top risk-decreasing features

Powered by SHAP for local interpretability.

---

## 🎯 Key Engineering Focus

This project emphasizes:

- Production-ready ML systems
- Model observability
- Logging & auditability
- Infrastructure integration
- Containerization
- Monitoring dashboards

It mirrors how ML systems operate in real healthcare environments.

---

## 🛠 Tech Stack

- Python
- FastAPI
- XGBoost
- Scikit-learn
- SHAP
- PostgreSQL
- Prometheus
- Grafana
- Docker / Docker Compose

---

## 👨‍💻 Author

**Tarun Badana**  
MS Computer Science  
University at Buffalo

---

## 🔮 Future Improvements

- Cloud deployment (AWS / GCP)
- CI/CD pipeline
- Model drift detection
- Authentication & role-based access
- Batch inference pipeline

---

⭐ If you found this interesting, feel free to connect or provide feedback!
