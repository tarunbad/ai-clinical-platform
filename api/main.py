from pathlib import Path
from typing import Dict, Any, List
import psycopg2
import json

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request


import os
from psycopg2.extras import RealDictCursor
templates = Jinja2Templates(directory="api/templates")


def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # fallback for local non-docker runs (optional)
        return psycopg2.connect(
            dbname="Tony",
            user=os.getenv("USER", "postgres"),
            host="localhost",
            port=5432
        )
    return psycopg2.connect(db_url)



def ensure_tables():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id SERIAL PRIMARY KEY,
            request_payload JSONB,
            risk_probability DOUBLE PRECISION,
            threshold DOUBLE PRECISION,
            prediction TEXT,
            top_risk_increasing JSONB,
            top_risk_decreasing JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()
        cur.close()
    finally:
        conn.close()


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_readmit_pipeline.joblib"

THRESHOLD = float(os.getenv("THRESHOLD", "0.40"))

app = FastAPI(title="AI Clinical Intelligence Platform", version="0.1")
ensure_tables()
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
  id SERIAL PRIMARY KEY,
  request_payload JSONB,
  risk_probability DOUBLE PRECISION,
  threshold DOUBLE PRECISION,
  prediction TEXT,
  top_risk_increasing JSONB,
  top_risk_decreasing JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

@app.on_event("startup")
def init_db():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        cur.close()
    finally:
        conn.close()



# Load once at startup
pipe = joblib.load(MODEL_PATH)
preprocessor = pipe.named_steps["pre"]
model = pipe.named_steps["model"]
explainer = shap.TreeExplainer(model)


class PatientFeatures(BaseModel):
    # flexible payload: you can send any feature columns as key/value
    data: Dict[str, Any]


def _to_dataframe(payload: PatientFeatures) -> pd.DataFrame:
    df = pd.DataFrame([payload.data])

    # match training-time behavior
    for c in ["encounter_id", "patient_nbr", "readmitted"]:
        if c in df.columns:
            df = df.drop(columns=c)

    return df



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PatientFeatures):
    df = _to_dataframe(payload)
    prob = float(pipe.predict_proba(df)[:, 1][0])
    pred = int(prob >= THRESHOLD)

    response = {
        "risk_probability": prob,
        "threshold": THRESHOLD,
        "prediction": "HIGH_RISK" if pred == 1 else "LOW_RISK"
    }

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO prediction_logs (request_payload, risk_probability, threshold, prediction)
            VALUES (%s, %s, %s, %s)
            """,
            (json.dumps(payload.data), prob, THRESHOLD, response["prediction"])
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()

    return response




@app.post("/explain")
def explain(payload: PatientFeatures):
    df = _to_dataframe(payload)
    prob = float(pipe.predict_proba(df)[:, 1][0])
    pred = int(prob >= THRESHOLD)

    X_transformed = preprocessor.transform(df)
    shap_vals = explainer.shap_values(X_transformed)[0]
    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame({"feature": feature_names, "shap_value": shap_vals})

    top_pos = shap_df.sort_values("shap_value", ascending=False).head(5)
    top_neg = shap_df.sort_values("shap_value", ascending=True).head(5)

    def rows_to_list(x: pd.DataFrame) -> List[dict]:
        return [{"feature": r["feature"], "impact": float(r["shap_value"])} for _, r in x.iterrows()]

    response = {
        "risk_probability": prob,
        "threshold": THRESHOLD,
        "prediction": "HIGH_RISK" if pred == 1 else "LOW_RISK",
        "top_risk_increasing": rows_to_list(top_pos),
        "top_risk_decreasing": rows_to_list(top_neg),
    }

    # Insert into DB
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO prediction_logs
            (request_payload, risk_probability, threshold, prediction,
            top_risk_increasing, top_risk_decreasing)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                json.dumps(payload.data),
                prob,
                THRESHOLD,
                response["prediction"],
                json.dumps(response["top_risk_increasing"]),
                json.dumps(response["top_risk_decreasing"])
            )
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


    return response

@app.get("/logs/recent")
def recent_logs(limit: int = 10):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, risk_probability, threshold, prediction, created_at
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    return [
        {
            "id": r[0],
            "risk_probability": r[1],
            "threshold": r[2],
            "prediction": r[3],
            "created_at": str(r[4]),
        }
        for r in rows
    ]

@app.get("/logs")
def logs(limit: int = 20):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, risk_probability, threshold, prediction, created_at
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    return [
        {
            "id": r[0],
            "risk_probability": r[1],
            "threshold": r[2],
            "prediction": r[3],
            "created_at": str(r[4]),
        }
        for r in rows
    ]

@app.get("/logs")
def get_logs(limit: int = 50):
    limit = max(1, min(limit, 500))  # safety clamp

    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT id, risk_probability, threshold, prediction, created_at,
                   top_risk_increasing, top_risk_decreasing
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT %s;
            """,
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()


from datetime import datetime, timedelta
from fastapi import Query, HTTPException

@app.get("/logs/{log_id}")
def get_log(log_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, request_payload, risk_probability, threshold, prediction,
                   top_risk_increasing, top_risk_decreasing, created_at
            FROM prediction_logs
            WHERE id = %s
            """,
            (log_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Log not found")

        return {
            "id": row[0],
            "request_payload": row[1],
            "risk_probability": row[2],
            "threshold": row[3],
            "prediction": row[4],
            "top_risk_increasing": row[5],
            "top_risk_decreasing": row[6],
            "created_at": str(row[7]),
        }
    finally:
        conn.close()


@app.get("/stats")
def stats(days: int = Query(7, ge=1, le=60)):
    """
    Simple analytics:
    - total predictions
    - high risk count + %
    - avg probability
    - daily counts for last N days
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # totals
        cur.execute("SELECT COUNT(*) FROM prediction_logs;")
        total = cur.fetchone()[0]

        # high risk
        cur.execute("SELECT COUNT(*) FROM prediction_logs WHERE prediction = 'HIGH_RISK';")
        high = cur.fetchone()[0]
        high_pct = (high / total) if total else 0.0

        # avg prob
        cur.execute("SELECT COALESCE(AVG(risk_probability), 0) FROM prediction_logs;")
        avg_prob = float(cur.fetchone()[0])

        # daily counts (last N days)
        cur.execute(
            """
            SELECT DATE(created_at) as day, COUNT(*) as cnt
            FROM prediction_logs
            WHERE created_at >= NOW() - (%s || ' days')::interval
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """,
            (days,)
        )
        daily = [{"day": str(r[0]), "count": r[1]} for r in cur.fetchall()]

        return {
            "total_predictions": total,
            "high_risk_count": high,
            "high_risk_rate": high_pct,
            "avg_risk_probability": avg_prob,
            "daily_counts": daily,
            "window_days": days,
        }
    finally:
        conn.close()

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )

from fastapi.responses import PlainTextResponse

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM prediction_logs;")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM prediction_logs WHERE prediction='HIGH_RISK';")
    high = cur.fetchone()[0]

    cur.execute("SELECT AVG(risk_probability) FROM prediction_logs;")
    avg_prob = cur.fetchone()[0] or 0.0

    cur.close()
    conn.close()

    # Prometheus-style metrics
    return f"""
# HELP total_predictions Total predictions made
# TYPE total_predictions counter
total_predictions {total}

# HELP high_risk_predictions Total high risk predictions
# TYPE high_risk_predictions counter
high_risk_predictions {high}

# HELP avg_risk_probability Average predicted probability
# TYPE avg_risk_probability gauge
avg_risk_probability {avg_prob}
"""
