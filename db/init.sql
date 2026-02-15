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
