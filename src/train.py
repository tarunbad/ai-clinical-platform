import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from src.data import load_raw, make_target, basic_clean, split
from src.features import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
TARGET_COL = "target_readmit_30"

def _clean_target(y: pd.Series) -> pd.Series:
    """Ensure y is clean binary 0/1 ints (no pd.NA)."""
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    if y.isna().any():
        raise ValueError(f"Target has NaNs after conversion. Sample:\n{y[y.isna()].head()}")
    y = y.astype(int)
    # Optional safety check:
    bad = set(y.unique()) - {0, 1}
    if bad:
        raise ValueError(f"Target contains non-binary values: {bad}")
    return y

def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas pd.NA/NaT to np.nan in a sklearn-friendly way."""
    # This avoids the sklearn SimpleImputer pd.NA ambiguity crash
    return X.astype("object").where(pd.notnull(X), np.nan)

def main():
    raw_path = ROOT / "data" / "raw" / "diabetic_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {raw_path}. Put the UCI CSV there named diabetic_data.csv"
        )

    df = load_raw(str(raw_path))
    df = basic_clean(df)
    df = make_target(df)

    # Split first (keeps your existing split logic)
    train_df, val_df, test_df = split(df)

    # ---- Build X/y properly (NO leakage) ----
    y_train = _clean_target(train_df[TARGET_COL])
    y_val   = _clean_target(val_df[TARGET_COL])
    y_test  = _clean_target(test_df[TARGET_COL])

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    X_val   = val_df.drop(columns=[TARGET_COL]).copy()
    X_test  = test_df.drop(columns=[TARGET_COL]).copy()

    # Clean only features
    X_train = _clean_features(X_train)
    X_val   = _clean_features(X_val)
    X_test  = _clean_features(X_test)

    # Class weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
    print("Scale_pos_weight:", scale_pos_weight)

    # Preprocessor should be built from training FEATURES only
    pre = build_preprocessor(pd.concat([X_train, y_train.rename(TARGET_COL)], axis=1))

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
        random_state=42,
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    # AUCs
    val_probs = pipe.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    test_probs = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    out_dir = ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "xgb_readmit_pipeline.joblib")

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        json.dumps({"val_auc": float(val_auc), "test_auc": float(test_auc)}, indent=2)
    )

    print(f"✅ Saved model: {out_dir / 'xgb_readmit_pipeline.joblib'}")
    print(f"📈 Validation AUC: {val_auc:.4f}")
    print(f"🧪 Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()
