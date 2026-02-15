import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.data import load_raw, make_target, basic_clean, split

ROOT = Path(__file__).resolve().parents[1]

def main():
    model_path = ROOT / "models" / "xgb_readmit_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train first: python -m src.train")

    df = load_raw(str(ROOT / "data" / "raw" / "diabetic_data.csv"))
    df = basic_clean(df)
    df = make_target(df)

    _, _, test_df = split(df)

    pipe = joblib.load(model_path)

    y_true = test_df["target_readmit_30"]
    probs = pipe.predict_proba(test_df)[:, 1]
    threshold = 0.4
    preds = (probs >= threshold).astype(int)


    auc = roc_auc_score(y_true, probs)
    print(f"🧪 Test AUC: {auc:.4f}\n")
    print("Confusion matrix:\n", confusion_matrix(y_true, preds), "\n")
    print(classification_report(y_true, preds, digits=4))

if __name__ == "__main__":
    main()
