from pathlib import Path
import joblib
import shap
import pandas as pd
import numpy as np

from src.data import load_raw, make_target, basic_clean, split

ROOT = Path(__file__).resolve().parents[1]

def main():
    model_path = ROOT / "models" / "xgb_readmit_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train first: python -m src.train")

    # Load full pipeline
    pipe = joblib.load(model_path)

    # Load data
    df = load_raw(str(ROOT / "data" / "raw" / "diabetic_data.csv"))
    df = basic_clean(df)
    df = make_target(df)

    _, _, test_df = split(df)

    # Pick one patient to explain
    sample = test_df.sample(1, random_state=42)
    y_true = sample["target_readmit_30"].values[0]

    # Get transformed features
    preprocessor = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    X_sample = sample.drop(columns=["readmitted", "target_readmit_30"])
    X_transformed = preprocessor.transform(sample)
    prob = pipe.predict_proba(sample)[:, 1][0]
    print("Predicted probability (readmit <30):", round(float(prob), 4))


    # Use SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    print("\nTrue Label:", y_true)
    print("Predicted probability (readmit <30):", round(float(prob), 4))

    # Sort for absolute contribution
    shap_df_sorted = shap_df.sort_values(by="shap_value", key=abs, ascending=False)

    # Top increasing (positive impact)
    top_pos = shap_df.sort_values("shap_value", ascending=False).head(5)

    # Top decreasing (negative impact)
    top_neg = shap_df.sort_values("shap_value", ascending=True).head(5)

    print("\nTop 5 risk INCREASING factors:\n")
    print(top_pos)

    print("\nTop 5 risk DECREASING factors:\n")
    print(top_neg)


if __name__ == "__main__":
    main()
