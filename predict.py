"""Generate churn probabilities for a CSV of new customer records."""
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
OUTPUT_PATH = ROOT / "reports" / "predictions.csv"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/predict.py path/to/new_customers.csv")
    input_path = Path(sys.argv[1])
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")
    data = pd.read_csv(input_path)
    data = data.drop(columns=[c for c in ["Churn", "customerID"] if c in data.columns])
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(data)[:, 1]
    result = pd.DataFrame({
        "churn_probability": probabilities,
        "predicted_churn_at_0_50": (probabilities >= 0.50).astype(int),
    })
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
