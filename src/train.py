"""Train and evaluate telecom churn classifiers."""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload the Telco Customer Churn CSV, "
            "rename it telco_churn.csv, and place it in data/raw/."
        )
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' was not found.")
    df = df.drop_duplicates()
    # Common IBM sample stores TotalCharges as text with blank strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df[df["Churn"].notna()].copy()
    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0}
    )
    df = df[df["Churn"].notna()].copy()
    df["Churn"] = df["Churn"].astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    drop_cols = [c for c in ["Churn", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["Churn"]

    if y.nunique() != 2:
        raise ValueError("The target must contain exactly two classes after cleaning.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        prob = pipeline.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "pr_auc": float(average_precision_score(y_test, prob)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = pipeline

    # Select by PR-AUC, a useful metric when the positive class is less common.
    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_model = fitted[best_name]
    best_prob = best_model.predict_proba(X_test)[:, 1]
    best_pred = (best_prob >= 0.50).astype(int)

    joblib.dump(best_model, MODEL_DIR / "churn_pipeline.joblib")
    output = {
        "selected_model": best_name,
        "threshold": 0.50,
        "test_rows": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
        "model_comparison": results,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(output, indent=2))
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, best_pred, target_names=["No churn", "Churn"])
    )
    print(json.dumps(output, indent=2))
    print(f"\nSaved model to: {MODEL_DIR / 'churn_pipeline.joblib'}")


if __name__ == "__main__":
    main()
