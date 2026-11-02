
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom Customer Churn Prediction")
st.write("Upload customer data and predict the possibility of customer churn.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Please run python src/train.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Upload Customer CSV")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Customer Data Preview")
    st.dataframe(data.head())

    features = data.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    ).copy()

    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"],
            errors="coerce"
        )

    if st.button("Predict Customer Churn", type="primary"):
        try:
            probabilities = model.predict_proba(features)[:, 1]

            results = pd.DataFrame({
                "churn_probability": probabilities,
                "predicted_churn_at_0_50": (
                    probabilities >= 0.50
                ).astype(int)
            })

            st.success("Predictions generated successfully!")

            churn_count = int(
                (results["predicted_churn_at_0_50"] == 1).sum()
            )
            no_churn_count = len(results) - churn_count

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Customers", len(results))
            col2.metric("Predicted Churn", churn_count)
            col3.metric("Predicted No Churn", no_churn_count)

            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            st.subheader("Churn Prediction Distribution")
            st.bar_chart(
                results["predicted_churn_at_0_50"].value_counts()
            )

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

else:
    st.info("Please upload a customer CSV file to start predictions.")

st.caption("Educational ML project | Telecom Customer Churn Prediction")
