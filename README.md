# Telecom Customer Churn Prediction

A beginner-friendly, reproducible Python ML project that predicts whether a telecom customer is likely to churn. It includes data validation, preprocessing, model comparison, evaluation, and model persistence.

## 1. Project status
This is a GitHub-ready starter implementation. The project plan is complete; model results will be generated after you add the dataset and run the pipeline. **Do not claim metrics or a trained model until you have run the code.**

## 2. Dataset
Use the IBM Telco Customer Churn sample dataset (CSV). Obtain it from a reputable source and check its usage terms. Dataset copies may differ.

1. Download the CSV.
2. Rename it to `telco_churn.csv`.
3. Place it at `data/raw/telco_churn.csv`.

Expected target column: `Churn` with values such as `Yes` and `No`. The commonly used sample also contains `customerID`, which is excluded from model features. The script handles the common `TotalCharges` blank-string issue.

## 3. Environment setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 4. Run the project

From the repository root:

```bash
python src/train.py
```

The script creates:
- `reports/metrics.json` — held-out test metrics and model comparison
- `reports/classification_report.txt` — precision/recall/F1
- `models/churn_pipeline.joblib` — fitted preprocessing + model pipeline

## 5. Predict for new customers

After training, create a CSV containing the same input feature columns as the training data (without `Churn` or `customerID`) and run:

```bash
python src/predict.py path/to/new_customers.csv
```

Predictions are written to `reports/predictions.csv`.

## 6. Repository structure

```text
Telecom-Customer-Churn-Prediction/
├── data/raw/                 # Put the downloaded CSV here (not committed)
├── models/                   # Trained model artifacts (not committed)
├── reports/                  # Generated reports (not committed)
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 7. Methodology
- Stratified train/test split (80/20)
- Numeric median imputation and standard scaling
- Categorical most-frequent imputation and one-hot encoding
- Logistic Regression and Random Forest comparison
- Evaluation with precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix
- The default classification threshold is 0.50; choose a business threshold only after reviewing costs and validation results.

## 8. Important limitations
This is an educational prototype, not a production churn system. A random split does not establish performance on future time periods. Predictive churn risk does not prove that a retention offer will prevent churn. Review privacy, fairness, representativeness, and operational requirements before real-world use.

## 9. Suggested next improvements
- Add cross-validation and hyperparameter tuning
- Add threshold selection based on retention-team capacity and error costs
- Add calibration and subgroup performance checks
- Add unit tests and experiment tracking (e.g., MLflow)
- Package batch scoring and monitoring for deployment

## 10. GitHub upload
Create a public or private repository named `Telecom-Customer-Churn-Prediction`, then upload these files. Do not upload customer data, credentials, `.venv`, or generated model artifacts.
