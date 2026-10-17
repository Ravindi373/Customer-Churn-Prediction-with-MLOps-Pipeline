from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/opt/airflow/project")
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Churn_Prediction_DataSet.csv"

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn"
]


def run_validation():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset missing: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    if df.shape[0] == 0:
        raise ValueError("Dataset is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic null check (TotalCharges often has blanks; allow but log count)
    null_counts = df.isna().sum().to_dict()
    print("Validation OK. Null counts:", null_counts)