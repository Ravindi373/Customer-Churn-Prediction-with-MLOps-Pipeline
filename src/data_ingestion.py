import argparse
import json
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--report_out", default="reports/data_quality_report.json")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Raw data not found: {in_path}")

    df = pd.read_csv(in_path)
    df.columns = [c.strip() for c in df.columns]

    # Validate required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic data quality metrics
    report = {
        "summary": {
            "total_rows": int(df.shape[0]),
            "total_cols": int(df.shape[1]),
        },
        "missing_values": df.isna().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_stats": df.describe().to_dict()
    }

    # Custom validation: Check for empty strings in TotalCharges which often causes issues
    empty_total_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    report["custom_checks"] = {
        "empty_total_charges_count": int(empty_total_charges)
    }

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Data validation complete. Report saved to: {args.report_out}")

if __name__ == "__main__":
    main()
