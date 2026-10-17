import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"
ID_COL = "customerID"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ],
        remainder="drop",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--outdir", required=True, help="Output dir for processed CSVs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df.columns = [c.strip() for c in df.columns]

    # Basic checks
    required = set(CAT_COLS + NUM_COLS + [TARGET_COL, ID_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    # Pre-cleaning: Fix TotalCharges (convert blanks to NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Map target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
        # Drop rows where target is missing if any
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Drop ID
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Split (stratified)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[TARGET_COL] if TARGET_COL in df.columns else None,
    )

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Fit preprocessor on train only
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Get feature names
    feature_names = []
    # numeric names
    feature_names.extend(NUM_COLS)
    # one-hot names
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    ohe_names = ohe.get_feature_names_out(CAT_COLS).tolist()
    feature_names.extend(ohe_names)

    # Save processed CSVs
    train_out = pd.DataFrame(X_train_p, columns=feature_names)
    train_out[TARGET_COL] = y_train.values

    test_out = pd.DataFrame(X_test_p, columns=feature_names)
    test_out[TARGET_COL] = y_test.values

    train_path = outdir / "train.csv"
    test_path = outdir / "test.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    # Save preprocessor + schema for API later
    Path("models").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    with open("models/feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print(f"Preprocessing complete. Saved to: {outdir}")
    print("Model artifacts saved to: models/")


if __name__ == "__main__":
    main()