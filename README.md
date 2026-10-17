# Churn MLOps Project

End-to-end customer churn prediction project with data validation, preprocessing, model training, evaluation, experiment tracking, orchestration, and API serving.

## Implemented Components

- Data validation report generation from the raw churn dataset
- Preprocessing pipeline with train/test split and saved preprocessing artifacts
- Model comparison across Logistic Regression, Random Forest, and XGBoost
- MLflow experiment logging for metrics, plots, and trained models
- DVC pipeline for ingestion, preprocessing, training, and evaluation
- Airflow DAG for orchestrating the full workflow
- FastAPI inference service for churn prediction
- Docker support for the API and Airflow services

## Project Structure

- `src/` - ingestion, preprocessing, training, and evaluation scripts
- `api/` - FastAPI inference service and API-specific Docker setup
- `airflow_dags/` - orchestration DAG and task wrappers
- `airflow/` - Docker Compose configuration for Airflow
- `data/raw/` - source dataset
- `data/processed/` - transformed training and test datasets
- `models/` - saved model, preprocessor, feature schema, and registry output
- `reports/` - evaluation metrics, plots, and validation reports

## Setup

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Place the churn dataset at:

```text
data/raw/Churn_Prediction_DataSet.csv
```

## Run the DVC Pipeline

```bat
dvc repro
```

This pipeline produces:

- `reports/data_quality_report.json`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/preprocessor.pkl`
- `models/feature_names.json`
- `models/best_model.pkl`
- `reports/metrics.json`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`

## Run Individual Steps

```bat
python src/data_ingestion.py --input data/raw/Churn_Prediction_DataSet.csv --report_out reports/data_quality_report.json
python src/preprocessing.py --input data/raw/Churn_Prediction_DataSet.csv --outdir data/processed
python src/train.py --train data/processed/train.csv --model_out models/best_model.pkl
python src/evaluate.py --test data/processed/test.csv --model models/best_model.pkl --outdir reports
```

## Run the API

```bat
uvicorn api.main:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`
- `GET /docs`

Example prediction payload:

```json
{
  "features": {
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 79.85,
    "TotalCharges": 956.4
  }
}
```

## Run the API with Docker

```bat
docker build -f api/Dockerfile.api -t churn-api .
docker run -p 8000:8000 churn-api
```

## Run Airflow

From the `airflow/` directory:

```bat
docker compose -f docker-compose.airflow.yml up
```

The DAG ID is `churn_mlops_end_to_end`..

## Notes

- The API expects the preprocessing artifacts and trained model to exist in `models/`.
- Airflow runs the DVC stages from inside the mounted project directory.
- MLflow artifacts are logged during training and evaluation.
- hello world