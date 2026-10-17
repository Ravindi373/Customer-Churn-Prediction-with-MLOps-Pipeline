import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Processed test CSV path")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--outdir", required=True, help="Output directory for metrics and plots")
    parser.add_argument("--experiment", default="churn-mlops", help="MLflow experiment name")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test)
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in processed test.csv")

    y_test = df["Churn"].astype(int)
    X_test = df.drop(columns=["Churn"])

    # Load model
    model = joblib.load(args.model)

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # Save metrics JSON
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = outdir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC-AUC: {metrics["roc_auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = outdir / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # Log to MLflow
    mlflow.set_experiment(args.experiment)
    # Start a new run for evaluation (or we could try to find the existing one, 
    # but a separate 'Evaluation' run is also clean)
    with mlflow.start_run(run_name="Evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(outdir / "metrics.json"))

    print(f"✅ Evaluation complete. Metrics: {metrics}")
    print(f"✅ Plots saved to: {outdir}")

if __name__ == "__main__":
    main()
