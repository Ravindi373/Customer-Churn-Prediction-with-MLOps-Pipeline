import argparse
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_and_log_model(model, name, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=name, nested=True):
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_prob)

        # Log params
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Save temporary artifact files
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        cm_path = reports_dir / f"{name}_confusion_matrix.png"
        roc_path = reports_dir / f"{name}_roc_curve.png"

        save_confusion_matrix(y_val, y_pred, cm_path)
        save_roc_curve(y_val, y_prob, roc_path)

        # Log artifacts
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))

        # Log model
        if name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{name} ROC-AUC: {roc_auc:.4f}")

        return roc_auc, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Processed train CSV path")
    parser.add_argument("--model_out", required=True, help="Output model path (.pkl)")
    parser.add_argument("--experiment", default="churn-mlops", help="MLflow experiment name")
    args = parser.parse_args()

    df = pd.read_csv(args.train)

    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in processed train.csv")

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment(args.experiment)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    best_score = -1
    best_model = None
    best_name = ""

    with mlflow.start_run(run_name="ModelSelection"):
        for name, model in models.items():
            score, trained_model = train_and_log_model(
                model, name, X_train, X_val, y_train, y_val
            )

            if score > best_score:
                best_score = score
                best_model = trained_model
                best_name = name

        mlflow.log_param("best_model_name", best_name)
        mlflow.log_metric("best_validation_roc_auc", best_score)

        out_path = Path(args.model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, out_path)

        print(f"\nBest Model: {best_name} with Validation ROC-AUC: {best_score:.4f}")
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()