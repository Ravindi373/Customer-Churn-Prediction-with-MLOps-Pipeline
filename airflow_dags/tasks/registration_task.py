from pathlib import Path
import json
import time

PROJECT_ROOT = Path("/opt/airflow/project")
REGISTRY_PATH = PROJECT_ROOT / "models" / "registry.json"


def run_registration():
    model_path = PROJECT_ROOT / "models" / "best_model.pkl"
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError("best_model.pkl not found. Train first.")
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json not found. Evaluate first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    record = {
        "model_file": str(model_path),
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "note": "Registered by Airflow DAG"
    }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Model registered -> models/registry.json")