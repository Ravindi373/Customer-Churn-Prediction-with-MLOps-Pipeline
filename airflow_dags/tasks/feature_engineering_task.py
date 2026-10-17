import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/opt/airflow/project")


def run_feature_engineering():
    cmd = ["python", "-m", "dvc", "repro", "preprocessing"]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)