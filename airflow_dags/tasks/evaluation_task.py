import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/opt/airflow/project")


def run_evaluation():
    cmd = ["python", "-m", "dvc", "repro", "evaluate"]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)