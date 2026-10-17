import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/opt/airflow/project")


def run_training():
    cmd = ["python", "-m", "dvc", "repro", "train"]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)