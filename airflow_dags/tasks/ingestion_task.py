import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/opt/airflow/project")


def run_ingestion():
    cmd = ["python", "-m", "dvc", "repro", "data_ingestion", "-v"]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True
    )

    print("---- DVC STDOUT ----")
    print(result.stdout)
    print("---- DVC STDERR ----")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"DVC failed with exit code {result.returncode}")