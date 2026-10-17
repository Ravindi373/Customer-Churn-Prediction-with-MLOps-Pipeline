# Airflow on Windows (Recommended: Docker)

Installing Apache Airflow directly on Windows is unreliable.
The simplest approach is to run Airflow with Docker Desktop.

## What to do
1) Ensure Docker Desktop is installed and running.
2) Use the provided `docker-compose.airflow.yml` file.
3) Put your DAG(s) in `airflow_dags/` (already in this repo).
4) Start Airflow:
   ```bash
   docker compose -f airflow/docker-compose.airflow.yml up -d
   ```
5) Open Airflow UI:
   - http://localhost:8080
   Default login:
   - user: airflow
   - pass: airflow

## Goal
Your DAG should run end-to-end and call scripts in `src/` (preprocessing, train, evaluate).
