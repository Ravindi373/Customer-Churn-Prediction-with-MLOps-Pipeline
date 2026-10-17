from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from tasks.ingestion_task import run_ingestion
from tasks.validation_task import run_validation
from tasks.feature_engineering_task import run_feature_engineering
from tasks.training_task import run_training
from tasks.evaluation_task import run_evaluation
from tasks.registration_task import run_registration

with DAG(
    dag_id="churn_mlops_end_to_end",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "churn"],
) as dag:

    t1 = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_ingestion,
    )

    t2 = PythonOperator(
        task_id="data_validation",
        python_callable=run_validation,
    )

    t3 = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering,
    )

    t4 = PythonOperator(
        task_id="model_training",
        python_callable=run_training,
    )

    t5 = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_evaluation,
    )

    t6 = PythonOperator(
        task_id="model_registration",
        python_callable=run_registration,
    )

    # Proper dependencies
    t1 >> t2 >> t3 >> t4 >> t5 >> t6