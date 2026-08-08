import json
import os
from datetime import datetime, timezone

from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

from airflow import DAG

default_args = {"owner": "mlops", "retries": 1}

DRIFT_REPORT_PATH = "/opt/airflow/monitoring/drift_result.json"


def drift_detected():
    if not os.path.exists(DRIFT_REPORT_PATH):
        return False
    with open(DRIFT_REPORT_PATH) as f:
        result = json.load(f)
    return result.get("drift_detected", False)


with DAG(
    dag_id="retrain_dag",
    default_args=default_args,
    description="Check drift report and retrain models if drift is detected",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["retrain", "drift", "mlflow"],
) as dag:

    check_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=drift_detected,
    )

    run_drift_report = BashOperator(
        task_id="run_drift_report",
        bash_command="python /opt/airflow/monitoring/drift_report.py",
        cwd="/opt/airflow",
    )

    retrain_logreg = BashOperator(
        task_id="retrain_logreg",
        bash_command="python /opt/airflow/training/train_logreg.py",
        cwd="/opt/airflow",
    )

    retrain_distilbert = BashOperator(
        task_id="retrain_distilbert",
        bash_command="python /opt/airflow/training/train_distilbert.py",
        cwd="/opt/airflow",
    )

    run_drift_report >> check_drift >> [retrain_logreg, retrain_distilbert]
