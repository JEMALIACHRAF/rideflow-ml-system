"""
Airflow DAG — Weekly model retraining pipeline.

Schedule: every Monday at 02:00.
Tasks:
    1. data_backfill      — generate / refresh last 7 days of data
    2. feature_pipeline   — rebuild feature store
    3. drift_check        — abort if drift is negligible (save compute)
    4. train_models       — run train_all.py with HPO disabled
    5. evaluate_champion  — compare new model vs production champion
    6. promote_model      — register new model in MLflow if better
    7. notify             — send Slack/email summary
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from loguru import logger

# ── DAG default args ──────────────────────────────────────────────────────────

default_args = {
    "owner":            "mlops",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ── Task functions ────────────────────────────────────────────────────────────

def task_data_backfill(**ctx):
    """Generate or refresh last 7 days of synthetic / real data."""
    import sys, os
    sys.path.insert(0, "/app")
    from producers.gps_producer import generate
    from config import settings
    logger.info("Backfilling last 7 days of data...")
    generate(n_days=7, output=settings.RAW_DIR)
    logger.success("Data backfill complete")


def task_feature_pipeline(**ctx):
    """Rebuild temporal and geospatial features for the full dataset."""
    import sys, pandas as pd
    sys.path.insert(0, "/app")
    from config import settings
    from features.temporal import build_temporal_features
    from features.geospatial import build_geospatial_features

    raw_path = settings.RAW_DIR / "rides_synthetic.parquet"
    df = pd.read_parquet(raw_path)
    df = df.sort_values(["zone", "timestamp"])
    df = build_temporal_features(df, lags=[1, 2, 3, 6, 12, 24, 48, 168],
                                  rolling_windows=[3, 6, 12, 24])
    df = build_geospatial_features(df)
    df = df.dropna()
    out = settings.PROCESSED_DIR / "features_latest.parquet"
    df.to_parquet(out, index=False)
    ctx["ti"].xcom_push(key="n_rows", value=len(df))
    logger.success(f"Features built — {len(df):,} rows → {out}")


def task_drift_check(**ctx):
    """
    Check if production model has drifted enough to warrant retraining.
    Returns 'train' if drift detected, 'skip_training' otherwise.
    """
    import sys, pandas as pd, numpy as np
    sys.path.insert(0, "/app")
    from config import settings
    from monitoring.drift_detector import DriftMonitor

    feat_path = settings.PROCESSED_DIR / "features_latest.parquet"
    ref_path  = settings.PROCESSED_DIR / "features_reference.parquet"

    if not ref_path.exists():
        logger.info("No reference data — training unconditionally")
        return "train_models"

    current   = pd.read_parquet(feat_path).sample(1000, random_state=42)
    reference = pd.read_parquet(ref_path).sample(1000, random_state=42)

    monitor = DriftMonitor(reference, drift_threshold=0.2)
    result  = monitor.run(current)
    ctx["ti"].xcom_push(key="drift_result", value=str(result["recommendation"]))

    if result["alert"] or len(result["drifted_features"]) > 3:
        logger.info(f"Drift detected in {len(result['drifted_features'])} features → retrain")
        return "train_models"
    else:
        logger.info("No significant drift → skipping retraining")
        return "skip_training"


def task_train_models(**ctx):
    """Run the full training pipeline."""
    import sys, subprocess
    result = subprocess.run(
        ["python", "/app/models/train_all.py",
         "--experiment", f"rideflow-weekly-{ctx['ds']}"],
        capture_output=True, text=True, timeout=7200
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")
    logger.success("Training complete")
    logger.info(result.stdout[-2000:])  # last 2k chars


def task_evaluate_champion(**ctx):
    """
    Compare newly trained model against the current production champion.
    Pushes 'promote' if new model is better by >1% MAPE.
    """
    import sys, pandas as pd
    sys.path.insert(0, "/app")
    import mlflow
    from config import settings

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # Get current production model metrics
    try:
        prod_versions = client.get_latest_versions("rideflow-demand", stages=["Production"])
        if not prod_versions:
            ctx["ti"].xcom_push(key="promote", value=True)
            return
        prod_run = client.get_run(prod_versions[0].run_id)
        prod_mape = float(prod_run.data.metrics.get("mape", 999))
    except Exception:
        ctx["ti"].xcom_push(key="promote", value=True)
        return

    # Get latest trained model metrics
    experiment = mlflow.get_experiment_by_name(f"rideflow-weekly-{ctx['ds']}")
    if not experiment:
        ctx["ti"].xcom_push(key="promote", value=False)
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                               order_by=["metrics.mape ASC"], max_results=1)
    if runs.empty:
        ctx["ti"].xcom_push(key="promote", value=False)
        return

    new_mape = float(runs.iloc[0].get("metrics.mape", 999))
    improvement = (prod_mape - new_mape) / prod_mape
    logger.info(f"Champion MAPE: {prod_mape:.4f} | Challenger MAPE: {new_mape:.4f} "
                f"| Improvement: {improvement:.2%}")
    ctx["ti"].xcom_push(key="promote", value=improvement > 0.01)
    ctx["ti"].xcom_push(key="new_mape", value=new_mape)
    ctx["ti"].xcom_push(key="improvement", value=improvement)


def task_promote_model(**ctx):
    """Register challenger as new Production model in MLflow."""
    import sys
    sys.path.insert(0, "/app")
    import mlflow
    from config import settings

    promote = ctx["ti"].xcom_pull(key="promote", task_ids="evaluate_champion")
    if not promote:
        logger.info("New model not better — keeping current champion")
        return

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    # Get latest version and transition to Production
    versions = client.get_latest_versions("rideflow-demand", stages=["Staging"])
    if versions:
        client.transition_model_version_stage(
            name="rideflow-demand",
            version=versions[0].version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.success(f"Promoted model v{versions[0].version} to Production")


def task_notify(**ctx):
    """Log training summary (extend to Slack/email as needed)."""
    promote    = ctx["ti"].xcom_pull(key="promote",     task_ids="evaluate_champion") or False
    new_mape   = ctx["ti"].xcom_pull(key="new_mape",    task_ids="evaluate_champion") or "N/A"
    improvement = ctx["ti"].xcom_pull(key="improvement", task_ids="evaluate_champion") or 0
    drift_msg  = ctx["ti"].xcom_pull(key="drift_result", task_ids="drift_check") or "unknown"
    n_rows     = ctx["ti"].xcom_pull(key="n_rows",      task_ids="feature_pipeline") or 0

    summary = (
        f"\n{'='*50}\n"
        f"RideFlow Weekly Retrain — {ctx['ds']}\n"
        f"{'='*50}\n"
        f"Data rows processed : {n_rows:,}\n"
        f"Drift status        : {drift_msg}\n"
        f"New model MAPE      : {new_mape}\n"
        f"Improvement         : {improvement:.2%}\n"
        f"Model promoted      : {'YES ✅' if promote else 'NO — kept champion'}\n"
        f"{'='*50}"
    )
    logger.info(summary)


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="rideflow_weekly_retrain",
    default_args=default_args,
    description="Weekly RideFlow model retraining pipeline",
    schedule_interval="0 2 * * 1",  # Monday 02:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "rideflow", "training"],
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    data_backfill = PythonOperator(
        task_id="data_backfill",
        python_callable=task_data_backfill,
    )
    feature_pipeline = PythonOperator(
        task_id="feature_pipeline",
        python_callable=task_feature_pipeline,
    )
    drift_check = BranchPythonOperator(
        task_id="drift_check",
        python_callable=task_drift_check,
    )
    skip_training = EmptyOperator(task_id="skip_training")

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
    )
    evaluate_champion = PythonOperator(
        task_id="evaluate_champion",
        python_callable=task_evaluate_champion,
    )
    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
    )
    notify = PythonOperator(
        task_id="notify",
        python_callable=task_notify,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    (
        start
        >> data_backfill
        >> feature_pipeline
        >> drift_check
        >> [train_models, skip_training]
    )
    train_models >> evaluate_champion >> promote_model >> notify >> end
    skip_training >> notify >> end
