import mlflow
import mlflow.sklearn
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

MLFLOW_TRACKING_DIR = os.path.join(os.path.dirname(__file__), "../../data/mlruns")
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_TRACKING_DIR)}")

EXPERIMENT_NAME = "time-series-forecasting"

def get_or_create_experiment(experiment_name: str = EXPERIMENT_NAME) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id

def start_training_run(dataset_name: str, model_type: str, hyperparameters: Dict[str, Any], run_name: Optional[str] = None) -> str:
    experiment_id = get_or_create_experiment()
    if run_name is None:
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_param("model_type", model_type)
    for key, value in hyperparameters.items():
        try:
            mlflow.log_param(key, value)
        except:
            mlflow.log_param(key, str(value))
    return run.info.run_id

def log_metrics(metrics: Dict[str, float]):
    for key, value in metrics.items():
        if value is not None and not (isinstance(value, float) and (value != value)):
            try:
                mlflow.log_metric(key, float(value))
            except:
                pass

def log_model(model: Any, model_name: str, model_type: str):
    try:
        if model_type in ["xgboost", "lightgbm"]:
            mlflow.sklearn.log_model(model, model_name)
        else:
            import joblib
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                joblib.dump(model, f.name)
                mlflow.log_artifact(f.name, model_name)
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")

def end_run(status: str = "FINISHED"):
    try:
        mlflow.end_run(status=status)
    except:
        pass

def get_all_runs(experiment_name: str = EXPERIMENT_NAME) -> list:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    return runs.to_dict('records') if not runs.empty else []
