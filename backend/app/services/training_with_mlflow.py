"""Training service with MLflow integration."""
from app.core.mlflow_tracking import (
    start_training_run, log_metrics, log_model, 
    log_feature_importance, end_run
)


def train_with_mlflow(
    dataset_name: str,
    model_type: str,
    model: object,
    hyperparameters: dict,
    metrics: dict,
    feature_importance: dict = None
):
    """
    Wrapper to train and log to MLflow.
    Call this after training a model.
    """
    try:
        # Start MLflow run
        run_id = start_training_run(
            dataset_name=dataset_name,
            model_type=model_type,
            hyperparameters=hyperparameters
        )
        
        # Log metrics
        log_metrics(metrics)
        
        # Log model
        log_model(model, f"{model_type}_model", model_type)
        
        # Log feature importance if available
        if feature_importance:
            log_feature_importance(feature_importance)
        
        # End run
        end_run("FINISHED")
        
        return run_id
    except Exception as e:
        print(f"MLflow logging error: {e}")
        end_run("FAILED")
        return None
