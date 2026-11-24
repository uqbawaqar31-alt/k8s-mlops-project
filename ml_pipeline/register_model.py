import json
from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient


def main():
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    model_path = artifacts_dir / "iris_model.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    tracking_uri = f"sqlite:///{(base_dir / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-k8s-mlops")

    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text())

    registered_model_name = "iris-classifier"

    with mlflow.start_run(run_name="iris-logreg-run") as run:
        mlflow.log_param("model_type", metrics.get("model_type", "LogisticRegression"))
        mlflow.log_param("dataset", metrics.get("dataset", "iris"))
        mlflow.log_metric("accuracy", float(metrics["accuracy"]))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    latest_version = max(int(v.version) for v in versions)

    # Transition Staging -> Production
    client.transition_model_version_stage(
        name=registered_model_name,
        version=str(latest_version),
        stage="Staging",
        archive_existing_versions=False,
    )
    client.transition_model_version_stage(
        name=registered_model_name,
        version=str(latest_version),
        stage="Production",
        archive_existing_versions=True,
    )

    # Set alias for modern MLflow compatibility alongside stage transition
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="production",
        version=str(latest_version),
    )

    prod_versions = client.get_latest_versions(registered_model_name, stages=["Production"])
    print(f"Registered model '{registered_model_name}' version {latest_version} in run {run.info.run_id}")
    print("Current Production versions:", [(v.version, v.current_stage) for v in prod_versions])


if __name__ == "__main__":
    main()
