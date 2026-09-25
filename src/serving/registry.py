"""MLflow model registry glue for the serving champion.

Registers a saved model bundle directory as run artifacts, creates the
registered model on first use, and points the "champion" alias at a
version — no MLflow stages, per current MLflow guidance. The serving
code depends only on load_champion and the Forecaster interface, never
on a concrete model_family or file layout.
"""

import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from config.mlflow_config import EXPERIMENT_NAME
from src.serving.bundle import ModelBundle, load_bundle

REGISTERED_MODEL_NAME = "wattcast_serving_lr_h6"
CHAMPION_ALIAS = "champion"

_ARTIFACT_PATH = "bundle"


def register_bundle(bundle_dir: Path, tracking_uri: str, tags: dict) -> tuple[str, int]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run(run_name=REGISTERED_MODEL_NAME) as run:
        mlflow.log_artifacts(str(bundle_dir), artifact_path=_ARTIFACT_PATH)
        run_id = run.info.run_id

    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    model_uri = f"runs:/{run_id}/{_ARTIFACT_PATH}"
    model_version = client.create_model_version(REGISTERED_MODEL_NAME, model_uri, run_id)

    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, key, str(value))

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, model_version.version)

    return run_id, int(model_version.version)


def load_champion(tracking_uri: str) -> ModelBundle:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = client.download_artifacts(model_version.run_id, _ARTIFACT_PATH, tmp_dir)
        return load_bundle(Path(local_dir))
