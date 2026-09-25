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

CHAMPION_ALIAS = "champion"

_ARTIFACT_PATH = "bundle"


# The already-registered LR champion (Stage 3, before this generalization)
# used the abbreviated name "wattcast_serving_lr_h6", not the full family
# key "linear_regression" — this mapping preserves that existing name so
# load_champion/register_bundle keep resolving to the exact same registered
# model. Families without an explicit abbreviation here use their full
# LOADERS-table key unabbreviated.
_MODEL_NAME_ABBREVIATIONS = {
    "linear_regression": "lr",
    "random_forest": "rf",
}


def registered_model_name(family: str) -> str:
    abbreviation = _MODEL_NAME_ABBREVIATIONS.get(family, family)
    return f"wattcast_serving_{abbreviation}_h6"


# Backward-compatible constant: scripts/train_final_lr.py (untouched, already
# registered LR) still imports this name directly. Its value is unchanged
# ("wattcast_serving_lr_h6") — it is now just registered_model_name("linear_regression").
REGISTERED_MODEL_NAME = registered_model_name("linear_regression")


def register_bundle(
    bundle_dir: Path, tracking_uri: str, tags: dict, family: str = "linear_regression"
) -> tuple[str, int]:
    model_name = registered_model_name(family)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_artifacts(str(bundle_dir), artifact_path=_ARTIFACT_PATH)
        run_id = run.info.run_id

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    model_uri = f"runs:/{run_id}/{_ARTIFACT_PATH}"
    model_version = client.create_model_version(model_name, model_uri, run_id)

    for key, value in tags.items():
        client.set_model_version_tag(model_name, model_version.version, key, str(value))

    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, model_version.version)

    return run_id, int(model_version.version)


def load_champion(tracking_uri: str, family: str) -> ModelBundle:
    model_name = registered_model_name(family)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = client.download_artifacts(model_version.run_id, _ARTIFACT_PATH, tmp_dir)
        return load_bundle(Path(local_dir))
