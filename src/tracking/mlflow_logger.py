"""
MLflow logging layer — a thin consumer of EvaluationResult, sitting
strictly outside the Forecaster interface, model wrappers, and
evaluate_forecaster() itself (see Phase 3 decisions log: MLflow
awareness never enters the core evaluation architecture).

One MLflow run per (model, horizon) pair. Metrics are flattened as
{partition}_{metric}. Model artifacts are logged only when the caller
explicitly supplies native_model (e.g. the underlying sklearn
estimator) — this file never branches on concrete Forecaster type.
"""

import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from config.mlflow_config import EXPERIMENT_NAME, TRACKING_URI
from src.evaluation.harness import EvaluationResult
from src.models.forecaster import Forecaster

_PARTITIONS = ("train", "val", "test")

# Model-internal types explicitly trusted for skops (de)serialization.
# Every model logged by this project is self-trained from our own
# pipeline on our own data — never loaded from an external/untrusted
# source — so this trust boundary is justified. If Phase 6's FastAPI
# serving layer is ever changed to load models from anywhere other
# than our own MLflow registry, this trust boundary must be revisited.
#
# sklearn.tree._tree.Tree: the shared node storage for tree-based
# models (DecisionTree*, RandomForest*, ExtraTrees*, GradientBoosting*).
# skops flags it by default because a maliciously crafted file could
# set out-of-bounds node indices, causing memory-unsafe behavior on
# load. Not a concern for models we generate ourselves.
_SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def _get_git_commit_sha() -> str:
    """
    Current Git commit SHA, for run traceability. Raises if git is
    unavailable or the repo has no commits yet — a run logged without
    a real commit reference would silently break "which code produced
    this result" traceability, so failing loudly here is preferred
    over logging a placeholder string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "Could not determine Git commit SHA for run traceability. "
            "Ensure this is run inside a Git repository with at least one commit."
        ) from e


def log_evaluation_result(
    result: EvaluationResult,
    forecaster: Forecaster,
    mape_threshold: float,
    scaler_path: str,
    native_model: Any | None = None,
    tracking_uri: str | None = None,
) -> str:
    """
    Log one EvaluationResult (single model, single horizon, single
    window) to MLflow. Returns the created run_id.

    scaler_path must point to an existing file — the scaler that
    produced the features this evaluation used. native_model, if
    provided, is logged via mlflow.sklearn.log_model(); if None, no
    model artifact is logged (e.g. naive baselines with no learned
    state). This function never inspects forecaster's concrete type.

    tracking_uri overrides the canonical config for testing —
    production callers omit it and use TRACKING_URI from
    config/mlflow_config.py.
    """
    scaler_file = Path(scaler_path)
    if not scaler_file.exists():
        raise FileNotFoundError(f"scaler_path does not exist: {scaler_path}")

    mlflow.set_tracking_uri(tracking_uri or TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"{result.model_name}_h{result.horizon}"

    with mlflow.start_run(run_name=run_name) as run:
        # tags — traceability, not model configuration
        mlflow.set_tags(
            {
                "model_name": result.model_name,
                "horizon": str(result.horizon),
                "git_commit_sha": _get_git_commit_sha(),
            }
        )

        # params — model config (from forecaster.params) + structural params
        mlflow.log_params(forecaster.params)
        mlflow.log_param("mape_threshold", mape_threshold)
        mlflow.log_param("required_columns", ",".join(forecaster.required_columns))

        # metrics — flattened {partition}_{metric}, straight from EvaluationResult
        for partition in _PARTITIONS:
            for metric_name, value in result.metrics[partition].items():
                mlflow.log_metric(f"{partition}_{metric_name}", value)

        # predictions/actuals artifact — no recomputation, straight from EvaluationResult
        predictions_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "partition": partition,
                        "prediction": result.predictions[partition],
                        "actual": result.actuals[partition],
                    }
                )
                for partition in _PARTITIONS
            ],
            ignore_index=True,
        )
        mlflow.log_table(predictions_df, artifact_file="predictions.json")

        # scaler artifact — tied to this specific run, per Phase 2's
        # open item (model version and scaler version must never drift apart)
        mlflow.log_artifact(str(scaler_file), artifact_path="scaler")

        # native model artifact — only if the caller explicitly provided one
        if native_model is not None:
            mlflow.sklearn.log_model(
                native_model,
                name="model",
                skops_trusted_types=_SKOPS_TRUSTED_TYPES,
            )

        return run.info.run_id
