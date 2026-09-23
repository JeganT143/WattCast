"""
Single-window evaluation harness — the Strategy-pattern consumer of
Forecaster. Fits once on train, predicts on train/val/test, scores with
the shared metrics module. Never branches on concrete model type.

Walk-forward validation (Phase 5) is a separate wrapper that calls this
function repeatedly across multiple windows — not implemented here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, mape, rmse
from src.models.forecaster import Forecaster

_PARTITIONS = ("train", "val", "test")
_METRIC_KEYS = ("mae", "rmse", "mape")


@dataclass
class EvaluationResult:
    """
    Single-window evaluation output for one (model, horizon) pair.

    __post_init__ enforces the harness's structural contract: metrics,
    predictions, and actuals must each have exactly the train/val/test
    keys, metrics must have exactly the mae/rmse/mape keys, and
    predictions/actuals must be 1-D and shape-matched per partition —
    this is what lets downstream consumers (e.g. mlflow_logger) trust
    the shape without re-validating it themselves.
    """

    model_name: str
    horizon: int
    metrics: dict[str, dict[str, float]]
    predictions: dict[str, np.ndarray]
    actuals: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        expected = set(_PARTITIONS)

        if set(self.metrics) != expected:
            raise ValueError(f"metrics must have exactly keys {expected}")
        if set(self.predictions) != expected:
            raise ValueError(f"predictions must have exactly keys {expected}")
        if set(self.actuals) != expected:
            raise ValueError(f"actuals must have exactly keys {expected}")

        expected_metric_keys = set(_METRIC_KEYS)
        for partition, m in self.metrics.items():
            if set(m) != expected_metric_keys:
                raise ValueError(
                    f"metrics['{partition}'] must have exactly keys {expected_metric_keys}"
                )

        for partition in _PARTITIONS:
            preds = self.predictions[partition]
            actuals = self.actuals[partition]
            if preds.ndim != 1 or actuals.ndim != 1:
                raise ValueError(f"{partition} predictions/actuals must be 1-D")
            if preds.shape != actuals.shape:
                raise ValueError(f"{partition} predictions/actuals shape mismatch")


def evaluate_forecaster(
    forecaster: Forecaster,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    model_name: str,
    horizon: int,
    mape_threshold: float,
) -> EvaluationResult:
    """
    Fit `forecaster` once on train_df, predict on train/val/test, and
    score each partition with the shared metrics module.

    Column selection is generic: X is built from
    train_df/val_df/test_df[forecaster.required_columns], so this
    function never branches on concrete Forecaster type. Returns a
    single validated EvaluationResult (see its __post_init__).
    """
    cols = forecaster.required_columns

    X_train, y_train = train_df[cols].to_numpy(), train_df[target_column].to_numpy()
    X_val, y_val = val_df[cols].to_numpy(), val_df[target_column].to_numpy()
    X_test, y_test = test_df[cols].to_numpy(), test_df[target_column].to_numpy()

    forecaster.fit(X_train, y_train)

    preds = {
        "train": forecaster.predict(X_train),
        "val": forecaster.predict(X_val),
        "test": forecaster.predict(X_test),
    }
    actuals = {"train": y_train, "val": y_val, "test": y_test}

    metrics = {
        partition: {
            "mae": mae(actuals[partition], preds[partition]),
            "rmse": rmse(actuals[partition], preds[partition]),
            "mape": mape(
                actuals[partition], preds[partition], threshold=mape_threshold
            ),
        }
        for partition in _PARTITIONS
    }

    return EvaluationResult(
        model_name=model_name,
        horizon=horizon,
        metrics=metrics,
        predictions=preds,
        actuals=actuals,
    )
