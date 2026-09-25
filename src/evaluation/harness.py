"""
Single-window evaluation harness — the Strategy-pattern consumer of
Forecaster. Fits once on train, predicts on train/val/test, scores with
the shared metrics module. Never branches on concrete model type.

Walk-forward validation (Phase 5) is a separate wrapper that calls this
function repeatedly across multiple windows — not implemented here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.purge import trailing_ineligible_count
from src.evaluation.context import (
    audit_leading_nans,
    drop_context_predictions,
    take_context,
)
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
    n_warmup_rows: int = 0
    n_ineligible_label_rows: int = 0

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

    @property
    def n_evaluated(self) -> dict[str, int]:
        """Rows actually scored per partition. predictions/actuals are already filtered."""
        return {p: int(len(self.predictions[p])) for p in _PARTITIONS}


def _predict_partition(
    forecaster: Forecaster,
    cols: list[str],
    previous_df: pd.DataFrame | None,
    current_df: pd.DataFrame,
) -> np.ndarray:
    """One prediction per row of current_df; NaN only where the model has no full window.

    Stage 1 audits the model on the array it actually received (context + current):
    exactly the first k positions must be NaN. Stage 2 drops the predictions that
    belong to borrowed context rows. What remains is k - len(context) NaNs:
    k for train (no context), 0 for val/test (context supplies all k rows), 0 for k = 0.
    """
    k = forecaster.required_history_length
    context = take_context(previous_df, current_df, k)  # raises if not contiguous
    X = np.concatenate([context[cols].to_numpy(), current_df[cols].to_numpy()], axis=0)

    raw = np.asarray(forecaster.predict(X))
    if raw.shape != (len(X),):
        raise ValueError(f"predict must return shape ({len(X)},), got {raw.shape}")
    audit_leading_nans(raw, k)
    return drop_context_predictions(raw, len(context))


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
    function never branches on concrete Forecaster type. val/test labels
    must be finite; train may carry a trailing block of non-finite
    (purged) labels, which is excluded from both fit and train scoring
    but still predicted over as context for later windows. Returns a
    single validated EvaluationResult (see its __post_init__).
    """
    cols = forecaster.required_columns

    y = {
        "train": train_df[target_column].to_numpy(),
        "val": val_df[target_column].to_numpy(),
        "test": test_df[target_column].to_numpy(),
    }

    for partition in ("val", "test"):
        if not np.isfinite(y[partition]).all():
            raise ValueError(
                f"{partition} contains non-finite labels; only train may contain purged (NaN) labels"
            )

    n_inel = trailing_ineligible_count(y["train"])
    n_train = len(y["train"])

    k = forecaster.required_history_length
    if k + n_inel >= n_train:
        raise ValueError(
            "warm-up rows and label-ineligible rows overlap or leave no train rows to score"
        )

    m = n_train - n_inel
    forecaster.fit(train_df[cols].to_numpy()[:m], y["train"][:m])

    # val borrows the tail of train; test borrows the tail of val; train has no predecessor.
    previous = {"train": None, "val": train_df, "test": val_df}
    current = {"train": train_df, "val": val_df, "test": test_df}

    preds, actuals = {}, {}
    for partition in _PARTITIONS:
        p = _predict_partition(
            forecaster, cols, previous[partition], current[partition]
        )
        valid = ~np.isnan(p) & np.isfinite(y[partition])
        preds[partition] = p[valid]
        actuals[partition] = y[partition][valid]

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
        n_warmup_rows=k,
        n_ineligible_label_rows=n_inel,
    )
