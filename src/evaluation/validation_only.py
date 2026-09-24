"""Validation-only evaluation of a Forecaster. It receives only training and validation data,
has no parameter for any other partition, and performs no file access (callers load the frames).
Its scoring must stay equivalent to the three-partition evaluation used for the baselines: the
same label checks, the same eligible-row fit, the same context rule, the same array layout, the
same metric functions in the same argument order.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.context import audit_leading_nans, drop_context_predictions, take_context
from src.evaluation.metrics import mae, rmse
from src.models.forecaster import Forecaster


@dataclass
class ValidationResult:
    metrics: dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    n_evaluated: int
    n_warmup_rows: int
    n_ineligible_label_rows: int


def evaluate_on_validation(
    forecaster: Forecaster, train_df: pd.DataFrame, val_df: pd.DataFrame, target_column: str
) -> ValidationResult:
    cols = forecaster.required_columns
    y_train = train_df[target_column].to_numpy()
    y_val = val_df[target_column].to_numpy()

    if not np.isfinite(y_val).all():
        raise ValueError("val contains non-finite labels; only train may contain purged (NaN) labels")

    ineligible = ~np.isfinite(y_train)
    n_inel = int(ineligible.sum())
    n_train = len(ineligible)
    if n_inel > 0 and not ineligible[n_train - n_inel :].all():
        raise ValueError(
            "non-finite train labels must form a trailing block; a non-trailing gap would open a hole inside sequence windows"
        )

    k = int(forecaster.required_history_length)
    if k + n_inel >= n_train:
        raise ValueError("warm-up rows and label-ineligible rows overlap or leave no train rows to score")

    m = n_train - n_inel
    forecaster.fit(train_df[cols].to_numpy()[:m], y_train[:m])

    # the preceding rows come from the full training frame, including the label-ineligible tail:
    # its feature values remain valid history, and excluding them would open a gap that the
    # contiguity check below rejects
    context = take_context(train_df, val_df, k)

    X = np.concatenate([context[cols].to_numpy(), val_df[cols].to_numpy()], axis=0)
    raw = np.asarray(forecaster.predict(X))
    if raw.shape != (len(X),):
        raise ValueError(f"predict must return shape ({len(X)},), got {raw.shape}")
    audit_leading_nans(raw, k)
    preds = drop_context_predictions(raw, len(context))

    valid = ~np.isnan(preds) & np.isfinite(y_val)
    n_evaluated = int(valid.sum())
    if n_evaluated != len(val_df):
        raise ValueError(
            f"evaluated {n_evaluated} of {len(val_df)} validation rows; every validation row must be scored"
        )
    predictions = preds[valid]
    actuals = y_val[valid]

    return ValidationResult(
        metrics={"mae": mae(actuals, predictions), "rmse": rmse(actuals, predictions)},
        predictions=predictions,
        actuals=actuals,
        n_evaluated=n_evaluated,
        n_warmup_rows=k,
        n_ineligible_label_rows=n_inel,
    )
