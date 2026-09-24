"""Temporal-context helpers for forecasters that declare required_history_length.

Contract for a forecaster with required_history_length == k:
  predict(X) returns len(X) values; the first k, and only the first k, are NaN
  (those rows have no full window). Everything here is pure: no model, no MLflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=10)


def take_context(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    k: int,
    timestamp_column: str = "date",
    step: pd.Timedelta = STEP,
) -> pd.DataFrame:
    """Last k rows of the immediately preceding partition, verified contiguous with `current`.

    previous=None means there is no preceding partition (train): returns an empty frame.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if previous is None:
        return current.iloc[0:0]
    if k > len(previous):
        raise ValueError(
            f"need {k} context rows but previous partition has {len(previous)}"
        )

    # len(previous) - k, not -k: previous.iloc[-0:] would return the whole frame.
    context = previous.iloc[len(previous) - k :]
    if len(context):
        ts = pd.concat([context[timestamp_column], current[timestamp_column].iloc[:1]])
        if not (ts.diff().dropna() == step).all():
            raise ValueError(
                "context is not contiguous with the current partition "
                f"(context ends {context[timestamp_column].iloc[-1]}, "
                f"partition starts {current[timestamp_column].iloc[0]})"
            )
    return context


def audit_leading_nans(predictions: np.ndarray, k: int) -> None:
    """Raise unless NaNs sit in exactly the first k positions of `predictions`."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 1:
        raise ValueError(f"predictions must be 1-D, got shape {predictions.shape}")
    expected = np.zeros(len(predictions), dtype=bool)
    expected[:k] = True
    actual = np.isnan(predictions)
    if not np.array_equal(actual, expected):
        bad = np.flatnonzero(actual != expected)
        raise ValueError(
            f"NaN pattern violates the k={k} contract at positions {bad[:10].tolist()}"
        )


def drop_context_predictions(predictions: np.ndarray, n_context: int) -> np.ndarray:
    """Discard the predictions that correspond to borrowed context rows."""
    return np.asarray(predictions)[n_context:]
