"""
Evaluation metrics — pure functions, no dependency on Forecaster,
DataFrames, or partitions. Shared identically across every model in
the harness, so the leaderboard is never subject to per-model scoring
asymmetries.
"""

import numpy as np


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must both be 1-D arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.shape[0] == 0:
        raise ValueError("y_true/y_pred must not be empty")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Mean Absolute Percentage Error, computed only over observations where
    |y_true| >= threshold. Ordinary MAPE is unstable/undefined near zero,
    which occurs in this dataset whenever appliances are idle (see Phase 2/3
    decisions log). threshold has no default — callers must supply a
    deliberately chosen, documented value.
    """
    _validate_inputs(y_true, y_pred)
    mask = np.abs(y_true) >= threshold
    if not mask.any():
        raise ValueError(
            f"No observations satisfy |y_true| >= {threshold}; cannot compute MAPE."
        )
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
