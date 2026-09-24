"""
Forecaster interface — the common contract every WattCast model
implements: naive baselines, sklearn wrappers, and (later) PyTorch
wrappers. The evaluation harness depends only on this interface and
never branches on concrete model type (Strategy pattern).

Input contract (owned by the harness, not any individual forecaster):
    X = df[forecaster.required_columns].to_numpy()
    X[:, i] corresponds to required_columns[i]
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Forecaster(ABC):
    """
    Strategy interface every WattCast model implements — naive
    baselines, sklearn adapters, and future PyTorch wrappers.

    Concrete implementations own no scaling/feature-engineering logic;
    those already happened upstream in the Phase 2 pipeline. A
    forecaster only declares which already-prepared columns it needs
    (required_columns) and how to fit/predict on them.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Forecaster":
        """Train on X (n_samples, len(required_columns)) and y (n_samples,)."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions, shape (n_samples,) — never (n_samples, 1)."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Model configuration/hyperparameters only — no MLflow/DVC metadata."""
        ...

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Column names this forecaster needs, in the order X's columns must follow."""
        ...

    @property
    def required_history_length(self) -> int:
        """Number of preceding rows this model needs as temporal context (default 0).

        Postcondition for k > 0: predict(X) still returns len(X) values, but the first k
        may be NaN (rows with no full window). A model declaring 0 must never return NaN.
        Sequence models must derive this from their own sequence length, never hardcode it.
        """
        return 0
