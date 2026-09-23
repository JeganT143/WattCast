"""
Naive baselines — the first concrete Forecaster implementations.

Both are deterministic: fit() is a legitimate no-op per the Forecaster
contract (fit() means "prepare for prediction using training data," not
"every implementation must learn parameters").
"""

from typing import Any

import numpy as np

from config.features import BASELINE_CONTEXT_LAGS, SEASONAL_PERIOD
from src.models.forecaster import Forecaster


class NaivePersistenceForecaster(Forecaster):
    """Predicts Appliances[t] held flat, for any forecast horizon."""

    @property
    def required_columns(self) -> list[str]:
        return ["Appliances"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaivePersistenceForecaster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0].astype(float)

    @property
    def params(self) -> dict[str, Any]:
        return {}


class NaiveSeasonalForecaster(Forecaster):
    """
    Predicts Appliances[t + h - SEASONAL_PERIOD] for forecast horizon h —
    "same time, one day ago." Requires the horizon-specific context lag
    column (e.g. Appliances_lag_143 for h=1, Appliances_lag_138 for h=6)
    to already exist in the assembled dataset.
    """

    def __init__(self, horizon: int) -> None:
        if horizon not in BASELINE_CONTEXT_LAGS:
            raise ValueError(
                f"Unsupported forecast horizon: {horizon}. "
                f"Configured horizons: {sorted(BASELINE_CONTEXT_LAGS)}"
            )
        self.horizon = horizon
        self._context_lag = BASELINE_CONTEXT_LAGS[horizon]

    @property
    def required_columns(self) -> list[str]:
        return [f"Appliances_lag_{self._context_lag}"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveSeasonalForecaster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0].astype(float)

    @property
    def params(self) -> dict[str, Any]:
        return {"horizon": self.horizon, "seasonal_period": SEASONAL_PERIOD}
