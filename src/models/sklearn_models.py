"""
Adapter wrappers around sklearn regressors, implementing the Forecaster
contract. Both are thin: fit/predict delegate directly to the sklearn
estimator, with no scaling or feature engineering of their own — that
already happened in the Phase 2 pipeline (SCALED_COLUMNS was fit-on-train
and applied before X ever reaches these wrappers). Re-scaling here would
be redundant and would risk the exact train/serve skew that
transform_with_scaler was designed to prevent.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config.features import FEATURE_COLUMNS
from src.models.forecaster import Forecaster


class LinearRegressionForecaster(Forecaster):
    """Forecaster adapter wrapping sklearn's LinearRegression over FEATURE_COLUMNS."""

    def __init__(self, fit_intercept: bool = True) -> None:
        self._fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)

    @property
    def required_columns(self) -> list[str]:
        return FEATURE_COLUMNS

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionForecaster":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.asarray(preds).reshape(-1).astype(float)

    @property
    def params(self) -> dict[str, Any]:
        return {"fit_intercept": self._fit_intercept}


class RandomForestForecaster(Forecaster):
    """Forecaster adapter wrapping sklearn's RandomForestRegressor over FEATURE_COLUMNS."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    @property
    def required_columns(self) -> list[str]:
        return FEATURE_COLUMNS

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestForecaster":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.asarray(preds).reshape(-1).astype(float)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "random_state": self._random_state,
        }
