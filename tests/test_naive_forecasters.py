"""Tests for NaivePersistenceForecaster and NaiveSeasonalForecaster."""

import numpy as np
import pandas as pd
import pytest

from src.models.naive import NaivePersistenceForecaster, NaiveSeasonalForecaster
from config.features import BASELINE_CONTEXT_LAGS

# ---------------------------------------------------------------------
# NaivePersistenceForecaster
# ---------------------------------------------------------------------


def test_persistence_required_columns():
    f = NaivePersistenceForecaster()
    assert f.required_columns == ["Appliances"]


def test_persistence_fit_is_noop_and_returns_self():
    f = NaivePersistenceForecaster()
    X = np.array([[10.0], [20.0]])
    y = np.array([15.0, 25.0])
    result = f.fit(X, y)
    assert result is f


def test_persistence_predict_returns_input_unchanged_as_1d_float():
    f = NaivePersistenceForecaster()
    X = np.array([[10.0], [20.0], [30.0]])
    preds = f.predict(X)

    assert preds.shape == (3,)
    assert preds.dtype == np.float64
    np.testing.assert_array_equal(preds, [10.0, 20.0, 30.0])


def test_persistence_params_is_empty():
    f = NaivePersistenceForecaster()
    assert f.params == {}


# ---------------------------------------------------------------------
# NaiveSeasonalForecaster
# ---------------------------------------------------------------------


def test_seasonal_required_columns_horizon_1():
    f = NaiveSeasonalForecaster(horizon=1)
    assert f.required_columns == ["Appliances_lag_143"]


def test_seasonal_required_columns_horizon_6():
    f = NaiveSeasonalForecaster(horizon=6)
    assert f.required_columns == ["Appliances_lag_138"]


def test_seasonal_rejects_unsupported_horizon():
    with pytest.raises(ValueError, match="Unsupported forecast horizon"):
        NaiveSeasonalForecaster(horizon=200)


def test_seasonal_fit_is_noop_and_returns_self():
    f = NaiveSeasonalForecaster(horizon=1)
    X = np.array([[50.0], [60.0]])
    y = np.array([55.0, 65.0])
    result = f.fit(X, y)
    assert result is f


def test_seasonal_predict_returns_input_unchanged_as_1d_float():
    f = NaiveSeasonalForecaster(horizon=6)
    X = np.array([[100.0], [200.0]])
    preds = f.predict(X)

    assert preds.shape == (2,)
    assert preds.dtype == np.float64
    np.testing.assert_array_equal(preds, [100.0, 200.0])


def test_seasonal_params_horizon_1():
    f = NaiveSeasonalForecaster(horizon=1)
    assert f.params == {"horizon": 1, "seasonal_period": 144}


def test_seasonal_params_horizon_6():
    f = NaiveSeasonalForecaster(horizon=6)
    assert f.params == {"horizon": 6, "seasonal_period": 144}


# ---------------------------------------------------------------------
# End-to-end: real assembled-dataset shape -> required_columns -> X -> predict
# ---------------------------------------------------------------------


def test_persistence_end_to_end_on_realistic_dataset_shape():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=4, freq="10min"),
            "Appliances": [50.0, 60.0, 70.0, 80.0],
            "target_t1": [60.0, 70.0, 80.0, 90.0],
        }
    )

    f = NaivePersistenceForecaster()
    X = df[f.required_columns].to_numpy()
    y = df["target_t1"].to_numpy()

    f.fit(X, y)
    preds = f.predict(X)

    # persistence prediction at t is simply Appliances[t]
    np.testing.assert_array_equal(preds, [50.0, 60.0, 70.0, 80.0])


def test_seasonal_end_to_end_on_realistic_dataset_shape():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=3, freq="10min"),
            "Appliances_lag_143": [111.0, 222.0, 333.0],
            "target_t1": [10.0, 20.0, 30.0],
        }
    )

    f = NaiveSeasonalForecaster(horizon=1)
    X = df[f.required_columns].to_numpy()
    y = df["target_t1"].to_numpy()

    f.fit(X, y)
    preds = f.predict(X)

    np.testing.assert_array_equal(preds, [111.0, 222.0, 333.0])
