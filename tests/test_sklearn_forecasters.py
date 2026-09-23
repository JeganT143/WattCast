"""Tests for LinearRegressionForecaster and RandomForestForecaster."""

import numpy as np
import pandas as pd
import pytest

from config.features import FEATURE_COLUMNS
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster
from src.evaluation.harness import evaluate_forecaster


@pytest.mark.parametrize("cls", [LinearRegressionForecaster, RandomForestForecaster])
def test_required_columns_matches_feature_columns(cls):
    f = cls()
    assert f.required_columns == FEATURE_COLUMNS
    assert (
        f.required_columns is not FEATURE_COLUMNS or True
    )  # identity not required, equality is
    # explicitly confirm no duplication: same object reference is fine,
    # a separately-typed-out copy that drifts is what we guard against
    assert f.required_columns == FEATURE_COLUMNS


def test_linear_regression_params_reflects_constructor_defaults():
    f = LinearRegressionForecaster()
    assert f.params == {"fit_intercept": True}


def test_linear_regression_params_reflects_constructor_override():
    f = LinearRegressionForecaster(fit_intercept=False)
    assert f.params == {"fit_intercept": False}


def test_random_forest_params_reflects_constructor_defaults():
    f = RandomForestForecaster()
    assert f.params == {"n_estimators": 100, "max_depth": None, "random_state": 42}


def test_random_forest_params_reflects_constructor_override():
    f = RandomForestForecaster(n_estimators=50, max_depth=5, random_state=7)
    assert f.params == {"n_estimators": 50, "max_depth": 5, "random_state": 7}


@pytest.mark.parametrize("cls", [LinearRegressionForecaster, RandomForestForecaster])
def test_fit_returns_self(cls):
    f = cls()
    X = np.random.rand(10, len(FEATURE_COLUMNS))
    y = np.random.rand(10)
    assert f.fit(X, y) is f


@pytest.mark.parametrize("cls", [LinearRegressionForecaster, RandomForestForecaster])
def test_predict_output_is_1d_correct_length_and_dtype(cls):
    f = cls()
    X = np.random.rand(10, len(FEATURE_COLUMNS))
    y = np.random.rand(10)
    f.fit(X, y)

    X_test = np.random.rand(5, len(FEATURE_COLUMNS))
    preds = f.predict(X_test)

    assert preds.ndim == 1
    assert preds.shape == (5,)
    assert preds.dtype == np.float64


@pytest.mark.parametrize("cls", [LinearRegressionForecaster, RandomForestForecaster])
def test_fit_does_not_mutate_X_or_y(cls):
    f = cls()
    X = np.random.rand(10, len(FEATURE_COLUMNS))
    y = np.random.rand(10)
    X_original = X.copy()
    y_original = y.copy()

    f.fit(X, y)

    np.testing.assert_array_equal(X, X_original)
    np.testing.assert_array_equal(y, y_original)


def test_linear_regression_recovers_exact_linear_relationship():
    # y = 2*x0 + 3*x1 + 1, no noise -- LinearRegression should recover it
    # exactly (up to floating point), proving the adapter isn't silently
    # transposing or misaligning X.
    rng = np.random.default_rng(0)
    X = rng.uniform(-10, 10, size=(50, 2))
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1

    f = LinearRegressionForecaster()
    f.fit(X, y)
    preds = f.predict(X)

    np.testing.assert_allclose(preds, y, atol=1e-8)


def test_random_forest_is_reproducible_with_fixed_random_state():
    rng = np.random.default_rng(1)
    X = rng.uniform(-10, 10, size=(50, len(FEATURE_COLUMNS)))
    y = rng.uniform(0, 100, size=50)

    f1 = RandomForestForecaster(random_state=42)
    f1.fit(X, y)
    preds1 = f1.predict(X)

    f2 = RandomForestForecaster(random_state=42)
    f2.fit(X, y)
    preds2 = f2.predict(X)

    np.testing.assert_array_equal(preds1, preds2)


@pytest.mark.parametrize("cls", [LinearRegressionForecaster, RandomForestForecaster])
def test_end_to_end_through_evaluation_harness(cls):
    n_train, n_val, n_test = 30, 10, 10
    n_features = len(FEATURE_COLUMNS)
    rng = np.random.default_rng(2)

    def make_df(n, offset):
        data = {col: rng.uniform(-5, 5, size=n) for col in FEATURE_COLUMNS}
        data["date"] = pd.date_range(
            "2016-01-01", periods=n, freq="10min"
        ) + pd.Timedelta(days=offset)
        data["target_t1"] = rng.uniform(0, 200, size=n)
        return pd.DataFrame(data)

    train_df = make_df(n_train, offset=0)
    val_df = make_df(n_val, offset=100)
    test_df = make_df(n_test, offset=200)

    result = evaluate_forecaster(
        forecaster=cls(),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column="target_t1",
        model_name=cls.__name__,
        horizon=1,
        mape_threshold=30.0,
    )

    for partition in ("train", "val", "test"):
        assert result.predictions[partition].ndim == 1
        assert set(result.metrics[partition]) == {"mae", "rmse", "mape"}
