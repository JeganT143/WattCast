"""Tests for evaluate_forecaster() and EvaluationResult's structural validation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.harness import EvaluationResult, evaluate_forecaster
from src.models.forecaster import Forecaster
from src.models.naive import NaivePersistenceForecaster

VALID_METRICS = {
    "train": {"mae": 1.0, "rmse": 1.0, "mape": 1.0},
    "val": {"mae": 1.0, "rmse": 1.0, "mape": 1.0},
    "test": {"mae": 1.0, "rmse": 1.0, "mape": 1.0},
}


def _valid_arrays(n=3):
    return {
        "train": np.zeros(n),
        "val": np.zeros(n),
        "test": np.zeros(n),
    }


# ---------------------------------------------------------------------
# EvaluationResult invariants
# ---------------------------------------------------------------------


def test_evaluation_result_rejects_missing_metrics_partition():
    bad_metrics = {k: v for k, v in VALID_METRICS.items() if k != "test"}
    with pytest.raises(ValueError, match="metrics must have exactly keys"):
        EvaluationResult("m", 1, bad_metrics, _valid_arrays(), _valid_arrays())


def test_evaluation_result_rejects_missing_metric_key():
    bad_metrics = {**VALID_METRICS, "test": {"mae": 1.0, "rmse": 1.0}}  # missing mape
    with pytest.raises(ValueError, match="must have exactly keys"):
        EvaluationResult("m", 1, bad_metrics, _valid_arrays(), _valid_arrays())


def test_evaluation_result_rejects_missing_predictions_partition():
    bad_preds = {k: v for k, v in _valid_arrays().items() if k != "val"}
    with pytest.raises(ValueError, match="predictions must have exactly keys"):
        EvaluationResult("m", 1, VALID_METRICS, bad_preds, _valid_arrays())


def test_evaluation_result_rejects_missing_actuals_partition():
    bad_actuals = {k: v for k, v in _valid_arrays().items() if k != "train"}
    with pytest.raises(ValueError, match="actuals must have exactly keys"):
        EvaluationResult("m", 1, VALID_METRICS, _valid_arrays(), bad_actuals)


def test_evaluation_result_rejects_mismatched_shapes():
    preds = _valid_arrays(n=3)
    actuals = _valid_arrays(n=5)  # mismatched
    with pytest.raises(ValueError, match="shape mismatch"):
        EvaluationResult("m", 1, VALID_METRICS, preds, actuals)


def test_evaluation_result_rejects_non_1d_predictions():
    preds = _valid_arrays()
    preds["test"] = np.zeros((3, 1))  # the PyTorch-shape gotcha
    with pytest.raises(ValueError, match="must be 1-D"):
        EvaluationResult("m", 1, VALID_METRICS, preds, _valid_arrays())


# ---------------------------------------------------------------------
# evaluate_forecaster end-to-end
# ---------------------------------------------------------------------


def _make_partition(appliances_values, target_values):
    n = len(appliances_values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=n, freq="10min"),
            "Appliances": appliances_values,
            "target_t1": target_values,
        }
    )


def test_evaluate_forecaster_with_naive_persistence_matches_hand_calculation():
    train_df = _make_partition([50.0, 60.0], [55.0, 65.0])
    val_df = _make_partition([100.0], [110.0])
    test_df = _make_partition([200.0], [190.0])

    result = evaluate_forecaster(
        forecaster=NaivePersistenceForecaster(),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column="target_t1",
        model_name="naive_persistence",
        horizon=1,
        mape_threshold=1.0,
    )

    # persistence predicts Appliances[t] itself
    # val: pred=100, actual=110 -> MAE=10, RMSE=10, MAPE=|110-100|/110*100
    assert result.metrics["val"]["mae"] == pytest.approx(10.0)
    assert result.metrics["val"]["rmse"] == pytest.approx(10.0)
    assert result.metrics["val"]["mape"] == pytest.approx(abs(110 - 100) / 110 * 100)

    # test: pred=200, actual=190
    assert result.metrics["test"]["mae"] == pytest.approx(10.0)


# ---------------------------------------------------------------------
# fit() called exactly once (spy Forecaster)
# ---------------------------------------------------------------------


class _SpyForecaster(Forecaster):
    """Stateful stub that counts fit() calls to prove the harness never
    refits per-partition — a no-op forecaster like NaivePersistence
    can't reveal this bug behaviorally, since it has no state to corrupt."""

    def __init__(self):
        self.fit_call_count = 0

    @property
    def required_columns(self):
        return ["Appliances"]

    def fit(self, X, y):
        self.fit_call_count += 1
        return self

    def predict(self, X):
        return X[:, 0].astype(float)

    @property
    def params(self):
        return {}


def test_evaluate_forecaster_calls_fit_exactly_once():
    train_df = _make_partition([50.0, 60.0], [55.0, 65.0])
    val_df = _make_partition([100.0], [110.0])
    test_df = _make_partition([200.0], [190.0])

    spy = _SpyForecaster()
    evaluate_forecaster(
        forecaster=spy,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column="target_t1",
        model_name="spy",
        horizon=1,
        mape_threshold=1.0,
    )

    assert spy.fit_call_count == 1
