"""Tests for src.evaluation.metrics: mae, rmse, mape."""

import numpy as np
import pytest

from src.evaluation.metrics import mae, mape, rmse


def test_mae_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # |diffs| = [2, 2, 3] -> mean = 7/3
    assert mae(y_true, y_pred) == pytest.approx(7 / 3)


def test_rmse_penalizes_large_outlier_more_than_mae():
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([11.0, 11.0, 100.0])  # one huge miss

    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)

    # RMSE must be pulled up disproportionately by the outlier vs MAE
    assert rmse_val > mae_val
    # hand-computed: sqrt((1 + 1 + 8100) / 3)
    assert rmse_val == pytest.approx(np.sqrt(8102 / 3))


def test_mape_known_values_above_threshold():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # |errors %| = [10%, 10%] -> mean = 10.0
    assert mape(y_true, y_pred, threshold=1.0) == pytest.approx(10.0)


def test_mape_raises_when_no_observations_meet_threshold():
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="No observations satisfy"):
        mape(y_true, y_pred, threshold=1000.0)


def test_mape_excludes_subthreshold_rows():
    y_true = np.array([0.0, 100.0, 200.0])  # first row is near-zero
    y_pred = np.array([5.0, 110.0, 180.0])
    threshold = 1.0

    result = mape(y_true, y_pred, threshold=threshold)

    # expected: computed ONLY over rows 1 and 2
    expected = mape(y_true[1:], y_pred[1:], threshold=threshold)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("metric_fn", [mae, rmse])
def test_metrics_reject_shape_mismatch(metric_fn):
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        metric_fn(y_true, y_pred)


def test_mape_rejects_shape_mismatch():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        mape(y_true, y_pred, threshold=1.0)


@pytest.mark.parametrize("metric_fn", [mae, rmse])
def test_metrics_reject_empty_input(metric_fn):
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError, match="must not be empty"):
        metric_fn(y_true, y_pred)


def test_mape_rejects_empty_input():
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError, match="must not be empty"):
        mape(y_true, y_pred, threshold=1.0)
