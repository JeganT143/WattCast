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


# ---------------------------------------------------------------------
# Sequence-model contract: context, NaN audit, population semantics
# ---------------------------------------------------------------------


def _series(n):
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=n, freq="10min"),
            "Appliances": np.arange(n, dtype=float) * 10.0,
            "target_t1": np.arange(n, dtype=float) * 10.0 + 5.0,
        }
    )


def _split(n_train=10, n_val=6, n_test=5):
    df = _series(n_train + n_val + n_test)
    a, b = n_train, n_train + n_val
    return (
        df.iloc[:a].reset_index(drop=True),
        df.iloc[a:b].reset_index(drop=True),
        df.iloc[b:].reset_index(drop=True),
    )


class _KBack(Forecaster):
    """Sequence-style stub: prediction at row i is the value k rows back.
    The first `nan_rows` rows are NaN (defaults to k, the honest behavior)."""

    def __init__(self, k, nan_rows=None):
        self.k = k
        self.nan_rows = k if nan_rows is None else nan_rows

    @property
    def required_columns(self):
        return ["Appliances"]

    @property
    def required_history_length(self):
        return self.k

    def fit(self, X, y):
        return self

    def predict(self, X):
        out = np.zeros(len(X))
        out[self.k :] = X[: len(X) - self.k, 0]
        out[: self.nan_rows] = np.nan
        return out

    @property
    def params(self):
        return {}


def _run(forecaster, train_df, val_df, test_df):
    return evaluate_forecaster(
        forecaster,
        train_df,
        val_df,
        test_df,
        target_column="target_t1",
        model_name="stub",
        horizon=1,
        mape_threshold=1.0,
    )


def test_population_semantics_for_a_sequence_model():
    train, val, test = _split(10, 6, 5)
    result = _run(_KBack(3), train, val, test)
    # train loses k warm-up rows; val and test are scored on every row
    assert result.n_evaluated == {"train": 10 - 3, "val": 6, "test": 5}


def test_population_semantics_for_a_zero_history_model():
    train, val, test = _split(10, 6, 5)
    result = _run(_KBack(0), train, val, test)
    assert result.n_evaluated == {"train": 10, "val": 6, "test": 5}


def test_val_borrows_train_tail_and_test_borrows_val_tail():
    train, val, test = _split(10, 6, 5)  # Appliances = 10 * row index, 0..20
    result = _run(_KBack(3), train, val, test)
    # first val prediction = train row 7 (value 70), not anything from val itself
    assert result.predictions["val"][0] == 70.0
    # first test prediction = val row 3 (value 130). A stale train tail would give 70.
    assert result.predictions["test"][0] == 130.0
    # train warm-up filtered from the FRONT; actuals stay aligned to their rows
    assert result.predictions["train"][0] == 0.0
    assert result.actuals["train"][0] == 35.0  # target of row 3
    assert result.actuals["val"][0] == 105.0  # target of first val row


def test_model_declaring_zero_history_but_emitting_nan_is_rejected():
    train, val, test = _split()
    with pytest.raises(ValueError, match="violates"):
        _run(_KBack(k=0, nan_rows=3), train, val, test)


def test_model_with_wrong_nan_count_is_rejected():
    train, val, test = _split()
    with pytest.raises(ValueError, match="violates"):
        _run(_KBack(k=3, nan_rows=2), train, val, test)


def test_non_contiguous_context_is_rejected():
    df = _series(21)
    train, val, test = df.iloc[:10], df.iloc[12:18], df.iloc[18:]  # rows 10-11 missing
    with pytest.raises(ValueError, match="not contiguous"):
        _run(_KBack(3), train, val, test)


# ---------------------------------------------------------------------
# Label eligibility: non-finite (purged) train labels vs warm-up rows
# ---------------------------------------------------------------------


class _RecordingKBack(_KBack):
    def fit(self, X, y):
        self.fit_X, self.fit_y = X, y
        return self


def _with_ineligible_tail(train, n):
    train = train.copy()
    train.loc[train.index[-n:], "target_t1"] = np.nan
    return train


def test_ineligible_train_labels_are_excluded_from_fit_and_train_scoring():
    train, val, test = _split(10, 6, 5)
    model = _RecordingKBack(0)
    result = _run(model, _with_ineligible_tail(train, 2), val, test)
    assert len(model.fit_X) == 8
    assert len(model.fit_y) == 8
    assert np.isfinite(model.fit_y).all()
    assert result.n_evaluated["train"] == 8
    assert result.n_ineligible_label_rows == 2


def test_warmup_and_ineligible_label_rows_are_counted_separately():
    train, val, test = _split(10, 6, 5)
    result = _run(_KBack(3), _with_ineligible_tail(train, 2), val, test)
    assert result.n_warmup_rows == 3
    assert result.n_ineligible_label_rows == 2
    assert result.n_evaluated == {"train": 10 - 3 - 2, "val": 6, "test": 5}


def test_train_actuals_and_predictions_stay_aligned_after_both_exclusions():
    train, val, test = _split(10, 6, 5)
    result = _run(_KBack(3), _with_ineligible_tail(train, 2), val, test)
    # rows 0-2: warm-up (no window); rows 8-9: label ineligible; rows 3..7 are scored
    assert result.actuals["train"].tolist() == [35.0, 45.0, 55.0, 65.0, 75.0]
    assert result.predictions["train"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]


@pytest.mark.parametrize("partition", ["val", "test"])
def test_non_finite_label_in_val_or_test_raises(partition):
    train, val, test = _split(10, 6, 5)
    frames = {"val": val.copy(), "test": test.copy()}
    frames[partition].loc[frames[partition].index[2], "target_t1"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _run(_KBack(0), train, frames["val"], frames["test"])


def test_non_trailing_ineligible_train_labels_raise():
    train, val, test = _split(10, 6, 5)
    train = train.copy()
    train.loc[train.index[4], "target_t1"] = np.nan
    with pytest.raises(ValueError, match="trailing"):
        _run(_KBack(0), train, val, test)


def test_overlapping_warmup_and_ineligible_rows_raise():
    train, val, test = _split(4, 6, 5)
    with pytest.raises(ValueError, match="overlap"):
        _run(_KBack(3), _with_ineligible_tail(train, 2), val, test)


def test_counts_are_zero_when_nothing_is_excluded():
    train, val, test = _split(10, 6, 5)
    result = _run(_KBack(0), train, val, test)
    assert result.n_warmup_rows == 0
    assert result.n_ineligible_label_rows == 0


# ---------------------------------------------------------------------
# Invalid label configurations must raise BEFORE any model is fitted
# ---------------------------------------------------------------------


class _FitCounter(_KBack):
    def __init__(self, k=0):
        super().__init__(k)
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        return self


@pytest.mark.parametrize("case", ["val_nan", "non_trailing", "overlap"])
def test_invalid_label_configurations_raise_before_fit(case):
    train, val, test = _split(10, 6, 5)
    k = 0
    if case == "val_nan":
        val = val.copy()
        val.loc[val.index[2], "target_t1"] = np.nan
    elif case == "non_trailing":
        train = train.copy()
        train.loc[train.index[4], "target_t1"] = np.nan
    else:
        train, val, test = _split(4, 6, 5)
        train = _with_ineligible_tail(train, 2)
        k = 3
    model = _FitCounter(k)
    with pytest.raises(ValueError):
        _run(model, train, val, test)
    assert model.fit_calls == 0
