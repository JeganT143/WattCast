import numpy as np
import pandas as pd
import pytest

from src.evaluation.context import audit_leading_nans, drop_context_predictions, take_context


def frame(start: str, n: int) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range(start, periods=n, freq="10min"), "x": np.arange(n)})


def test_take_context_returns_last_k_rows_of_previous():
    prev, cur = frame("2016-04-01 00:00", 10), frame("2016-04-01 01:40", 5)
    ctx = take_context(prev, cur, k=3)
    assert ctx["x"].tolist() == [7, 8, 9]


def test_take_context_k_zero_is_empty_not_the_whole_frame():
    prev, cur = frame("2016-04-01 00:00", 10), frame("2016-04-01 01:40", 5)
    assert len(take_context(prev, cur, k=0)) == 0


def test_take_context_no_previous_partition_gives_empty_context():
    assert len(take_context(None, frame("2016-04-01", 5), k=3)) == 0


def test_take_context_raises_when_previous_is_shorter_than_k():
    with pytest.raises(ValueError, match="need 5"):
        take_context(frame("2016-04-01 00:00", 3), frame("2016-04-01 00:30", 5), k=5)


def test_take_context_raises_on_gap_between_context_and_partition():
    prev = frame("2016-04-01 00:00", 10).iloc[:-2]   # simulate a purge that deleted rows
    cur = frame("2016-04-01 01:40", 5)
    with pytest.raises(ValueError, match="not contiguous"):
        take_context(prev, cur, k=3)


def test_audit_accepts_nans_in_exactly_the_first_k_positions():
    audit_leading_nans(np.array([np.nan, np.nan, 1.0, 2.0]), k=2)


def test_audit_rejects_nan_in_wrong_position():
    with pytest.raises(ValueError, match="violates"):
        audit_leading_nans(np.array([np.nan, 1.0, np.nan, 2.0]), k=2)


def test_audit_rejects_too_few_leading_nans():
    with pytest.raises(ValueError, match="violates"):
        audit_leading_nans(np.array([np.nan, 1.0, 2.0]), k=2)


def test_forgetful_model_declaring_zero_but_emitting_nan_is_rejected():
    with pytest.raises(ValueError, match="violates"):
        audit_leading_nans(np.array([np.nan, 1.0, 2.0]), k=0)


def test_drop_context_predictions_leaves_one_value_per_current_row():
    preds = np.array([np.nan, np.nan, 5.0, 6.0, 7.0])
    assert drop_context_predictions(preds, n_context=2).tolist() == [5.0, 6.0, 7.0]