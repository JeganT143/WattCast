from pathlib import Path

test_path = PROJECT_ROOT / "tests" / "test_features.py"
test_path.parent.mkdir(parents=True, exist_ok=True)

test_path.write_text("""
import pandas as pd
import pytest

from src.features import add_lag_features


def make_test_df():
    index = pd.date_range(
        "2026-01-01 00:00",
        periods=5,
        freq="10min",
    )

    return pd.DataFrame(
        {"Appliances": [10, 20, 30, 40, 50]},
        index=index,
    )


def test_lag_values_are_correct():
    df = make_test_df()

    result = add_lag_features(
        df,
        column="Appliances",
        lags=[1, 2],
    )

    expected_lag_1 = pd.Series(
        [float("nan"), 10.0, 20.0, 30.0, 40.0],
        index=df.index,
        name="lag_1",
    )

    expected_lag_2 = pd.Series(
        [float("nan"), float("nan"), 10.0, 20.0, 30.0],
        index=df.index,
        name="lag_2",
    )

    pd.testing.assert_series_equal(
        result["lag_1"],
        expected_lag_1,
    )

    pd.testing.assert_series_equal(
        result["lag_2"],
        expected_lag_2,
    )


def test_input_dataframe_is_not_mutated():
    df = make_test_df()
    original_df = df.copy(deep=True)

    add_lag_features(
        df,
        column="Appliances",
        lags=[1, 2],
    )

    pd.testing.assert_frame_equal(
        df,
        original_df,
    )


def test_unsorted_index_raises_value_error():
    df = make_test_df()
    unsorted_df = df.iloc[[2, 0, 1, 3, 4]]

    with pytest.raises(
        ValueError,
        match="chronologically ordered",
    ):
        add_lag_features(
            unsorted_df,
            column="Appliances",
            lags=[1, 2],
        )


def test_missing_column_raises_value_error():
    df = make_test_df()

    with pytest.raises(
        ValueError,
        match="Column 'WrongColumn' not found",
    ):
        add_lag_features(
            df,
            column="WrongColumn",
            lags=[1],
        )
""")

print(f"Created: {test_path}")
