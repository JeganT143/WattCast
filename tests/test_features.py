import pandas as pd
import pytest

from src.features.lag import add_lag_features


def make_test_df():
    return pd.DataFrame(
        {
            "Appliances": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
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
        name="Appliances_lag_1",
    )

    expected_lag_2 = pd.Series(
        [float("nan"), float("nan"), 10.0, 20.0, 30.0],
        name="Appliances_lag_2",
    )

    pd.testing.assert_series_equal(
        result["Appliances_lag_1"],
        expected_lag_1,
    )

    pd.testing.assert_series_equal(
        result["Appliances_lag_2"],
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


def test_lag_features_reject_zero_lag():
    df = make_test_df()

    with pytest.raises(ValueError, match="lag must be positive"):
        add_lag_features(
            df,
            column="Appliances",
            lags=[0],
        )


def test_lag_features_reject_negative_lag():
    df = make_test_df()

    with pytest.raises(ValueError, match="lag must be positive"):
        add_lag_features(
            df,
            column="Appliances",
            lags=[-1],
        )
