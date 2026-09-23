"""Tests for src.features.lag.add_lag_features (leakage-safety and shift correctness)."""

import pandas as pd
import pytest

from src.features.lag import add_lag_features


def test_lag_features_use_past_values():
    df = pd.DataFrame(
        {
            "Appliances": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    result = add_lag_features(
        df,
        column="Appliances",
        lags=[1, 3],
    )

    # lag 1: current row gets previous row's value
    assert pd.isna(result.loc[0, "Appliances_lag_1"])
    assert result.loc[1, "Appliances_lag_1"] == 10.0
    assert result.loc[4, "Appliances_lag_1"] == 40.0

    # lag 3: current row gets value from three rows earlier
    assert result.loc[3, "Appliances_lag_3"] == 10.0
    assert result.loc[4, "Appliances_lag_3"] == 20.0

    # First lag rows do not have enough history.
    assert result["Appliances_lag_3"].iloc[:3].isna().all()


def test_lag_features_do_not_mutate_input():
    df = pd.DataFrame(
        {
            "Appliances": [10.0, 20.0, 30.0],
        }
    )

    original = df.copy(deep=True)

    add_lag_features(
        df,
        column="Appliances",
        lags=[1],
    )

    pd.testing.assert_frame_equal(df, original)


def test_lag_features_reject_non_positive_lag():
    df = pd.DataFrame(
        {
            "Appliances": [10.0, 20.0, 30.0],
        }
    )

    with pytest.raises(ValueError):
        add_lag_features(
            df,
            column="Appliances",
            lags=[0],
        )

    with pytest.raises(ValueError):
        add_lag_features(
            df,
            column="Appliances",
            lags=[-1],
        )
