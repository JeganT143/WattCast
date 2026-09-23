"""Tests for src.features.rolling.add_rolling_features (backward-only window correctness)."""

import pandas as pd
import pytest

from src.features.rolling import add_rolling_features


def test_rolling_features_are_backward_only():
    df = pd.DataFrame({
        "Appliances": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    before = add_rolling_features(
        df,
        column="Appliances",
        windows=[3],
    )

    # Change a future observation.
    df_future_changed = df.copy()
    df_future_changed.loc[4, "Appliances"] = 999999.0

    after = add_rolling_features(
        df_future_changed,
        column="Appliances",
        windows=[3],
    )

    # Row 3 uses rows 1, 2, 3 — row 4 must have no influence.
    assert before.loc[3, "Appliances_roll3_mean"] == \
        after.loc[3, "Appliances_roll3_mean"]

    # Original expected value: (20 + 30 + 40) / 3 = 30.
    assert before.loc[3, "Appliances_roll3_mean"] == 30.0


def test_rolling_features_reject_non_positive_window():
    df = pd.DataFrame({"Appliances": [10.0, 20.0, 30.0]})

    with pytest.raises(ValueError):
        add_rolling_features(
            df,
            column="Appliances",
            windows=[0],
        )