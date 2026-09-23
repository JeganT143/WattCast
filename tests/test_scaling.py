import numpy as np
import pandas as pd
import pytest

from src.preprocessing.scaling import (
    fit_scaler,
    transform_with_scaler,
)


def test_scaler_uses_training_rows_only():
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 100.0],
        }
    )

    train_mask = pd.Series(
        [True, True, False, False],
        index=df.index,
    )

    scaler_a = fit_scaler(
        df,
        columns=["feature"],
        train_mask=train_mask,
    )

    # Change only non-training data.
    df_changed = df.copy()
    df_changed.loc[2:, "feature"] = 1_000_000.0

    scaler_b = fit_scaler(
        df_changed,
        columns=["feature"],
        train_mask=train_mask,
    )

    # Validation/test values must have no influence
    # on the fitted scaler statistics.
    np.testing.assert_array_equal(
        scaler_a.mean_,
        scaler_b.mean_,
    )

    np.testing.assert_array_equal(
        scaler_a.scale_,
        scaler_b.scale_,
    )

    # Training mean = (1 + 2) / 2 = 1.5
    assert scaler_a.mean_[0] == 1.5


def test_transform_uses_fitted_scaler_without_refitting():
    train_df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    train_mask = pd.Series(
        [True, True, True],
        index=train_df.index,
    )

    scaler = fit_scaler(
        train_df,
        columns=["feature"],
        train_mask=train_mask,
    )

    # Save the scaler parameters before transformation.
    original_mean = scaler.mean_.copy()
    original_scale = scaler.scale_.copy()

    new_df = pd.DataFrame(
        {
            "feature": [4.0, 5.0],
        }
    )

    transformed = transform_with_scaler(
        new_df,
        scaler,
        columns=["feature"],
    )

    # Transform must NOT refit the scaler.
    np.testing.assert_array_equal(
        scaler.mean_,
        original_mean,
    )

    np.testing.assert_array_equal(
        scaler.scale_,
        original_scale,
    )

    # StandardScaler uses population standard deviation:
    #
    # mean = 2
    # variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3
    #          = 2 / 3
    #
    # Therefore:
    # 4 -> (4 - 2) / sqrt(2/3)
    # 5 -> (5 - 2) / sqrt(2/3)
    expected = (new_df["feature"].to_numpy() - 2.0) / np.sqrt(2.0 / 3.0)

    np.testing.assert_allclose(
        transformed["feature"].to_numpy(),
        expected,
    )


def test_transform_preserves_nan_values():
    train_df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    train_mask = pd.Series(
        [True, True, True],
        index=train_df.index,
    )

    scaler = fit_scaler(
        train_df,
        columns=["feature"],
        train_mask=train_mask,
    )

    df = pd.DataFrame(
        {
            "feature": [4.0, np.nan],
        }
    )

    transformed = transform_with_scaler(
        df,
        scaler,
        columns=["feature"],
    )

    assert pd.isna(transformed.loc[1, "feature"])


def test_fit_scaler_raises_on_empty_train_mask():
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    empty_mask = pd.Series(
        [False, False, False],
        index=df.index,
    )

    with pytest.raises(ValueError):
        fit_scaler(
            df,
            columns=["feature"],
            train_mask=empty_mask,
        )
