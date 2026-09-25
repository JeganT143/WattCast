"""Tests for the final train+validation training window (Phase 6 registration,
DECISIONS.md "Phase 6: serving registration, 2026-09-25")."""

import numpy as np
import pandas as pd
import pytest

from config.features import SCALED_COLUMNS
from config.paths import RAW_DATA_PATH
from src.training.final_window import build_final_window

BOUNDARY = pd.Timestamp("2016-04-30")


@pytest.fixture(scope="module")
def df_raw():
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])


def test_registered_counts_and_dates(df_raw):
    window = build_final_window(df_raw, horizon=6, boundary=BOUNDARY)

    assert window.n_mask_rows == 15738
    assert window.n_lag144_finite == 15594
    assert len(window.train_df) == 15594
    assert window.n_fittable == 15588
    assert window.last_date == pd.Timestamp("2016-04-29 23:50")
    assert window.first_date == pd.Timestamp("2016-01-12 17:00")
    assert (window.train_df["date"] < BOUNDARY).all()

    target_col = "target_t6"
    labels = window.train_df[target_col]
    assert labels.iloc[-6:].isna().all()
    assert labels.iloc[:-6].notna().all()


def test_scaler_n_samples_seen(df_raw):
    window = build_final_window(df_raw, horizon=6, boundary=BOUNDARY)
    seen = window.scaler.n_samples_seen_

    for lag in [1, 2, 3, 4, 5, 6, 144]:
        col = f"Appliances_lag_{lag}"
        idx = SCALED_COLUMNS.index(col)
        assert seen[idx] == 15738 - lag, col

    for window_size in [6, 18]:
        for stat in ["mean", "std"]:
            col = f"Appliances_roll{window_size}_{stat}"
            idx = SCALED_COLUMNS.index(col)
            assert seen[idx] == 15738 - (window_size - 1), col


def test_independent_of_test_partition_values(df_raw):
    window_a = build_final_window(df_raw, horizon=6, boundary=BOUNDARY)

    corrupted = df_raw.copy()
    corrupted.loc[corrupted["date"] >= BOUNDARY, "Appliances"] = 1e9
    window_b = build_final_window(corrupted, horizon=6, boundary=BOUNDARY)

    pd.testing.assert_frame_equal(window_a.train_df, window_b.train_df)
    assert np.array_equal(window_a.scaler.mean_, window_b.scaler.mean_)
    assert np.array_equal(window_a.scaler.scale_, window_b.scaler.scale_)


def test_synthetic_small_frame_trailing_labels_masked():
    n_rows = 200
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="10min")
    rng = np.random.default_rng(0)
    synthetic = pd.DataFrame(
        {"date": dates, "Appliances": rng.uniform(10, 200, size=n_rows)}
    )
    boundary = dates[-1] + pd.Timedelta(minutes=10)

    window = build_final_window(synthetic, horizon=6, boundary=boundary)

    labels = window.train_df["target_t6"]
    assert labels.iloc[-6:].isna().all()
    assert labels.iloc[:-6].notna().all()
