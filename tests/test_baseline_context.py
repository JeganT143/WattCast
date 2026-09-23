"""
Tests for baseline-specific seasonal context lags (Appliances_lag_143,
Appliances_lag_138) — separate from the canonical learned-model
FEATURE_COLUMNS, added specifically to support the seasonal-naive
baseline's required_columns for each forecast horizon.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.pipeline import run_pipeline
from config.features import BASELINE_CONTEXT_LAGS


def make_synthetic_raw_data():
    dates = pd.date_range(
        "2016-01-11 17:00",
        "2016-05-27 18:00",
        freq="10min",
    )
    n_rows = len(dates)
    appliances = 100 + 10 * np.sin(np.arange(n_rows) / 10) + np.arange(n_rows) * 0.01
    return pd.DataFrame({"date": dates, "Appliances": appliances})


# ---------------------------------------------------------------------
# 1 & 2: correct lag values
# ---------------------------------------------------------------------


def test_baseline_context_lag_143_matches_shift_143():
    df_raw = make_synthetic_raw_data()
    df_features = build_features(df_raw)

    expected = df_raw["Appliances"].shift(143)
    pd.testing.assert_series_equal(
        df_features["Appliances_lag_143"],
        expected,
        check_names=False,
    )


def test_baseline_context_lag_138_matches_shift_138():
    df_raw = make_synthetic_raw_data()
    df_features = build_features(df_raw)

    expected = df_raw["Appliances"].shift(138)
    pd.testing.assert_series_equal(
        df_features["Appliances_lag_138"],
        expected,
        check_names=False,
    )


# ---------------------------------------------------------------------
# 3 & 4: correct per-horizon context columns in assembled datasets
# ---------------------------------------------------------------------


def test_t1_datasets_contain_lag_143_context_not_lag_138():
    result = run_pipeline(make_synthetic_raw_data())

    for dataset in [result.train_t1, result.val_t1, result.test_t1]:
        assert "Appliances" in dataset.columns
        assert "Appliances_lag_143" in dataset.columns
        assert "Appliances_lag_138" not in dataset.columns


def test_t6_datasets_contain_lag_138_context_not_lag_143():
    result = run_pipeline(make_synthetic_raw_data())

    for dataset in [result.train_t6, result.val_t6, result.test_t6]:
        assert "Appliances" in dataset.columns
        assert "Appliances_lag_138" in dataset.columns
        assert "Appliances_lag_143" not in dataset.columns


# ---------------------------------------------------------------------
# 5: baseline context columns remain unscaled
# ---------------------------------------------------------------------


def test_baseline_context_lags_are_unscaled():
    df_raw = make_synthetic_raw_data()
    result = run_pipeline(df_raw)

    raw_lookup = df_raw.set_index("date")["Appliances"]

    # t+1: Appliances_lag_143[t] should equal raw Appliances at (t - 143 steps)
    sample = result.train_t1.iloc[500]  # comfortably clear of any burn-in
    expected_lag_143 = raw_lookup.iloc[raw_lookup.index.get_loc(sample["date"]) - 143]
    assert sample["Appliances_lag_143"] == expected_lag_143

    # t+6: same check for lag_138
    sample6 = result.train_t6.iloc[500]
    expected_lag_138 = raw_lookup.iloc[raw_lookup.index.get_loc(sample6["date"]) - 138]
    assert sample6["Appliances_lag_138"] == expected_lag_138


# ---------------------------------------------------------------------
# 6: NaN handling — burn-in rows correctly dropped by assembly
# ---------------------------------------------------------------------


def test_baseline_context_lags_produce_no_nans_after_assembly():
    result = run_pipeline(make_synthetic_raw_data())

    for dataset in [
        result.train_t1,
        result.val_t1,
        result.test_t1,
        result.train_t6,
        result.val_t6,
        result.test_t6,
    ]:
        assert dataset.isna().sum().sum() == 0


# ---------------------------------------------------------------------
# 7: no future leakage into baseline context lags
# ---------------------------------------------------------------------


def test_baseline_context_lag_143_has_no_future_leakage():
    df_raw = make_synthetic_raw_data()
    before = build_features(df_raw)

    df_future_changed = df_raw.copy()
    # mutate a value strictly after row 1000
    df_future_changed.loc[1500, "Appliances"] = 999999.0
    after = build_features(df_future_changed)

    # row 1000's lag_143 looks back to row 857 — unaffected by row 1500
    assert (
        before.loc[1000, "Appliances_lag_143"] == after.loc[1000, "Appliances_lag_143"]
    )


def test_baseline_context_lag_138_has_no_future_leakage():
    df_raw = make_synthetic_raw_data()
    before = build_features(df_raw)

    df_future_changed = df_raw.copy()
    df_future_changed.loc[1500, "Appliances"] = 999999.0
    after = build_features(df_future_changed)

    # row 1000's lag_138 looks back to row 862 — unaffected by row 1500
    assert (
        before.loc[1000, "Appliances_lag_138"] == after.loc[1000, "Appliances_lag_138"]
    )
