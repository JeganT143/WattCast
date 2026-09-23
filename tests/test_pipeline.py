import numpy as np
import pandas as pd

from src.pipeline import PipelineResult, run_pipeline


def make_synthetic_raw_data():
    dates = pd.date_range(
        "2016-01-11 17:00",
        "2016-05-27 18:00",
        freq="10min",
    )

    n_rows = len(dates)

    appliances = 100 + 10 * np.sin(np.arange(n_rows) / 10) + np.arange(n_rows) * 0.01

    return pd.DataFrame(
        {
            "date": dates,
            "Appliances": appliances,
        }
    )


def test_run_pipeline_returns_complete_result():
    df = make_synthetic_raw_data()

    result = run_pipeline(df)

    assert isinstance(result, PipelineResult)

    assert result.train_t1 is not None
    assert result.val_t1 is not None
    assert result.test_t1 is not None

    assert result.train_t6 is not None
    assert result.val_t6 is not None
    assert result.test_t6 is not None

    assert result.scaler is not None


def test_run_pipeline_outputs_have_expected_columns():
    df = make_synthetic_raw_data()

    result = run_pipeline(df)

    expected_features = {
        "date",
        "Appliances_lag_1",
        "Appliances_lag_2",
        "Appliances_lag_3",
        "Appliances_lag_4",
        "Appliances_lag_5",
        "Appliances_lag_6",
        "Appliances_lag_144",
        "Appliances_roll6_mean",
        "Appliances_roll6_std",
        "Appliances_roll18_mean",
        "Appliances_roll18_std",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "minute_of_day_sin",
        "minute_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    }

    for dataset in [
        result.train_t1,
        result.val_t1,
        result.test_t1,
    ]:
        assert set(dataset.columns) == expected_features | {"target_t1"}

    for dataset in [
        result.train_t6,
        result.val_t6,
        result.test_t6,
    ]:
        assert set(dataset.columns) == expected_features | {"target_t6"}


def test_run_pipeline_outputs_contain_no_nans():
    df = make_synthetic_raw_data()

    result = run_pipeline(df)

    datasets = [
        result.train_t1,
        result.val_t1,
        result.test_t1,
        result.train_t6,
        result.val_t6,
        result.test_t6,
    ]

    for dataset in datasets:
        assert dataset.isna().sum().sum() == 0


def test_run_pipeline_preserves_chronological_partition_order():
    df = make_synthetic_raw_data()

    result = run_pipeline(df)

    for horizon in ["t1", "t6"]:
        train = getattr(result, f"train_{horizon}")
        val = getattr(result, f"val_{horizon}")
        test = getattr(result, f"test_{horizon}")

        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

        assert train["date"].is_monotonic_increasing
        assert val["date"].is_monotonic_increasing
        assert test["date"].is_monotonic_increasing


def test_run_pipeline_horizon_row_count_difference():
    df = make_synthetic_raw_data()

    result = run_pipeline(df)

    # Train and validation end before the raw dataset ends,
    # so both horizons have enough future observations inside
    # their respective partitions.
    assert len(result.train_t1) == len(result.train_t6)
    assert len(result.val_t1) == len(result.val_t6)

    # The test partition reaches the end of the raw dataset.
    # t+6 therefore loses five more rows than t+1.
    assert len(result.test_t1) - len(result.test_t6) == 5


def test_run_pipeline_fits_scaler():
    df = make_synthetic_raw_data()

    result = run_pipeline(df)

    assert hasattr(result.scaler, "mean_")
    assert hasattr(result.scaler, "scale_")

    assert result.scaler.n_features_in_ == 11
