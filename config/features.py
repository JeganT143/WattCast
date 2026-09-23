"""
Feature engineering configuration — single source of truth for the feature
schema used across training (notebooks/src) and serving (Phase 6).

Changing any of these constants changes the feature contract and should be
treated as a versioned decision: new lag/window/feature -> new processed
dataset version -> new scaler version -> new model version.
"""

LAG_STEPS = [1, 2, 3, 4, 5, 6, 144]
ROLLING_WINDOWS = [6, 18]

SEASONAL_PERIOD = 144


def _derive_baseline_context_lags(
    horizons: list[int],
    seasonal_period: int,
) -> dict[int, int]:
    """
    Derive the seasonal-naive baseline's required historical lag for each
    target horizon.

    Predicting Appliances[t+h] using the value from one seasonal period
    earlier requires Appliances[t+h-seasonal_period].

    Relative to forecast origin t, the required lag is:

        seasonal_period - horizon

    Valid horizons must satisfy:

        0 < horizon < seasonal_period
    """
    lags = {}

    for horizon in horizons:
        if not (0 < horizon < seasonal_period):
            raise ValueError(
                f"horizon must satisfy 0 < horizon < seasonal_period "
                f"({seasonal_period}); got {horizon}"
            )

        lags[horizon] = seasonal_period - horizon

    return lags


TARGET_HORIZONS = [1, 6]

BASELINE_CONTEXT_LAGS = _derive_baseline_context_lags(
    TARGET_HORIZONS,
    SEASONAL_PERIOD,
)


FEATURE_COLUMNS = [
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
]


SCALED_COLUMNS = [
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
]


SPLIT_TRAIN_END = "2016-04-18"
SPLIT_VAL_END = "2016-04-30"

# MAPE evaluation threshold (Wh). Conservative cutoff to avoid unstable/
# disproportionate percentage errors at the lowest observed consumption
# values — NOT a claim that 30 Wh is a scientifically established idle
# boundary. Appliances never reaches exactly 0 in this dataset (observed
# floor: 10 train / 20 val,test); median is 60 across all partitions.
#
# Known limitation: exclusion rate is asymmetric across partitions due
# to the dataset's non-stationary amplitude (see Phase 1 STL finding,
# Phase 2 split rationale). At this threshold: train excludes 2.34%,
# val excludes 0.58%, test excludes 0.38% of observations. MAE and RMSE
# are unaffected — they are computed over all observations; only MAPE
# applies this threshold.
MAPE_THRESHOLD = 30.0
