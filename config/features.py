"""
Feature engineering configuration — single source of truth for the feature
schema used across training (notebooks/src) and serving (Phase 6).

Changing any of these constants changes the feature contract and should be
treated as a versioned decision: new lag/window/feature -> new processed
dataset version -> new scaler version -> new model version.
"""

LAG_STEPS = [1, 2, 3, 4, 5, 6, 144]
ROLLING_WINDOWS = [6, 18]
TARGET_HORIZONS = [1, 6]

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

# subset of FEATURE_COLUMNS that gets StandardScaler treatment —
# excludes binary (is_weekend), cyclical (sin/cos), and raw calendar
# integer columns per the Phase 2 scaling design discussion.
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