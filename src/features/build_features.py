"""
Feature pipeline orchestration.

This is the single canonical sequence for turning raw sensor data into
the full feature-engineered dataframe. Training and serving must both
call this function — never reimplement the sequence independently
(Decorator/DRY discipline per project rules: train/serve skew is a
silent, hard-to-debug failure mode).
"""

import pandas as pd

from src.features.lag import add_lag_features
from src.features.rolling import add_rolling_features
from src.features.temporal import add_time_features
from src.features.targets import add_targets
from config.features import LAG_STEPS, ROLLING_WINDOWS, TARGET_HORIZONS


def build_features(
    df: pd.DataFrame,
    date_column: str = "date",
    target_source_column: str = "Appliances",
    target_horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on raw sensor data.

    Precondition: df must be sorted chronologically ascending on
    `date_column` (enforced here, once, at the pipeline boundary —
    individual feature functions trust this and do not re-check it).

    Returns a dataframe with all lag/rolling/temporal features and all
    target columns added. NaN rows at the head (insufficient lag/rolling
    history) and tail (insufficient future horizon) are NOT dropped here
    — that's horizon-specific and handled by data/assemble.py.
    """
    if target_horizons is None:
        target_horizons = TARGET_HORIZONS

    df = df.sort_values(date_column).reset_index(drop=True)

    df = add_lag_features(df, column=target_source_column, lags=LAG_STEPS)
    df = add_rolling_features(df, column=target_source_column, windows=ROLLING_WINDOWS)
    df = add_time_features(df, date_column=date_column)
    df = add_targets(df, column=target_source_column, horizons=target_horizons)

    return df