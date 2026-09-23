"""
Time-based and cyclical calendar features.

Unlike lag/rolling features, these carry no leakage risk — calendar
position at any timestamp t is always knowable in advance, regardless of
forecast horizon (see Phase 1 governing rule).
"""

import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Add raw calendar features and their cyclical (sin/cos) encodings.

    Raw integer/binary columns:
      hour_of_day, day_of_week, is_weekend
      (kept for tree-based models, which split on raw thresholds fine
      and don't benefit from cyclical distance)

    Cyclical encodings (clock-face distance, e.g. 23:59 close to 00:00):
      minute_of_day_sin/cos — period 1440 (true 10-min data resolution,
        not hour_of_day's period 24, which would collapse six distinct
        10-minute slots per hour into one identical value)
      day_of_week_sin/cos — period 7
    """
    df = df.copy()

    df["hour_of_day"] = df[date_column].dt.hour
    df["day_of_week"] = df[date_column].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    minute_of_day = df[date_column].dt.hour * 60 + df[date_column].dt.minute
    df["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df