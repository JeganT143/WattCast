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
from config.features import (
    LAG_STEPS,
    ROLLING_WINDOWS,
    TARGET_HORIZONS,
    BASELINE_CONTEXT_LAGS,
)


def build_features(
    df: pd.DataFrame,
    date_column: str = "date",
    target_source_column: str = "Appliances",
    target_horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run the canonical feature-engineering sequence on `df`.

    Order: sort chronologically -> canonical lag features (LAG_STEPS)
    -> baseline-only seasonal context lags (BASELINE_CONTEXT_LAGS,
    deduped against LAG_STEPS) -> rolling features (ROLLING_WINDOWS)
    -> calendar/cyclical features -> future targets (target_horizons).
    Every step is leakage-safe by construction (see lag.py, rolling.py,
    temporal.py, targets.py); this function only sequences them and
    must not be reimplemented elsewhere.

    Returns the full-width dataframe with NaN in feature/target columns
    at the series' edges — dropped later at horizon-dataset assembly
    time (see data/assemble.py), not here.
    """
    if target_horizons is None:
        target_horizons = TARGET_HORIZONS

    df = df.sort_values(date_column).reset_index(drop=True)
    df = add_lag_features(df, column=target_source_column, lags=LAG_STEPS)

    # baseline-specific seasonal context lags (e.g. lag_143, lag_138) —
    # NOT part of the canonical learned-model feature set, added via the
    # same leakage-safe add_lag_features mechanism, deduped in case any
    # value coincides with LAG_STEPS
    baseline_lags = sorted(set(BASELINE_CONTEXT_LAGS.values()) - set(LAG_STEPS))
    if baseline_lags:
        df = add_lag_features(df, column=target_source_column, lags=baseline_lags)

    df = add_rolling_features(df, column=target_source_column, windows=ROLLING_WINDOWS)
    df = add_time_features(df, date_column=date_column)
    df = add_targets(df, column=target_source_column, horizons=target_horizons)
    return df
