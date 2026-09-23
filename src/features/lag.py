"""
Lag feature construction.

Governing rule (per Phase 1 EDA): any feature must be available at or
before the forecast origin t. shift(N) with positive N pulls values from
row (t - N) into row t — i.e., strictly the past. Never use a negative N
here; that produces a target, not a feature (see features/targets.py).
"""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list[int],
) -> pd.DataFrame:
    """
    Add lagged versions of `column` to `df`.

    For each lag N in `lags`, creates a new column f"{column}_lag_{N}"
    where row t holds the value of `column` at row (t - N).

    Assumes df is already sorted chronologically ascending — this function
    does not sort, since sorting is the caller's responsibility (done once,
    upstream, not re-checked at every feature step).

    Rows without sufficient history (the first `max(lags)` rows) will have
    NaN in the new columns — this is expected and handled at dataset
    assembly time (see data/assemble.py), not here.
    """
    df = df.copy()
    for lag in lags:
        if lag <= 0:
            raise ValueError(
                f"lag must be positive (a feature looks backward); got {lag}. "
                f"Negative shifts produce targets, not features."
            )
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df