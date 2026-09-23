"""
Rolling window statistics — backward-only, by construction.

Invariant: rolling windows must never use center=True or any forward-
looking construction. A centered window is not just "a bad choice" here
- it is undefined at serving time, since a live rolling context buffer
has no future rows to center around (see Phase 2 leak demo).
"""

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int],
) -> pd.DataFrame:
    """
    Add backward-looking rolling mean and std of `column` to `df`.

    For each window size N in `windows`, creates:
      f"{column}_roll{N}_mean"
      f"{column}_roll{N}_std"

    where row t's value is computed over rows [t-N+1, t] inclusive —
    i.e., it includes the current row, which is valid because the
    current row's `column` value is observed at the forecast origin t
    (see Phase 2 design discussion: Appliances[t] is known at time t).

    Assumes df is already sorted chronologically ascending (same
    precondition as add_lag_features).

    Rows without `N` prior observations will have NaN — expected,
    handled at dataset assembly time.
    """
    df = df.copy()
    for window in windows:
        if window <= 0:
            raise ValueError(f"window must be positive; got {window}")
        roll = df[column].rolling(window=window)  # center=False is the default — deliberately never overridden
        df[f"{column}_roll{window}_mean"] = roll.mean()
        df[f"{column}_roll{window}_std"] = roll.std()
    return df