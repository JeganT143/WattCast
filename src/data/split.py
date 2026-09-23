"""
Time-aware split boundaries.

Returns boolean masks, not separate dataframes — feature construction
(lags, rolling stats) must run on the full continuous series first, since
splitting before feature construction would break temporal context at
split boundaries (e.g. val's first row would have no access to train's
tail for computing lag_144).
"""

import pandas as pd


def create_time_masks(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    date_column: str = "date",
) -> dict[str, pd.Series]:
    """
    Build boolean masks for a chronological train/val/test split.

    train: date < train_end
    val:   train_end <= date < val_end
    test:  date >= val_end

    Boundaries are exclusive on the upper end, consistent with standard
    half-open interval convention — avoids double-counting a row that
    lands exactly on a boundary timestamp.
    """
    if train_end >= val_end:
        raise ValueError(f"train_end ({train_end}) must be before val_end ({val_end})")

    train_mask = df[date_column] < train_end
    val_mask = (df[date_column] >= train_end) & (df[date_column] < val_end)
    test_mask = df[date_column] >= val_end

    return {"train": train_mask, "val": val_mask, "test": test_mask}
