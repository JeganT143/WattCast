"""Masks (sets to NaN) the labels of rows whose label time falls at or after a partition boundary,
so a partition's labels never come from a later, held-out period. Rows are never deleted: the feature
rows remain valid history, and deleting them would open a gap in the next partition's context.
"""

import pandas as pd


def mask_ineligible_labels(
    df: pd.DataFrame,
    target_column: str,
    horizon: int,
    boundary: pd.Timestamp,
    step: pd.Timedelta = pd.Timedelta(minutes=10),
    date_column: str = "date",
) -> pd.DataFrame:
    """Returns a copy, never drops rows, never touches any column other than target_column.
    The boundary is inclusive: label_time >= boundary is masked.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    out = df.copy()
    label_time = out[date_column] + horizon * step
    out[target_column] = out[target_column].where(label_time < boundary)
    return out
