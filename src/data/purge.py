"""Masks (sets to NaN) the labels of rows whose label time falls at or after a partition boundary,
so a partition's labels never come from a later, held-out period. Rows are never deleted: the feature
rows remain valid history, and deleting them would open a gap in the next partition's context.
"""

import numpy as np
import pandas as pd


def trailing_ineligible_count(y_train: np.ndarray) -> int:
    """Returns the number of trailing non-finite (purged) train labels.

    Raises if the non-finite rows do not form a trailing block, since a
    non-trailing gap would open a hole inside sequence windows.
    """
    ineligible = ~np.isfinite(y_train)
    n_inel = int(ineligible.sum())
    n_train = len(ineligible)
    if n_inel > 0 and not ineligible[n_train - n_inel :].all():
        raise ValueError(
            "non-finite train labels must form a trailing block; a non-trailing gap would open a hole inside sequence windows"
        )
    return n_inel


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
