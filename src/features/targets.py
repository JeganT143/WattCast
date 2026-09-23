"""
Target construction — deliberately the mirror image of lag.py.

Governing rule: shift(-N) pulls values from row (t + N) into row t. This
is correct and required here, since a target must represent the future
by definition. The leakage boundary is features-vs-target, not
past-vs-future in isolation (see Phase 2 design discussion).
"""

import pandas as pd


def add_targets(
    df: pd.DataFrame,
    column: str,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Add future target columns for `column`.

    For each horizon N in `horizons`, creates f"target_t{N}" where row t
    holds the value of `column` at row (t + N).

    Assumes df is already sorted chronologically ascending.

    The last `max(horizons)` rows will have NaN — expected, handled at
    dataset assembly time.
    """
    df = df.copy()
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(
                f"horizon must be positive (a target looks forward); got {horizon}."
            )
        df[f"target_t{horizon}"] = df[column].shift(-horizon)
    return df