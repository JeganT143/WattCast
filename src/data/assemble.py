"""
Horizon-specific dataset assembly — the final step before a modeling
dataset is ready for training.

Deliberately separate from targets.py: target construction and dataset
assembly are different operations (target creation happens once, for
all horizons, during build_features; assembly happens once per horizon,
selecting the right columns and dropping only the rows that horizon
actually requires).
"""

import pandas as pd


def build_horizon_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Assemble the final modeling dataset for one forecast horizon.

    Selects date + feature_columns + target_column, then drops rows
    with NaN in feature_columns or target_column ONLY — a column not
    in this selection (e.g. the other horizon's target, or an unused
    raw sensor column) can have NaN elsewhere in df without causing
    rows to be dropped here (avoids the blanket-dropna over-trim
    problem discussed in Phase 2).
    """
    required_cols = feature_columns + [target_column]
    selected = df[[date_column] + required_cols].copy()
    result = selected.dropna(subset=required_cols).reset_index(drop=True)
    return result
