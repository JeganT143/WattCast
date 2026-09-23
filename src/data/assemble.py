"""
Horizon-specific dataset assembly — the final step before a modeling
dataset is ready for training.

Assembles the complete downstream data boundary for a given forecast
horizon: canonical learned-model features, plus any raw observed
"context" columns a non-learned baseline needs (e.g. the naive
persistence baseline needs Appliances[t] itself, which is deliberately
excluded from FEATURE_COLUMNS). Context columns are observed at or
before the forecast origin t and carry no leakage risk by inclusion
alone — leakage only occurs if a model consumes information from after
t, not from a column's mere presence in this dataset.
"""

import pandas as pd


def build_horizon_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    context_columns: list[str] | None = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Assemble the final modeling dataset for one forecast horizon.

    Selects date + feature_columns + context_columns + target_column,
    then drops rows with NaN in feature_columns, context_columns, or
    target_column ONLY. A column not in this selection (e.g. the other
    horizon's target, or an unused raw sensor column) can have NaN
    elsewhere in df without causing rows to be dropped here.

    context_columns defaults to an empty list — most horizon datasets
    need no raw context beyond the engineered features. Only add a
    column here when a concrete Forecaster's required_columns actually
    needs it (YAGNI: currently just ["Appliances"] for the naive
    persistence baseline).
    """
    if context_columns is None:
        context_columns = []

    required_cols = feature_columns + context_columns + [target_column]
    selected = df[[date_column] + required_cols].copy()
    result = selected.dropna(subset=required_cols).reset_index(drop=True)
    return result
