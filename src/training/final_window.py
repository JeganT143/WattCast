"""Final train+validation training window for the Phase 6 serving champion
(DECISIONS.md "Phase 6: serving registration, 2026-09-25").

Composed entirely from existing pipeline pieces (build_features, fit_scaler,
transform_with_scaler, build_horizon_dataset, mask_ineligible_labels,
trailing_ineligible_count) — none of them are reimplemented here.

The test partition (date >= boundary) is never read: raw rows are trimmed
to date < boundary before build_features ever runs, so no feature or
target value derived from a held-out row can enter the returned frame.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS, SPLIT_VAL_END, TARGET_HORIZONS
from src.data.assemble import build_horizon_dataset
from src.data.purge import mask_ineligible_labels, trailing_ineligible_count
from src.features.build_features import build_features
from src.preprocessing.scaling import fit_scaler, transform_with_scaler


@dataclass(frozen=True)
class FinalWindow:
    train_df: pd.DataFrame
    scaler: StandardScaler
    n_mask_rows: int
    n_lag144_finite: int
    n_fittable: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp


def build_final_window(
    df_raw: pd.DataFrame,
    horizon: int = 6,
    boundary: pd.Timestamp = pd.Timestamp(SPLIT_VAL_END),
    date_column: str = "date",
) -> FinalWindow:
    target_column = f"target_t{horizon}"

    trimmed = df_raw[df_raw[date_column] < boundary].copy()
    n_mask_rows = len(trimmed)

    df_features = build_features(
        trimmed, date_column=date_column, target_horizons=TARGET_HORIZONS
    )
    n_lag144_finite = int(df_features["Appliances_lag_144"].notna().sum())

    train_mask = pd.Series(True, index=df_features.index)
    scaler = fit_scaler(df_features, SCALED_COLUMNS, train_mask)
    scaled = transform_with_scaler(df_features, scaler, SCALED_COLUMNS)

    # The trailing `horizon` rows have no target yet (their label would fall
    # at or beyond the boundary, i.e. inside the untouched test partition).
    # build_horizon_dataset drops NaN-target rows, which would delete that
    # trailing block instead of just masking its label — so it is filled
    # with a placeholder here and re-masked by date afterward, preserving
    # the "rows are never deleted" purge convention (src/data/purge.py).
    scaled_for_assembly = scaled.copy()
    scaled_for_assembly[target_column] = scaled_for_assembly[target_column].fillna(0.0)

    train_df = build_horizon_dataset(
        scaled_for_assembly,
        FEATURE_COLUMNS,
        target_column,
        context_columns=[],
        date_column=date_column,
    )
    train_df = mask_ineligible_labels(
        train_df, target_column, horizon, boundary, date_column=date_column
    )

    n_fittable = len(train_df) - trailing_ineligible_count(
        train_df[target_column].to_numpy()
    )

    return FinalWindow(
        train_df=train_df,
        scaler=scaler,
        n_mask_rows=n_mask_rows,
        n_lag144_finite=n_lag144_finite,
        n_fittable=n_fittable,
        first_date=train_df[date_column].iloc[0],
        last_date=train_df[date_column].iloc[-1],
    )
