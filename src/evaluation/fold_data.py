"""Per-fold train-only scaling and label purge for the registered Phase 5
walk-forward scheme (decisions.md, ADR-009).

Composed entirely from the existing pipeline pieces (fit_scaler,
transform_with_scaler, build_horizon_dataset, mask_ineligible_labels) —
none of them are reimplemented here. Pure: no I/O, does not mutate the
input df_features."""

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from src.data.assemble import build_horizon_dataset
from src.data.purge import mask_ineligible_labels
from src.evaluation.folds import FoldSpec
from src.pipeline import _context_columns_for_horizon
from src.preprocessing.scaling import fit_scaler, transform_with_scaler


@dataclass(frozen=True)
class FoldData:
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    scaler: StandardScaler


def build_fold_datasets(
    df_features: pd.DataFrame,
    spec: FoldSpec,
    horizon: int,
    date_column: str = "date",
) -> FoldData:
    target_column = f"target_t{horizon}"

    train_mask = df_features[date_column] < spec.eval_start
    eval_mask = (df_features[date_column] >= spec.eval_start) & (
        df_features[date_column] < spec.eval_end
    )

    scaler = fit_scaler(df_features, SCALED_COLUMNS, train_mask)

    df_train_scaled = transform_with_scaler(
        df_features.loc[train_mask], scaler, SCALED_COLUMNS
    )
    df_eval_scaled = transform_with_scaler(
        df_features.loc[eval_mask], scaler, SCALED_COLUMNS
    )

    context_columns = _context_columns_for_horizon(horizon)
    train_df = build_horizon_dataset(
        df_train_scaled, FEATURE_COLUMNS, target_column, context_columns=context_columns
    )
    eval_df = build_horizon_dataset(
        df_eval_scaled, FEATURE_COLUMNS, target_column, context_columns=context_columns
    )

    train_df = mask_ineligible_labels(
        train_df, target_column, horizon, pd.Timestamp(spec.eval_start)
    )

    if len(train_df) != spec.n_train:
        raise ValueError(
            f"fold {spec.fold}: built {len(train_df)} train rows, expected {spec.n_train}"
        )
    if len(eval_df) != spec.n_eval:
        raise ValueError(
            f"fold {spec.fold}: built {len(eval_df)} eval rows, expected {spec.n_eval}"
        )
    if train_df[date_column].iloc[0] != spec.train_start:
        raise ValueError(
            f"fold {spec.fold}: train_df starts at {train_df[date_column].iloc[0]}, "
            f"expected {spec.train_start}"
        )
    if eval_df[date_column].iloc[0] != spec.eval_start:
        raise ValueError(
            f"fold {spec.fold}: eval_df starts at {eval_df[date_column].iloc[0]}, "
            f"expected {spec.eval_start}"
        )
    expected_last = spec.eval_end - pd.Timedelta(minutes=10)
    if eval_df[date_column].iloc[-1] != expected_last:
        raise ValueError(
            f"fold {spec.fold}: eval_df ends at {eval_df[date_column].iloc[-1]}, "
            f"expected {expected_last}"
        )

    return FoldData(train_df=train_df, eval_df=eval_df, scaler=scaler)
