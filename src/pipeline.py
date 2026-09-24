"""
End-to-end Phase 2 data pipeline orchestration.

This module owns the sequence of operations, but not the implementation
of any individual transformation.

Pipeline:
    raw dataframe
        -> feature construction on full continuous series (including
           baseline-specific seasonal context lags, e.g. lag_143/lag_138)
        -> chronological split masks
        -> train-only scaler fitting
        -> transform train/val/test with the same scaler
        -> horizon-specific dataset assembly (features + raw context
           columns needed by non-learned baselines: Appliances[t] for
           the naive persistence baseline, and Appliances[t+h-144] for
           the naive seasonal baseline, per horizon)
        -> purge of train labels that point past the train boundary
           (rows kept, labels set to NaN)
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.features import (
    FEATURE_COLUMNS,
    SCALED_COLUMNS,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    TARGET_HORIZONS,
    BASELINE_CONTEXT_LAGS,
)
from src.data.assemble import build_horizon_dataset
from src.data.purge import mask_ineligible_labels
from src.data.split import create_time_masks
from src.features.build_features import build_features
from src.preprocessing.scaling import fit_scaler, transform_with_scaler


@dataclass
class PipelineResult:
    """Final outputs of the Phase 2 feature/preprocessing pipeline."""

    scaler: StandardScaler

    train_t1: pd.DataFrame
    val_t1: pd.DataFrame
    test_t1: pd.DataFrame

    train_t6: pd.DataFrame
    val_t6: pd.DataFrame
    test_t6: pd.DataFrame


def _context_columns_for_horizon(horizon: int) -> list[str]:
    """
    Raw/baseline-only context columns required for a given forecast
    horizon, on top of the canonical FEATURE_COLUMNS:

    - "Appliances": Appliances[t], needed by the naive persistence
      baseline (predict = last observed value, held flat).
    - "Appliances_lag_{N}", where N = BASELINE_CONTEXT_LAGS[horizon]:
      Appliances[t+horizon-144], needed by the naive seasonal baseline
      (predict = same time yesterday's cycle position).

    Not part of FEATURE_COLUMNS or SCALED_COLUMNS — these are baseline
    context only, per the Forecaster.required_columns design.
    """
    seasonal_lag = BASELINE_CONTEXT_LAGS[horizon]
    return ["Appliances", f"Appliances_lag_{seasonal_lag}"]


def run_pipeline(df_raw: pd.DataFrame) -> PipelineResult:
    """
    Run the complete Phase 2 feature engineering and preprocessing pipeline.

    Feature construction runs on the full continuous time series before
    splitting so lag and rolling features retain temporal context across
    partition boundaries.

    The scaler is fitted only on training rows and then reused unchanged
    for train, validation, and test transformations.

    Final modeling datasets are assembled independently for each horizon
    and partition, and each carries the raw context columns required by
    the naive baselines (see _context_columns_for_horizon) alongside the
    engineered FEATURE_COLUMNS — without adding raw/baseline-only values
    to the learned-model feature contract itself.

    After assembly, train labels that point at or beyond SPLIT_TRAIN_END are masked to NaN (rows are kept).
    """

    # 1. Feature engineering on the full continuous series. This includes
    #    baseline-specific seasonal context lags (e.g. lag_143, lag_138),
    #    added inside build_features via the same leakage-safe mechanism
    #    used for the canonical LAG_STEPS.
    df_features = build_features(
        df_raw,
        target_horizons=TARGET_HORIZONS,
    )

    # 2. Create chronological train/validation/test masks.
    masks = create_time_masks(
        df_features,
        train_end=pd.Timestamp(SPLIT_TRAIN_END),
        val_end=pd.Timestamp(SPLIT_VAL_END),
    )

    # 3. Fit scaler using training rows only.
    scaler = fit_scaler(
        df_features,
        SCALED_COLUMNS,
        masks["train"],
    )

    # 4. Transform each partition using the same fitted scaler.
    df_train_scaled = transform_with_scaler(
        df_features.loc[masks["train"]],
        scaler,
        SCALED_COLUMNS,
    )

    df_val_scaled = transform_with_scaler(
        df_features.loc[masks["val"]],
        scaler,
        SCALED_COLUMNS,
    )

    df_test_scaled = transform_with_scaler(
        df_features.loc[masks["test"]],
        scaler,
        SCALED_COLUMNS,
    )

    # 5. Assemble horizon-specific modeling datasets.
    #    context_columns carries Appliances (persistence baseline) and
    #    the horizon-specific seasonal lag (seasonal-naive baseline)
    #    through unscaled — StandardScaler was only ever fit on
    #    SCALED_COLUMNS, which excludes all context columns.
    train_t1 = build_horizon_dataset(
        df_train_scaled,
        FEATURE_COLUMNS,
        "target_t1",
        context_columns=_context_columns_for_horizon(1),
    )
    val_t1 = build_horizon_dataset(
        df_val_scaled,
        FEATURE_COLUMNS,
        "target_t1",
        context_columns=_context_columns_for_horizon(1),
    )
    test_t1 = build_horizon_dataset(
        df_test_scaled,
        FEATURE_COLUMNS,
        "target_t1",
        context_columns=_context_columns_for_horizon(1),
    )

    train_t6 = build_horizon_dataset(
        df_train_scaled,
        FEATURE_COLUMNS,
        "target_t6",
        context_columns=_context_columns_for_horizon(6),
    )
    val_t6 = build_horizon_dataset(
        df_val_scaled,
        FEATURE_COLUMNS,
        "target_t6",
        context_columns=_context_columns_for_horizon(6),
    )
    test_t6 = build_horizon_dataset(
        df_test_scaled,
        FEATURE_COLUMNS,
        "target_t6",
        context_columns=_context_columns_for_horizon(6),
    )

    # 6. Purge label-ineligible train targets. The last h train rows carry labels whose
    #    timestamps (date + h * 10 min) fall in the validation period. The rows stay:
    #    their features are valid history and deleting them would open a gap in the
    #    validation context. Only the labels are masked (NaN). Val and test are untouched.
    boundary = pd.Timestamp(SPLIT_TRAIN_END)
    train_t1 = mask_ineligible_labels(train_t1, "target_t1", 1, boundary)
    train_t6 = mask_ineligible_labels(train_t6, "target_t6", 6, boundary)

    return PipelineResult(
        scaler=scaler,
        train_t1=train_t1,
        val_t1=val_t1,
        test_t1=test_t1,
        train_t6=train_t6,
        val_t6=val_t6,
        test_t6=test_t6,
    )
