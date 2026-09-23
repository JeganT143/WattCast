"""
End-to-end Phase 2 data pipeline orchestration.

This module owns the sequence of operations, but not the implementation
of any individual transformation.

Pipeline:
    raw dataframe
        -> feature construction on full continuous series
        -> chronological split masks
        -> train-only scaler fitting
        -> transform train/val/test with the same scaler
        -> horizon-specific dataset assembly
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
)
from src.data.assemble import build_horizon_dataset
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


def run_pipeline(df_raw: pd.DataFrame) -> PipelineResult:
    """
    Run the complete Phase 2 feature engineering and preprocessing pipeline.

    Feature construction runs on the full continuous time series before
    splitting so lag and rolling features retain temporal context across
    partition boundaries.

    The scaler is fitted only on training rows and then reused unchanged
    for train, validation, and test transformations.

    Final modeling datasets are assembled independently for each horizon
    and partition.
    """

    # 1. Feature engineering on the full continuous series.
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
    train_t1 = build_horizon_dataset(
        df_train_scaled,
        FEATURE_COLUMNS,
        "target_t1",
    )

    val_t1 = build_horizon_dataset(
        df_val_scaled,
        FEATURE_COLUMNS,
        "target_t1",
    )

    test_t1 = build_horizon_dataset(
        df_test_scaled,
        FEATURE_COLUMNS,
        "target_t1",
    )

    train_t6 = build_horizon_dataset(
        df_train_scaled,
        FEATURE_COLUMNS,
        "target_t6",
    )

    val_t6 = build_horizon_dataset(
        df_val_scaled,
        FEATURE_COLUMNS,
        "target_t6",
    )

    test_t6 = build_horizon_dataset(
        df_test_scaled,
        FEATURE_COLUMNS,
        "target_t6",
    )

    return PipelineResult(
        scaler=scaler,
        train_t1=train_t1,
        val_t1=val_t1,
        test_t1=test_t1,
        train_t6=train_t6,
        val_t6=val_t6,
        test_t6=test_t6,
    )
