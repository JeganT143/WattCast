"""Tests for src.evaluation.fold_data.build_fold_datasets — per-fold train-only
scaling and label purge, composed from the existing pipeline pieces, for the
registered walk-forward scheme (decisions.md, ADR-009)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from config.features import (
    FEATURE_COLUMNS,
    SCALED_COLUMNS,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    TARGET_HORIZONS,
)
from config.paths import RAW_DATA_PATH
from config.walk_forward import EVAL_DAYS, FIRST_EVAL_START, FREQ_MINUTES, N_FOLDS
from src.evaluation.fold_data import FoldData, build_fold_datasets
from src.evaluation.folds import FoldSpec, make_folds
from src.features.build_features import build_features
from src.pipeline import run_pipeline

pytestmark = pytest.mark.requires_raw_data

REAL_N_TRAIN = [6954, 7962, 8970, 9978, 10986, 11994, 13002, 14010]


@pytest.fixture(scope="module")
def df_raw():
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def df_features(df_raw):
    return build_features(df_raw, target_horizons=TARGET_HORIZONS)


@pytest.fixture(scope="module")
def result(df_raw):
    return run_pipeline(df_raw)


@pytest.fixture(scope="module")
def real_specs(result):
    timestamps = pd.DatetimeIndex(
        pd.concat(
            [
                result.train_t6["date"],
                result.val_t6["date"],
                result.test_t6["date"],
            ],
            ignore_index=True,
        )
    )
    return make_folds(
        timestamps,
        first_eval_start=FIRST_EVAL_START,
        n_folds=N_FOLDS,
        eval_days=EVAL_DAYS,
        freq_minutes=FREQ_MINUTES,
        holdout_start=pd.Timestamp(SPLIT_VAL_END),
    )


def test_equivalence_at_global_boundary_both_horizons(df_features, result):
    eval_start = pd.Timestamp(SPLIT_TRAIN_END)
    eval_end = pd.Timestamp(SPLIT_VAL_END)

    spec_t6 = FoldSpec(
        fold=0,
        train_start=result.train_t6["date"].iloc[0],
        train_end=eval_start,
        eval_start=eval_start,
        eval_end=eval_end,
        n_train=len(result.train_t6),
        n_eval=len(result.val_t6),
    )
    fd6 = build_fold_datasets(df_features, spec_t6, 6)
    pd.testing.assert_frame_equal(fd6.train_df, result.train_t6, check_exact=True)
    pd.testing.assert_frame_equal(fd6.eval_df, result.val_t6, check_exact=True)
    assert np.array_equal(fd6.scaler.mean_, result.scaler.mean_)
    assert np.array_equal(fd6.scaler.scale_, result.scaler.scale_)

    spec_t1 = FoldSpec(
        fold=0,
        train_start=result.train_t1["date"].iloc[0],
        train_end=eval_start,
        eval_start=eval_start,
        eval_end=eval_end,
        n_train=len(result.train_t1),
        n_eval=len(result.val_t1),
    )
    fd1 = build_fold_datasets(df_features, spec_t1, 1)
    pd.testing.assert_frame_equal(fd1.train_df, result.train_t1, check_exact=True)
    pd.testing.assert_frame_equal(fd1.eval_df, result.val_t1, check_exact=True)
    assert np.array_equal(fd1.scaler.mean_, result.scaler.mean_)
    assert np.array_equal(fd1.scaler.scale_, result.scaler.scale_)


def test_fold_scaler_is_train_only_by_independent_arithmetic(
    df_features, result, real_specs
):
    spec = real_specs[2]
    fd = build_fold_datasets(df_features, spec, 6)

    train_slice = df_features.loc[df_features["date"] < spec.eval_start, SCALED_COLUMNS]
    expected_mean = train_slice.mean().values
    expected_std = train_slice.std(ddof=0).values

    assert np.allclose(fd.scaler.mean_, expected_mean, rtol=1e-9)
    assert np.allclose(fd.scaler.scale_, expected_std, rtol=1e-9)
    assert not np.allclose(fd.scaler.mean_, result.scaler.mean_)


def test_leak_tripwire_eval_mutation_does_not_reach_train(df_features, real_specs):
    spec = real_specs[2]
    fd_original = build_fold_datasets(df_features, spec, 6)

    mutated = df_features.copy()
    future_mask = mutated["date"] >= spec.eval_start
    mutate_cols = SCALED_COLUMNS + ["target_t1", "target_t6"]
    mutated.loc[future_mask, mutate_cols] = mutated.loc[future_mask, mutate_cols] * 1000

    fd_mutated = build_fold_datasets(mutated, spec, 6)

    pd.testing.assert_frame_equal(fd_mutated.train_df, fd_original.train_df, check_exact=True)
    assert np.array_equal(fd_mutated.scaler.mean_, fd_original.scaler.mean_)
    assert np.array_equal(fd_mutated.scaler.scale_, fd_original.scaler.scale_)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(fd_mutated.eval_df, fd_original.eval_df, check_exact=True)


def test_purge_masks_exactly_the_last_h_train_labels(df_features, real_specs):
    spec = real_specs[2]

    fd6 = build_fold_datasets(df_features, spec, 6)
    labels6 = fd6.train_df["target_t6"].to_numpy()
    assert np.isnan(labels6[-6:]).all()
    assert np.isfinite(labels6[:-6]).all()
    assert not fd6.train_df[FEATURE_COLUMNS].isna().any().any()
    assert not fd6.eval_df[FEATURE_COLUMNS].isna().any().any()
    assert not fd6.eval_df["target_t6"].isna().any()

    fd1 = build_fold_datasets(df_features, spec, 1)
    labels1 = fd1.train_df["target_t1"].to_numpy()
    assert np.isnan(labels1[-1:]).all()
    assert np.isfinite(labels1[:-1]).all()
    assert not fd1.train_df[FEATURE_COLUMNS].isna().any().any()
    assert not fd1.eval_df[FEATURE_COLUMNS].isna().any().any()
    assert not fd1.eval_df["target_t1"].isna().any()


def test_spec_consistency_raises_on_bad_specs(df_features, real_specs):
    spec = real_specs[2]

    with pytest.raises(ValueError):
        build_fold_datasets(df_features, dataclasses.replace(spec, n_train=spec.n_train + 1), 6)

    with pytest.raises(ValueError):
        build_fold_datasets(df_features, dataclasses.replace(spec, n_eval=spec.n_eval + 1), 6)

    with pytest.raises(ValueError):
        build_fold_datasets(
            df_features,
            dataclasses.replace(
                spec,
                eval_start=spec.eval_start + pd.Timedelta(minutes=10),
            ),
            6,
        )


def test_input_purity_df_features_unchanged(df_features, real_specs):
    before = df_features.copy(deep=True)
    build_fold_datasets(df_features, real_specs[2], 6)
    build_fold_datasets(df_features, real_specs[7], 6)
    pd.testing.assert_frame_equal(df_features, before, check_exact=True)


def test_all_8_real_folds_build(df_features, real_specs):
    assert len(real_specs) == 8
    for spec, expected_n_train in zip(real_specs, REAL_N_TRAIN):
        fd = build_fold_datasets(df_features, spec, 6)
        assert isinstance(fd, FoldData)
        assert len(fd.train_df) == spec.n_train
        assert len(fd.eval_df) == 1008
        assert spec.n_train == expected_n_train
