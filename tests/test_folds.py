"""Tests for src.evaluation.folds.make_folds — pure expanding-window fold generator
for the registered Phase 5 walk-forward scheme (DECISIONS.md)."""

import dataclasses

import pandas as pd
import pytest

from config.walk_forward import EVAL_DAYS, FIRST_EVAL_START, FREQ_MINUTES, N_FOLDS
from src.evaluation.folds import FoldSpec, make_folds

SYNTHETIC_INDEX = pd.date_range(
    "2016-01-12 17:00", "2016-05-27 17:00", freq="10min"
)
HOLDOUT_START = pd.Timestamp("2016-04-30 00:00")


def _make_folds(timestamps=SYNTHETIC_INDEX, holdout_start=HOLDOUT_START):
    return make_folds(
        timestamps,
        first_eval_start=FIRST_EVAL_START,
        n_folds=N_FOLDS,
        eval_days=EVAL_DAYS,
        freq_minutes=FREQ_MINUTES,
        holdout_start=holdout_start,
    )


def test_fold_count_and_eval_boundaries():
    folds = _make_folds()
    assert len(folds) == 8
    assert folds[0].eval_start == pd.Timestamp("2016-03-01 00:00")
    assert folds[0].eval_end == pd.Timestamp("2016-03-08 00:00")
    assert folds[7].eval_start == pd.Timestamp("2016-04-19 00:00")
    assert folds[7].eval_end == pd.Timestamp("2016-04-26 00:00")


def test_every_fold_has_1008_eval_rows():
    folds = _make_folds()
    for fold in folds:
        assert fold.n_eval == 1008


def test_train_start_fixed_and_train_grows_by_1008_per_fold():
    folds = _make_folds()
    for fold in folds:
        assert fold.train_start == SYNTHETIC_INDEX[0]
        assert fold.train_end == fold.eval_start
    assert folds[0].n_train == 42 + 48 * 144
    for k in range(1, 8):
        assert folds[k].n_train == folds[k - 1].n_train + 1008


def test_eval_windows_contiguous_and_non_overlapping():
    folds = _make_folds()
    for k in range(7):
        assert folds[k].eval_end == folds[k + 1].eval_start


def test_holdout_start_before_last_fold_end_raises():
    with pytest.raises(ValueError):
        _make_folds(holdout_start=pd.Timestamp("2016-04-25 00:00"))


def test_unsorted_index_raises():
    shuffled = pd.DatetimeIndex(SYNTHETIC_INDEX.to_list()[::-1])
    with pytest.raises(ValueError):
        _make_folds(timestamps=shuffled)


def test_duplicate_timestamp_raises():
    with_dup = pd.DatetimeIndex(
        list(SYNTHETIC_INDEX) + [SYNTHETIC_INDEX[-1]]
    )
    with pytest.raises(ValueError):
        _make_folds(timestamps=with_dup)


def test_gap_inside_fold_3_eval_window_raises():
    fold_3_start = pd.Timestamp("2016-03-01 00:00") + pd.Timedelta(days=14)
    victim = fold_3_start + pd.Timedelta(minutes=30)
    gapped = SYNTHETIC_INDEX[SYNTHETIC_INDEX != victim]
    with pytest.raises(ValueError):
        _make_folds(timestamps=gapped)


def test_folds_are_frozen_and_input_index_unchanged():
    before = SYNTHETIC_INDEX.copy()
    folds = _make_folds()
    assert isinstance(folds, tuple)
    for fold in folds:
        assert isinstance(fold, FoldSpec)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fold.fold = 99
    assert SYNTHETIC_INDEX.equals(before)


def test_registered_constants():
    assert FIRST_EVAL_START == "2016-03-01 00:00"
    assert N_FOLDS == 8
    assert EVAL_DAYS == 7
    assert FREQ_MINUTES == 10
