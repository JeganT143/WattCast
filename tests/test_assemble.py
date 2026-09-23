"""Tests for horizon dataset assembly: column selection and NaN-drop scoping."""

import pandas as pd
import pytest

from src.data.assemble import build_horizon_dataset


def test_build_horizon_dataset_selects_required_columns_and_drops_required_nans():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4),
            "feature_a": [1.0, 2.0, None, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
            "target_t1": [100.0, 200.0, 300.0, None],
            "unrelated": [None, 2.0, 3.0, 4.0],
        }
    )

    result = build_horizon_dataset(
        df,
        feature_columns=["feature_a", "feature_b"],
        target_column="target_t1",
    )

    assert list(result.columns) == [
        "date",
        "feature_a",
        "feature_b",
        "target_t1",
    ]
    assert len(result) == 2
    assert result["date"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]
    assert result.index.tolist() == [0, 1]


def test_build_horizon_dataset_includes_context_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "feature_a": [1.0, 2.0, 3.0],
            "Appliances": [50.0, 60.0, 70.0],
            "target_t1": [100.0, 200.0, 300.0],
        }
    )

    result = build_horizon_dataset(
        df,
        feature_columns=["feature_a"],
        target_column="target_t1",
        context_columns=["Appliances"],
    )

    assert list(result.columns) == [
        "date",
        "feature_a",
        "Appliances",
        "target_t1",
    ]
    assert result["Appliances"].tolist() == [50.0, 60.0, 70.0]


def test_build_horizon_dataset_drops_rows_with_context_column_nan():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "feature_a": [1.0, 2.0, 3.0],
            "Appliances": [50.0, None, 70.0],
            "target_t1": [100.0, 200.0, 300.0],
        }
    )

    result = build_horizon_dataset(
        df,
        feature_columns=["feature_a"],
        target_column="target_t1",
        context_columns=["Appliances"],
    )

    # row 1 must be dropped because its context column (Appliances) is NaN,
    # even though feature_a and target_t1 are both valid there
    assert len(result) == 2
    assert result["Appliances"].tolist() == [50.0, 70.0]


def test_build_horizon_dataset_ignores_unrelated_column_nans():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "feature_a": [1.0, 2.0, 3.0],
            "Appliances": [50.0, 60.0, 70.0],
            "target_t1": [100.0, 200.0, 300.0],
            "unrelated": [None, None, None],
        }
    )

    result = build_horizon_dataset(
        df,
        feature_columns=["feature_a"],
        target_column="target_t1",
        context_columns=["Appliances"],
    )

    # all 3 rows survive — 'unrelated' is not in feature/context/target
    assert len(result) == 3


def test_build_horizon_dataset_does_not_mutate_input():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2),
            "feature": [1.0, 2.0],
            "Appliances": [10.0, 20.0],
            "target_t1": [10.0, 20.0],
        }
    )
    original = df.copy(deep=True)

    build_horizon_dataset(
        df,
        feature_columns=["feature"],
        target_column="target_t1",
        context_columns=["Appliances"],
    )

    pd.testing.assert_frame_equal(df, original)


def test_build_horizon_dataset_raises_key_error_for_missing_column():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2),
            "feature": [1.0, 2.0],
            "target_t1": [10.0, 20.0],
        }
    )

    with pytest.raises(KeyError):
        build_horizon_dataset(
            df,
            feature_columns=["missing_feature"],
            target_column="target_t1",
        )


def test_build_horizon_dataset_raises_key_error_for_missing_context_column():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2),
            "feature": [1.0, 2.0],
            "target_t1": [10.0, 20.0],
        }
    )

    with pytest.raises(KeyError):
        build_horizon_dataset(
            df,
            feature_columns=["feature"],
            target_column="target_t1",
            context_columns=["missing_context"],
        )
