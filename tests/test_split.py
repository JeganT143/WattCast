import pandas as pd
import pytest

from src.data.split import create_time_masks


def test_create_time_masks_respects_boundaries():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01 23:50",
                    "2026-01-02 00:00",
                    "2026-01-02 12:00",
                    "2026-01-03 00:00",
                ]
            ),
        }
    )

    train_end = pd.Timestamp("2026-01-02")
    val_end = pd.Timestamp("2026-01-03")

    masks = create_time_masks(
        df,
        train_end=train_end,
        val_end=val_end,
    )

    assert masks["train"].tolist() == [
        True,
        False,
        False,
        False,
    ]

    assert masks["val"].tolist() == [
        False,
        True,
        True,
        False,
    ]

    assert masks["test"].tolist() == [
        False,
        False,
        False,
        True,
    ]


def test_create_time_masks_assigns_each_row_to_exactly_one_partition():
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=10,
                freq="D",
            ),
        }
    )

    train_end = pd.Timestamp("2026-01-04")
    val_end = pd.Timestamp("2026-01-07")

    masks = create_time_masks(
        df,
        train_end=train_end,
        val_end=val_end,
    )

    total_assignments = (
        masks["train"].astype(int)
        + masks["val"].astype(int)
        + masks["test"].astype(int)
    )

    assert (total_assignments == 1).all()


def test_create_time_masks_rejects_invalid_boundaries():
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
        }
    )

    with pytest.raises(ValueError):
        create_time_masks(
            df,
            train_end=pd.Timestamp("2026-01-03"),
            val_end=pd.Timestamp("2026-01-02"),
        )

    with pytest.raises(ValueError):
        create_time_masks(
            df,
            train_end=pd.Timestamp("2026-01-02"),
            val_end=pd.Timestamp("2026-01-02"),
        )
