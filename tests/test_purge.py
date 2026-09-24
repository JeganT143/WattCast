import numpy as np
import pandas as pd
import pytest

from src.data.purge import mask_ineligible_labels

STEP = pd.Timedelta(minutes=10)
BOUNDARY = pd.Timestamp("2016-04-18")


def _train_like(n=30):
    dates = pd.date_range(end=BOUNDARY - STEP, periods=n, freq="10min")
    return pd.DataFrame(
        {
            "date": dates,
            "feat": np.arange(n, dtype=float),
            "target": np.arange(n, dtype=float) + 1000.0,
        }
    )


@pytest.mark.parametrize("h", [1, 6])
def test_masks_exactly_the_last_h_rows_of_a_partition_ending_at_the_boundary(h):
    out = mask_ineligible_labels(_train_like(), "target", h, BOUNDARY)
    assert out["target"].isna().sum() == h
    assert out["target"].iloc[-h:].isna().all()
    assert out["target"].iloc[:-h].notna().all()


@pytest.mark.parametrize("h", [1, 6])
def test_untouched_rows_keep_their_labels_exactly(h):
    df = _train_like()
    out = mask_ineligible_labels(df, "target", h, BOUNDARY)
    assert out["target"].iloc[:-h].equals(df["target"].iloc[:-h])


def test_feature_rows_are_kept_not_deleted():
    df = _train_like()
    out = mask_ineligible_labels(df, "target", 6, BOUNDARY)
    assert len(out) == len(df)
    assert out[["date", "feat"]].equals(df[["date", "feat"]])


def test_label_time_equal_to_boundary_is_masked_and_one_step_earlier_is_kept():
    df = pd.DataFrame(
        {"date": [BOUNDARY - 4 * STEP, BOUNDARY - 3 * STEP], "target": [1.0, 2.0]}
    )
    out = mask_ineligible_labels(df, "target", 3, BOUNDARY)
    assert out["target"].iloc[0] == 1.0
    assert np.isnan(out["target"].iloc[1])


def test_input_frame_is_not_mutated():
    df = _train_like()
    before = df.copy()
    mask_ineligible_labels(df, "target", 6, BOUNDARY)
    assert df.equals(before)


def test_partition_far_from_the_boundary_is_unchanged():
    df = _train_like()
    out = mask_ineligible_labels(df, "target", 6, BOUNDARY + pd.Timedelta(days=30))
    assert out.equals(df)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_horizon_raises(bad):
    with pytest.raises(ValueError, match="horizon"):
        mask_ineligible_labels(_train_like(), "target", bad, BOUNDARY)
