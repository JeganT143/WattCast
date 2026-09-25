"""Cross-checks the app's static narrative facts (src/ui/content.py) against
the configuration and, when it is present, the raw dataset."""

import pandas as pd
import pytest

from config.features import SPLIT_TRAIN_END, SPLIT_VAL_END
from config.paths import RAW_DATA_PATH
from src.ui.content import DATASET, PARTITIONS


@pytest.mark.requires_raw_data
def test_dataset_facts_match_the_raw_csv():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    assert len(df) == DATASET["n_rows"]
    assert df.shape[1] == DATASET["n_columns"]
    assert str(df["date"].min().date()) == DATASET["start"]
    assert str(df["date"].max().date()) == DATASET["end"]
    assert (df["date"].diff().dropna() == pd.Timedelta(minutes=DATASET["interval_minutes"])).all()
    assert df[DATASET["target"]].max() == 1080


def test_partition_periods_match_the_configured_split():
    train, val, test = PARTITIONS
    day = pd.Timedelta(days=1)
    assert train["Period"].endswith(str((pd.Timestamp(SPLIT_TRAIN_END) - day).date()))
    assert val["Period"].startswith(SPLIT_TRAIN_END)
    assert val["Period"].endswith(str((pd.Timestamp(SPLIT_VAL_END) - day).date()))
    assert test["Period"].startswith(SPLIT_VAL_END)
    assert test["Period"].endswith(DATASET["end"])
