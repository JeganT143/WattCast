"""Guard that the DVC-tracked processed CSVs contain every column the four baselines consume."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.naive import NaivePersistenceForecaster, NaiveSeasonalForecaster
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _forecasters(h):
    return [
        NaivePersistenceForecaster(),
        NaiveSeasonalForecaster(h),
        LinearRegressionForecaster(),
        RandomForestForecaster(),
    ]


@pytest.mark.parametrize("h", [1, 6])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_processed_csv_has_required_columns(split, h):
    path = PROCESSED / f"{split}_t{h}.csv"
    if not path.exists():
        pytest.skip(f"{path} not found; run `make data`")

    columns = set(pd.read_csv(path, nrows=0).columns)
    assert f"target_t{h}" in columns

    for forecaster in _forecasters(h):
        missing = set(forecaster.required_columns) - columns
        assert not missing, (
            f"{type(forecaster).__name__} is missing columns {sorted(missing)} in {path}"
        )
