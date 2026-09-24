from src.models.naive import NaivePersistenceForecaster, NaiveSeasonalForecaster
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster


def test_all_existing_baselines_require_no_history():
    forecasters = [
        NaivePersistenceForecaster(),
        NaiveSeasonalForecaster(1),
        NaiveSeasonalForecaster(6),
        LinearRegressionForecaster(),
        RandomForestForecaster(),
    ]
    assert [f.required_history_length for f in forecasters] == [0] * 5
