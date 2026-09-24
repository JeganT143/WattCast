"""Factory table for the seven models the walk-forward run evaluates: four deterministic
baselines built once, and three neural sequence models built fresh per seed at the
registered configuration (50 epochs, Huber delta 40, everything else at model defaults).
"""

from typing import Callable

from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.forecaster import Forecaster
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.models.naive import NaivePersistenceForecaster, NaiveSeasonalForecaster
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster

LSTM_MAX_EPOCHS = 50
HUBER_DELTA = 40.0


def deterministic_factories(horizon: int) -> dict[str, Callable[[], Forecaster]]:
    return {
        "naive_persistence": lambda: NaivePersistenceForecaster(),
        "naive_seasonal": lambda: NaiveSeasonalForecaster(horizon),
        "linear_regression": lambda: LinearRegressionForecaster(),
        "random_forest": lambda: RandomForestForecaster(),
    }


def seeded_factories() -> dict[str, Callable[[int], Forecaster]]:
    return {
        "lstm": lambda seed: LSTMForecaster(
            max_epochs=LSTM_MAX_EPOCHS, huber_delta=HUBER_DELTA, seed=seed
        ),
        "gru": lambda seed: GRUForecaster(
            max_epochs=LSTM_MAX_EPOCHS, huber_delta=HUBER_DELTA, seed=seed
        ),
        "cnn_lstm": lambda seed: CNNLSTMForecaster(
            max_epochs=LSTM_MAX_EPOCHS, huber_delta=HUBER_DELTA, seed=seed
        ),
    }
