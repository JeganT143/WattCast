import json

from src.evaluation.walk_forward_models import (
    HUBER_DELTA,
    LSTM_MAX_EPOCHS,
    deterministic_factories,
    seeded_factories,
)
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.models.naive import NaivePersistenceForecaster, NaiveSeasonalForecaster
from src.models.sklearn_models import LinearRegressionForecaster, RandomForestForecaster

_NEURAL_KEYS = [
    "L", "hidden_size", "num_layers", "dropout", "learning_rate", "batch_size",
    "max_epochs", "loss", "huber_delta", "output_bias_init", "seed",
    "torch_num_threads", "optimizer",
]


def test_a_keys_and_order():
    det = deterministic_factories(6)
    assert list(det) == ["naive_persistence", "naive_seasonal", "linear_regression", "random_forest"]

    seeded = seeded_factories()
    assert list(seeded) == ["lstm", "gru", "cnn_lstm"]

    assert set(det.keys()).isdisjoint(set(seeded.keys()))


def test_b_classes_and_freshness():
    det = deterministic_factories(6)
    assert isinstance(det["naive_persistence"](), NaivePersistenceForecaster)
    assert isinstance(det["naive_seasonal"](), NaiveSeasonalForecaster)
    assert isinstance(det["linear_regression"](), LinearRegressionForecaster)
    assert isinstance(det["random_forest"](), RandomForestForecaster)
    assert det["naive_persistence"]() is not det["naive_persistence"]()

    seeded = seeded_factories()
    assert isinstance(seeded["lstm"](1), LSTMForecaster)
    assert isinstance(seeded["gru"](1), GRUForecaster)
    assert isinstance(seeded["cnn_lstm"](1), CNNLSTMForecaster)
    assert seeded["lstm"](1) is not seeded["lstm"](1)

    det2 = deterministic_factories(6)
    assert det2 is not det
    det2.pop("naive_persistence")
    assert "naive_persistence" in det


def test_c_linear_regression_and_random_forest_match_golden():
    # tests/golden.py constructs these with no explicit arguments:
    #   LinearRegressionForecaster()
    #   RandomForestForecaster()
    det = deterministic_factories(6)
    assert det["linear_regression"]().params == LinearRegressionForecaster().params
    assert det["random_forest"]().params == RandomForestForecaster().params


def test_d_seasonal_horizon():
    det6 = deterministic_factories(6)
    seasonal6 = det6["naive_seasonal"]()
    assert seasonal6.params["horizon"] == 6
    assert "Appliances_lag_138" in seasonal6.required_columns

    det1 = deterministic_factories(1)
    seasonal1 = det1["naive_seasonal"]()
    assert "Appliances_lag_143" in seasonal1.required_columns


def test_e_registered_neural_configuration():
    seeded = seeded_factories()
    lstm = seeded["lstm"](42)
    assert lstm.params == {
        "max_epochs": 50,
        "huber_delta": 40.0,
        "seed": 42,
        "L": 18,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "loss": "huber",
        "output_bias_init": "train_target_median",
        "torch_num_threads": 4,
        "optimizer": "adam",
    }


def test_f_gru_and_cnn_lstm_share_lstm_keys():
    seeded = seeded_factories()
    lstm_params = seeded["lstm"](42).params
    gru_params = seeded["gru"](42).params
    cnn_lstm_params = seeded["cnn_lstm"](42).params

    lstm_subset = {k: lstm_params[k] for k in _NEURAL_KEYS}
    assert {k: gru_params[k] for k in _NEURAL_KEYS} == lstm_subset
    assert {k: cnn_lstm_params[k] for k in _NEURAL_KEYS} == lstm_subset

    assert cnn_lstm_params["conv_channels"] == 32
    assert cnn_lstm_params["conv_kernel_size"] == 3
    assert cnn_lstm_params["conv_padding"] == 1
    assert cnn_lstm_params["conv_activation"] == "relu"


def test_g_seed_passthrough():
    seeded = seeded_factories()
    for seed in (42, 43, 44):
        assert seeded["lstm"](seed).params["seed"] == seed
        assert seeded["gru"](seed).params["seed"] == seed
        assert seeded["cnn_lstm"](seed).params["seed"] == seed


def test_h_registered_constants():
    assert LSTM_MAX_EPOCHS == 50
    assert HUBER_DELTA == 40.0


def test_i_crosscheck_phase4_selection():
    with open("results/tuning_h6_validation.json") as f:
        selected = json.load(f)["selected"]
    assert selected["max_epochs"] == LSTM_MAX_EPOCHS
    assert selected["huber_delta"] == HUBER_DELTA
