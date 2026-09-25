"""Tests for the torch-specific bundle serialization path used by sequence
model families (DECISIONS.md "Phase 6: deep-model final deployment
decision, 2026-09-25"). Deliberately separate from the skops path used by
the sklearn-based families (linear_regression, random_forest) — these
tests confirm that separation, not just that loading "works".

Parametrized across all three sequence families (lstm, gru, cnn_lstm): the
mechanism (state_dict + architecture reconstruction) is shared, but each
family gets its own genuine bit-exact round trip and prediction-equality
check rather than inferring correctness from LSTM's tests passing."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.serving.bundle import ModelBundle, load_bundle, save_bundle

N_FEATURES = 16
L = 4

_FAMILY_CLASSES = {
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "cnn_lstm": CNNLSTMForecaster,
}


def _tiny_forecaster(family: str, seed: int = 0):
    cls = _FAMILY_CLASSES[family]
    forecaster = cls(
        max_epochs=2,
        huber_delta=40.0,
        seed=seed,
        L=L,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=8,
        torch_num_threads=1,
    )
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(40, N_FEATURES))
    y = rng.uniform(0, 100, size=40)
    forecaster.fit(X, y)
    return forecaster


def _tiny_bundle(family: str, seed: int = 0) -> ModelBundle:
    forecaster = _tiny_forecaster(family, seed)
    schema = {
        "model_family": family,
        "feature_columns": FEATURE_COLUMNS,
        "scaled_columns": SCALED_COLUMNS,
        "architecture": {
            "class": type(forecaster._net).__name__,
            "n_features": N_FEATURES,
            "params": forecaster.params,
        },
    }
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(
        pd.DataFrame(rng.uniform(0, 500, size=(20, len(SCALED_COLUMNS))), columns=SCALED_COLUMNS)
    )
    return ModelBundle(forecaster=forecaster, scaler=scaler, schema=schema)


@pytest.mark.parametrize("family", ["lstm", "gru", "cnn_lstm"])
def test_torch_state_dict_round_trip_bit_exact(tmp_path, family):
    bundle = _tiny_bundle(family)
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)

    loaded = load_bundle(directory)

    original_state = bundle.forecaster._net.state_dict()
    loaded_state = loaded.forecaster._net.state_dict()

    assert set(original_state.keys()) == set(loaded_state.keys())
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key]), key


@pytest.mark.parametrize("family", ["lstm", "gru", "cnn_lstm"])
def test_torch_predictions_bit_identical_after_round_trip(tmp_path, family):
    bundle = _tiny_bundle(family)
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)
    loaded = load_bundle(directory)

    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(20, N_FEATURES))

    original_preds = bundle.forecaster.predict(X)
    loaded_preds = loaded.forecaster.predict(X)

    assert np.array_equal(original_preds, loaded_preds, equal_nan=True)


@pytest.mark.parametrize("family", ["lstm", "gru", "cnn_lstm"])
def test_load_bundle_never_touches_skops_for_the_model(tmp_path, family):
    bundle = _tiny_bundle(family)
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)

    calls = []
    from src.serving.bundle import sio

    original_load = sio.load

    def _tracking_load(file, *args, **kwargs):
        calls.append(str(file))
        return original_load(file, *args, **kwargs)

    with patch("src.serving.bundle.sio.load", side_effect=_tracking_load):
        load_bundle(directory)

    assert not any("model_state" in c or "estimator" in c for c in calls), calls
    assert any("scaler" in c for c in calls), "the scaler is expected to still use skops"


@pytest.mark.parametrize("family", ["lstm", "gru", "cnn_lstm"])
def test_loaded_required_history_and_columns_match_pre_serialization(tmp_path, family):
    bundle = _tiny_bundle(family)
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)
    loaded = load_bundle(directory)

    assert loaded.forecaster.required_history_length == bundle.forecaster.required_history_length
    assert loaded.forecaster.required_columns == bundle.forecaster.required_columns
    assert loaded.forecaster.L == bundle.forecaster.L


@pytest.mark.parametrize("family", ["lstm", "gru", "cnn_lstm"])
def test_load_bundle_raises_without_model_state_file(tmp_path, family):
    bundle = _tiny_bundle(family)
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)
    (directory / "model_state.pt").unlink()

    with pytest.raises(FileNotFoundError):
        load_bundle(directory)


def test_cnn_lstm_architecture_params_includes_conv_fields_not_accepted_by_constructor():
    """Documents the one real difference from LSTM/GRU: CNNLSTMForecaster.params
    (used verbatim as schema["architecture"]["params"]) includes conv_channels/
    conv_kernel_size/conv_padding/conv_activation, which SequenceForecaster.__init__
    does not accept as keyword arguments (they are module-level constants in
    cnn_lstm.py, not per-instance configurable). The bundle loader must filter
    these out before reconstructing the forecaster; naively splatting
    schema["architecture"]["params"] into the constructor raises TypeError."""
    forecaster = _tiny_forecaster("cnn_lstm")
    params = forecaster.params
    for extra_key in ("conv_channels", "conv_kernel_size", "conv_padding", "conv_activation"):
        assert extra_key in params

    with pytest.raises(TypeError):
        CNNLSTMForecaster(**params)
