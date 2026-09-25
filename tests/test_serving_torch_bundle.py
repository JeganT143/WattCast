"""Tests for the torch-specific bundle serialization path used by sequence
model families (DECISIONS.md "Phase 6: deep-model final deployment
decision, 2026-09-25"). Deliberately separate from the skops path used by
the sklearn-based families (linear_regression, random_forest) — these
tests confirm that separation, not just that loading "works"."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from src.models.lstm import LSTMForecaster
from src.serving.bundle import ModelBundle, load_bundle, save_bundle

N_FEATURES = 16
L = 4


def _tiny_forecaster(seed: int = 0) -> LSTMForecaster:
    forecaster = LSTMForecaster(
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


def _tiny_bundle(seed: int = 0) -> ModelBundle:
    forecaster = _tiny_forecaster(seed)
    schema = {
        "model_family": "lstm",
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


def test_torch_state_dict_round_trip_bit_exact(tmp_path):
    bundle = _tiny_bundle()
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)

    loaded = load_bundle(directory)

    original_state = bundle.forecaster._net.state_dict()
    loaded_state = loaded.forecaster._net.state_dict()

    assert set(original_state.keys()) == set(loaded_state.keys())
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key]), key


def test_torch_predictions_bit_identical_after_round_trip(tmp_path):
    bundle = _tiny_bundle()
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)
    loaded = load_bundle(directory)

    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(20, N_FEATURES))

    original_preds = bundle.forecaster.predict(X)
    loaded_preds = loaded.forecaster.predict(X)

    assert np.array_equal(original_preds, loaded_preds, equal_nan=True)


def test_load_bundle_for_lstm_never_touches_skops_for_the_model(tmp_path):
    bundle = _tiny_bundle()
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


def test_loaded_lstm_required_history_and_columns_match_pre_serialization(tmp_path):
    bundle = _tiny_bundle()
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)
    loaded = load_bundle(directory)

    assert loaded.forecaster.required_history_length == bundle.forecaster.required_history_length
    assert loaded.forecaster.required_columns == bundle.forecaster.required_columns
    assert loaded.forecaster.L == bundle.forecaster.L


def test_load_bundle_raises_without_model_state_file_for_lstm(tmp_path):
    bundle = _tiny_bundle()
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)
    (directory / "model_state.pt").unlink()

    with pytest.raises(FileNotFoundError):
        load_bundle(directory)
