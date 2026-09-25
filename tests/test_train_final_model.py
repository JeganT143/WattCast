"""Tests for scripts/train_final_model.py, the generalized (multi-family)
successor to scripts/train_final_lr.py (decisions.md,
ADR-012). linear_regression is excluded here — it is
already registered via the original script. random_forest, lstm, gru, and
cnn_lstm are all supported in this stage.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from config.mlflow_config import EXPERIMENT_NAME, TRACKING_URI
from scripts import train_final_model
from src.serving.bundle import load_bundle
from src.serving.registry import CHAMPION_ALIAS, load_champion, registered_model_name


def _synthetic_raw(n_rows: int = 250) -> pd.DataFrame:
    seed_end = pd.Timestamp("2016-04-29 23:50:00")
    dates = pd.date_range(end=seed_end, periods=n_rows, freq="10min")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"date": dates, "Appliances": rng.uniform(10, 200, size=n_rows)})


def test_family_linear_regression_refuses_no_mlflow_call(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("MlflowClient must not be instantiated for a refused family")

    monkeypatch.setattr(train_final_model, "MlflowClient", _boom)

    with pytest.raises(ValueError, match="linear_regression"):
        train_final_model.check_can_run(
            "linear_regression", results_path=None, tracking_uri="sqlite:///unused.db"
        )


def test_family_unsupported_refuses_no_mlflow_call(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("MlflowClient must not be instantiated for a refused family")

    monkeypatch.setattr(train_final_model, "MlflowClient", _boom)

    with pytest.raises(ValueError, match="xgboost"):
        train_final_model.check_can_run(
            "xgboost", results_path=None, tracking_uri="sqlite:///unused.db"
        )


def test_random_forest_main_produces_registered_bundle(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "artifacts")
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)

    results_path = tmp_path / "final_model_registration_random_forest.json"

    payload = train_final_model.main(
        family="random_forest",
        results_path=results_path,
        tracking_uri=tracking_uri,
        df_raw=_synthetic_raw(),
    )

    assert results_path.exists()
    assert payload["model_family"] == "random_forest"

    champion = load_champion(tracking_uri, family="random_forest")
    assert champion.schema["model_family"] == "random_forest"

    model_name = registered_model_name("random_forest")
    mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    assert int(mv.version) == payload["model_version"]


def test_refuses_when_results_file_already_exists(tmp_path):
    results_path = tmp_path / "final_model_registration_random_forest.json"
    results_path.write_text("{}")

    with pytest.raises(FileExistsError):
        train_final_model.check_can_run(
            "random_forest", results_path=results_path, tracking_uri="sqlite:///unused.db"
        )


@pytest.mark.requires_registry
def test_load_champion_linear_regression_unaffected_by_signature_change():
    """Critical regression check for the registry.py signature change: the
    new function-based load_champion(family=...) must resolve to the exact
    same already-registered LR champion the old hardcoded-constant path
    resolved to, with bit-identical predictions on a fixed input."""
    old_style_client = MlflowClient(tracking_uri=TRACKING_URI)
    old_model_name = "wattcast_serving_lr_h6"
    old_mv = old_style_client.get_model_version_by_alias(old_model_name, CHAMPION_ALIAS)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = old_style_client.download_artifacts(old_mv.run_id, "bundle", tmp_dir)
        old_bundle = load_bundle(Path(local_dir))

    new_bundle = load_champion(TRACKING_URI, family="linear_regression")

    assert new_bundle.schema["n_fit_rows"] == 15588
    assert new_bundle.schema["n_fit_rows"] == old_bundle.schema["n_fit_rows"]

    n_cols = len(old_bundle.forecaster.required_columns)
    X = np.linspace(-1, 1, num=5 * n_cols).reshape(5, n_cols)
    assert np.array_equal(old_bundle.forecaster.predict(X), new_bundle.forecaster.predict(X))


def _synthetic_raw_for_lstm(n_rows: int = 250) -> pd.DataFrame:
    return _synthetic_raw(n_rows)


def test_family_lstm_end_to_end_produces_registered_bundle(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "artifacts")
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)

    results_path = tmp_path / "final_model_registration_lstm.json"

    payload = train_final_model.main(
        family="lstm",
        results_path=results_path,
        tracking_uri=tracking_uri,
        df_raw=_synthetic_raw_for_lstm(),
    )

    assert results_path.exists()
    assert payload["model_family"] == "lstm"
    assert payload["seed"] == 42
    assert payload["epochs"] == train_final_model.LSTM_MAX_EPOCHS

    champion = load_champion(tracking_uri, family="lstm")
    assert champion.schema["model_family"] == "lstm"
    assert champion.schema["seed"] == 42
    assert champion.schema["epochs"] == train_final_model.LSTM_MAX_EPOCHS

    # (e) reconstruction must not silently hardcode required_history_length /
    # required_columns wrong: compare against what SequenceForecaster/
    # LSTMForecaster themselves report for the exact same construction params.
    from src.models.lstm import LSTMForecaster

    fresh = LSTMForecaster(**champion.schema["architecture"]["params"])
    assert champion.forecaster.required_history_length == fresh.required_history_length
    assert champion.forecaster.required_columns == fresh.required_columns


def test_allowed_families_is_random_forest_and_all_sequence_families():
    assert train_final_model.ALLOWED_FAMILIES == {
        "random_forest",
        "lstm",
        "gru",
        "cnn_lstm",
    }


@pytest.mark.parametrize("family", ["gru", "cnn_lstm"])
def test_family_sequence_end_to_end_produces_registered_bundle(tmp_path, family):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "artifacts")
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)

    results_path = tmp_path / f"final_model_registration_{family}.json"

    payload = train_final_model.main(
        family=family,
        results_path=results_path,
        tracking_uri=tracking_uri,
        df_raw=_synthetic_raw(),
    )

    assert results_path.exists()
    assert payload["model_family"] == family
    assert payload["seed"] == 42
    assert payload["epochs"] == train_final_model.LSTM_MAX_EPOCHS

    champion = load_champion(tracking_uri, family=family)
    assert champion.schema["model_family"] == family
    assert champion.schema["seed"] == 42
    assert champion.schema["epochs"] == train_final_model.LSTM_MAX_EPOCHS

    # (e) reconstruction must not silently hardcode required_history_length /
    # required_columns wrong: compare against a fresh instance of the same
    # class built from the constructor-accepted subset of the recorded
    # architecture params (mirrors what the generic bundle loader itself
    # does — CNNLSTMForecaster.params includes extra conv_* fields that its
    # own __init__ does not accept as keyword arguments).
    from src.serving.bundle import _SEQUENCE_FORECASTER_CLASSES

    cls = _SEQUENCE_FORECASTER_CLASSES[family]
    ctor_params = inspect.signature(cls.__init__).parameters
    filtered = {
        k: v for k, v in champion.schema["architecture"]["params"].items() if k in ctor_params
    }
    fresh = cls(**filtered)
    assert champion.forecaster.required_history_length == fresh.required_history_length
    assert champion.forecaster.required_columns == fresh.required_columns
