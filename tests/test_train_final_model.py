"""Tests for scripts/train_final_model.py, the generalized (multi-family)
successor to scripts/train_final_lr.py (DECISIONS.md "Phase 6: serving
registration, 2026-09-25"). linear_regression is excluded here — it is
already registered via the original script — and only "random_forest"
is allowed in this stage; deep sequence-model families are a later stage.
"""

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

    with pytest.raises(ValueError, match="cnn_lstm"):
        train_final_model.check_can_run(
            "cnn_lstm", results_path=None, tracking_uri="sqlite:///unused.db"
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
