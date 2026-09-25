"""Tests for the serving model bundle and MLflow registry loader
(DECISIONS.md "Phase 6: serving registration, 2026-09-25")."""

import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from config.mlflow_config import EXPERIMENT_NAME
from src.models.sklearn_models import LinearRegressionForecaster
from src.serving.bundle import (
    ModelBundle,
    feature_schema_version,
    load_bundle,
    required_raw_history,
    save_bundle,
)
from src.serving.registry import CHAMPION_ALIAS, load_champion, register_bundle


def _tiny_bundle(seed: int = 0) -> ModelBundle:
    rng = np.random.default_rng(seed)
    n_cols = len(FEATURE_COLUMNS)
    X = rng.uniform(-1, 1, size=(50, n_cols))
    y = rng.uniform(0, 100, size=50)

    forecaster = LinearRegressionForecaster()
    forecaster.fit(X, y)

    scaler = StandardScaler().fit(X)

    schema = {
        "model_family": "linear_regression",
        "horizon": 6,
        "feature_columns": FEATURE_COLUMNS,
        "scaled_columns": SCALED_COLUMNS,
        "required_columns": FEATURE_COLUMNS,
        "required_raw_history": required_raw_history(),
        "train_start": "2016-01-12 17:00:00",
        "train_end": "2016-04-29 22:50:00",
        "seed_end": "2016-04-29 23:50:00",
        "n_fit_rows": 50,
        "scaler_convention": "test-fixture",
        "code_sha": "deadbeef",
        "feature_schema_version": feature_schema_version(FEATURE_COLUMNS),
    }
    return ModelBundle(forecaster=forecaster, scaler=scaler, schema=schema)


def _fixed_X(n_cols: int) -> np.ndarray:
    return np.linspace(-1, 1, num=5 * n_cols).reshape(5, n_cols)


def test_save_load_round_trip_bit_identical(tmp_path):
    bundle = _tiny_bundle()
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)

    loaded = load_bundle(directory)

    n_cols = len(FEATURE_COLUMNS)
    X = _fixed_X(n_cols)
    original_preds = bundle.forecaster.predict(X)
    loaded_preds = loaded.forecaster.predict(X)

    assert np.array_equal(original_preds, loaded_preds)
    assert loaded.schema == bundle.schema


def test_load_bundle_raises_on_changed_feature_columns(tmp_path):
    bundle = _tiny_bundle()
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)

    import json

    schema_path = directory / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema["feature_columns"] = ["not_a_real_column"]
    schema_path.write_text(json.dumps(schema))

    with pytest.raises(ValueError):
        load_bundle(directory)


def test_load_bundle_raises_on_unknown_model_family(tmp_path):
    bundle = _tiny_bundle()
    directory = tmp_path / "bundle"
    save_bundle(bundle, directory)

    import json

    schema_path = directory / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema["model_family"] = "some_unregistered_family"
    schema_path.write_text(json.dumps(schema))

    with pytest.raises(ValueError):
        load_bundle(directory)


def test_register_bundle_then_load_champion(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "artifacts")

    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)

    bundle = _tiny_bundle()
    bundle_dir = tmp_path / "bundle_to_register"
    save_bundle(bundle, bundle_dir)

    run_id, version = register_bundle(
        bundle_dir, tracking_uri, tags={"selection_basis": "test"}
    )

    assert isinstance(run_id, str) and run_id
    assert version == 1

    champion = load_champion(tracking_uri)

    n_cols = len(FEATURE_COLUMNS)
    X = _fixed_X(n_cols)
    assert np.array_equal(champion.forecaster.predict(X), bundle.forecaster.predict(X))

    mv = client.get_model_version_by_alias(
        "wattcast_serving_lr_h6", CHAMPION_ALIAS
    )
    assert int(mv.version) == version
