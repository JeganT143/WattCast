"""
Tests for the MLflow logging layer. Every test uses an isolated,
tmp_path-based SQLite store — never the real db/mlflow.db.
"""

from pathlib import Path

import mlflow
import numpy as np
import pytest

from src.evaluation.harness import EvaluationResult
from src.models.naive import NaivePersistenceForecaster
from src.models.sklearn_models import LinearRegressionForecaster
from src.tracking.mlflow_logger import log_evaluation_result


def _make_result(model_name="naive_persistence", horizon=1):
    metrics = {
        "train": {"mae": 30.84, "rmse": 74.17, "mape": 25.47},
        "val": {"mae": 25.24, "rmse": 65.30, "mape": 21.78},
        "test": {"mae": 26.50, "rmse": 66.14, "mape": 21.66},
    }
    arrays = {p: np.array([1.0, 2.0, 3.0]) for p in ("train", "val", "test")}
    return EvaluationResult(
        model_name=model_name,
        horizon=horizon,
        metrics=metrics,
        predictions=arrays,
        actuals=arrays,
    )


def _make_scaler_file(tmp_path):
    scaler_path = tmp_path / "scaler_train_fit.joblib"
    scaler_path.write_bytes(
        b"fake scaler content"
    )  # content irrelevant, only existence matters
    return str(scaler_path)


@pytest.fixture
def tracking_uri(tmp_path):
    db_path = tmp_path / "test_mlflow.db"
    return f"sqlite:///{db_path}"


def _get_client(tracking_uri):
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


# ---------------------------------------------------------------------
# Core logging behavior
# ---------------------------------------------------------------------


def test_experiment_name_is_wattcast(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    experiments = client.search_experiments()
    names = [e.name for e in experiments]
    assert "WattCast" in names


def test_exactly_one_run_created(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    exp = client.get_experiment_by_name("WattCast")
    runs = client.search_runs(exp.experiment_id)
    assert len(runs) == 1


def test_run_name_follows_convention(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result(model_name="naive_persistence", horizon=6)
    scaler_path = _make_scaler_file(tmp_path)

    log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    exp = client.get_experiment_by_name("WattCast")
    run = client.search_runs(exp.experiment_id)[0]
    assert run.data.tags.get("mlflow.runName") == "naive_persistence_h6"


def test_forecaster_params_are_logged(tmp_path, tracking_uri):
    forecaster = LinearRegressionForecaster(fit_intercept=False)
    result = _make_result(model_name="linear_regression")
    scaler_path = _make_scaler_file(tmp_path)

    log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    exp = client.get_experiment_by_name("WattCast")
    run = client.search_runs(exp.experiment_id)[0]
    assert run.data.params.get("fit_intercept") == "False"


def test_mape_threshold_and_required_columns_are_logged(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    exp = client.get_experiment_by_name("WattCast")
    run = client.search_runs(exp.experiment_id)[0]
    assert run.data.params.get("mape_threshold") == "30.0"
    assert run.data.params.get("required_columns") == "Appliances"


def test_all_nine_metrics_logged_correctly(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    exp = client.get_experiment_by_name("WattCast")
    run = client.search_runs(exp.experiment_id)[0]

    expected = {
        "train_mae": 30.84,
        "train_rmse": 74.17,
        "train_mape": 25.47,
        "val_mae": 25.24,
        "val_rmse": 65.30,
        "val_mape": 21.78,
        "test_mae": 26.50,
        "test_rmse": 66.14,
        "test_mape": 21.66,
    }
    for key, value in expected.items():
        assert run.data.metrics.get(key) == pytest.approx(value)


def test_git_commit_sha_tag_present_and_nonempty(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    exp = client.get_experiment_by_name("WattCast")
    run = client.search_runs(exp.experiment_id)[0]
    sha = run.data.tags.get("git_commit_sha")
    assert sha is not None
    assert len(sha) > 0


# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------


def test_predictions_table_artifact_exists(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    run_id = log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "predictions.json" in artifacts


def test_scaler_artifact_exists(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    run_id = log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )

    client = _get_client(tracking_uri)
    artifacts = [a.path for a in client.list_artifacts(run_id, path="scaler")]
    assert any("scaler_train_fit.joblib" in a for a in artifacts)


def test_native_model_logged_when_supplied(tmp_path, tracking_uri):
    from sklearn.linear_model import LinearRegression

    forecaster = LinearRegressionForecaster()
    result = _make_result(model_name="linear_regression")
    scaler_path = _make_scaler_file(tmp_path)

    native_model = LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0])

    run_id = log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        native_model=native_model,
        tracking_uri=tracking_uri,
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )  # search_logged_models needs an active tracking URI, not just a client
    exp = mlflow.get_experiment_by_name("WattCast")
    logged_models = mlflow.search_logged_models(
        experiment_ids=[exp.experiment_id],
        filter_string=f"source_run_id='{run_id}'",
    )
    assert len(logged_models) == 1


def test_no_model_artifact_when_native_model_none(tmp_path, tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()
    scaler_path = _make_scaler_file(tmp_path)

    run_id = log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        native_model=None,
        tracking_uri=tracking_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name("WattCast")
    logged_models = mlflow.search_logged_models(
        experiment_ids=[exp.experiment_id],
        filter_string=f"source_run_id='{run_id}'",
    )
    assert len(logged_models) == 0


# ---------------------------------------------------------------------
# Isolation and failure guard
# ---------------------------------------------------------------------


def test_logging_never_touches_real_tracking_store(tmp_path, tracking_uri):
    """Guards the store-fragmentation incident (decisions.md, ADR-010): with a
    tracking_uri override, the real store file must be neither created nor
    written. Checked by file state only, so the guard itself never opens it."""
    from config.mlflow_config import TRACKING_URI as REAL_TRACKING_URI

    real_store = Path(REAL_TRACKING_URI.removeprefix("sqlite:///"))

    def state():
        return real_store.stat().st_mtime_ns if real_store.exists() else None

    before = state()
    log_evaluation_result(
        _make_result(),
        NaivePersistenceForecaster(),
        mape_threshold=30.0,
        scaler_path=_make_scaler_file(tmp_path),
        tracking_uri=tracking_uri,
    )
    assert state() == before


def test_missing_scaler_path_raises_file_not_found(tracking_uri):
    forecaster = NaivePersistenceForecaster()
    result = _make_result()

    with pytest.raises(FileNotFoundError):
        log_evaluation_result(
            result,
            forecaster,
            mape_threshold=30.0,
            scaler_path="/nonexistent/path/scaler.joblib",
            tracking_uri=tracking_uri,
        )


def test_random_forest_native_model_logged_when_supplied(tmp_path, tracking_uri):
    from src.models.sklearn_models import RandomForestForecaster
    import numpy as np

    forecaster = RandomForestForecaster(n_estimators=5)  # small, fast for a test
    result = _make_result(model_name="random_forest")
    scaler_path = _make_scaler_file(tmp_path)

    X = np.random.rand(20, 1)
    y = np.random.rand(20)
    forecaster.fit(
        X, y
    )  # must actually fit — an unfitted RF has no real Tree structure to trigger the skops check

    run_id = log_evaluation_result(
        result,
        forecaster,
        mape_threshold=30.0,
        scaler_path=scaler_path,
        native_model=forecaster.model,
        tracking_uri=tracking_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name("WattCast")
    logged_models = mlflow.search_logged_models(
        experiment_ids=[exp.experiment_id],
        filter_string=f"source_run_id='{run_id}'",
    )
    assert len(logged_models) == 1
