"""Tests for the thin FastAPI adapter (DECISIONS.md "Phase 6: serving
registration, 2026-09-25"). Latency/live values used here are synthetic;
no test-partition value is read or compared."""

import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.bundle import ModelBundle
from src.serving.buffer import (
    DuplicateTimestampError,
    InsufficientHistoryError,
    NonSuccessorTimestampError,
)
from src.serving.service import PredictionResult


class _FakeBuffer:
    def __init__(self, capacity=145, last_timestamp=None, is_ready=True):
        self.capacity = capacity
        self.last_timestamp = last_timestamp
        self.is_ready = is_ready


class _FakeService:
    def __init__(self, mode="ok"):
        self.mode = mode
        schema = {"model_family": "linear_regression", "horizon": 6, "model_version": 1}
        self.bundle = ModelBundle(forecaster=None, scaler=None, schema=schema)
        self.buffer = _FakeBuffer(last_timestamp=pd.Timestamp("2016-04-29 23:50:00"))

    def predict(self, ts, appliances):
        if self.mode == "duplicate":
            raise DuplicateTimestampError(ts)
        if self.mode == "non_successor":
            raise NonSuccessorTimestampError(ts, ts + pd.Timedelta(minutes=10))
        if self.mode == "insufficient":
            raise InsufficientHistoryError(have=100, need=145)
        return PredictionResult(
            origin_timestamp=ts,
            forecast_timestamp=ts + pd.Timedelta(minutes=60),
            prediction_wh=42.0,
        )


def _client_for(mode="ok"):
    app = create_app(load=lambda: _FakeService(mode))
    return TestClient(app)


def test_predict_ok():
    with _client_for("ok") as client:
        resp = client.post(
            "/predict", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction_wh"] == 42.0
        assert body["horizon_steps"] == 6
        assert body["model_family"] == "linear_regression"
        assert body["model_version"] == 1
        assert body["forecast_timestamp"] == "2016-04-30T01:00:00"


def test_predict_duplicate_409():
    with _client_for("duplicate") as client:
        resp = client.post(
            "/predict", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert resp.status_code == 409
        assert resp.json()["status"] == "duplicate"


def test_predict_non_successor_409():
    with _client_for("non_successor") as client:
        resp = client.post(
            "/predict", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert resp.status_code == 409
        assert resp.json()["status"] == "non_successor"
        assert "expected_next" in resp.json()


def test_predict_insufficient_history_503():
    with _client_for("insufficient") as client:
        resp = client.post(
            "/predict", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body == {"status": "insufficient_history", "have": 100, "need": 145}


def test_predict_non_finite_value_422():
    with _client_for("ok") as client:
        resp = client.post(
            "/predict",
            content='{"timestamp": "2016-04-30T00:00:00", "appliances": NaN}',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422


def test_predict_timezone_aware_timestamp_422():
    with _client_for("ok") as client:
        resp = client.post(
            "/predict",
            json={"timestamp": "2016-04-30T00:00:00+00:00", "appliances": 60.0},
        )
        assert resp.status_code == 422


def test_predict_malformed_body_422():
    with _client_for("ok") as client:
        resp = client.post("/predict", json={"timestamp": "not-a-date", "appliances": 60.0})
        assert resp.status_code == 422

        resp2 = client.post("/predict", json={"appliances": 60.0})
        assert resp2.status_code == 422


def test_health_route():
    with _client_for("ok") as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_family"] == "linear_regression"
        assert body["model_version"] == 1
        assert body["horizon"] == 6
        assert body["ready"] is True
        assert body["required_raw_history"] == 145


def test_model_route():
    with _client_for("ok") as client:
        resp = client.get("/model")
        assert resp.status_code == 200
        assert resp.json()["model_family"] == "linear_regression"


@pytest.mark.slow
def test_real_champion_end_to_end():
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ready"] is True

        resp = client.post(
            "/predict", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert resp.status_code == 200
        assert resp.json()["forecast_timestamp"] == "2016-04-30T01:00:00"

        repeat = client.post(
            "/predict", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert repeat.status_code == 409

        gap = client.post(
            "/predict", json={"timestamp": "2016-04-30T00:20:00", "appliances": 60.0}
        )
        assert gap.status_code == 409


def test_serving_uses_no_torch_model_or_tensors():
    """DECISIONS.md "Phase 6: serving dependency claim amendment, 2026-09-25":
    skops/sklearn transitively import torch via all_estimators() in this
    environment, so the literal 'torch' not in sys.modules assertion was
    withdrawn. This checks what actually matters instead: the LR serving
    path never touches a torch model, checkpoint file, or tensor.
    """
    script = (
        "import os\n"
        "import types\n"
        "from fastapi.testclient import TestClient\n"
        "from sklearn.linear_model import LinearRegression\n"
        "import src.serving.app as m\n"
        "c = TestClient(m.app)\n"
        "c.__enter__()\n"
        "resp = c.get('/health')\n"
        "assert resp.status_code == 200\n"
        "\n"
        "service = m.app.state.service\n"
        "forecaster = service.bundle.forecaster\n"
        "\n"
        "import torch\n"
        "\n"
        "assert type(forecaster.model) is LinearRegression, type(forecaster.model)\n"
        "assert not isinstance(forecaster.model, torch.nn.Module)\n"
        "\n"
        "for name in dir(service.bundle):\n"
        "    if name.startswith('_'):\n"
        "        continue\n"
        "    value = getattr(service.bundle, name)\n"
        "    assert not isinstance(value, torch.nn.Module), name\n"
        "    assert not isinstance(value, torch.Tensor), name\n"
        "\n"
        "for name in dir(forecaster):\n"
        "    if name.startswith('_'):\n"
        "        continue\n"
        "    try:\n"
        "        value = getattr(forecaster, name)\n"
        "    except Exception:\n"
        "        continue\n"
        "    if isinstance(value, types.MethodType):\n"
        "        continue\n"
        "    assert not isinstance(value, torch.nn.Module), name\n"
        "    assert not isinstance(value, torch.Tensor), name\n"
        "\n"
        "predict_resp = c.post('/predict', json={'timestamp': '2016-04-30T00:00:00', 'appliances': 60.0})\n"
        "for value in predict_resp.json().values():\n"
        "    assert not isinstance(value, torch.Tensor), value\n"
        "\n"
        "import config.paths as paths\n"
        "artifact_root = str(paths.PROJECT_ROOT / 'models')\n"
        "found_checkpoint = []\n"
        "if os.path.isdir(artifact_root):\n"
        "    for root, _dirs, files in os.walk(artifact_root):\n"
        "        for fname in files:\n"
        "            if fname.endswith('.pt') or fname.endswith('.pth'):\n"
        "                found_checkpoint.append(os.path.join(root, fname))\n"
        "assert found_checkpoint == [], found_checkpoint\n"
        "\n"
        "print('no torch model, checkpoint, or tensor in the LR serving path')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=None,
    )
    assert "no torch model, checkpoint, or tensor in the LR serving path" in result.stdout, (
        result.stdout + result.stderr
    )
