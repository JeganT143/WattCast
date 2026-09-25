"""Tests for the thin FastAPI adapter (DECISIONS.md "Phase 6: serving
registration, 2026-09-25"). Latency/live values used here are synthetic;
no test-partition value is read or compared.

Contract: POST /ingest is the only endpoint that mutates the rolling
buffer (validate + commit, no prediction). POST /predict is read-only:
it computes a prediction per requested model_family against the current
buffer state and never commits.
"""

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


class _FakeBuffer:
    def __init__(self, capacity=145, last_timestamp=None, is_ready=True, have=145):
        self.capacity = capacity
        self.last_timestamp = last_timestamp
        self.is_ready = is_ready
        self._have = have

    def __len__(self):
        return self._have


class _FakeService:
    def __init__(self, mode="ok", ready=True):
        self.mode = mode
        schema = {"model_family": "linear_regression", "horizon": 6, "model_version": 1}
        self.bundle = ModelBundle(forecaster=None, scaler=None, schema=schema)
        self.buffer = _FakeBuffer(
            is_ready=ready,
            last_timestamp=pd.Timestamp("2016-04-29 23:50:00") if ready else None,
            have=145 if ready else 100,
        )

    def ingest(self, ts, appliances):
        if self.mode == "duplicate":
            raise DuplicateTimestampError(ts)
        if self.mode == "non_successor":
            raise NonSuccessorTimestampError(ts, ts + pd.Timedelta(minutes=10))
        self.buffer.last_timestamp = ts
        self.buffer._have += 1
        self.buffer.is_ready = True

    def predict(self, model_families):
        if not self.buffer.is_ready:
            raise InsufficientHistoryError(have=len(self.buffer), need=self.buffer.capacity)

        origin = self.buffer.last_timestamp
        results = []
        for family in model_families:
            if family != "linear_regression":
                results.append({"model_family": family, "error": "not_available"})
                continue
            results.append(
                {
                    "model_family": family,
                    "origin_timestamp": origin,
                    "forecast_timestamp": origin + pd.Timedelta(minutes=60),
                    "prediction_wh": 42.0 + origin.minute,
                    "model_version": 1,
                }
            )
        return results


def _client_for(mode="ok", ready=True):
    app = create_app(load=lambda: _FakeService(mode, ready=ready))
    return TestClient(app)


# --- /ingest ---


def test_ingest_ok():
    with _client_for("ok") as client:
        resp = client.post(
            "/ingest", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["origin_timestamp"] == "2016-04-30T00:00:00"
        assert body["buffer_ready"] is True
        assert body["have"] == 146
        assert body["need"] == 145


def test_ingest_duplicate_409():
    with _client_for("duplicate") as client:
        resp = client.post(
            "/ingest", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert resp.status_code == 409
        assert resp.json()["status"] == "duplicate"


def test_ingest_non_successor_409_with_expected_next():
    with _client_for("non_successor") as client:
        resp = client.post(
            "/ingest", json={"timestamp": "2016-04-30T00:20:00", "appliances": 60.0}
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body["status"] == "non_successor"
        assert body["expected_next"] == "2016-04-30 00:30:00"


def test_ingest_non_finite_value_422():
    with _client_for("ok") as client:
        resp = client.post(
            "/ingest",
            content='{"timestamp": "2016-04-30T00:00:00", "appliances": NaN}',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422


def test_ingest_timezone_aware_timestamp_422():
    with _client_for("ok") as client:
        resp = client.post(
            "/ingest",
            json={"timestamp": "2016-04-30T00:00:00+00:00", "appliances": 60.0},
        )
        assert resp.status_code == 422


def test_ingest_malformed_body_422():
    with _client_for("ok") as client:
        resp = client.post("/ingest", json={"timestamp": "not-a-date", "appliances": 60.0})
        assert resp.status_code == 422

        resp2 = client.post("/ingest", json={"appliances": 60.0})
        assert resp2.status_code == 422


# --- /predict ---


def test_predict_ok():
    with _client_for("ok") as client:
        resp = client.post("/predict", json={"model_families": ["linear_regression"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == [
            {
                "model_family": "linear_regression",
                "origin_timestamp": "2016-04-29T23:50:00",
                "forecast_timestamp": "2016-04-30T00:50:00",
                "prediction_wh": 92.0,
                "model_version": 1,
            }
        ]


def test_predict_unregistered_family_returns_error_entry_not_failure():
    with _client_for("ok") as client:
        resp = client.post(
            "/predict", json={"model_families": ["linear_regression", "random_forest"]}
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
        by_family = {r["model_family"]: r for r in results}
        assert by_family["linear_regression"]["prediction_wh"] == 92.0
        assert by_family["random_forest"] == {
            "model_family": "random_forest",
            "error": "not_available",
        }


def test_predict_empty_model_families_422():
    with _client_for("ok") as client:
        resp = client.post("/predict", json={"model_families": []})
        assert resp.status_code == 422


def test_predict_insufficient_history_503_regardless_of_families():
    with _client_for("ok", ready=False) as client:
        resp = client.post(
            "/predict", json={"model_families": ["linear_regression", "random_forest"]}
        )
        assert resp.status_code == 503
        assert resp.json() == {"status": "insufficient_history", "have": 100, "need": 145}


def test_predict_does_not_mutate_buffer_via_http():
    with _client_for("ok") as client:
        have_before = client.get("/health").json()["have"]

        client.post("/predict", json={"model_families": ["linear_regression"]})
        client.post("/predict", json={"model_families": ["linear_regression"]})

        health_after = client.get("/health").json()
        assert health_after["have"] == have_before
        assert health_after["buffer_end"] == "2016-04-29T23:50:00"


def test_predict_twice_without_ingest_between_is_identical_via_http():
    with _client_for("ok") as client:
        first = client.post("/predict", json={"model_families": ["linear_regression"]}).json()
        second = client.post("/predict", json={"model_families": ["linear_regression"]}).json()
        assert first == second


def test_ingest_then_predict_then_ingest_then_predict_differs():
    with _client_for("ok") as client:
        client.post("/ingest", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0})
        first = client.post("/predict", json={"model_families": ["linear_regression"]}).json()

        client.post("/ingest", json={"timestamp": "2016-04-30T00:10:00", "appliances": 70.0})
        second = client.post("/predict", json={"model_families": ["linear_regression"]}).json()

        assert first["results"][0]["forecast_timestamp"] != second["results"][0]["forecast_timestamp"]


# --- /health, /model ---


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
        assert body["buffer_ready"] is True
        assert body["have"] == 145
        assert body["need"] == 145


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
        assert health.json()["buffer_ready"] is True

        ingest_resp = client.post(
            "/ingest", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert ingest_resp.status_code == 200
        assert ingest_resp.json()["buffer_ready"] is True

        predict_resp = client.post(
            "/predict", json={"model_families": ["linear_regression"]}
        )
        assert predict_resp.status_code == 200
        assert predict_resp.json()["results"][0]["forecast_timestamp"] == "2016-04-30T01:00:00"

        repeat_ingest = client.post(
            "/ingest", json={"timestamp": "2016-04-30T00:00:00", "appliances": 60.0}
        )
        assert repeat_ingest.status_code == 409

        gap_ingest = client.post(
            "/ingest", json={"timestamp": "2016-04-30T00:20:00", "appliances": 60.0}
        )
        assert gap_ingest.status_code == 409


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
        "ingest_resp = c.post('/ingest', json={'timestamp': '2016-04-30T00:00:00', 'appliances': 60.0})\n"
        "assert ingest_resp.status_code == 200\n"
        "predict_resp = c.post('/predict', json={'model_families': ['linear_regression']})\n"
        "for result in predict_resp.json()['results']:\n"
        "    for value in result.values():\n"
        "        assert not isinstance(value, torch.Tensor), value\n"
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
