"""Tests for the pure, testable logic behind the Streamlit UI
(src/ui/client.py): starting/health-polling the FastAPI serving subprocess
and the thin HTTP client over its /health, /model, /ingest, /predict
contract. The Streamlit script itself (streamlit_app.py) is not covered
here — it is not meaningfully unit-testable and is smoke-tested manually
instead.

All timestamps/values used are SYNTHETIC (2016-04-30 00:00 onward, on the
10-minute grid) — no test-partition row is read or displayed.
"""

import math
import socket
import subprocess

import pytest

from config.paths import PROJECT_ROOT
from src.ui.client import (
    ApiClient,
    ApiError,
    compute_next_timestamp,
    find_free_port,
    start_server_subprocess,
    wait_for_health,
)

HEALTH_TIMEOUT_S = 30.0
ALL_FAMILIES = ["linear_regression", "random_forest", "lstm", "gru", "cnn_lstm"]


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def test_find_free_port_returns_a_port_that_is_actually_bindable():
    port = find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))  # must not raise


def test_start_server_subprocess_becomes_healthy_then_terminates_cleanly():
    port = find_free_port()
    proc = start_server_subprocess(port, cwd=PROJECT_ROOT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        assert wait_for_health(base_url, timeout=HEALTH_TIMEOUT_S) is True
    finally:
        _terminate(proc)

    assert proc.poll() is not None  # confirmed gone


@pytest.fixture(scope="module")
def live_server():
    port = find_free_port()
    proc = start_server_subprocess(port, cwd=PROJECT_ROOT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        assert wait_for_health(base_url, timeout=HEALTH_TIMEOUT_S) is True
        yield base_url
    finally:
        _terminate(proc)


def test_api_client_health_and_model_report_expected_shape(live_server):
    client = ApiClient(live_server)

    health = client.health()
    assert health["ready"] is True
    assert health["buffer_ready"] is True
    assert "buffer_end" in health

    model = client.model()
    assert sorted(model["available_families"]) == sorted(ALL_FAMILIES)


def test_api_client_predict_all_five_families_returns_finite_predictions(live_server):
    client = ApiClient(live_server)
    response = client.predict(ALL_FAMILIES)
    results = response["results"]

    assert len(results) == 5
    for r in results:
        assert "error" not in r, r
        assert math.isfinite(r["prediction_wh"])


def test_api_client_ingest_success_then_duplicate_raises_clear_error(live_server):
    client = ApiClient(live_server)
    health = client.health()
    next_ts = compute_next_timestamp(health)

    result = client.ingest(next_ts, 60.0)
    assert result["buffer_ready"] is True
    assert result["origin_timestamp"] == next_ts.isoformat()

    with pytest.raises(ApiError) as excinfo:
        client.ingest(next_ts, 60.0)
    message = str(excinfo.value)
    assert "409" in message
    assert "duplicate" in message.lower()


def test_compute_next_timestamp_is_genuinely_what_the_server_accepts_next(live_server):
    """Don't just assert a hardcoded expected value: actually ingest at the
    computed timestamp and confirm the server accepts it (200, not 409)."""
    client = ApiClient(live_server)
    health = client.health()
    computed = compute_next_timestamp(health)

    result = client.ingest(computed, 77.0)

    assert result["origin_timestamp"] == computed.isoformat()
    assert result["buffer_ready"] is True

    # And the timestamp right after that is the one the server accepted,
    # confirmed against a fresh /health call, not the client's own guess.
    fresh_health = client.health()
    assert fresh_health["buffer_end"] == computed.isoformat()
