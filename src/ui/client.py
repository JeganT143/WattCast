"""Pure, pytest-testable logic behind the Streamlit UI: starting and
health-polling the FastAPI serving subprocess, and a thin HTTP client over
its existing /health, /model, /ingest, /predict contract (src/serving/app.py
— unchanged by this module). Kept separate from streamlit_app.py, which is
UI glue and not meaningfully unit-testable on its own.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

STEP = pd.Timedelta(minutes=10)


def find_free_port() -> int:
    """Binds to port 0 to let the OS pick a free ephemeral port, then
    releases it immediately — the standard find-then-race pattern used
    elsewhere in this repo (scripts/measure_serving_latency.py)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server_subprocess(port: int, cwd: str | Path | None = None) -> subprocess.Popen:
    """Launches the existing FastAPI app under uvicorn as a subprocess.
    Returns the Popen immediately — callers must poll wait_for_health
    before using it, and are responsible for terminating it."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.serving.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(cwd) if cwd is not None else None,
    )


def wait_for_health(base_url: str, timeout: float = 15.0) -> bool:
    """Polls GET /health until it reports ready, or timeout elapses."""
    deadline = time.monotonic() + timeout
    with httpx.Client(base_url=base_url, timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = client.get("/health")
                if resp.status_code == 200 and resp.json().get("ready"):
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    return False


class ApiError(Exception):
    """Raised for any non-2xx response; the message includes the response
    status and body so a caller (or the Streamlit UI) never has to guess
    what the server actually said."""


class ApiClient:
    """Thin wrapper over the serving API's existing HTTP contract. Never
    reimplements validation or business logic — every rule (successor
    timestamps, finite values, family availability) lives in
    src/serving/app.py and src/serving/service.py; this just calls them
    and surfaces errors clearly."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _handle(self, resp: httpx.Response) -> dict:
        if resp.status_code >= 400:
            raise ApiError(
                f"{resp.status_code} from {resp.request.method} {resp.request.url}: {resp.text}"
            )
        return resp.json()

    def health(self) -> dict:
        return self._handle(self._client.get("/health"))

    def model(self) -> dict:
        return self._handle(self._client.get("/model"))

    def ingest(self, timestamp, appliances: float) -> dict:
        ts = pd.Timestamp(timestamp)
        return self._handle(
            self._client.post(
                "/ingest", json={"timestamp": ts.isoformat(), "appliances": appliances}
            )
        )

    def predict(self, model_families: list[str]) -> dict:
        return self._handle(
            self._client.post("/predict", json={"model_families": model_families})
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def compute_next_timestamp(health_response: dict) -> pd.Timestamp:
    """The timestamp the UI should default its next-ingest input to: the
    server only ever accepts buffer_end + 10 minutes as the next successor
    timestamp (RollingBuffer.check_next, src/serving/buffer.py), so that is
    exactly what this derives from the real /health response's buffer_end
    field — never a hardcoded or guessed value."""
    buffer_end = health_response.get("buffer_end")
    if buffer_end is None:
        raise ValueError(
            "health response has no buffer_end; cannot compute the next ingest timestamp"
        )
    return pd.Timestamp(buffer_end) + STEP
