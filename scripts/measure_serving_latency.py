"""Measure serving latency of the LR champion: startup, in-process
prediction, and HTTP round-trip (DECISIONS.md "Phase 6: serving
registration, 2026-09-25").

All timestamps used for the timed prediction calls are synthetic,
starting at 2016-04-30 00:00 and moving forward on the 10-minute grid,
with synthetic Appliances values (60.0 + 10.0 * (i % 7)). No row with
date >= 2016-04-30 (the held-out test partition) is ever read or
compared against — this measures latency only, not accuracy.
"""

import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from config.mlflow_config import TRACKING_URI
from config.paths import PROJECT_ROOT, RAW_DATA_PATH
from src.serving.buffer import required_raw_history, seed_buffer
from src.serving.registry import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, load_champion
from src.serving.service import ServingService

RESULTS_PATH = PROJECT_ROOT / "results" / "serving_latency.json"
STEP = pd.Timedelta(minutes=10)
FIRST_LIVE_TS = pd.Timestamp("2016-04-30 00:00:00")
N_CORE = 1000
N_HTTP = 500
HEALTH_TIMEOUT_S = 15.0


def _stats_ms(times_ms: list[float]) -> dict:
    arr = np.array(times_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "n": len(arr),
    }


def _get_git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _measure_startup(n_runs: int = 5) -> list[float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        bundle = load_champion(TRACKING_URI, family="linear_regression")
        raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
        buffer = seed_buffer(raw, required_raw_history(), bundle.schema["seed_end"])
        ServingService(bundle, buffer)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _build_service() -> ServingService:
    bundle = load_champion(TRACKING_URI, family="linear_regression")
    raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    buffer = seed_buffer(raw, required_raw_history(), bundle.schema["seed_end"])
    return ServingService(bundle, buffer)


def _measure_core(service: ServingService, start_ts: pd.Timestamp, n: int) -> tuple[list[float], pd.Timestamp]:
    times = []
    ts = start_ts
    for i in range(n):
        value = 60.0 + 10.0 * (i % 7)
        t0 = time.perf_counter()
        service.predict(ts, value)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        ts = ts + STEP
    return times, ts


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _measure_http(start_ts: pd.Timestamp, n: int, n_warmup: int) -> list[float]:
    port = _find_free_port()
    proc = subprocess.Popen(
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
        cwd=str(PROJECT_ROOT),
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        ready = False
        with httpx.Client(base_url=base_url, timeout=2.0) as probe:
            while time.monotonic() < deadline:
                try:
                    resp = probe.get("/health")
                    if resp.status_code == 200 and resp.json().get("ready"):
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        if not ready:
            raise RuntimeError(
                f"uvicorn subprocess (pid {proc.pid}) did not become ready within {HEALTH_TIMEOUT_S}s"
            )

        # This subprocess boots with its own freshly seeded buffer (anchored
        # at the same fixed historical seed_end as the in-process core
        # measurement), so it always accepts FIRST_LIVE_TS first — there is
        # no way to seed a live server past that anchor except by genuinely
        # replaying calls through it. To have the *timed* HTTP calls
        # continue from where the core measurement left off (start_ts)
        # without reusing timestamps for the timed portion, first replay
        # the same untimed volume the core measurement used to advance this
        # server's buffer to the same state, then measure only the
        # continuation.
        with httpx.Client(base_url=base_url, timeout=5.0) as warmup_client:
            warmup_ts = FIRST_LIVE_TS
            for i in range(n_warmup):
                value = 60.0 + 10.0 * (i % 7)
                resp = warmup_client.post(
                    "/predict",
                    json={"timestamp": warmup_ts.isoformat(), "appliances": value},
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP warmup /predict failed at i={i}: {resp.status_code} {resp.text}"
                    )
                warmup_ts = warmup_ts + STEP
            if warmup_ts != start_ts:
                raise RuntimeError(
                    f"warmup did not reach the expected continuation point: "
                    f"{warmup_ts} != {start_ts}"
                )

        times = []
        ts = start_ts
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            for i in range(n):
                value = 60.0 + 10.0 * (i % 7)
                payload = {"timestamp": ts.isoformat(), "appliances": value}
                t0 = time.perf_counter()
                resp = client.post("/predict", json=payload)
                t1 = time.perf_counter()
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP /predict failed at i={i}: {resp.status_code} {resp.text}"
                    )
                times.append((t1 - t0) * 1000.0)
                ts = ts + STEP
        return times
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def main() -> dict:
    if RESULTS_PATH.exists():
        print(f"refusing to overwrite existing results file: {RESULTS_PATH}", file=sys.stderr)
        sys.exit(1)

    client = MlflowClient(tracking_uri=TRACKING_URI)
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    startup_times = _measure_startup(n_runs=5)

    service = _build_service()
    core_times, next_ts = _measure_core(service, FIRST_LIVE_TS, N_CORE)

    http_times = _measure_http(next_ts, N_HTTP, n_warmup=N_CORE)

    package_versions = {
        name: importlib.metadata.version(dist)
        for name, dist in [
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("httpx", "httpx"),
            ("mlflow", "mlflow"),
            ("scikit-learn", "scikit-learn"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
        ]
    }

    payload = {
        "startup_ms": {
            "runs": startup_times,
            "mean": float(np.mean(startup_times)),
            "max": float(np.max(startup_times)),
        },
        "core_predict_ms": _stats_ms(core_times),
        "http_predict_ms": _stats_ms(http_times),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "package_versions": package_versions,
        "code_sha": _get_git_commit_sha(),
        "model_version": int(mv.version),
        "run_id": mv.run_id,
        "timestamp_of_measurement": datetime.now(timezone.utc).isoformat(),
        "synthetic_data": True,
        "synthetic_data_note": (
            "All predict() calls used synthetic timestamps (2016-04-30 00:00 onward, "
            "10-minute grid) and synthetic Appliances values; no row with date >= "
            "2016-04-30 was read or compared."
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, sort_keys=True))
