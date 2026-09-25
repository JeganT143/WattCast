"""Measure serving latency of the linear-regression champion: startup,
in-process ingest+predict, and HTTP round-trip through the FastAPI app.

All timestamps are synthetic, starting at 2016-04-30 00:00 on the 10-minute
grid, with synthetic Appliances values (60.0 + 10.0 * (i % 7)). No row of
the held-out test partition is read — this measures latency, not accuracy.

    python -m scripts.measure_serving_latency [--out PATH]

Refuses to overwrite an existing results file.
"""

import argparse
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from config.paths import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, SEED_HISTORY_PATH
from src.serving.bundle import load_bundle, load_bundles
from src.serving.service import ServingService, create_service

RESULTS_PATH = RESULTS_DIR / "serving_latency.json"
FAMILY = "linear_regression"
STEP = pd.Timedelta(minutes=10)
FIRST_LIVE_TS = pd.Timestamp("2016-04-30 00:00:00")
N_CORE = 1000
N_HTTP = 500
HEALTH_TIMEOUT_S = 60.0


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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float) -> bool:
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


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()


def _synthetic_value(i: int) -> float:
    return 60.0 + 10.0 * (i % 7)


def _build_lr_service() -> ServingService:
    bundles = {FAMILY: load_bundle(MODELS_DIR / FAMILY)}
    seed = pd.read_csv(SEED_HISTORY_PATH, parse_dates=["date"])
    return create_service(bundles, seed)


def _measure_startup(n_runs: int = 5) -> list[float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _build_lr_service()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _measure_core(service: ServingService, n: int) -> list[float]:
    times = []
    ts = FIRST_LIVE_TS
    for i in range(n):
        t0 = time.perf_counter()
        service.ingest(ts, _synthetic_value(i))
        service.predict([FAMILY])
        times.append((time.perf_counter() - t0) * 1000.0)
        ts += STEP
    return times


def _measure_http(n: int) -> list[float]:
    """Each call is one /ingest plus one /predict, the same unit of work as
    the in-process measurement. The subprocess seeds its own fresh buffer,
    so its timestamps restart at FIRST_LIVE_TS."""
    port = _find_free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.serving.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT),
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        if not _wait_for_health(base_url, timeout=HEALTH_TIMEOUT_S):
            raise RuntimeError(f"API subprocess did not become ready within {HEALTH_TIMEOUT_S}s")
        times = []
        ts = FIRST_LIVE_TS
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            for i in range(n):
                t0 = time.perf_counter()
                ingest = client.post("/ingest", json={"timestamp": ts.isoformat(), "appliances": _synthetic_value(i)})
                predict = client.post("/predict", json={"model_families": [FAMILY]})
                times.append((time.perf_counter() - t0) * 1000.0)
                if ingest.status_code != 200 or predict.status_code != 200:
                    raise RuntimeError(f"HTTP call {i} failed: {ingest.text} / {predict.text}")
                ts += STEP
        return times
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(out: Path = RESULTS_PATH) -> dict:
    if out.exists():
        sys.exit(f"refusing to overwrite existing results file: {out}")

    startup_times = _measure_startup()
    core_times = _measure_core(_build_lr_service(), N_CORE)
    http_times = _measure_http(N_HTTP)
    schema = load_bundles(MODELS_DIR)[FAMILY].schema

    payload = {
        "startup_ms": {"runs": startup_times, "mean": float(np.mean(startup_times)), "max": float(np.max(startup_times))},
        "core_ingest_predict_ms": _stats_ms(core_times),
        "http_ingest_predict_ms": _stats_ms(http_times),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("fastapi", "uvicorn", "httpx", "scikit-learn", "numpy", "pandas")
        },
        "code_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "model_version": schema.get("model_version"),
        "run_id": schema.get("run_id"),
        "timestamp_of_measurement": datetime.now(timezone.utc).isoformat(),
        "synthetic_data": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    print(json.dumps(main(parser.parse_args().out), indent=2, sort_keys=True))
