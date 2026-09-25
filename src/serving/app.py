"""Thin FastAPI adapter over ServingService (DECISIONS.md "Phase 6: serving
registration, 2026-09-25"). Routes depend only on ServingService and the
Forecaster interface via its bundle's schema dict — there is no model
class, model path, or model_family branching here.

Importing this module must not load anything: the champion, the raw
history, and the rolling buffer are all constructed lazily inside the
app's lifespan, never at import time.
"""

import math
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from config.mlflow_config import TRACKING_URI
from config.paths import RAW_DATA_PATH
from src.serving.bundle import LOADERS, ModelBundle
from src.serving.buffer import (
    SEQUENCE_MODEL_RAW_HISTORY,
    DuplicateTimestampError,
    InsufficientHistoryError,
    NonSuccessorTimestampError,
    seed_buffer,
)
from src.serving.registry import CHAMPION_ALIAS, load_champion, registered_model_name
from src.serving.service import ServingService


class IngestRequest(BaseModel):
    timestamp: datetime
    appliances: float


class PredictRequest(BaseModel):
    model_families: list[str]


def _load_bundle_with_version(client: MlflowClient, family: str) -> ModelBundle:
    bundle = load_champion(TRACKING_URI, family=family)
    model_version = client.get_model_version_by_alias(
        registered_model_name(family), CHAMPION_ALIAS
    ).version
    return ModelBundle(
        forecaster=bundle.forecaster,
        scaler=bundle.scaler,
        schema={**bundle.schema, "model_version": int(model_version)},
    )


def _default_load() -> ServingService:
    # Preloads every family with a registered bundle loader (LOADERS is the
    # existing single source of truth for which families the bundle
    # mechanism supports — src/serving/bundle.py), not just linear_regression:
    # /predict accepts any requested model_families, so startup must resolve
    # all of them, not only the one the service happens to hold as primary.
    client = MlflowClient(tracking_uri=TRACKING_URI)
    bundles = {family: _load_bundle_with_version(client, family) for family in LOADERS}
    primary = bundles["linear_regression"]
    extra_bundles = [b for family, b in bundles.items() if family != "linear_regression"]

    raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    buffer = seed_buffer(raw, SEQUENCE_MODEL_RAW_HISTORY, primary.schema["seed_end"])
    return ServingService(primary, buffer, extra_bundles=extra_bundles)


def create_app(load: Callable[[], ServingService] = _default_load) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = load()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/ingest")
    async def ingest(payload: IngestRequest, request: Request) -> Any:
        service: ServingService = request.app.state.service

        timestamp = payload.timestamp
        if timestamp.tzinfo is not None:
            return JSONResponse(
                status_code=422,
                content={"status": "invalid_timestamp", "detail": "timestamp must be timezone-naive"},
            )
        if not math.isfinite(payload.appliances):
            return JSONResponse(
                status_code=422,
                content={"status": "invalid_appliances", "detail": "appliances must be finite"},
            )

        try:
            service.ingest(pd.Timestamp(timestamp), payload.appliances)
        except DuplicateTimestampError as e:
            return JSONResponse(
                status_code=409,
                content={"status": "duplicate", "timestamp": str(e.timestamp)},
            )
        except NonSuccessorTimestampError as e:
            return JSONResponse(
                status_code=409,
                content={"status": "non_successor", "expected_next": str(e.expected_next)},
            )

        return {
            "origin_timestamp": timestamp.isoformat(),
            "buffer_ready": service.buffer.is_ready,
            "have": len(service.buffer),
            "need": service.buffer.capacity,
        }

    @app.post("/predict")
    async def predict(payload: PredictRequest, request: Request) -> Any:
        service: ServingService = request.app.state.service

        if len(payload.model_families) == 0:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "invalid_model_families",
                    "detail": "model_families must be non-empty",
                },
            )

        try:
            results = service.predict(payload.model_families)
        except InsufficientHistoryError as e:
            return JSONResponse(
                status_code=503,
                content={"status": "insufficient_history", "have": e.have, "need": e.need},
            )

        formatted = []
        for r in results:
            if "error" in r:
                formatted.append({"model_family": r["model_family"], "error": r["error"]})
                continue
            formatted.append(
                {
                    "model_family": r["model_family"],
                    "origin_timestamp": r["origin_timestamp"].isoformat(),
                    "forecast_timestamp": r["forecast_timestamp"].isoformat(),
                    "prediction_wh": r["prediction_wh"],
                    "model_version": r["model_version"],
                }
            )
        return {"results": formatted}

    @app.get("/health")
    async def health(request: Request) -> Any:
        service: ServingService = request.app.state.service
        schema = service.bundle.schema
        last = service.buffer.last_timestamp
        return {
            "model_family": schema["model_family"],
            "model_version": schema["model_version"],
            "horizon": schema["horizon"],
            "buffer_end": last.isoformat() if last is not None else None,
            "ready": service.buffer.is_ready,
            "required_raw_history": service.buffer.capacity,
            "buffer_ready": service.buffer.is_ready,
            "have": len(service.buffer),
            "need": service.buffer.capacity,
        }

    @app.get("/model")
    async def model(request: Request) -> Any:
        service: ServingService = request.app.state.service
        return {**service.bundle.schema, "available_families": service.available_families}

    return app


app = create_app()
