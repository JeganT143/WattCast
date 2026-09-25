"""FastAPI adapter over ServingService (decisions.md, ADR-014 and ADR-016).

Routes depend only on ServingService and each bundle's schema dict — there
is no model class, model path, or model_family branching here.

Importing this module loads nothing: the bundles in models/ and the seed
history are read inside the app's lifespan. Run with:

    uvicorn src.serving.app:app
"""

import math
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.paths import MODELS_DIR, SEED_HISTORY_PATH
from src.serving.bundle import load_bundles
from src.serving.buffer import (
    DuplicateTimestampError,
    InsufficientHistoryError,
    NonSuccessorTimestampError,
)
from src.serving.service import ServingService, create_service


class IngestRequest(BaseModel):
    timestamp: datetime
    appliances: float


class PredictRequest(BaseModel):
    model_families: list[str]


def _default_load() -> ServingService:
    bundles = load_bundles(MODELS_DIR)
    seed_history = pd.read_csv(SEED_HISTORY_PATH, parse_dates=["date"])
    return create_service(bundles, seed_history)


def create_app(load: Callable[[], ServingService] = _default_load) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = load()
        yield

    app = FastAPI(title="WattCast forecasting API", lifespan=lifespan)

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
            "model_version": schema.get("model_version"),
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
