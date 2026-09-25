"""The serving core, split into a state-mutating ingest and a read-only
predict (decisions.md, ADR-014).

ingest() is the only operation that ever advances the RollingBuffer:
validate the successor/duplicate timestamp rule and the finite-value
rule, then commit — no prediction happens here. predict() never mutates
the buffer: it builds the feature row once from the currently committed
state and runs it through each requested model_family's forecaster, so
multiple families see identical input and repeated calls without an
intervening ingest are bit-identical. An unregistered model_family
never raises or fails the whole request — it appears in the results
list as {"model_family": ..., "error": "not_available"}.
"""

import threading

import numpy as np
import pandas as pd

from config.features import SCALED_COLUMNS
from src.preprocessing.scaling import transform_with_scaler
from src.serving.bundle import ModelBundle
from src.serving.buffer import (
    SEQUENCE_MODEL_RAW_HISTORY,
    InsufficientHistoryError,
    RollingBuffer,
    seed_buffer,
)
from src.serving.features import serving_feature_frame, serving_feature_row

STEP = pd.Timedelta(minutes=10)
PRIMARY_FAMILY = "linear_regression"


class ServingService:
    def __init__(
        self,
        bundle: ModelBundle,
        buffer: RollingBuffer,
        extra_bundles: list[ModelBundle] | None = None,
    ):
        self.bundle = bundle
        self.buffer = buffer
        self._bundles = {bundle.schema["model_family"]: bundle}
        for extra in extra_bundles or []:
            self._bundles[extra.schema["model_family"]] = extra
        self._lock = threading.Lock()

    @property
    def available_families(self) -> list[str]:
        return list(self._bundles)

    def ingest(self, ts: pd.Timestamp, appliances: float) -> None:
        with self._lock:
            self.buffer.check_next(ts)
            if not np.isfinite(appliances):
                raise ValueError("appliances value must be finite")
            self.buffer.commit(ts, appliances)

    def predict(self, model_families: list[str]) -> list[dict]:
        with self._lock:
            if not self.buffer.is_ready:
                raise InsufficientHistoryError(
                    have=len(self.buffer), need=self.buffer.capacity
                )

            frame = self.buffer.committed_frame()
            raw_row = serving_feature_row(frame)
            row_df = pd.DataFrame([raw_row])
            origin_timestamp = self.buffer.last_timestamp

            # Built lazily, only if a requested family actually needs it: a
            # windowed multi-row feature frame for sequence models
            # (LSTM/GRU/CNN-LSTM), which — unlike LR/RF — need
            # required_history_length preceding rows to produce a non-NaN
            # prediction (SequenceForecaster.predict windows internally via
            # make_windows(X, L); handing it a single row always yields NaN).
            windowed_frame = None

            results = []
            for family in model_families:
                bundle = self._bundles.get(family)
                if bundle is None:
                    results.append({"model_family": family, "error": "not_available"})
                    continue

                window_size = bundle.forecaster.required_history_length + 1
                if window_size > 1:
                    if windowed_frame is None:
                        windowed_frame = serving_feature_frame(frame)
                    window_raw = windowed_frame.tail(window_size)
                    if not np.isfinite(window_raw.to_numpy(dtype=float)).all():
                        raise ValueError("serving feature window contains non-finite values")
                    scaled = transform_with_scaler(window_raw, bundle.scaler, SCALED_COLUMNS)
                    X = scaled[bundle.forecaster.required_columns].to_numpy()
                else:
                    scaled_df = transform_with_scaler(row_df, bundle.scaler, SCALED_COLUMNS)
                    X = scaled_df[bundle.forecaster.required_columns].to_numpy()

                pred_arr = bundle.forecaster.predict(X)
                pred = pred_arr[-1:]
                if pred.shape != (1,) or not np.isfinite(pred).all():
                    raise ValueError("forecaster returned an invalid prediction")

                horizon = bundle.schema["horizon"]
                results.append(
                    {
                        "model_family": family,
                        "origin_timestamp": origin_timestamp,
                        "forecast_timestamp": origin_timestamp + horizon * STEP,
                        "prediction_wh": float(pred[0]),
                        "model_version": bundle.schema.get("model_version"),
                    }
                )
            return results


def create_service(
    bundles: dict[str, ModelBundle],
    seed_history: pd.DataFrame,
    primary_family: str = PRIMARY_FAMILY,
) -> ServingService:
    """Builds a ready-to-serve ServingService: every bundle is registered,
    and a fresh buffer is seeded from pre-test history ending at the primary
    bundle's seed_end. The buffer is sized for the sequence models, the
    largest history requirement of any served family."""
    if primary_family not in bundles:
        raise ValueError(f"primary family {primary_family!r} has no bundle")
    primary = bundles[primary_family]
    buffer = seed_buffer(seed_history, SEQUENCE_MODEL_RAW_HISTORY, primary.schema["seed_end"])
    extra = [bundle for family, bundle in bundles.items() if family != primary_family]
    return ServingService(primary, buffer, extra_bundles=extra)
