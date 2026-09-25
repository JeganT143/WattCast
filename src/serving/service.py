"""The serving core: one atomic append-and-predict operation over a
RollingBuffer and a ModelBundle (DECISIONS.md "Phase 6: serving
registration, 2026-09-25", section 3).

Any failure during predict() leaves the buffer unchanged — commit only
happens after a finite prediction has been produced, so the same
timestamp can always be retried.
"""

import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.features import SCALED_COLUMNS
from src.preprocessing.scaling import transform_with_scaler
from src.serving.bundle import ModelBundle
from src.serving.buffer import InsufficientHistoryError, RollingBuffer
from src.serving.features import serving_feature_row

STEP = pd.Timedelta(minutes=10)


@dataclass(frozen=True)
class PredictionResult:
    origin_timestamp: pd.Timestamp
    forecast_timestamp: pd.Timestamp
    prediction_wh: float


class ServingService:
    def __init__(self, bundle: ModelBundle, buffer: RollingBuffer):
        self.bundle = bundle
        self.buffer = buffer
        self._lock = threading.Lock()

    def predict(self, ts: pd.Timestamp, appliances: float) -> PredictionResult:
        with self._lock:
            self.buffer.check_next(ts)

            if not self.buffer.is_ready:
                raise InsufficientHistoryError(
                    have=len(self.buffer), need=self.buffer.capacity
                )

            if not np.isfinite(appliances):
                raise ValueError("appliances value must be finite")

            frame = self.buffer.tentative_frame(ts, appliances)
            raw_row = serving_feature_row(frame)

            row_df = pd.DataFrame([raw_row])
            scaled_df = transform_with_scaler(row_df, self.bundle.scaler, SCALED_COLUMNS)
            X = scaled_df[self.bundle.forecaster.required_columns].to_numpy()

            pred = self.bundle.forecaster.predict(X)
            if pred.shape != (1,) or not np.isfinite(pred).all():
                raise ValueError("forecaster returned an invalid prediction")

            self.buffer.commit(ts, appliances)

            horizon = self.bundle.schema["horizon"]
            forecast_timestamp = ts + horizon * STEP
            return PredictionResult(
                origin_timestamp=ts,
                forecast_timestamp=forecast_timestamp,
                prediction_wh=float(pred[0]),
            )
