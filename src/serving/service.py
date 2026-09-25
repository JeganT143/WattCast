"""The serving core, split into a state-mutating ingest and a read-only
predict (Phase 6 follow-on: /ingest + /predict contract split).

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
from src.serving.buffer import InsufficientHistoryError, RollingBuffer
from src.serving.features import serving_feature_row

STEP = pd.Timedelta(minutes=10)


class ServingService:
    def __init__(self, bundle: ModelBundle, buffer: RollingBuffer):
        self.bundle = bundle
        self.buffer = buffer
        self._bundles = {bundle.schema["model_family"]: bundle}
        self._lock = threading.Lock()

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

            results = []
            for family in model_families:
                bundle = self._bundles.get(family)
                if bundle is None:
                    results.append({"model_family": family, "error": "not_available"})
                    continue

                scaled_df = transform_with_scaler(row_df, bundle.scaler, SCALED_COLUMNS)
                X = scaled_df[bundle.forecaster.required_columns].to_numpy()

                pred = bundle.forecaster.predict(X)
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
