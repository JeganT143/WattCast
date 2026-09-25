"""Tests for the pure, framework-free serving core: rolling buffer,
serving feature builder, and ServingService (DECISIONS.md "Phase 6:
serving registration, 2026-09-25", sections 3-4)."""

import threading

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS
from config.paths import RAW_DATA_PATH
from src.serving.bundle import ModelBundle
from src.serving.buffer import (
    DuplicateTimestampError,
    InsufficientHistoryError,
    NonSuccessorTimestampError,
    RollingBuffer,
    required_raw_history,
    seed_buffer,
)
from src.serving.features import serving_feature_row
from src.serving.service import ServingService

SEED_END = pd.Timestamp("2016-04-29 23:50:00")
FIRST_LIVE_TS = pd.Timestamp("2016-04-30 00:00:00")


@pytest.fixture(scope="module")
def df_raw():
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])


class _StubForecaster:
    def __init__(self, value: float = 42.0, raise_on_predict: bool = False):
        self.value = value
        self.raise_on_predict = raise_on_predict
        self.calls = 0

    @property
    def required_columns(self):
        return FEATURE_COLUMNS

    def predict(self, X):
        self.calls += 1
        if self.raise_on_predict:
            raise RuntimeError("boom")
        return np.array([self.value])


def _fake_bundle(horizon: int = 6, forecaster=None) -> ModelBundle:
    if forecaster is None:
        forecaster = _StubForecaster()
    rng = np.random.default_rng(0)
    fit_frame = pd.DataFrame(
        rng.uniform(0, 500, size=(50, len(SCALED_COLUMNS))), columns=SCALED_COLUMNS
    )
    scaler = StandardScaler().fit(fit_frame)
    schema = {"horizon": horizon}
    return ModelBundle(forecaster=forecaster, scaler=scaler, schema=schema)


def test_required_raw_history_default_is_145():
    assert required_raw_history() == 145


def test_required_raw_history_changes_with_inputs():
    default = required_raw_history()
    assert required_raw_history(lag_steps=[1, 2], rolling_windows=[3]) != default
    assert required_raw_history(lag_steps=[1, 2], rolling_windows=[3]) == 4


def test_seed_buffer_real_history(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)

    assert len(buffer) == capacity
    assert buffer.last_timestamp == SEED_END
    assert buffer.is_ready


def test_seed_buffer_rejects_wrong_seed_end(df_raw):
    capacity = required_raw_history()
    with pytest.raises(ValueError):
        seed_buffer(df_raw, capacity, SEED_END - pd.Timedelta(minutes=10))


def test_seed_buffer_insufficient_history_raises(df_raw):
    with pytest.raises(InsufficientHistoryError):
        seed_buffer(df_raw, capacity=100000, seed_end=SEED_END)


def test_first_accepted_timestamp_after_seed(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    buffer.check_next(FIRST_LIVE_TS)  # must not raise


def test_duplicate_timestamp_rejected(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    with pytest.raises(DuplicateTimestampError):
        buffer.check_next(SEED_END)


def test_non_successor_timestamp_rejected_gap(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    with pytest.raises(NonSuccessorTimestampError):
        buffer.check_next(SEED_END + pd.Timedelta(minutes=20))


def test_non_successor_timestamp_rejected_earlier(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    with pytest.raises(NonSuccessorTimestampError):
        buffer.check_next(SEED_END - pd.Timedelta(minutes=10))


def test_short_buffer_insufficient_history(df_raw):
    capacity = required_raw_history()
    pre = df_raw[df_raw["date"] < pd.Timestamp("2016-04-30")].sort_values("date")
    short_tail = pre.tail(100)

    buffer = RollingBuffer(capacity)
    for _, row in short_tail.iterrows():
        buffer.commit(row["date"], row["Appliances"])

    assert not buffer.is_ready
    service = ServingService(_fake_bundle(), buffer)
    with pytest.raises(InsufficientHistoryError):
        service.predict(short_tail["date"].iloc[-1] + pd.Timedelta(minutes=10), 60.0)


def test_serving_feature_row_columns(df_raw):
    pre = df_raw[df_raw["date"] < pd.Timestamp("2016-04-30")].sort_values("date")
    row = serving_feature_row(pre.tail(200))

    assert list(row.index) == FEATURE_COLUMNS
    assert "Appliances_lag_138" not in row.index
    assert "Appliances_lag_143" not in row.index


def test_predict_forecast_timestamp_is_origin_plus_60min(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(_fake_bundle(horizon=6), buffer)

    result = service.predict(FIRST_LIVE_TS, 60.0)

    assert result.origin_timestamp == FIRST_LIVE_TS
    assert result.forecast_timestamp == FIRST_LIVE_TS + pd.Timedelta(minutes=60)


def test_failed_predict_leaves_buffer_unchanged_and_is_retryable(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    forecaster = _StubForecaster(raise_on_predict=True)
    service = ServingService(_fake_bundle(forecaster=forecaster), buffer)

    with pytest.raises(RuntimeError):
        service.predict(FIRST_LIVE_TS, 60.0)

    assert buffer.last_timestamp == SEED_END
    assert len(buffer) == capacity

    forecaster.raise_on_predict = False
    result = service.predict(FIRST_LIVE_TS, 60.0)
    assert result.origin_timestamp == FIRST_LIVE_TS
    assert buffer.last_timestamp == FIRST_LIVE_TS


def test_concurrent_same_timestamp_yields_one_success(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(_fake_bundle(), buffer)

    results = []
    errors = []

    def worker():
        try:
            results.append(service.predict(FIRST_LIVE_TS, 60.0))
        except (DuplicateTimestampError, NonSuccessorTimestampError) as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1
    assert len(errors) == 1
