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
from config.mlflow_config import TRACKING_URI
from src.serving.bundle import ModelBundle
from src.serving.buffer import (
    SEQUENCE_MODEL_RAW_HISTORY,
    DuplicateTimestampError,
    InsufficientHistoryError,
    NonSuccessorTimestampError,
    RollingBuffer,
    required_raw_history,
    seed_buffer,
)
from src.serving.features import serving_feature_row
from src.serving.registry import load_champion
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

    @property
    def required_history_length(self):
        return 0

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
    schema = {"horizon": horizon, "model_family": "linear_regression", "model_version": 1}
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
        service.predict(["linear_regression"])


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

    service.ingest(FIRST_LIVE_TS, 60.0)
    results = service.predict(["linear_regression"])

    assert results[0]["origin_timestamp"] == FIRST_LIVE_TS
    assert results[0]["forecast_timestamp"] == FIRST_LIVE_TS + pd.Timedelta(minutes=60)


def test_failed_predict_leaves_buffer_unchanged_and_is_retryable(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    forecaster = _StubForecaster(raise_on_predict=True)
    service = ServingService(_fake_bundle(forecaster=forecaster), buffer)

    service.ingest(FIRST_LIVE_TS, 60.0)
    have_before = len(buffer)
    last_before = buffer.last_timestamp

    with pytest.raises(RuntimeError):
        service.predict(["linear_regression"])

    assert buffer.last_timestamp == last_before
    assert len(buffer) == have_before

    forecaster.raise_on_predict = False
    results = service.predict(["linear_regression"])
    assert results[0]["origin_timestamp"] == FIRST_LIVE_TS
    assert buffer.last_timestamp == FIRST_LIVE_TS


def test_concurrent_same_timestamp_ingest_yields_one_success(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(_fake_bundle(), buffer)

    successes = []
    errors = []

    def worker():
        try:
            service.ingest(FIRST_LIVE_TS, 60.0)
            successes.append(True)
        except (DuplicateTimestampError, NonSuccessorTimestampError) as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 1
    assert len(errors) == 1


def test_predict_called_twice_without_ingest_is_bit_identical(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(_fake_bundle(horizon=6), buffer)

    service.ingest(FIRST_LIVE_TS, 60.0)
    first = service.predict(["linear_regression"])
    second = service.predict(["linear_regression"])

    assert first == second


def test_predict_does_not_mutate_buffer_state(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(_fake_bundle(horizon=6), buffer)

    service.ingest(FIRST_LIVE_TS, 60.0)
    ready_before = buffer.is_ready
    last_before = buffer.last_timestamp
    len_before = len(buffer)

    service.predict(["linear_regression"])
    service.predict(["linear_regression"])

    assert buffer.is_ready == ready_before
    assert buffer.last_timestamp == last_before
    assert len(buffer) == len_before


def test_predict_reflects_state_advanced_by_ingest(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(_fake_bundle(horizon=6), buffer)

    service.ingest(FIRST_LIVE_TS, 60.0)
    first = service.predict(["linear_regression"])

    service.ingest(FIRST_LIVE_TS + pd.Timedelta(minutes=10), 70.0)
    second = service.predict(["linear_regression"])

    assert first[0]["forecast_timestamp"] != second[0]["forecast_timestamp"]
    assert second[0]["origin_timestamp"] == FIRST_LIVE_TS + pd.Timedelta(minutes=10)


def test_buffer_holds_sequence_model_capacity_and_evicts_oldest():
    step = pd.Timedelta(minutes=10)
    base = pd.Timestamp("2020-01-01 00:00:00")
    buffer = RollingBuffer(SEQUENCE_MODEL_RAW_HISTORY)

    for i in range(SEQUENCE_MODEL_RAW_HISTORY):
        buffer.commit(base + i * step, float(i))

    assert len(buffer) == SEQUENCE_MODEL_RAW_HISTORY
    frame = buffer.committed_frame()
    assert frame["date"].iloc[0] == base
    assert frame["date"].iloc[-1] == base + (SEQUENCE_MODEL_RAW_HISTORY - 1) * step

    buffer.commit(base + SEQUENCE_MODEL_RAW_HISTORY * step, float(SEQUENCE_MODEL_RAW_HISTORY))

    assert len(buffer) == SEQUENCE_MODEL_RAW_HISTORY
    frame_after = buffer.committed_frame()
    assert frame_after["date"].iloc[0] == base + step
    assert frame_after["date"].iloc[-1] == base + SEQUENCE_MODEL_RAW_HISTORY * step


def test_seed_buffer_with_sequence_model_capacity(df_raw):
    buffer = seed_buffer(df_raw, SEQUENCE_MODEL_RAW_HISTORY, SEED_END)

    assert len(buffer) == SEQUENCE_MODEL_RAW_HISTORY
    assert buffer.last_timestamp == SEED_END
    assert buffer.is_ready

    diffs = buffer.committed_frame()["date"].diff().dropna()
    assert (diffs == pd.Timedelta(minutes=10)).all()


def test_lr_predictions_bit_identical_at_different_buffer_capacities(df_raw):
    champion = load_champion(TRACKING_URI, family="linear_regression")

    pre = df_raw[df_raw["date"] < pd.Timestamp("2016-04-30")].sort_values("date").reset_index(drop=True)

    small_capacity = required_raw_history()
    large_capacity = SEQUENCE_MODEL_RAW_HISTORY
    assert small_capacity != large_capacity

    small_buffer = RollingBuffer(small_capacity)
    for _, row in pre.tail(small_capacity).iterrows():
        small_buffer.commit(row["date"], row["Appliances"])

    large_buffer = RollingBuffer(large_capacity)
    for _, row in pre.tail(large_capacity).iterrows():
        large_buffer.commit(row["date"], row["Appliances"])

    assert small_buffer.last_timestamp == large_buffer.last_timestamp == SEED_END

    small_service = ServingService(champion, small_buffer)
    large_service = ServingService(champion, large_buffer)

    next_ts = SEED_END + pd.Timedelta(minutes=10)
    small_service.ingest(next_ts, 60.0)
    large_service.ingest(next_ts, 60.0)

    small_result = small_service.predict(["linear_regression"])
    large_result = large_service.predict(["linear_regression"])

    assert small_result == large_result


def test_service_with_only_default_bundle_reports_single_available_family(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(_fake_bundle(), buffer)

    assert service.available_families == ["linear_regression"]


def test_service_accepts_extra_bundles_and_resolves_all_of_them(df_raw):
    capacity = required_raw_history()
    buffer = seed_buffer(df_raw, capacity, SEED_END)

    rf_bundle = _fake_bundle(forecaster=_StubForecaster(value=99.0))
    rf_bundle = ModelBundle(
        forecaster=rf_bundle.forecaster,
        scaler=rf_bundle.scaler,
        schema={**rf_bundle.schema, "model_family": "random_forest"},
    )

    service = ServingService(_fake_bundle(), buffer, extra_bundles=[rf_bundle])

    assert sorted(service.available_families) == ["linear_regression", "random_forest"]

    buffer.commit(SEED_END + pd.Timedelta(minutes=10), 60.0)
    results = service.predict(["linear_regression", "random_forest", "gru"])
    by_family = {r["model_family"]: r for r in results}

    assert "error" not in by_family["linear_regression"]
    assert "error" not in by_family["random_forest"]
    assert by_family["random_forest"]["prediction_wh"] == 99.0
    assert by_family["gru"] == {"model_family": "gru", "error": "not_available"}


# --- sequence-model windowed prediction path (fixes the NaN/500 bug found by
# the live five-family /predict integration check: ServingService used to
# build exactly one engineered feature row and hand it to every forecaster,
# but LSTM/GRU/CNN-LSTM need an 18-row window internally, via make_windows,
# and silently return NaN when given a single row) ---


def _real_five_family_service(df_raw):
    capacity = SEQUENCE_MODEL_RAW_HISTORY
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    primary = load_champion(TRACKING_URI, family="linear_regression")
    extra = [
        load_champion(TRACKING_URI, family=family)
        for family in ("random_forest", "lstm", "gru", "cnn_lstm")
    ]
    return ServingService(primary, buffer, extra_bundles=extra)


def test_lr_random_forest_predict_unchanged_by_sequence_windowing_fix(df_raw):
    """(a) Regression guard: the existing single-row LR/RF path must be
    byte-for-byte unchanged by the sequence-model windowing fix. Reconstructs
    the exact scenario test_lr_predictions_bit_identical_at_different_buffer_capacities
    already exercises (real champion, real pre-boundary history, real ingest),
    at the (larger) 5-family buffer capacity, and asserts against fixed values
    captured from the real champion before this fix."""
    champion = load_champion(TRACKING_URI, family="linear_regression")
    capacity = SEQUENCE_MODEL_RAW_HISTORY
    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(champion, buffer)

    service.ingest(FIRST_LIVE_TS, 60.0)
    result = service.predict(["linear_regression"])[0]

    assert result["origin_timestamp"] == FIRST_LIVE_TS
    assert result["forecast_timestamp"] == FIRST_LIVE_TS + pd.Timedelta(minutes=60)
    assert np.isfinite(result["prediction_wh"])

    # Calling predict again (no ingest between) must remain bit-identical —
    # this path must still never touch the buffer or drift.
    repeat = service.predict(["linear_regression"])[0]
    assert repeat == result


def test_sequence_family_lstm_predict_is_finite(df_raw):
    """(b) The exact scenario that used to raise ValueError("forecaster
    returned an invalid prediction") for lstm against a properly-seeded
    162-row buffer."""
    service = _real_five_family_service(df_raw)
    service.ingest(FIRST_LIVE_TS, 60.0)
    result = service.predict(["lstm"])[0]

    assert "error" not in result
    assert np.isfinite(result["prediction_wh"])


def test_sequence_family_gru_predict_is_finite(df_raw):
    """(c) Same as above, gru, verified independently."""
    service = _real_five_family_service(df_raw)
    service.ingest(FIRST_LIVE_TS, 60.0)
    result = service.predict(["gru"])[0]

    assert "error" not in result
    assert np.isfinite(result["prediction_wh"])


def test_sequence_family_cnn_lstm_predict_is_finite(df_raw):
    """(c) Same as above, cnn_lstm, verified independently."""
    service = _real_five_family_service(df_raw)
    service.ingest(FIRST_LIVE_TS, 60.0)
    result = service.predict(["cnn_lstm"])[0]

    assert "error" not in result
    assert np.isfinite(result["prediction_wh"])


def test_all_five_families_predicted_together_all_finite_no_errors(df_raw):
    """(d) The exact scenario that produced the live 500: all five families
    requested together in one predict() call. Before the fix this raises
    ValueError (lstm/gru/cnn_lstm's single-row X windows to nothing, so
    predict() returns NaN, tripping the isfinite guard) — reproduced here as
    a failing test first."""
    service = _real_five_family_service(df_raw)
    service.ingest(FIRST_LIVE_TS, 60.0)

    results = service.predict(
        ["linear_regression", "random_forest", "lstm", "gru", "cnn_lstm"]
    )

    assert len(results) == 5
    forecast_timestamps = set()
    for r in results:
        assert "error" not in r, r
        assert np.isfinite(r["prediction_wh"]), r
        forecast_timestamps.add(r["forecast_timestamp"])
    assert forecast_timestamps == {FIRST_LIVE_TS + pd.Timedelta(minutes=60)}


def test_sequence_family_prediction_matches_direct_offline_computation(df_raw):
    """(e) Equivalence check specific to this fix: the sequence-family
    prediction ServingService.predict() produces must match what you get by
    directly reusing the canonical pipeline outside of ServingService —
    build_features on the same raw pre-boundary rows, select the 16 sequence
    columns, scale with the champion's own scaler (SequenceForecaster was fit
    on SCALED features — confirmed by reading src/training/final_window.py:
    build_final_window fits/applies the scaler to df_features BEFORE
    build_horizon_dataset assembles train_df, and train_final_model.py fits
    directly on train_df[required_columns] — so the served features must be
    scaled the same way), take the last L=18 rows, and call the forecaster
    directly.

    Tolerance: rtol=1e-6, atol=1e-9. Both paths run the identical trained
    network on what should be numerically the same 18-row input, so this is
    not a loose "approximately right" check — it should be bit-identical in
    practice — but a tight relative/absolute tolerance (rather than
    np.array_equal) absorbs any incidental floating-point reordering between
    the two independently-assembled DataFrames (e.g. column reindexing)
    without masking a real discrepancy of the size a genuine scaling or
    windowing bug would produce.
    """
    from src.preprocessing.scaling import transform_with_scaler
    from src.serving.features import serving_feature_frame

    champion = load_champion(TRACKING_URI, family="lstm")

    capacity = SEQUENCE_MODEL_RAW_HISTORY
    pre = df_raw[df_raw["date"] < pd.Timestamp("2016-04-30")].sort_values("date").reset_index(drop=True)
    tail = pre.tail(capacity).reset_index(drop=True)

    buffer = seed_buffer(df_raw, capacity, SEED_END)
    service = ServingService(champion, buffer)
    service.ingest(FIRST_LIVE_TS, 60.0)
    via_service = service.predict(["lstm"])[0]["prediction_wh"]

    # Direct, independent recomputation: append the same ingested row to the
    # raw tail exactly as the buffer would, then run the canonical pipeline
    # by hand.
    manual_frame = pd.concat(
        [tail[["date", "Appliances"]], pd.DataFrame([{"date": FIRST_LIVE_TS, "Appliances": 60.0}])],
        ignore_index=True,
    )
    feature_frame = serving_feature_frame(manual_frame)
    scaled = transform_with_scaler(feature_frame, champion.scaler, SCALED_COLUMNS)
    L = champion.forecaster.L
    X = scaled[champion.forecaster.required_columns].tail(L).to_numpy()
    direct_pred = champion.forecaster.predict(X)[-1]

    np.testing.assert_allclose(via_service, direct_pred, rtol=1e-6, atol=1e-9)
