"""Registered feature-row and prediction equivalence between the offline
canonical pipeline and the live serving path, using the real registered
LR champion (DECISIONS.md "Phase 6: serving registration, 2026-09-25",
section 5).

Only rows with date < 2016-04-30 are ever read. Latency/API tests use
synthetic values elsewhere; this file reads real pre-boundary history
because the equivalence claim is about the real feature pipeline.
"""

import numpy as np
import pandas as pd

from config.features import FEATURE_COLUMNS, SCALED_COLUMNS, TARGET_HORIZONS
from config.mlflow_config import TRACKING_URI
from config.paths import RAW_DATA_PATH
from src.features.build_features import build_features
from src.preprocessing.scaling import transform_with_scaler
from src.serving.buffer import RollingBuffer, required_raw_history
from src.serving.features import serving_feature_row
from src.serving.registry import load_champion
from src.serving.service import ServingService

BOUNDARY = pd.Timestamp("2016-04-30")
RAW_FLOOR = 1e-12
PREDICTION_FLOOR = 1e-10
RELATIVE_TOLERANCE = 1e-12

# DECISIONS.md "Phase 6: equivalence bar amendment, 2026-09-25": the
# original 1e-12 relative/floor bar was violated only by rolling standard
# deviation (roll6_std, roll18_std), a floating-point accumulation-path
# effect (offline computes over the full series, serving over the
# 145-row buffer) rather than a serving-logic discrepancy. Scoped to
# those two columns only; every other feature bar and the prediction
# bar are unchanged.
AMENDED_STD_COLUMNS = ("Appliances_roll6_std", "Appliances_roll18_std")
AMENDED_RELATIVE_TOLERANCE = 1e-10
AMENDED_FLOOR = 1e-10


def _relative_tolerance_and_floor(column: str) -> tuple[float, float]:
    if column in AMENDED_STD_COLUMNS:
        return AMENDED_RELATIVE_TOLERANCE, AMENDED_FLOOR
    return RELATIVE_TOLERANCE, RAW_FLOOR


def _bar(reference: np.ndarray, relative_tolerance: float, floor: float) -> np.ndarray:
    return np.maximum(relative_tolerance * np.abs(reference), floor)


def _report(label: str, a: np.ndarray, b: np.ndarray, relative_tolerance: float, floor: float) -> None:
    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / np.maximum(np.abs(b), 1e-300)
    n_equal = int(np.sum(a == b))
    print(
        f"[{label}] n={len(a)} max_abs_diff={abs_diff.max():.3e} "
        f"max_rel_diff={rel_diff.max():.3e} n_exactly_equal={n_equal}"
    )
    bar = _bar(b, relative_tolerance, floor)
    assert np.all(abs_diff <= bar), (
        f"{label}: equivalence bar failed, max_abs_diff={abs_diff.max():.3e}, "
        f"floor={floor}"
    )


def _selected_positions(n_rows: int) -> np.ndarray:
    # The offline feature-row completeness threshold is index >= 144 (144
    # prior raw rows + itself = 145 = required_raw_history()). Driving the
    # live ServingService.predict() gate additionally requires the buffer
    # to already hold a full 145-row window *before* the call (its
    # steady-state invariant, set by seed_buffer/commit), i.e. one row
    # earlier than that: index >= required_raw_history().
    first = required_raw_history()
    last = n_rows - 1
    positions = np.linspace(first, last, 100).astype(int)
    return np.unique(positions)


def test_feature_and_prediction_equivalence():
    df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    pre = df_raw[df_raw["date"] < BOUNDARY].sort_values("date").reset_index(drop=True)

    df_features_offline = build_features(pre, target_horizons=TARGET_HORIZONS)

    champion = load_champion(TRACKING_URI, family="linear_regression")
    capacity = required_raw_history()

    positions = _selected_positions(len(df_features_offline))
    print(f"selected positions: n={len(positions)} first={positions[0]} last={positions[-1]}")

    offline_raw_rows = df_features_offline.loc[positions, FEATURE_COLUMNS].to_numpy(dtype=float)

    serving_raw_rows = []
    serving_predictions = []
    for pos in positions:
        history = pre.iloc[pos - capacity : pos]
        current = pre.iloc[pos]

        buffer = RollingBuffer(capacity)
        for _, row in history.iterrows():
            buffer.commit(row["date"], row["Appliances"])

        frame = buffer.tentative_frame(current["date"], current["Appliances"])
        raw_row = serving_feature_row(frame)
        serving_raw_rows.append(raw_row[FEATURE_COLUMNS].to_numpy(dtype=float))

        service = ServingService(champion, buffer)
        service.ingest(current["date"], current["Appliances"])
        result = service.predict(["linear_regression"])[0]
        serving_predictions.append(result["prediction_wh"])

    serving_raw_rows = np.array(serving_raw_rows)
    serving_predictions = np.array(serving_predictions)

    n_positions = len(positions)
    relative_matrix = np.zeros_like(offline_raw_rows)
    floor_matrix = np.zeros_like(offline_raw_rows)
    for i, col in enumerate(FEATURE_COLUMNS):
        rel, floor = _relative_tolerance_and_floor(col)
        relative_matrix[:, i] = rel
        floor_matrix[:, i] = floor

        col_abs_diff = np.abs(offline_raw_rows[:, i] - serving_raw_rows[:, i])
        col_rel_diff = col_abs_diff / np.maximum(np.abs(offline_raw_rows[:, i]), 1e-300)
        col_bar = np.maximum(rel * np.abs(offline_raw_rows[:, i]), floor)
        print(
            f"[raw feature] {col}: n={n_positions} max_abs_diff={col_abs_diff.max():.3e} "
            f"max_rel_diff={col_rel_diff.max():.3e} "
            f"n_exactly_equal={int(np.sum(offline_raw_rows[:, i] == serving_raw_rows[:, i]))} "
            f"bar={'amended 1e-10' if col in AMENDED_STD_COLUMNS else 'registered 1e-12'}"
        )
        assert np.all(col_abs_diff <= col_bar), f"raw feature {col}: equivalence bar failed"

    abs_diff = np.abs(serving_raw_rows - offline_raw_rows)
    bar = np.maximum(relative_matrix * np.abs(offline_raw_rows), floor_matrix)
    print(
        f"[raw features (flattened)] n={abs_diff.size} max_abs_diff={abs_diff.max():.3e} "
        f"n_exactly_equal={int(np.sum(serving_raw_rows == offline_raw_rows))}"
    )
    assert np.all(abs_diff <= bar), "raw features: equivalence bar failed"

    offline_scaled = transform_with_scaler(
        df_features_offline.loc[positions, FEATURE_COLUMNS].copy(), champion.scaler, SCALED_COLUMNS
    )
    X_offline = offline_scaled[champion.forecaster.required_columns].to_numpy()
    offline_predictions = champion.forecaster.predict(X_offline)

    _report(
        "predictions",
        serving_predictions,
        offline_predictions,
        RELATIVE_TOLERANCE,
        PREDICTION_FLOOR,
    )


def test_constant_window_rolling_stats():
    n_rows = 300
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="10min")
    constant_frame = pd.DataFrame({"date": dates, "Appliances": 100.0})

    offline_features = build_features(constant_frame, target_horizons=[])
    offline_last_row = offline_features.iloc[-1]

    capacity = required_raw_history()
    tail = constant_frame.iloc[-capacity:]
    buffer = RollingBuffer(capacity)
    for _, row in tail.iloc[:-1].iterrows():
        buffer.commit(row["date"], row["Appliances"])
    frame = buffer.tentative_frame(tail["date"].iloc[-1], tail["Appliances"].iloc[-1])
    serving_row = serving_feature_row(frame)

    for col in FEATURE_COLUMNS:
        offline_value = float(offline_last_row[col])
        serving_value = float(serving_row[col])
        abs_diff = abs(offline_value - serving_value)
        rel, floor = _relative_tolerance_and_floor(col)
        print(
            f"[constant-window] {col}: offline={offline_value!r} serving={serving_value!r} "
            f"abs_diff={abs_diff:.3e} "
            f"bar={'amended 1e-10' if col in AMENDED_STD_COLUMNS else 'registered 1e-12'}"
        )
        bar = max(rel * abs(offline_value), floor)
        assert abs_diff <= bar, f"constant-window equivalence bar failed for {col}"

    print(
        f"[constant-window] roll6_std offline={offline_last_row['Appliances_roll6_std']!r} "
        f"serving={serving_row['Appliances_roll6_std']!r}"
    )
    print(
        f"[constant-window] roll18_std offline={offline_last_row['Appliances_roll18_std']!r} "
        f"serving={serving_row['Appliances_roll18_std']!r}"
    )
