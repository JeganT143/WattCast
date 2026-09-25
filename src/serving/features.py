"""Serving-time feature row builder — runs the same canonical build_features
sequence used offline, never a reimplementation (DECISIONS.md "Phase 6:
serving registration, 2026-09-25", section 4).
"""

import numpy as np
import pandas as pd

from config.features import FEATURE_COLUMNS
from src.features.build_features import build_features


def serving_feature_frame(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Build the canonical engineered feature frame for every row `frame` can
    produce — including leading rows whose lag_144 (or other lookback) isn't
    yet finite for short history. Callers select the rows/columns they need
    and validate finiteness themselves; this is the single canonical
    feature-engineering call site, shared by the single-row path
    (serving_feature_row) and the multi-row windowed path sequence models
    need (ServingService.predict).

    Selects FEATURE_COLUMNS by explicit name and order — never "everything
    except targets", since build_features(..., target_horizons=[]) still
    produces baseline-only context columns (e.g. lag_138, lag_143).
    """
    features = build_features(frame, date_column=date_column, target_horizons=[])
    return features[FEATURE_COLUMNS]


def serving_feature_row(frame: pd.DataFrame, date_column: str = "date") -> pd.Series:
    """Build the canonical feature row for the last timestamp in `frame`."""
    row = serving_feature_frame(frame, date_column=date_column).iloc[-1]

    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("serving feature row contains non-finite values")

    return row
