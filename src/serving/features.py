"""Serving-time feature row builder — runs the same canonical build_features
sequence used offline, never a reimplementation (DECISIONS.md "Phase 6:
serving registration, 2026-09-25", section 4).
"""

import numpy as np
import pandas as pd

from config.features import FEATURE_COLUMNS
from src.features.build_features import build_features


def serving_feature_row(frame: pd.DataFrame, date_column: str = "date") -> pd.Series:
    """Build the canonical feature row for the last timestamp in `frame`.

    Selects FEATURE_COLUMNS by explicit name and order — never "everything
    except targets", since build_features(..., target_horizons=[]) still
    produces baseline-only context columns (e.g. lag_138, lag_143).
    """
    features = build_features(frame, date_column=date_column, target_horizons=[])
    last_row = features.iloc[-1]
    row = last_row[FEATURE_COLUMNS]

    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("serving feature row contains non-finite values")

    return row
