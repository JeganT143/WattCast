"""Serving model bundle: a fitted forecaster, its scaler, and the schema
that pins the feature contract they were trained against
(DECISIONS.md "Phase 6: serving registration, 2026-09-25").

Serialization uses skops, the same convention as the existing MLflow
logger (src/tracking/mlflow_logger.py): no pickle, and any untrusted
type surfaced by skops on load must be checked explicitly rather than
blanket-trusted.

required_raw_history lives here (not in src/serving/buffer.py, which
needs it too) because the bundle schema must be able to record it at
training time, before the serving buffer module exists at runtime;
buffer.py imports it from here rather than redefining it.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import skops.io as sio
from sklearn.preprocessing import StandardScaler

from config.features import FEATURE_COLUMNS, LAG_STEPS, ROLLING_WINDOWS, SCALED_COLUMNS
from src.models.forecaster import Forecaster
from src.models.sklearn_models import LinearRegressionForecaster


def required_raw_history(
    lag_steps: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> int:
    """Minimum contiguous raw observations needed to produce one feature row."""
    if lag_steps is None:
        lag_steps = LAG_STEPS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS
    return max(max(lag_steps), max(rolling_windows)) + 1


def feature_schema_version(feature_columns: list[str]) -> str:
    """First 12 hex chars of the sha256 of the ordered feature columns, comma-joined."""
    digest = hashlib.sha256(",".join(feature_columns).encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass(frozen=True)
class ModelBundle:
    forecaster: Forecaster
    scaler: StandardScaler
    schema: dict


def _dump_skops(obj, path: Path) -> None:
    sio.dump(obj, path)


def _load_skops_trusted(path: Path):
    untrusted = sio.get_untrusted_types(file=path)
    if untrusted:
        raise ValueError(
            f"skops file {path} contains untrusted types not explicitly reviewed: {untrusted}"
        )
    return sio.load(path, trusted=[])


def save_bundle(bundle: ModelBundle, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _dump_skops(bundle.forecaster.model, directory / "estimator.skops")
    _dump_skops(bundle.scaler, directory / "scaler.skops")
    with open(directory / "schema.json", "w") as f:
        json.dump(bundle.schema, f, indent=2, sort_keys=True)


def _load_linear_regression(directory: Path, schema: dict) -> Forecaster:
    estimator = _load_skops_trusted(directory / "estimator.skops")
    forecaster = LinearRegressionForecaster(fit_intercept=bool(estimator.fit_intercept))
    forecaster.model = estimator
    return forecaster


LOADERS: dict[str, Callable[[Path, dict], Forecaster]] = {
    "linear_regression": _load_linear_regression,
}


def load_bundle(directory: Path) -> ModelBundle:
    directory = Path(directory)
    with open(directory / "schema.json") as f:
        schema = json.load(f)

    if schema["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            "bundle schema feature_columns differs from config/features.py FEATURE_COLUMNS"
        )
    if schema["scaled_columns"] != SCALED_COLUMNS:
        raise ValueError(
            "bundle schema scaled_columns differs from config/features.py SCALED_COLUMNS"
        )

    model_family = schema["model_family"]
    if model_family not in LOADERS:
        raise ValueError(f"no loader registered for model_family={model_family!r}")

    forecaster = LOADERS[model_family](directory, schema)
    scaler = _load_skops_trusted(directory / "scaler.skops")

    return ModelBundle(forecaster=forecaster, scaler=scaler, schema=schema)
